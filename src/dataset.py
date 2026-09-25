"""Fixed subject splits and model-ready NIfTI loading; no registration or skull stripping."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import LABEL_TO_INDEX, REQUIRED_MANIFEST_COLUMNS, SPLIT_NAMES


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_index(data_config, resolve_splits=SPLIT_NAMES):
    """Validate the entire partition definition; resolve images only for requested splits.

    A filename template may contain {subject_id} and {image_id}, plus glob wildcards.
    Zero or multiple matches are errors: choosing the first scan would hide ambiguity.
    """
    frame = pd.read_csv(data_config["manifest_path"], dtype=str, keep_default_na=False)
    missing = REQUIRED_MANIFEST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    for col in REQUIRED_MANIFEST_COLUMNS:
        frame[col] = frame[col].str.strip()
        if (frame[col] == "").any():
            raise ValueError(f"Empty manifest values in {col}")
    frame["image_id"] = frame.image_id.str.replace(r"^I", "", regex=True)
    if not frame.image_id.str.fullmatch(r"[0-9]+").all():
        raise ValueError("image_id must contain integer ADNI IDs (optional I prefix)")
    for col in ("subject_id", "image_id"):
        if frame[col].duplicated().any():
            raise ValueError(f"Duplicate {col} in baseline manifest")
    if not set(frame.diagnosis).issubset(LABEL_TO_INDEX):
        raise ValueError("Unknown diagnosis in manifest")
    with Path(data_config["splits_path"]).open() as handle:
        splits = json.load(handle)
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError("Split JSON must contain exactly train, val, test")
    membership = {}
    for split, ids in splits.items():
        if not isinstance(ids, list) or not all(
            isinstance(s, str) and s == s.strip() for s in ids
        ):
            raise ValueError(f"Invalid subject list for {split}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate subject in {split}")
        if len(ids) != data_config["expected_split_sizes"][split]:
            raise ValueError(f"Unexpected {split} size: {len(ids)}")
        for subject in ids:
            if subject in membership:
                raise ValueError(f"Subject overlap across splits: {subject}")
            membership[subject] = split
    if set(membership) != set(frame.subject_id):
        raise ValueError("Split subjects do not exactly cover the manifest")
    frame["split"] = frame.subject_id.map(membership)
    frame["label"] = frame.diagnosis.map(LABEL_TO_INDEX)
    for split in SPLIT_NAMES:
        if set(frame.loc[frame.split == split, "diagnosis"]) != set(LABEL_TO_INDEX):
            raise ValueError(f"Primary {split} partition must contain all four classes")
    frame["path"] = ""
    root = Path(data_config["volume_dir"]).resolve()
    paths = set()
    for i, row in frame.iterrows():
        if row["split"] not in resolve_splits:
            continue
        pattern = data_config["file_pattern"].format(**row.to_dict())
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError("file_pattern must remain inside volume_dir")
        matches = [
            p.resolve()
            for p in root.glob(pattern)
            if p.is_file() and str(p).endswith((".nii", ".nii.gz"))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{row.subject_id}: expected one volume, found {len(matches)} for {pattern}"
            )
        path = matches[0]
        if not path.is_relative_to(root) or path in paths:
            raise ValueError(f"Shared volume or path outside volume_dir: {path}")
        paths.add(path)
        frame.at[i, "path"] = str(path)
    return frame


def load_volume(path, data_config):
    """Validate geometry and normalize foreground values, preserving zero background.

    Nonzero foreground is only a normalization mask; it does not establish brain extraction.
    Geometry matching also does not prove anatomical registration. External visual QC is required.
    """
    image = nib.load(str(path))
    if image.shape != tuple(data_config["expected_shape"]):
        raise ValueError(f"Wrong shape for {path}: {image.shape}")
    if not np.allclose(
        image.header.get_zooms()[:3],
        data_config["expected_spacing"],
        atol=data_config["spacing_tolerance"],
        rtol=0,
    ):
        raise ValueError(f"Wrong voxel spacing: {path}")
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError(f"NIfTI spatial units must explicitly be mm: {path}")
    if (
        not np.isfinite(image.affine).all()
        or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8
    ):
        raise ValueError(f"Invalid affine: {path}")
    orientation = "".join(nib.aff2axcodes(image.affine))
    if orientation != data_config["orientation"]:
        raise ValueError(f"Wrong orientation for {path}: {orientation}")
    reference = data_config.get("reference_image")
    if reference:
        template = nib.load(str(reference))
        if template.shape != image.shape or not np.allclose(
            template.affine, image.affine, atol=1e-3, rtol=0
        ):
            raise ValueError(f"Volume does not match reference grid: {path}")
    array = image.get_fdata(dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"NaN or Inf voxel in {path}")
    foreground = array != 0
    values = array[foreground]
    if values.size < 2 or float(np.ptp(values)) < 1e-8:
        raise ValueError(f"Empty or constant foreground: {path}")
    method = data_config["normalization"]
    if method == "minmax":
        array[foreground] = (values - values.min()) / np.ptp(values)
    elif method == "zscore":
        array[foreground] = (values - values.mean()) / max(float(values.std()), 1e-8)
    elif method != "none":
        raise ValueError(f"Unknown normalization: {method}")
    return np.ascontiguousarray(array), image.affine.copy()


class MRIDataset(Dataset):
    """One MRI and manifest label per subject; augmentation is strictly train-only."""

    def __init__(self, index, split, data_config, transform=None):
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split: {split}")
        if split != "train" and transform is not None:
            raise ValueError("Augmentation is allowed only on train")
        self.records = index[index.split == split].to_dict("records")
        self.data_config = data_config
        self.transform = transform
        if not self.records:
            raise ValueError(f"Empty dataset for {split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        volume, affine = load_volume(record["path"], self.data_config)
        tensor = torch.from_numpy(volume).unsqueeze(0)
        if self.transform is not None:
            import torchio as tio

            tensor = self.transform(tio.ScalarImage(tensor=tensor, affine=affine)).data
        return {**record, "image": tensor.float(), "label": int(record["label"])}


def audit_dataset(index, data_config, splits=SPLIT_NAMES):
    """Load every requested image and record failures and content hashes without skipping them."""
    report = {
        "counts": index.groupby("split").size().to_dict(),
        "class_counts": {
            s: index[index.split == s].diagnosis.value_counts().to_dict()
            for s in SPLIT_NAMES
        },
        "manifest_sha256": sha256_file(data_config["manifest_path"]),
        "splits_sha256": sha256_file(data_config["splits_path"]),
        "reference_sha256": sha256_file(data_config["reference_image"])
        if data_config.get("reference_image")
        else None,
        "preprocessing_confirmed": data_config.get("preprocessing_confirmed", False),
        "audited_splits": list(splits),
        "volumes": [],
        "errors": [],
    }
    selected = index[index.split.isin(splits)].to_dict("records")
    for number, row in enumerate(selected, 1):
        try:
            load_volume(row["path"], data_config)
            report["volumes"].append(
                {"subject_id": row["subject_id"], "sha256": sha256_file(row["path"])}
            )
        except Exception as exc:
            report["errors"].append(
                {"subject_id": row["subject_id"], "error": str(exc)}
            )
        if number % 100 == 0:
            print(f"Audited {number}/{len(selected)} volumes", flush=True)
    report["passed"] = not report["errors"]
    return report
