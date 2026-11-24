"""PyTorch datasets for the 3D-LDM: preprocessed 128^3 NIfTI scans and cached latents.

Every dataset refuses records outside ``allowed_splits`` (train/val). Test subjects
are never loadable here, and the stored z-score intensity domain is kept verbatim.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.dataset import load_volume as _load_classifier_volume

from .common import CONCEPTS

DEV_SPLITS = ("train", "val")


def check_records(records, allowed=DEV_SPLITS):
    if not records:
        raise ValueError("Empty record list")
    for r in records:
        if r["split"] not in allowed or r["split"] == "test":
            raise ValueError(f"Record {r['subject_id']} is in forbidden split {r['split']!r}")
    return records


def volume_config(data_cfg):
    """Map the diffusion data section onto the classifier ``load_volume`` contract."""
    keys = ("expected_shape", "expected_spacing", "spacing_tolerance", "orientation",
            "normalization", "reference_image", "reference_grid_mode",
            "reference_fov_tolerance_mm")
    return {k: data_cfg.get(k) for k in keys if k in data_cfg}


def load_scan(path, data_cfg):
    """Return (float32 array in stored z-score domain, affine). No clipping or rescaling."""
    array, affine = _load_classifier_volume(path, volume_config(data_cfg))
    return array, affine


def load_mask(path, expected_shape):
    image = nib.load(str(path))
    if tuple(image.shape) != tuple(expected_shape):
        raise ValueError(f"Wrong mask shape for {path}: {image.shape}")
    mask = np.asarray(image.dataobj) > 0
    if not mask.any():
        raise ValueError(f"Empty brain mask: {path}")
    return mask


class MorphologyEncoder:
    """Standardize six measurements with Train-CN medians and robust scales.

    ``encode`` returns a 12-vector: six clipped z-values (0 when unavailable) and
    six availability flags. With no table, every token is the all-zero vector,
    i.e. class-only conditioning.
    """

    def __init__(self, median=None, scale=None):
        self.median = {k: float(v) for k, v in (median or {}).items()}
        self.scale = {k: float(v) for k, v in (scale or {}).items()}

    @classmethod
    def fit(cls, table, records):
        """table: {image_id: {concept: value}}; fit on Train-CN records only."""
        median, scale = {}, {}
        for c in CONCEPTS:
            values = [table[r["image_id"]][c] for r in records
                      if r["split"] == "train" and r["label"] == 0
                      and r["image_id"] in table and np.isfinite(table[r["image_id"]].get(c, np.nan))]
            if len(values) >= 5:
                q1, med, q3 = np.percentile(values, [25, 50, 75])
                median[c], scale[c] = float(med), float(max((q3 - q1) / 1.349, 1e-6))
        return cls(median, scale)

    def encode(self, values=None):
        out = np.zeros(2 * len(CONCEPTS), dtype=np.float32)
        if not values:
            return out
        for i, c in enumerate(CONCEPTS):
            v = values.get(c)
            if v is None or not np.isfinite(v) or c not in self.median:
                continue
            out[i] = float(np.clip((v - self.median[c]) / self.scale[c], -5, 5))
            out[len(CONCEPTS) + i] = 1.0
        return out

    def state_dict(self):
        return {"median": self.median, "scale": self.scale}


def read_morphometry(path):
    """Read image_id,HA,ECT,VE,THD,GCT,AA. Empty cells mean measurement_unavailable."""
    if not path:
        return {}
    table = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            iid = str(int(row["image_id"].strip().removeprefix("I")))
            table[iid] = {c: (float(row[c]) if row.get(c, "") not in ("", None) else float("nan"))
                          for c in CONCEPTS}
    return table


class DiffusionVolumeDataset(Dataset):
    """Scan + companion brain mask + class label (+ optional morphology token input)."""

    def __init__(self, records, data_cfg, morph_table=None, encoder=None, allowed=DEV_SPLITS):
        self.records = check_records(list(records), allowed)
        self.data_cfg = data_cfg
        self.table = morph_table or {}
        self.encoder = encoder or MorphologyEncoder()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        volume, _ = load_scan(r["volume_path"], self.data_cfg)
        brain = load_mask(r["mask_path"], self.data_cfg["expected_shape"])
        return {
            "image": torch.from_numpy(volume).unsqueeze(0).float(),
            "brain": torch.from_numpy(brain.astype(np.float32)).unsqueeze(0),
            "label": int(r["label"]),
            "morph": torch.from_numpy(self.encoder.encode(self.table.get(r["image_id"]))),
            "subject_id": r["subject_id"], "image_id": r["image_id"], "split": r["split"],
        }


class LatentDataset(Dataset):
    """Cached scaled latents ``[4,16,16,16]`` with labels and morphology inputs."""

    def __init__(self, records, cache_dir, morph_table=None, encoder=None, allowed=DEV_SPLITS):
        self.records = check_records(list(records), allowed)
        self.cache_dir = Path(cache_dir)
        self.table = morph_table or {}
        self.encoder = encoder or MorphologyEncoder()

    def path(self, r):
        return self.cache_dir / r["split"] / f"{r['subject_id']}_{r['image_id']}.npy"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        z = np.load(self.path(r), allow_pickle=False)
        if not np.isfinite(z).all():
            raise ValueError(f"Nonfinite cached latent: {self.path(r)}")
        return {
            "latent": torch.from_numpy(z).float(), "label": int(r["label"]),
            "morph": torch.from_numpy(self.encoder.encode(self.table.get(r["image_id"]))),
            "subject_id": r["subject_id"], "image_id": r["image_id"],
        }


class SubjectRoundSampler(Sampler):
    """Patient-balanced epochs: one visit per subject per round, visits rotate between rounds."""

    def __init__(self, records, seed=42, length=None):
        self.by_subject = {}
        for i, r in enumerate(records):
            self.by_subject.setdefault(r["subject_id"], []).append(i)
        self.length = length or len(records)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.length

    def __iter__(self):
        rng = random.Random(self.seed * 100003 + self.epoch)
        order = {s: rng.sample(v, len(v)) for s, v in self.by_subject.items()}
        pointer = dict.fromkeys(order, 0)
        out = []
        while len(out) < self.length:
            subjects = list(order)
            rng.shuffle(subjects)
            for s in subjects:
                out.append(order[s][pointer[s] % len(order[s])])
                pointer[s] += 1
                if len(out) == self.length:
                    break
        return iter(out)
