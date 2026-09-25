import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from src.dataset import MRIDataset, audit_dataset, build_index, load_volume
from src.transforms import build_transform


def test_dataset_shapes_and_labels(config):
    index = build_index(config["data"])
    dataset = MRIDataset(index, "train", config["data"])
    item = dataset[0]
    assert item["image"].shape == (1, 16, 16, 16)
    assert item["image"].dtype == torch.float32
    assert 0 <= item["image"].min() <= item["image"].max() <= 1
    assert item["label"] == 0
    assert torch.equal(
        MRIDataset(index, "val", config["data"])[0]["image"],
        MRIDataset(index, "val", config["data"])[0]["image"],
    )
    assert audit_dataset(index, config["data"])["passed"]


@pytest.mark.parametrize(
    "problem",
    [
        "shape",
        "spacing",
        "nan",
        "inf",
        "empty",
        "constant",
        "units",
        "orientation",
        "affine",
    ],
)
def test_bad_volumes_rejected(config, problem, tmp_path):
    path = Path(config["data"]["volume_dir"]) / "synthetic_000.nii.gz"
    old = nib.load(path)
    array, affine = old.get_fdata(dtype=np.float32), old.affine.copy()
    if problem == "shape":
        array = array[:-1]
    elif problem == "spacing":
        affine[0, 0] = 1.0
    elif problem in {"nan", "inf"}:
        array[0, 0, 0] = float(problem)
    elif problem == "empty":
        array[:] = 0
    elif problem == "constant":
        array[:] = 1
    elif problem == "orientation":
        affine[0, 0] = -2.0
    elif problem == "affine":
        reference = tmp_path / "ref.nii.gz"
        nib.save(old, reference)
        config["data"]["reference_image"] = str(reference)
        affine[0, 3] = 20
    new = nib.Nifti1Image(array, affine)
    new.header.set_xyzt_units("unknown" if problem == "units" else "mm")
    nib.save(new, path)
    with pytest.raises(ValueError):
        load_volume(path, config["data"])


def test_no_augmentation_outside_train(config):
    index = build_index(config["data"])
    config["augmentation"]["enabled"] = True
    transform = build_transform(config["augmentation"])
    with pytest.raises(ValueError, match="train"):
        MRIDataset(index, "test", config["data"], transform)
    item = MRIDataset(index, "train", config["data"], transform)[0]
    assert item["image"].shape == (1, 16, 16, 16)
    assert torch.isfinite(item["image"]).all()


def test_missing_and_ambiguous_files(config):
    root = Path(config["data"]["volume_dir"])
    original = root / "synthetic_000.nii.gz"
    duplicate = root / "synthetic_000_copy.nii.gz"
    shutil.copyfile(original, duplicate)
    config["data"]["file_pattern"] = "{subject_id}*.nii.gz"
    with pytest.raises(ValueError, match="found 2"):
        build_index(config["data"])
    config["data"]["file_pattern"] = "{subject_id}_missing.nii.gz"
    with pytest.raises(ValueError, match="found 0"):
        build_index(config["data"])


def test_test_images_not_required_for_training(config):
    for path in Path(config["data"]["volume_dir"]).glob("synthetic_01[2-5].nii.gz"):
        path.unlink()
    index = build_index(config["data"], ("train", "val"))
    assert audit_dataset(index, config["data"], ("train", "val"))["passed"]


def test_zscore_and_background(config):
    config["data"]["normalization"] = "zscore"
    volume, _ = load_volume(
        Path(config["data"]["volume_dir"]) / "synthetic_000.nii.gz", config["data"]
    )
    assert volume[0].sum() == 0
    assert abs(float(volume[volume != 0].mean())) < 1e-5
