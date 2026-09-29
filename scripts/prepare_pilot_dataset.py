"""Prepare a 29-subject pilot dataset (manifest and split) for pipeline testing."""

import csv
import json
import os
import shutil
from pathlib import Path

# Paths
DEFAULT_DRIVE_DIR = Path(r"G:\My Drive\DG-TCAV_FYP_NUST\02_Cohort_A_Classifier_Baseline")
REPO_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PILOT_DIR = REPO_DIR / "data" / "pilot"


def prepare_pilot(drive_dir: Path = DEFAULT_DRIVE_DIR, output_dir: Path = OUTPUT_PILOT_DIR):
    drive_dir = Path(drive_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = drive_dir / "cohort_a_baseline_manifest.csv"
    preprocessed_dir = drive_dir / "preprocessed"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"Preprocessed dir not found at {preprocessed_dir}")

    # Discover preprocessed volume files
    volume_files = sorted(preprocessed_dir.glob("*_preprocessed_mni152_2mm.nii.gz"))
    if not volume_files:
        raise ValueError(f"No *_preprocessed_mni152_2mm.nii.gz files found in {preprocessed_dir}")

    # Extract subject IDs
    subject_ids = set()
    for f in volume_files:
        name = f.name
        # format: {subject_id}_preprocessed_mni152_2mm.nii.gz
        sid = name.replace("_preprocessed_mni152_2mm.nii.gz", "")
        subject_ids.add(sid)

    print(f"Discovered {len(subject_ids)} preprocessed subject IDs.")

    # Read and filter manifest
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        fieldnames = reader.fieldnames

    pilot_rows = [row for row in all_rows if row["subject_id"] in subject_ids]
    print(f"Matched {len(pilot_rows)} rows in cohort manifest.")

    # Save pilot manifest
    out_manifest = output_dir / "cohort_a_pilot_manifest.csv"
    with out_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pilot_rows)
    print(f"Saved pilot manifest -> {out_manifest}")

    # Diagnosis distribution
    by_dx = {}
    for r in pilot_rows:
        by_dx.setdefault(r["diagnosis"], []).append(r["subject_id"])

    for dx, sids in by_dx.items():
        print(f"  {dx}: {len(sids)} subjects")

    # Construct split
    # Total: 29. AD: 1 (002_S_5018). Place AD in train so training contains all 4 classes!
    # CN: 14 -> 8 train, 3 val, 3 test
    # EMCI: 7 -> 5 train, 1 val, 1 test
    # LMCI: 7 -> 5 train, 1 val, 1 test
    # AD: 1   -> 1 train, 0 val, 0 test
    # Total: train = 19, val = 5, test = 5 (Total = 29)
    train_ids, val_ids, test_ids = [], [], []

    for dx, sids in sorted(by_dx.items()):
        if dx == "AD":
            train_ids.extend(sids)  # 1 AD in train
        elif dx == "CN":  # 13 CN
            train_ids.extend(sids[:7])
            val_ids.extend(sids[7:10])
            test_ids.extend(sids[10:13])
        elif dx == "EMCI":  # 8 EMCI
            train_ids.extend(sids[:6])
            val_ids.append(sids[6])
            test_ids.append(sids[7])
        elif dx == "LMCI":  # 7 LMCI
            train_ids.extend(sids[:5])
            val_ids.append(sids[5])
            test_ids.append(sids[6])

    splits = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }

    out_splits = output_dir / "cohort_a_pilot_splits.json"
    out_splits.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(f"Saved pilot splits -> {out_splits}")
    print(f"Split sizes: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Copy one scan as reference.nii.gz if not already present
    ref_target = output_dir / "reference.nii.gz"
    if not ref_target.exists() and volume_files:
        print(f"Copying reference image from {volume_files[0].name} -> {ref_target}...")
        shutil.copyfile(volume_files[0], ref_target)
        print("Reference image created successfully.")

    return out_manifest, out_splits


if __name__ == "__main__":
    prepare_pilot()
