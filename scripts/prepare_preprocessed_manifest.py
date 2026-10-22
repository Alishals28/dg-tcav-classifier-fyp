"""Create classifier inputs from the final preprocessing attempt per subject."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.constants import REQUIRED_MANIFEST_COLUMNS, SPLIT_NAMES


def prepare(
    baseline_manifest, splits_path, preprocessing_manifest, output_dir, overwrite=False
):
    baseline = pd.read_csv(baseline_manifest, dtype=str, keep_default_na=False)
    preprocessing = pd.read_csv(
        preprocessing_manifest, dtype=str, keep_default_na=False
    )
    missing = REQUIRED_MANIFEST_COLUMNS - set(baseline.columns)
    if missing:
        raise ValueError(f"Baseline manifest is missing columns: {sorted(missing)}")
    for column in REQUIRED_MANIFEST_COLUMNS:
        baseline[column] = baseline[column].str.strip()
        if (baseline[column] == "").any():
            raise ValueError(f"Baseline manifest has empty {column}")
    if baseline.subject_id.duplicated().any():
        raise ValueError("Baseline manifest has duplicate subjects")
    if "subject_id" not in preprocessing or "status" not in preprocessing:
        raise ValueError("Preprocessing manifest must contain subject_id and status")
    preprocessing["subject_id"] = preprocessing.subject_id.str.strip()
    preprocessing["status"] = preprocessing.status.str.strip().str.upper()
    if (preprocessing.subject_id == "").any():
        raise ValueError("Preprocessing manifest has an empty subject_id")
    if not set(preprocessing.status).issubset({"SUCCESS", "FAILED"}):
        raise ValueError("Preprocessing status values must be SUCCESS or FAILED")
    unknown = set(preprocessing.subject_id) - set(baseline.subject_id)
    if unknown:
        raise ValueError(
            f"Preprocessing manifest contains subjects absent from baseline manifest: "
            f"{sorted(unknown)[:10]}"
        )
    attempt_rows = len(preprocessing)
    subjects_with_retries = int(
        preprocessing.loc[
            preprocessing.subject_id.duplicated(keep=False), "subject_id"
        ].nunique()
    )
    # The preprocessing report has no attempt timestamp. Its row order is the
    # retry order, and the user-selected policy is that the last row wins.
    preprocessing = preprocessing.drop_duplicates("subject_id", keep="last")

    with Path(splits_path).open(encoding="utf-8") as handle:
        original_splits = json.load(handle)
    if set(original_splits) != set(SPLIT_NAMES):
        raise ValueError("Split JSON must contain exactly train, val, and test")
    membership = {}
    for split, subjects in original_splits.items():
        if len(subjects) != len(set(subjects)):
            raise ValueError(f"Duplicate subject in original {split} split")
        for subject in subjects:
            if subject in membership:
                raise ValueError(f"Original splits overlap at {subject}")
            membership[subject] = split
    if set(membership) != set(baseline.subject_id):
        raise ValueError("Original splits must exactly cover the baseline manifest")

    successful = set(
        preprocessing.loc[preprocessing.status == "SUCCESS", "subject_id"]
    )
    eligible = baseline[baseline.subject_id.isin(successful)].copy()
    eligible["split"] = eligible.subject_id.map(membership)
    filtered_splits = {
        split: [subject for subject in original_splits[split] if subject in successful]
        for split in SPLIT_NAMES
    }
    counts = {split: len(subjects) for split, subjects in filtered_splits.items()}
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"No successful scans remain in one or more splits: {counts}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=overwrite)
    eligible.to_csv(output / "classifier_manifest.csv", index=False)
    (output / "classifier_splits.json").write_text(
        json.dumps(filtered_splits, indent=2) + "\n", encoding="utf-8"
    )
    missing_status = set(baseline.subject_id) - set(preprocessing.subject_id)
    failed = set(
        preprocessing.loc[preprocessing.status == "FAILED", "subject_id"]
    )
    summary = {
        "preprocessing_attempt_rows": attempt_rows,
        "subjects_with_retries": subjects_with_retries,
        "successful_subjects": len(successful),
        "failed_subjects": len(failed),
        "missing_status_subjects": len(missing_status),
        "split_counts": counts,
        "class_counts_by_split": {
            split: eligible.loc[eligible.split == split, "diagnosis"]
            .value_counts()
            .to_dict()
            for split in SPLIT_NAMES
        },
        "failed_subject_ids": sorted(failed),
        "missing_status_subject_ids": sorted(missing_status),
    }
    (output / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--preprocessing-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if the directory already exists",
    )
    args = parser.parse_args()
    summary = prepare(
        args.baseline_manifest,
        args.splits,
        args.preprocessing_manifest,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
