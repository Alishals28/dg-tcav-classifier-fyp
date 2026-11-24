"""Build Train/Val-only diffusion indices without resolving or opening any test scan.

Appendix B.4 of the blueprint. The master subject split (``cohort_a_splits.json``)
partitions the 5,971-visit Cohort A + B superset at the subject level. Test
subjects are a denylist: their scans are skipped before any path is resolved.

CLI:
    python -m src.diffusion.index_diffusion qc-index ...   # write the QC CSV for today's scans
    python -m src.diffusion.index_diffusion audit --config configs/diffusion_ldm_128.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from csv import DictReader
from pathlib import Path

import pandas as pd

from .common import (
    PIPELINE_VERSION,
    TEMPLATE_SHA256,
    load_ldm_config,
    sha256_file,
    write_json,
)

LABELS = {"CN": 0, "EMCI": 1, "LMCI": 1, "MCI": 1, "AD": 2}
VERSION = PIPELINE_VERSION
QC_COLUMNS = ["subject_id", "image_id", "pipeline_version", "template_sha256",
              "status", "volume_path", "mask_path"]
BASELINE_VISITS = {"scmri", "sc", "bl"}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(DictReader(handle))


def scan_key(row):
    image_id = row["image_id"].strip().removeprefix("I")
    if not image_id.isascii() or not image_id.isdigit():
        raise ValueError("Invalid image ID")
    return row["subject_id"].strip(), str(int(image_id))


def file_digest(path):
    return sha256_file(path)


def load_master_split(master_split):
    """Validate the master split and return ({partition: ids}, {subject: partition})."""
    splits = json.loads(Path(master_split).read_text())
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Invalid master split keys")
    owner = {}
    for partition, ids in splits.items():
        if not isinstance(ids, list) or not ids:
            raise ValueError("Expected a non-empty JSON subject list")
        for subject in ids:
            if (not isinstance(subject, str) or not subject
                    or subject != subject.strip() or subject in owner):
                raise ValueError("Invalid or overlapping master subject IDs")
            owner[subject] = partition
    return splits, owner


def generate_index(master_split, longitudinal_manifest, qc_index, volume_root,
                   template_sha256=TEMPLATE_SHA256, label_map=LABELS,
                   require_all_classes=True, expected_test_subjects=None,
                   pipeline_version=VERSION):
    splits, owner = load_master_split(master_split)
    if expected_test_subjects is not None and len(splits["test"]) != expected_test_subjects:
        raise ValueError(
            f"Expected {expected_test_subjects} master test subjects, found {len(splits['test'])}"
        )

    manifest = read_csv(longitudinal_manifest)
    rows = {scan_key(row): row for row in manifest}
    if len(rows) != len(manifest) or len({key[1] for key in rows}) != len(rows):
        raise ValueError("Duplicate source image ID")
    if {key[0] for key in rows} != set(owner):
        raise ValueError("Master split and longitudinal manifest disagree")
    visits = {(key[0], row["visit"]) for key, row in rows.items()}
    if len(visits) != len(rows):
        raise ValueError("More than one scan in a subject visit")
    qc = {}
    for row in read_csv(qc_index):
        key = scan_key(row)
        if key not in rows or key in qc:
            raise ValueError("QC index requires a unique final row per original scan")
        qc[key] = row

    root = Path(volume_root).resolve(strict=True)
    # Test-owned paths are compared as strings only. They are never opened or hashed.
    test_paths = set()
    for key, report in qc.items():
        if owner[key[0]] == "test":
            for column in ("volume_path", "mask_path"):
                if report.get(column):
                    test_paths.add(os.path.normpath(str(root / report[column])))

    seen_paths = set()
    image_hashes = {}
    result = {"train": [], "val": []}
    excluded = []

    def get_file(relative):
        p = Path(relative)
        if not relative or p.is_absolute() or ".." in p.parts:
            raise ValueError("Files must be relative to volume_root")
        if os.path.normpath(str(root / p)) in test_paths:
            raise ValueError("A development scan aliases a test-owned file path")
        candidate = root / p
        if not candidate.is_file():
            return None
        p = candidate.resolve(strict=True)
        if not p.is_relative_to(root) or p in seen_paths:
            raise ValueError("Invalid or reused volume path")
        seen_paths.add(p)
        return p

    for key, row in sorted(rows.items()):
        partition = owner[key[0]]
        if partition == "test":
            continue
        report = qc.get(key)
        if report is None:
            excluded.append((key, "missing_qc_row"))
            continue
        if report["status"].strip().upper() != "SUCCESS":
            excluded.append((key, "qc_status_not_success"))
            continue
        if report["pipeline_version"] != pipeline_version:
            excluded.append((key, "pipeline_version_mismatch"))
            continue
        if report["template_sha256"] != template_sha256:
            excluded.append((key, "template_digest_mismatch"))
            continue
        volume = get_file(report["volume_path"])
        mask = get_file(report["mask_path"])
        if volume is None or mask is None:
            excluded.append((key, "file_missing"))
            continue
        if row["visit_diagnosis"] not in label_map:
            raise ValueError(f"Unmapped diagnosis {row['visit_diagnosis']!r}")
        digest = file_digest(volume)
        if digest in image_hashes:
            raise ValueError("Repeated MRI bytes within or across development splits")
        image_hashes[digest] = key
        result[partition].append({
            "subject_id": key[0], "image_id": key[1], "split": partition,
            "visit": row.get("visit", ""), "phase": row.get("phase", ""),
            "label": label_map[row["visit_diagnosis"]],
            "source_label": row["visit_diagnosis"],
            "baseline_label": row.get("baseline_diagnosis", ""),
            "visit_is_baseline": row.get("visit", "") in BASELINE_VISITS,
            "volume_path": str(volume), "mask_path": str(mask),
            "volume_sha256": digest, "mask_sha256": file_digest(mask),
        })
    expected_labels = set(label_map.values())
    for partition, data in result.items():
        if not data:
            raise ValueError(f"No accepted scans in {partition}")
        if require_all_classes and {row["label"] for row in data} != expected_labels:
            raise ValueError(f"Missing configured classes in {partition}")
    assert_no_test_leak(result, splits["test"])
    return {
        "partitions": result,
        "excluded_development_scans": [
            {"subject_id": k[0], "image_id": k[1], "reason": r} for k, r in excluded
        ],
        "master_split_sha256": file_digest(Path(master_split)),
        "manifest_sha256": file_digest(Path(longitudinal_manifest)),
        "qc_index_sha256": file_digest(Path(qc_index)),
        "template_sha256": template_sha256,
        "pipeline_version": pipeline_version,
        "test_denylist_size": len(splits["test"]),
    }


def assert_no_test_leak(partitions, test_subjects):
    """Raise if any record (index, latent cache, probe plan...) names a master test subject."""
    deny = set(test_subjects)
    records = partitions.values() if isinstance(partitions, dict) else [partitions]
    for data in records:
        for row in data:
            if row["subject_id"] in deny:
                raise ValueError(f"Test subject {row['subject_id']} in a development artifact")


def flat_records(index, splits=("train", "val")):
    out = []
    for split in splits:
        if split not in index["partitions"]:
            raise ValueError(f"Split {split!r} is not available in the development index")
        out.extend(index["partitions"][split])
    return out


def index_summary(index):
    summary = {}
    for split, data in index["partitions"].items():
        summary[split] = {
            "visits": len(data),
            "subjects": len({r["subject_id"] for r in data}),
            "classes": dict(sorted(Counter(r["label"] for r in data).items())),
        }
    summary["excluded_reasons"] = dict(
        Counter(e["reason"] for e in index["excluded_development_scans"])
    )
    return summary


def save_index(index, path):
    write_json(path, index)
    return path


def load_index(path):
    index = json.loads(Path(path).read_text())
    if set(index["partitions"]) - {"train", "val"}:
        raise ValueError("A diffusion index may contain only train and val partitions")
    return index


def build_qc_index(baseline_manifest, processing_manifest, volume_root, output,
                   template_sha256=TEMPLATE_SHA256, pipeline_version=VERSION,
                   layout="baseline", file_pattern="{subject_id}_preprocessed_128.nii.gz",
                   mask_suffix="_mask"):
    """Turn the preprocessing report into the visit-keyed QC CSV (one final row per scan).

    ``layout='baseline'`` joins today's ``{subject}_preprocessed_128.nii.gz`` files to
    the Cohort A baseline (subject_id, image_id). ``layout='visit'`` expects the
    processing report to carry ``image_id`` and files at ``volumes/<subject>/<image>_...``.
    The final row per scan wins. No file content is read here.
    """
    baseline = pd.read_csv(baseline_manifest, dtype=str, keep_default_na=False)
    report = pd.read_csv(processing_manifest, dtype=str, keep_default_na=False)
    report["subject_id"] = report.subject_id.str.strip()
    report["status"] = report.status.str.strip().str.upper()
    root = Path(volume_root)
    rows = []
    if layout == "baseline":
        final = report.drop_duplicates("subject_id", keep="last").set_index("subject_id")
        for _, b in baseline.iterrows():
            sid, iid = b.subject_id.strip(), b.image_id.strip().removeprefix("I")
            status = final.status.get(sid, "MISSING")
            name = file_pattern.format(subject_id=sid, image_id=iid)
            stem = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
            rows.append({
                "subject_id": sid, "image_id": iid,
                "pipeline_version": final["pipeline_version"].get(sid, pipeline_version)
                if "pipeline_version" in final else pipeline_version,
                "template_sha256": template_sha256,
                "status": status if status in {"SUCCESS", "FAILED"} else "MISSING",
                "volume_path": name, "mask_path": f"{stem}{mask_suffix}.nii.gz",
            })
    elif layout == "visit":
        if "image_id" not in report:
            raise ValueError("Visit layout needs an image_id column in the processing report")
        report["image_id"] = report.image_id.str.strip().str.removeprefix("I")
        final = report.drop_duplicates(["subject_id", "image_id"], keep="last")
        for _, r in final.iterrows():
            stem = f"{r.subject_id}/{r.image_id}_preprocessed_128"
            rows.append({
                "subject_id": r.subject_id, "image_id": r.image_id,
                "pipeline_version": r.get("pipeline_version", pipeline_version) or pipeline_version,
                "template_sha256": template_sha256, "status": r.status,
                "volume_path": f"{stem}.nii.gz", "mask_path": f"{stem}{mask_suffix}.nii.gz",
            })
    else:
        raise ValueError("layout must be 'baseline' or 'visit'")
    frame = pd.DataFrame(rows, columns=QC_COLUMNS)
    frame.to_csv(output, index=False)
    del root
    return frame


def audit_inputs(cfg, deep=False, write_index=True):
    """Build the index from a config and return (index, report). ``deep`` reads headers."""
    d = cfg["data"]
    index = generate_index(
        d["master_split_path"], d["longitudinal_manifest_path"], d["qc_index_path"],
        d["volume_root"], d["template_sha256"], cfg["classes"]["label_map"],
        d.get("require_all_classes", True), d.get("expected_master_test_subjects"),
        d["pipeline_version"],
    )
    report = {"summary": index_summary(index), "errors": []}
    expected = d.get("expected_visits") or {}
    report["expected_visits_vs_accepted"] = {
        s: {"expected": expected.get(s), "accepted": len(index["partitions"][s])}
        for s in ("train", "val")
    }
    if deep:
        import nibabel as nib
        import numpy as np

        for split in ("train", "val"):
            for r in index["partitions"][split]:
                img = nib.load(r["volume_path"])
                ok = (img.shape == tuple(d["expected_shape"])
                      and np.allclose(img.header.get_zooms()[:3], d["expected_spacing"],
                                      atol=d["spacing_tolerance"], rtol=0)
                      and "".join(nib.aff2axcodes(img.affine)) == d["orientation"])
                if not ok:
                    report["errors"].append({"image_id": r["image_id"], "error": "geometry"})
    report["passed"] = not report["errors"]
    if write_index:
        save_index(index, d["index_path"])
        write_json(Path(d["index_path"]).with_name("audit_report.json"), report)
    return index, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Build the Train/Val index and print the audit report")
    audit.add_argument("--config", required=True)
    audit.add_argument("--deep", action="store_true", help="Read every NIfTI header")
    qc = sub.add_parser("qc-index", help="Write the visit-keyed QC CSV")
    qc.add_argument("--baseline-manifest", required=True)
    qc.add_argument("--processing-manifest", required=True)
    qc.add_argument("--volume-root", required=True)
    qc.add_argument("--output", required=True)
    qc.add_argument("--layout", choices=["baseline", "visit"], default="baseline")
    args = parser.parse_args()
    if args.command == "audit":
        _, report = audit_inputs(load_ldm_config(args.config), deep=args.deep)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["passed"] else 1)
    frame = build_qc_index(args.baseline_manifest, args.processing_manifest,
                           args.volume_root, args.output, layout=args.layout)
    print(frame.status.value_counts().to_dict())


if __name__ == "__main__":
    main()
