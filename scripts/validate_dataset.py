"""Check preprocessing requirements and write a scan-integrity report."""

import argparse
from pathlib import Path

from src.config import load_config, require_preprocessing
from src.dataset import audit_dataset, build_index
from src.train import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/dataset_audit.json")
    parser.add_argument(
        "--include-test-qc",
        action="store_true",
        help="Geometry/content QC only, no model predictions",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    require_preprocessing(config)
    splits = ("train", "val", "test") if args.include_test_qc else ("train", "val")
    index = build_index(config["data"], splits)
    report = audit_dataset(index, config["data"], splits)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, report)
    print(f"Audit passed={report['passed']}; report: {path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
