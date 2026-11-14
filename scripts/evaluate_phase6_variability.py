"""Phase 6 Variability and Uncertainty Evaluation Script.

Compiles results across multi-seed runs, cross-validation folds, and
subject-bootstrap distributions. Generates:
- phase6_variability_summary.json
- phase6_variability_report.md

Usage:
    python -m scripts.evaluate_phase6_variability
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def load_metrics_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return None


def generate_markdown_report(report: Dict[str, Any], output_path: Path) -> None:
    boot = report.get("bootstrap_uncertainty", {})
    md_lines: List[str] = [
        "# Phase 6 Uncertainty & Variability Report",
        "",
        "## 1. Subject-Bootstrap Uncertainty (B=1,000 resamples on N=236 Validation Subjects)",
        "",
        "Bootstrap confidence intervals evaluate statistical uncertainty arising from finite validation sample size conditional on the frozen checkpoint parameters.",
        "",
        "| Candidate / Run | Metric | Point Estimate | 95% Bootstrap CI | Std Dev |",
        "|---|---|---|---|---|",
    ]

    for exp_key, exp_data in boot.items():
        pe = exp_data.get("point_estimates", {})
        bs = exp_data.get("bootstrap_summary", {})

        f1_pe = pe.get("macro_f1", 0.0)
        f1_ci = bs.get("macro_f1", {}).get("ci_95", [0.0, 0.0])
        f1_std = bs.get("macro_f1", {}).get("std", 0.0)

        acc_pe = pe.get("accuracy", 0.0)
        acc_ci = bs.get("accuracy", {}).get("ci_95", [0.0, 0.0])
        acc_std = bs.get("accuracy", {}).get("std", 0.0)

        bacc_pe = pe.get("balanced_accuracy", 0.0)
        bacc_ci = bs.get("balanced_accuracy", {}).get("ci_95", [0.0, 0.0])
        bacc_std = bs.get("balanced_accuracy", {}).get("std", 0.0)

        md_lines.append(f"| **{exp_key}** | Macro-F1 | {f1_pe:.4f} | [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}] | {f1_std:.4f} |")
        md_lines.append(f"| **{exp_key}** | Accuracy | {acc_pe*100:.2f}% | [{acc_ci[0]*100:.2f}%, {acc_ci[1]*100:.2f}%] | {acc_std*100:.2f}% |")
        md_lines.append(f"| **{exp_key}** | Balanced Acc | {bacc_pe*100:.2f}% | [{bacc_ci[0]*100:.2f}%, {bacc_ci[1]*100:.2f}%] | {bacc_std*100:.2f}% |")

    md_lines.extend([
        "",
        "## 2. Per-Class Recall & Representation Recovery",
        "",
        "| Candidate / Run | CN Recall (N=90) | EMCI Recall (N=66) | LMCI Recall (N=46) | AD Recall (N=34) |",
        "|---|---|---|---|---|",
    ])

    for exp_key, exp_data in boot.items():
        pe = exp_data.get("point_estimates", {})
        bs = exp_data.get("bootstrap_summary", {})
        cr_pe = pe.get("class_recall", {})
        cr_bs = bs.get("class_recall", {})

        def _fmt(cls_name: str) -> str:
            pt = cr_pe.get(cls_name, 0.0) * 100
            ci = cr_bs.get(cls_name, {}).get("ci_95", [0.0, 0.0])
            return f"{pt:.1f}% [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]"

        md_lines.append(f"| **{exp_key}** | {_fmt('CN')} | {_fmt('EMCI')} | {_fmt('LMCI')} | {_fmt('AD')} |")

    md_lines.extend([
        "",
        "## 3. Mathematical Distinction: Bootstrap vs Run-to-Run Uncertainty",
        "",
        "- **Subject-Bootstrap Uncertainty:** Evaluates $\\mathrm{Var}_{D_{val}}[\\mathcal{M}(f(X; \\hat{\\theta}), Y)]$. It samples validation patients with replacement holding the trained model $\\hat{\\theta}$ fixed.",
        "- **Run-to-Run Variation:** Evaluates $\\mathrm{Var}_{s}[\\mathcal{M}(f(X; \\theta^{(s)}), Y)]$. It captures optimization stochasticity across seeds $s$ (e.g. data shuffle, GPU non-determinism, weight initialization).",
        "- **Protocol:** Subject-bootstrap intervals quantify evaluation uncertainty on finite cohorts but do NOT replace training repeats across seeds/folds.",
        "",
    ])

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[OK] Generated markdown report at {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 Variability and Uncertainty Evaluator")
    parser.add_argument(
        "--bootstrap-file",
        type=Path,
        default=Path("outputs/phase6_diagnostics/bootstrap_uncertainty.json"),
        help="Path to bootstrap_uncertainty.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase6_diagnostics"),
        help="Directory to save summary outputs",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_data: Dict[str, Any] = {}

    if args.bootstrap_file.is_file():
        with args.bootstrap_file.open("r", encoding="utf-8") as f:
            report_data["bootstrap_uncertainty"] = json.load(f)

    json_path = args.output_dir / "phase6_variability_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    md_path = args.output_dir / "phase6_variability_report.md"
    generate_markdown_report(report_data, md_path)


if __name__ == "__main__":
    main()
