"""Compile and evaluate Phase 5 Controlled Experiment Matrix results against Baseline A.

Outputs comparison table, guardrail checks, and per-class diagnostics strictly on
development validation data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

EXPERIMENT_PATTERNS = {
    "A (Baseline)": "medicalnet_resnet18_128_colab_seed42_*",
    "B (Natural Sampling)": "medicalnet_resnet18_exp_b_natural_sampling_*",
    "C (Weighted Loss)": "medicalnet_resnet18_exp_c_weighted_loss_*",
    "D (Mild Augmentation)": "medicalnet_resnet18_exp_d_mild_augmentation_*",
    "E (Lower LR 5e-5)": "medicalnet_resnet18_exp_e_lower_lr_5e5_*",
    "F (Combined Candidate)": "medicalnet_resnet18_exp_f_combined_candidate_*",
}

GUARDRAILS = {
    "macro_f1_gain": 0.3993,
    "emci_recall_min": 0.2500,
    "all_class_recall_min": 0.2500,
    "balanced_acc_min": 0.4000,
}


def find_latest_run(output_dirs: list[Path], pattern: str) -> Path | None:
    matched = []
    for base in output_dirs:
        if base.exists():
            matched.extend(base.glob(pattern))
    if not matched:
        return None
    # Sort by mtime descending
    matched.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matched[0]


def evaluate_run(run_dir: Path) -> dict:
    hist_file = run_dir / "history.csv"
    pred_file = run_dir / "validation_predictions.csv"
    cfg_file = run_dir / "config.yaml"

    if not pred_file.exists():
        return {"status": "INCOMPLETE", "path": str(run_dir)}

    preds_df = pd.read_csv(pred_file)
    true_col = "true_label" if "true_label" in preds_df.columns else "label"
    pred_col = "predicted_label" if "predicted_label" in preds_df.columns else "prediction"

    y_true = preds_df[true_col].to_numpy(dtype=int)
    y_pred = preds_df[pred_col].to_numpy(dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    recalls = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3])
    f1s = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3])

    best_epoch = None
    best_val_loss = None
    if hist_file.exists():
        hist_df = pd.read_csv(hist_file)
        if "val_macro_f1" in hist_df.columns:
            best_idx = hist_df["val_macro_f1"].idxmax()
            best_epoch = int(hist_df.loc[best_idx, "epoch"])
            best_val_loss = float(hist_df.loc[best_idx, "val_loss"])

    return {
        "status": "COMPLETED",
        "path": str(run_dir),
        "best_epoch": best_epoch,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "val_loss": best_val_loss,
        "recalls": {
            "CN": float(recalls[0]),
            "EMCI": float(recalls[1]),
            "LMCI": float(recalls[2]),
            "AD": float(recalls[3]),
        },
        "f1_scores": {
            "CN": float(f1s[0]),
            "EMCI": float(f1s[1]),
            "LMCI": float(f1s[2]),
            "AD": float(f1s[3]),
        },
        "guardrail_checks": {
            "macro_f1_exceeds_baseline": bool(macro_f1 > GUARDRAILS["macro_f1_gain"]),
            "emci_recall_ge_25": bool(recalls[1] >= GUARDRAILS["emci_recall_min"]),
            "min_class_recall_ge_25": bool(np.all(recalls >= GUARDRAILS["all_class_recall_min"])),
            "balanced_acc_ge_40": bool(bal_acc >= GUARDRAILS["balanced_acc_min"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 Experiment Matrix")
    parser.add_argument(
        "--output-dirs",
        nargs="+",
        default=[
            "G:/My Drive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs",
            "outputs",
        ],
        help="Directories to search for experiment outputs",
    )
    parser.add_argument(
        "--report-file",
        default="outputs/phase5_matrix_report.md",
        help="Path to save markdown report",
    )
    args = parser.parse_args()

    search_dirs = [Path(p) for p in args.output_dirs]
    results = {}

    for label, pattern in EXPERIMENT_PATTERNS.items():
        run_dir = find_latest_run(search_dirs, pattern)
        if run_dir is None:
            results[label] = {"status": "NOT_STARTED", "path": None}
        else:
            results[label] = evaluate_run(run_dir)

    # Print summary table
    print("\n" + "=" * 90)
    print("PHASE 5 CONTROLLED EXPERIMENT MATRIX STATUS & RESULTS")
    print("=" * 90)

    rows = []
    for label, res in results.items():
        st = res["status"]
        if st == "COMPLETED":
            f1 = f"{res['macro_f1']:.4f}"
            acc = f"{res['accuracy']*100:.2f}%"
            bal = f"{res['balanced_accuracy']*100:.2f}%"
            emci = f"{res['recalls']['EMCI']*100:.2f}%"
            ep = f"{res['best_epoch']}"
            all_gr = all(res["guardrail_checks"].values())
            gr_str = "PASSED" if all_gr else "FAILED"
        else:
            f1 = acc = bal = emci = ep = "-"
            gr_str = st
        rows.append({
            "Experiment": label,
            "Status": st,
            "Best Epoch": ep,
            "Val Macro-F1": f1,
            "Val Accuracy": acc,
            "Balanced Acc": bal,
            "EMCI Recall": emci,
            "Guardrails": gr_str,
        })

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    print("=" * 90 + "\n")

    # Generate Markdown Report
    # Format manual markdown table without requiring 'tabulate'
    headers = list(summary_df.columns)
    md_header = "| " + " | ".join(headers) + " |\n"
    md_sep = "| " + " | ".join(["---"] * len(headers)) + " |\n"
    md_rows = ""
    for _, row in summary_df.iterrows():
        md_rows += "| " + " | ".join(str(val) for val in row) + " |\n"
    table_md = md_header + md_sep + md_rows

    lines = [
        "# Phase 5 Experiment Matrix Evaluation Report\n\n",
        "Evaluated on development validation cohort ($N=236$). Zero test split access.\n\n",
        table_md,
        "\n## Predeclared Phase 4 Guardrails:\n\n",
        f"- **Primary Metric Target:** Validation Macro-F1 $> {GUARDRAILS['macro_f1_gain']:.4f}$ (Baseline)\n",
        f"- **EMCI Recall Guardrail:** $\\ge {GUARDRAILS['emci_recall_min']*100:.1f}\\%$\n",
        f"- **All Classes Min Recall:** $\\ge {GUARDRAILS['all_class_recall_min']*100:.1f}\\%$\n",
        f"- **Balanced Accuracy:** $\\ge {GUARDRAILS['balanced_acc_min']*100:.1f}\\%$\n\n",
    ]

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"Report saved to: {report_path}")

    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"JSON results saved to: {json_path}")


if __name__ == "__main__":
    main()
