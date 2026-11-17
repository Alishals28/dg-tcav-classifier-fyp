"""Authentic Weighted Probability-Based Ensemble Evaluation per Yang et al. (PLOS ONE 2025, Page 6-7).

Implements:
- Equations (1), (2), and (3) on Page 7:
    P^i_norm = P^i / max(P^i)
    c = argmax_k sum_{i=1}^3 W^i * P^i_norm,k
- Table 4 (Page 10) Ensemble Ablations:
    - Ensemble 1: 3D ResNet-50 + 3D ResNet-18
    - Ensemble 2: 3D ResNet-50 + 3D ResNet-34
    - Ensemble 3: 3D ResNet-34 + 3D ResNet-18
    - Full Ensemble: 3D ResNet-18 + 3D ResNet-34 + 3D ResNet-50 with W = (0.4, 0.3, 0.3)
      plus weight sensitivity check (0.5, 0.3, 0.2), (0.6, 0.2, 0.2), (0.7, 0.2, 0.1) from Page 7.
- Table 5 & Table 6 (Page 10) Tasks:
    - 4-Class: NC vs. EMCI vs. LMCI vs. AD
    - Binary Sub-tasks: MCI vs. NC, MCI vs. AD, EMCI vs. LMCI, and AD vs. NC
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def yang_peak_normalize(P: np.ndarray) -> np.ndarray:
    """Equation (2) on Page 7 of Yang et al. (2025): P^i = P^i / max(P^i)."""
    P = np.asarray(P, dtype=np.float64)
    max_p = np.max(P, axis=1, keepdims=True)
    return P / np.maximum(max_p, 1e-12)


def yang_ensemble_fuse(prob_list, weights=(0.4, 0.3, 0.3), normalize_peak=True) -> np.ndarray:
    """Equations (1)-(3) on Page 7 of Yang et al. (2025)."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    normed = [yang_peak_normalize(P) if normalize_peak else np.asarray(P, dtype=np.float64) for P in prob_list]
    fused = sum(w * p for w, p in zip(weights, normed))
    return fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-12)


def print_yang_table6_and_table5(y: np.ndarray, P: np.ndarray, name: str):
    """Print Table 6 (4-Class) and Table 5 (Binary MCI vs NC, MCI vs AD, EMCI vs LMCI) metrics."""
    pred = np.argmax(P, axis=1)
    acc = accuracy_score(y, pred)
    prec, rec, f1_c, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2, 3], zero_division=0)
    macro_rec = float(np.mean(rec))
    macro_prec = float(np.mean(prec))
    macro_f1 = float(np.mean(f1_c))
    ovr_auc = roc_auc_score(y, P, multi_class="ovr", average="macro")

    print(f"\n==================== {name} (N={len(y)}) ====================")
    print(
        f"[Table 6 Multiclass 4-Class] Accuracy: {acc*100:.2f}% | Recall: {macro_rec:.4f} | "
        f"Precision: {macro_prec:.4f} | F1-score: {macro_f1:.4f} | OvR AUC: {ovr_auc:.4f}"
    )
    for idx, cname in enumerate(["NC", "EMCI", "LMCI", "AD"]):
        print(f"  {cname:4s} -> Recall: {rec[idx]*100:5.2f}% | Precision: {prec[idx]*100:5.2f}% | F1: {f1_c[idx]:.4f}")
    print("  Confusion Matrix (rows=true NC,EMCI,LMCI,AD; cols=pred):")
    print(confusion_matrix(y, pred, labels=[0, 1, 2, 3]))

    # Table 5 Binary Tasks:
    # 1. MCI (EMCI+LMCI) vs. NC
    mci_nc = np.isin(y, [0, 1, 2])
    y_mn = (y[mci_nc] != 0).astype(int)
    p_mci = P[mci_nc, 1] + P[mci_nc, 2]
    s_mn = p_mci / np.maximum(P[mci_nc, 0] + p_mci, 1e-12)
    pred_mn = (s_mn >= 0.5).astype(int)
    p_mn, r_mn, f1_mn, _ = precision_recall_fscore_support(y_mn, pred_mn, average="binary", zero_division=0)
    auc_mn = roc_auc_score(y_mn, s_mn)

    # 2. MCI (EMCI+LMCI) vs. AD
    mci_ad = np.isin(y, [1, 2, 3])
    y_ma = (y[mci_ad] == 3).astype(int)
    p_mci_a = P[mci_ad, 1] + P[mci_ad, 2]
    s_ma = P[mci_ad, 3] / np.maximum(p_mci_a + P[mci_ad, 3], 1e-12)
    pred_ma = (s_ma >= 0.5).astype(int)
    p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(y_ma, pred_ma, average="binary", zero_division=0)
    auc_ma = roc_auc_score(y_ma, s_ma)

    # 3. EMCI vs. LMCI
    em_lm = np.isin(y, [1, 2])
    y_el = (y[em_lm] == 2).astype(int)
    s_el = P[em_lm, 2] / np.maximum(P[em_lm, 1] + P[em_lm, 2], 1e-12)
    pred_el = (s_el >= 0.5).astype(int)
    p_el, r_el, f1_el, _ = precision_recall_fscore_support(y_el, pred_el, average="binary", zero_division=0)
    auc_el = roc_auc_score(y_el, s_el)

    # 4. AD vs. NC
    ad_nc = np.isin(y, [0, 3])
    y_an = (y[ad_nc] == 3).astype(int)
    s_an = P[ad_nc, 3] / np.maximum(P[ad_nc, 0] + P[ad_nc, 3], 1e-12)
    pred_an = (s_an >= 0.5).astype(int)
    p_an, r_an, f1_an, _ = precision_recall_fscore_support(y_an, pred_an, average="binary", zero_division=0)
    auc_an = roc_auc_score(y_an, s_an)

    print("[Table 5 Binary Sub-Tasks]")
    print(f"  MCI vs. NC   -> Acc: {accuracy_score(y_mn, pred_mn)*100:5.2f}% | AUC: {auc_mn:.4f} | Recall: {r_mn:.4f} | Precision: {p_mn:.4f} | F1: {f1_mn:.4f}")
    print(f"  MCI vs. AD   -> Acc: {accuracy_score(y_ma, pred_ma)*100:5.2f}% | AUC: {auc_ma:.4f} | Recall: {r_ma:.4f} | Precision: {p_ma:.4f} | F1: {f1_ma:.4f}")
    print(f"  EMCI vs. LMCI-> Acc: {accuracy_score(y_el, pred_el)*100:5.2f}% | AUC: {auc_el:.4f} | Recall: {r_el:.4f} | Precision: {p_el:.4f} | F1: {f1_el:.4f}")
    print(f"  AD vs. NC    -> Acc: {accuracy_score(y_an, pred_an)*100:5.2f}% | AUC: {auc_an:.4f} | Recall: {r_an:.4f} | Precision: {p_an:.4f} | F1: {f1_an:.4f}")


def find_latest_csv(base_dir: str, pattern: str, filename: str = "validation_predictions.csv"):
    matches = sorted(glob.glob(os.path.join(base_dir, pattern, filename)))
    return matches[-1] if matches else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default="/content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs",
    )
    parser.add_argument("--r18-pattern", default="*yang2025_resnet18_cbam*")
    parser.add_argument("--r34-pattern", default="*yang2025_resnet34_cbam*")
    parser.add_argument("--r50-pattern", default="*yang2025_resnet50_cbam*")
    args = parser.parse_args()

    cols = ["prob_cn", "prob_emci", "prob_lmci", "prob_ad"]
    p18_path = find_latest_csv(args.base_dir, args.r18_pattern)
    p34_path = find_latest_csv(args.base_dir, args.r34_pattern)
    p50_path = find_latest_csv(args.base_dir, args.r50_pattern)

    loaded = {}
    for label, path in [("3D ResNet-18 + CBAM", p18_path), ("3D ResNet-34 + CBAM", p34_path), ("3D ResNet-50 + CBAM", p50_path)]:
        if path:
            print(f"Loaded {label}: {path}")
            df = pd.read_csv(path).sort_values("subject_id").reset_index(drop=True)
            loaded[label] = (df["true_label"].to_numpy(dtype=int), df[cols].to_numpy(dtype=np.float64))
        else:
            print(f"Not found yet: {label}")

    if not loaded:
        raise FileNotFoundError("No prediction CSVs found. Train at least one model first.")

    for label, (y, P) in loaded.items():
        print_yang_table6_and_table5(y, P, label)

    if len(loaded) == 3:
        y = loaded["3D ResNet-18 + CBAM"][0]
        P18 = loaded["3D ResNet-18 + CBAM"][1]
        P34 = loaded["3D ResNet-34 + CBAM"][1]
        P50 = loaded["3D ResNet-50 + CBAM"][1]

        # Table 4 Pairwise Ensembles
        print_yang_table6_and_table5(y, yang_ensemble_fuse([P50, P18], (0.5, 0.5)), "Table 4 — Ensemble 1 (ResNet-50 + ResNet-18)")
        print_yang_table6_and_table5(y, yang_ensemble_fuse([P50, P34], (0.5, 0.5)), "Table 4 — Ensemble 2 (ResNet-50 + ResNet-34)")
        print_yang_table6_and_table5(y, yang_ensemble_fuse([P34, P18], (0.5, 0.5)), "Table 4 — Ensemble 3 (ResNet-34 + ResNet-18)")

        # Page 7 Optimal 3-Model Ensemble W = (0.4, 0.3, 0.3)
        P_ens = yang_ensemble_fuse([P18, P34, P50], weights=(0.4, 0.3, 0.3), normalize_peak=True)
        print_yang_table6_and_table5(y, P_ens, "Yang et al. (2025) Full 3-Model Ensemble W=(0.4, 0.3, 0.3)")


if __name__ == "__main__":
    main()
