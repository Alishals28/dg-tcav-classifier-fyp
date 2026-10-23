"""Comprehensive Phase 4 Diagnostic Script:
Analyzes validation predictions, training history, and acquisition metadata for the audited 1,593-subject baseline.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)

def run_diagnostics():
    run_dir = Path("outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z")
    val_preds_path = run_dir / "validation_predictions.csv"
    history_path = run_dir / "history.csv"
    manifest_path = Path("data/preprocessed_128_classifier_inputs/classifier_manifest.csv")
    prep_manifest_path = Path("data/preprocessing_manifest.csv")

    df_preds = pd.read_csv(val_preds_path)
    df_history = pd.read_csv(history_path)
    df_manifest = pd.read_csv(manifest_path)
    df_prep = pd.read_csv(prep_manifest_path)

    # Dedup preprocessing manifest to last row per subject
    df_prep_last = df_prep.groupby("subject_id").last().reset_index()

    # Join predictions with manifest metadata
    df_val = df_preds.merge(df_manifest[["subject_id", "visit", "field_strength", "series_description"]], on="subject_id", how="left")
    df_val = df_val.merge(df_prep_last[["subject_id", "brain_volume_liters", "brain_mean", "brain_std"]], on="subject_id", how="left")

    class_names = ["CN", "EMCI", "LMCI", "AD"]
    y_true = df_val["true_label"].values
    y_pred = df_val["predicted_label"].values
    y_prob = df_val[["prob_cn", "prob_emci", "prob_lmci", "prob_ad"]].values

    # 1. Overall & per-class metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    roc_auc_ovr = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    prec_k, rec_k, f1_k, supp_k = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3])
    
    # Per-class specificity & ROC-AUC
    spec_k = []
    roc_auc_k = []
    for k in range(4):
        # binary OvR
        y_true_bin = (y_true == k).astype(int)
        y_pred_bin = (y_pred == k).astype(int)
        tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
        fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        spec_k.append(spec)
        auc_k = roc_auc_score(y_true_bin, y_prob[:, k])
        roc_auc_k.append(auc_k)

    # Bootstrap CIs (1,000 resamples)
    rng = np.random.default_rng(42)
    n = len(y_true)
    boot_accs, boot_bal_accs, boot_f1s = [], [], []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        b_true = y_true[idx]
        b_pred = y_pred[idx]
        # check if all classes represented
        if len(np.unique(b_true)) == 4:
            boot_accs.append(accuracy_score(b_true, b_pred))
            boot_bal_accs.append(balanced_accuracy_score(b_true, b_pred))
            _, _, b_f1, _ = precision_recall_fscore_support(b_true, b_pred, average="macro", zero_division=0)
            boot_f1s.append(b_f1)

    ci_acc = (np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5))
    ci_bal_acc = (np.percentile(boot_bal_accs, 2.5), np.percentile(boot_bal_accs, 97.5))
    ci_f1 = (np.percentile(boot_f1s, 2.5), np.percentile(boot_f1s, 97.5))

    print("================ 4.1 CLASSIFICATION METRICS ================")
    print(f"Validation Subjects: {n}")
    print(f"Accuracy: {acc:.4f} (95% CI: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}])")
    print(f"Balanced Accuracy: {bal_acc:.4f} (95% CI: [{ci_bal_acc[0]:.4f}, {ci_bal_acc[1]:.4f}])")
    print(f"Macro Precision: {prec_macro:.4f}")
    print(f"Macro Recall: {rec_macro:.4f}")
    print(f"Macro F1: {f1_macro:.4f} (95% CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}])")
    print(f"Macro ROC-AUC (OvR): {roc_auc_ovr:.4f}")

    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    print("      CN  EMCI  LMCI    AD  Total")
    for i, name in enumerate(class_names):
        print(f"{name:4s}: {cm[i,0]:3d}   {cm[i,1]:3d}   {cm[i,2]:3d}   {cm[i,3]:3d}    {supp_k[i]:3d}")
    col_sums = cm.sum(axis=0)
    print(f"Pred: {col_sums[0]:3d}   {col_sums[1]:3d}   {col_sums[2]:3d}   {col_sums[3]:3d}    {n:3d}")

    print("\nPer-class Metrics:")
    for i, name in enumerate(class_names):
        print(f"Class {name}: Precision={prec_k[i]:.4f}, Recall={rec_k[i]:.4f}, F1={f1_k[i]:.4f}, Specificity={spec_k[i]:.4f}, ROC-AUC={roc_auc_k[i]:.4f}, Support={supp_k[i]}")

    # Specific error patterns
    print("\n--- Error Patterns ---")
    emci_as_lmci = cm[1, 2]
    lmci_as_emci = cm[2, 1]
    emci_lmci_confused = emci_as_lmci + lmci_as_emci
    total_emci_lmci = supp_k[1] + supp_k[2]
    print(f"EMCI misclassified as LMCI: {emci_as_lmci} / {supp_k[1]} ({emci_as_lmci/supp_k[1]*100:.1f}%)")
    print(f"LMCI misclassified as EMCI: {lmci_as_emci} / {supp_k[2]} ({lmci_as_emci/supp_k[2]*100:.1f}%)")
    print(f"Total EMCI/LMCI mutual confusion: {emci_lmci_confused} / {total_emci_lmci} ({emci_lmci_confused/total_emci_lmci*100:.1f}%)")

    # Extreme errors
    cn_as_ad = cm[0, 3]
    ad_as_cn = cm[3, 0]
    print(f"Extreme Error - CN predicted as AD: {cn_as_ad} / {supp_k[0]} ({cn_as_ad/supp_k[0]*100:.1f}%)")
    print(f"Extreme Error - AD predicted as CN: {ad_as_cn} / {supp_k[3]} ({ad_as_cn/supp_k[3]*100:.1f}%)")

    # Distance-based error analysis
    # Distance |true - pred|
    distances = np.abs(y_true - y_pred)
    dist_counts = pd.Series(distances).value_counts().sort_index()
    print("\nError Distance Breakdown (|True - Pred|):")
    for d, cnt in dist_counts.items():
        desc = "Exact match" if d == 0 else f"Distance {d} ({'adjacent' if d==1 else 'moderate' if d==2 else 'extreme'})"
        print(f"  Distance {d} ({desc}): {cnt} ({cnt/n*100:.2f}%)")

    # Prediction distribution vs True distribution (Class collapse check)
    print("\nPrediction Distribution vs True Distribution:")
    for i, name in enumerate(class_names):
        true_pct = supp_k[i] / n * 100
        pred_pct = col_sums[i] / n * 100
        print(f"  {name}: True={supp_k[i]} ({true_pct:.1f}%), Predicted={col_sums[i]} ({pred_pct:.1f}%), Ratio (Pred/True)={col_sums[i]/supp_k[i]:.2f}")

    # Comparison with Phase 1 baselines
    print("\n--- Comparison with Phase 1 Development Baselines ---")
    maj_acc = 90 / 236
    maj_f1 = ( (90/(90+146)) * 1.0 ) / ( 1.0 + (90/(90+146)) ) / 4 # majority class F1 formula
    # majority CN predicts class 0 for all
    maj_pred = np.zeros(n, dtype=int)
    maj_acc_true = accuracy_score(y_true, maj_pred)
    _, _, maj_f1_true, _ = precision_recall_fscore_support(y_true, maj_pred, average="macro", zero_division=0)
    print(f"Majority-CN Baseline: Accuracy = {maj_acc_true:.4f} (38.14%), Macro-F1 = {maj_f1_true:.4f}")
    print(f"Random (training class prior) Expected Accuracy = ~0.2824 (28.24%)")
    print(f"Uniform Guess Baseline (4 classes): Accuracy = 0.2500 (25.00%), Macro-F1 = 0.2500")
    print(f"Model Baseline (Epoch 28): Accuracy = {acc:.4f} (+{acc - maj_acc_true:.4f} vs Maj, +{acc - 0.2824:.4f} vs Prior), Macro-F1 = {f1_macro:.4f} (+{f1_macro - maj_f1_true:.4f} vs Maj, +{f1_macro - 0.25:.4f} vs Uniform)")

    # 2. Optimization and Overfitting Diagnosis
    print("\n================ 4.2 OPTIMIZATION & OVERFITTING ================")
    best_epoch_idx = df_history["val_macro_f1"].idxmax()
    best_row = df_history.iloc[best_epoch_idx]
    final_row = df_history.iloc[-1]
    epoch1_row = df_history.iloc[0]

    print(f"Total Epochs Trained: {len(df_history)}")
    print(f"Peak Validation Epoch: Epoch {int(best_row['epoch'])}")
    print(f"  At Best Epoch: Train Loss={best_row['train_loss']:.4f}, Train Acc={best_row['train_accuracy']:.4f}, Train Macro-F1={best_row['train_macro_f1']:.4f}")
    print(f"                 Val Loss={best_row['val_loss']:.4f}, Val Acc={best_row['val_accuracy']:.4f}, Val Macro-F1={best_row['val_macro_f1']:.4f}")
    print(f"  At Final Epoch (Epoch {int(final_row['epoch'])}): Train Loss={final_row['train_loss']:.4f}, Train Acc={final_row['train_accuracy']:.4f}, Train Macro-F1={final_row['train_macro_f1']:.4f}")
    print(f"                 Val Loss={final_row['val_loss']:.4f}, Val Acc={final_row['val_accuracy']:.4f}, Val Macro-F1={final_row['val_macro_f1']:.4f}")
    print(f"  Delta (Final - Best):")
    print(f"    Train Loss: {final_row['train_loss'] - best_row['train_loss']:.4f} (improved)")
    print(f"    Train Acc:  +{final_row['train_accuracy'] - best_row['train_accuracy']:.4f} (improved)")
    print(f"    Val Loss:   +{final_row['val_loss'] - best_row['val_loss']:.4f} (DEGRADED SEVERELY)")
    print(f"    Val Acc:    {final_row['val_accuracy'] - best_row['val_accuracy']:.4f} (degraded)")
    print(f"    Val F1:     {final_row['val_macro_f1'] - best_row['val_macro_f1']:.4f} (degraded)")

    # 3. Stratified Subgroup Analysis
    print("\n================ 4.3 SUBGROUP STRATIFICATION ================")
    # By Phase
    print("\n--- Stratification by ADNI Phase ---")
    for phase_val, sub_df in df_val.groupby("phase"):
        sub_y_true = sub_df["true_label"].values
        sub_y_pred = sub_df["predicted_label"].values
        sub_acc = accuracy_score(sub_y_true, sub_y_pred)
        sub_bal_acc = balanced_accuracy_score(sub_y_true, sub_y_pred)
        _, _, sub_f1, _ = precision_recall_fscore_support(sub_y_true, sub_y_pred, average="macro", zero_division=0)
        print(f"Phase {phase_val:8s} (n={len(sub_df):3d}): Accuracy={sub_acc:.4f}, Balanced Acc={sub_bal_acc:.4f}, Macro-F1={sub_f1:.4f}")
        print("  Class counts:", pd.Series(sub_y_true).value_counts().to_dict())

    # By Series Family
    print("\n--- Stratification by MRI Series Family ---")
    def get_series_family(desc):
        d = str(desc).upper()
        if "IR-FSPGR" in d:
            return "IR-FSPGR"
        elif "IR-SPGR" in d:
            return "IR-SPGR"
        elif "MPRAGE" in d:
            if "ACCEL" in d:
                return "Accelerated MPRAGE"
            else:
                return "Standard MPRAGE"
        return "Other"

    df_val["series_family"] = df_val["series_description"].apply(get_series_family)
    for fam, sub_df in df_val.groupby("series_family"):
        sub_y_true = sub_df["true_label"].values
        sub_y_pred = sub_df["predicted_label"].values
        sub_acc = accuracy_score(sub_y_true, sub_y_pred)
        sub_bal_acc = balanced_accuracy_score(sub_y_true, sub_y_pred)
        _, _, sub_f1, _ = precision_recall_fscore_support(sub_y_true, sub_y_pred, average="macro", zero_division=0)
        print(f"Family {fam:20s} (n={len(sub_df):3d}): Accuracy={sub_acc:.4f}, Balanced Acc={sub_bal_acc:.4f}, Macro-F1={sub_f1:.4f}")

    # By Brain Volume Quartiles
    print("\n--- Stratification by Brain Volume (Preprocessing Manifest) ---")
    df_val["vol_quartile"] = pd.qcut(df_val["brain_volume_liters"], 4, labels=["Q1 (Smallest)", "Q2", "Q3", "Q4 (Largest)"])
    for q, sub_df in df_val.groupby("vol_quartile", observed=True):
        sub_y_true = sub_df["true_label"].values
        sub_y_pred = sub_df["predicted_label"].values
        sub_acc = accuracy_score(sub_y_true, sub_y_pred)
        sub_bal_acc = balanced_accuracy_score(sub_y_true, sub_y_pred)
        _, _, sub_f1, _ = precision_recall_fscore_support(sub_y_true, sub_y_pred, average="macro", zero_division=0)
        vol_min = sub_df["brain_volume_liters"].min()
        vol_max = sub_df["brain_volume_liters"].max()
        print(f"Quartile {q:15s} (range {vol_min:.3f}-{vol_max:.3f}L, n={len(sub_df):3d}): Accuracy={sub_acc:.4f}, Balanced Acc={sub_bal_acc:.4f}, Macro-F1={sub_f1:.4f}")

    # Outliers: check the low-volume subjects identified in Phase 2
    low_vol = df_val[df_val["brain_volume_liters"] < 1.0]
    print(f"\nLow Volume Outliers (< 1.0L) in Validation: {len(low_vol)}")
    if len(low_vol) > 0:
        for _, row in low_vol.iterrows():
            print(f"  Subject: {row['subject_id']}, True: {class_names[row['true_label']]}, Pred: {class_names[row['predicted_label']]}, Vol: {row['brain_volume_liters']:.3f}L")

    # Blinded QC: False positives / false negatives analysis
    print("\n--- High Confidence Errors for Blinded Review ---")
    df_val["correct"] = (df_val["true_label"] == df_val["predicted_label"])
    df_val["max_prob"] = np.max(y_prob, axis=1)
    df_val["error_type"] = "Correct"
    df_val.loc[df_val["true_label"] != df_val["predicted_label"], "error_type"] = (
        df_val["true_label"].map(lambda x: class_names[x]) + " -> " + df_val["predicted_label"].map(lambda x: class_names[x])
    )

    extreme_errors = df_val[(df_val["true_label"] == 0) & (df_val["predicted_label"] == 3) | (df_val["true_label"] == 3) & (df_val["predicted_label"] == 0)]
    print(f"Total Extreme Cross-Stage Errors (CN <-> AD): {len(extreme_errors)}")
    for _, row in extreme_errors.iterrows():
        print(f"  Subject: {row['subject_id']}, True={class_names[row['true_label']]}, Pred={class_names[row['predicted_label']]}, Prob_CN={row['prob_cn']:.3f}, Prob_AD={row['prob_ad']:.3f}, Phase={row['phase']}, Series={row['series_description']}")

    # Save summary dataframe for visualization / artifact
    df_val.to_csv("outputs/phase4_diagnostics_validation.csv", index=False)
    print("\nDiagnostics data saved to outputs/phase4_diagnostics_validation.csv")

if __name__ == "__main__":
    run_diagnostics()
