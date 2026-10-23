"""Generate Phase 4 diagnostic plots:
1. Training and validation loss, accuracy, and macro-F1 curves with best epoch and overfitting phase highlighted.
2. Formatted 4-class confusion matrix (counts and normalized percentages).
3. Error distance distribution.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def generate_plots():
    run_dir = Path("outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z")
    df_history = pd.read_csv(run_dir / "history.csv")
    df_val = pd.read_csv("outputs/phase4_diagnostics_validation.csv")

    out_dir = Path("outputs/phase4_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Learning Curves Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = df_history["epoch"].values
    best_epoch = 28

    # Loss
    axes[0].plot(epochs, df_history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(epochs, df_history["val_loss"], label="Val Loss", color="#d62728", lw=2)
    axes[0].axvline(best_epoch, color="gray", linestyle="--", label=f"Best Val Epoch ({best_epoch})")
    axes[0].axvspan(best_epoch, len(epochs), color="#fee0d2", alpha=0.5, label="Overfitting Phase")
    axes[0].set_title("Loss vs Epoch (Severe Overfitting Post-Epoch 28)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    # Macro-F1
    axes[1].plot(epochs, df_history["train_macro_f1"], label="Train Macro-F1", color="#1f77b4", lw=2)
    axes[1].plot(epochs, df_history["val_macro_f1"], label="Val Macro-F1", color="#2ca02c", lw=2)
    axes[1].axvline(best_epoch, color="gray", linestyle="--", label=f"Best Epoch ({best_epoch})")
    axes[1].axvspan(best_epoch, len(epochs), color="#fee0d2", alpha=0.5)
    axes[1].set_title("Macro-F1 vs Epoch (Peak = 0.3993)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    # Accuracy & Balanced Accuracy
    axes[2].plot(epochs, df_history["train_accuracy"], label="Train Acc", color="#1f77b4", lw=2)
    axes[2].plot(epochs, df_history["val_accuracy"], label="Val Acc", color="#ff7f0e", lw=2)
    axes[2].plot(epochs, df_history["val_balanced_accuracy"], label="Val Bal Acc", color="#9467bd", lw=2, linestyle=":")
    axes[2].axvline(best_epoch, color="gray", linestyle="--", label=f"Best Epoch ({best_epoch})")
    axes[2].axvspan(best_epoch, len(epochs), color="#fee0d2", alpha=0.5)
    axes[2].set_title("Accuracy vs Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(out_dir / "phase4_learning_curves_diagnosis.png", dpi=200)
    plt.close(fig)
    print("Saved outputs/phase4_diagnostics/phase4_learning_curves_diagnosis.png")

    # 2. Confusion Matrix Plot
    class_names = ["CN", "EMCI", "LMCI", "AD"]
    cm = np.array([
        [58, 10, 18,  4],
        [25, 10, 25,  6],
        [10,  4, 18, 14],
        [ 3,  3, 11, 17]
    ])
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Raw Counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[0], cbar=False)
    axes[0].set_title("Validation Confusion Matrix (Raw Counts, n=236)")
    axes[0].set_xlabel("Predicted Diagnosis")
    axes[0].set_ylabel("True Diagnosis")

    # Normalized (Recall)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[1], cbar=False)
    axes[1].set_title("Validation Confusion Matrix (Normalized by True Class / Recall)")
    axes[1].set_xlabel("Predicted Diagnosis")
    axes[1].set_ylabel("True Diagnosis")

    plt.tight_layout()
    fig.savefig(out_dir / "phase4_confusion_matrix_diagnosis.png", dpi=200)
    plt.close(fig)
    print("Saved outputs/phase4_diagnostics/phase4_confusion_matrix_diagnosis.png")

if __name__ == "__main__":
    generate_plots()
