"""Non-interactive figures with explicit class order and undefined-curve handling."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from .constants import LABEL_TO_INDEX


def plot_history(history, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(axes.flat, ("loss", "macro_f1", "balanced_accuracy")):
        for split in ("train", "val"):
            ax.plot(history.epoch, history[f"{split}_{metric}"], label=split)
        ax.set(xlabel="Epoch", ylabel=metric)
        ax.legend()
    axes[1, 1].plot(history.epoch, history.learning_rate)
    axes[1, 1].set(xlabel="Epoch", ylabel="Learning rate")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_evaluation(labels, probabilities, metrics, output):
    names = list(LABEL_TO_INDEX)
    matrix = np.asarray(metrics["confusion_matrix"])
    for normalized in (False, True):
        displayed = (
            matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
            if normalized
            else matrix
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(displayed, cmap="Blues")
        for i in range(4):
            for j in range(4):
                value = f"{displayed[i, j]:.2f}" if normalized else str(displayed[i, j])
                ax.text(
                    j,
                    i,
                    value,
                    ha="center",
                    va="center",
                    color="white" if displayed[i, j] > displayed.max() / 2 else "black",
                )
        ax.set(
            xticks=range(4),
            yticks=range(4),
            xticklabels=names,
            yticklabels=names,
            xlabel="Predicted",
            ylabel="True",
            title="Row-normalized confusion" if normalized else "Confusion matrix",
        )
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(
            output
            / (
                "confusion_matrix_normalized.png"
                if normalized
                else "confusion_matrix.png"
            ),
            dpi=150,
        )
        plt.close(fig)
    for kind in ("roc", "precision_recall"):
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, name in enumerate(names):
            binary = labels == i
            if binary.all() or not binary.any():
                continue
            if kind == "roc":
                x, y, _ = roc_curve(binary, probabilities[:, i])
                ax.plot(
                    x,
                    y,
                    label=f"{name} (AUC={roc_auc_score(binary, probabilities[:, i]):.3f})",
                )
            else:
                precision, recall, _ = precision_recall_curve(
                    binary, probabilities[:, i]
                )
                ax.plot(recall, precision, label=name)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set(
            xlabel="False positive rate" if kind == "roc" else "Recall",
            ylabel="True positive rate" if kind == "roc" else "Precision",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{kind}_curves.png", dpi=150)
        plt.close(fig)
