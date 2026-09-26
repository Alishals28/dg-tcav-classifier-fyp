"""Four-class metrics, subject bootstrap intervals and pairwise summaries."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .constants import LABEL_TO_INDEX


def classification_metrics(labels, probabilities):
    """Return metrics with None for AUCs that lack positive or negative examples."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if labels.size == 0 or probabilities.shape != (labels.size, 4):
        raise ValueError("Expected nonempty labels and an N x 4 probability matrix")
    if not np.isin(labels, range(4)).all() or not np.isfinite(probabilities).all():
        raise ValueError("Invalid labels or probabilities")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError("Probabilities must be nonnegative and sum to one")
    predicted = probabilities.argmax(1)
    matrix = confusion_matrix(labels, predicted, labels=range(4))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=range(4), zero_division=0
    )
    per_class = {}
    for name, i in LABEL_TO_INDEX.items():
        positive = labels == i
        auc = (
            float(roc_auc_score(positive, probabilities[:, i]))
            if 0 < positive.sum() < len(labels)
            else None
        )
        fp = matrix[:, i].sum() - matrix[i, i]
        tn = matrix.sum() - matrix[i, :].sum() - fp
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "roc_auc": auc,
            "support": int(support[i]),
        }
    aucs = [v["roc_auc"] for v in per_class.values()]
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(recall.mean()) if np.all(support > 0) else None,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "macro_roc_auc_ovr": float(np.mean(aucs))
        if all(v is not None for v in aucs)
        else None,
        "brier_multiclass": float(
            np.square(probabilities - np.eye(4)[labels]).sum(1).mean()
        ),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def bootstrap_intervals(labels, probabilities, samples=1000, seed=42):
    """Estimate 95% intervals by resampling subjects within each class.

    Cohort A has one scan per subject. Stratification fixes observed class counts.
    These intervals describe the fitted model, not variability across training seeds.
    """
    rng = np.random.default_rng(seed)
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if samples < 1:
        return {}
    groups = [np.flatnonzero(labels == i) for i in range(4)]
    if any(len(g) == 0 for g in groups):
        return {"unavailable": "All four classes are required for stratified bootstrap"}
    metrics = {
        k: []
        for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_roc_auc_ovr")
    }
    for _ in range(samples):
        ids = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        result = classification_metrics(labels[ids], probabilities[ids])
        for k in metrics:
            metrics[k].append(result[k])
    return {
        k: {
            "lower": float(np.percentile(v, 2.5)),
            "upper": float(np.percentile(v, 97.5)),
        }
        for k, v in metrics.items()
    }


def pairwise_metrics(labels, probabilities):
    """Distinguish four-class predictions on a subset from a forced two-class decision."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    result = {}
    for name, a, b in (("AD_vs_CN", 0, 3), ("EMCI_vs_LMCI", 1, 2)):
        mask = np.isin(labels, [a, b])
        y, p = labels[mask], probabilities[mask]
        if not len(y):
            continue
        # Conditional probability among this pair; not a separately trained binary model.
        score = p[:, b] / np.maximum(p[:, a] + p[:, b], 1e-12)
        pred = np.where(p[:, b] > p[:, a], b, a)
        result[name] = {
            "n": int(len(y)),
            "four_class_subset_accuracy": float((p.argmax(1) == y).mean()),
            "forced_pair_accuracy": float((pred == y).mean()),
            "conditional_pair_auc": float(roc_auc_score(y == b, score))
            if len(np.unique(y)) == 2
            else None,
        }
    return result
