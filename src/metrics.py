"""Four-class metrics, 3-class (CN vs MCI vs AD) metrics, prior calibration, and pairwise summaries."""

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
    three_cls = three_class_metrics(labels, probabilities)
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
        "three_class_cn_mci_ad": three_cls,
    }


def three_class_metrics(labels, probabilities):
    """Collapse EMCI (1) + LMCI (2) into MCI to eliminate arbitrary Logical Memory II cutoff vote-splitting."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    y3 = np.where(labels == 0, 0, np.where(labels == 3, 2, 1))
    p3 = np.stack(
        [probabilities[:, 0], probabilities[:, 1] + probabilities[:, 2], probabilities[:, 3]],
        axis=1,
    )
    pred3 = p3.argmax(1)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y3, pred3, labels=[0, 1, 2], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y3, pred3)),
        "balanced_accuracy": float(rec.mean()) if np.all(sup > 0) else None,
        "macro_f1": float(f1.mean()),
        "per_class_recall": {"CN": float(rec[0]), "MCI": float(rec[1]), "AD": float(rec[2])},
        "per_class_f1": {"CN": float(f1[0]), "MCI": float(f1[1]), "AD": float(f1[2])},
    }


def apply_prior_calibration(probabilities, multipliers):
    """Apply class-prior decision multipliers m = [m0, m1, m2, m3] and renormalize."""
    p = np.asarray(probabilities, dtype=np.float64) * np.asarray(multipliers, dtype=np.float64).reshape(1, 4)
    denom = np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return p / denom


def fit_prior_calibration(labels, probabilities, grid_steps=17):
    """Fit post-hoc class multipliers on validation predictions to resolve MCI vote-splitting."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    grid = np.linspace(0.65, 1.85, grid_steps)
    best_m = np.ones(4, dtype=np.float64)
    best_pred = probabilities.argmax(1)
    _, _, f1, _ = precision_recall_fscore_support(labels, best_pred, labels=range(4), zero_division=0)
    best_score = float(f1.mean())
    # Coordinate ascent over (m1, m2, m3) with m0=1.0 fixed (2 passes)
    for _ in range(2):
        for c in (1, 2, 3):
            for val in grid:
                cand = best_m.copy()
                cand[c] = val
                pred = (probabilities * cand.reshape(1, 4)).argmax(1)
                _, rec, f1_c, _ = precision_recall_fscore_support(
                    labels, pred, labels=range(4), zero_division=0
                )
                score = float(f1_c.mean())
                if score > best_score and float(rec.min()) >= 0.25:
                    best_score = score
                    best_m = cand
    cal_probs = apply_prior_calibration(probabilities, best_m)
    cal_metrics = classification_metrics(labels, cal_probs)
    return best_m.tolist(), cal_metrics


def bootstrap_intervals(labels, probabilities, samples=1000, seed=42):
    """Estimate 95% intervals by resampling subjects within each class."""
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
    # Add MCI (EMCI+LMCI) vs CN binary sub-task
    mci_cn_mask = np.isin(labels, [0, 1, 2])
    y_mc = labels[mci_cn_mask]
    p_mc = probabilities[mci_cn_mask]
    if len(y_mc):
        y_bin = (y_mc != 0).astype(int)
        p_mci = p_mc[:, 1] + p_mc[:, 2]
        score_mc = p_mci / np.maximum(p_mc[:, 0] + p_mci, 1e-12)
        pred_mc = (p_mci > p_mc[:, 0]).astype(int)
        result["MCI_vs_CN"] = {
            "n": int(len(y_mc)),
            "forced_pair_accuracy": float((pred_mc == y_bin).mean()),
            "conditional_pair_auc": float(roc_auc_score(y_bin, score_mc))
            if len(np.unique(y_bin)) == 2
            else None,
        }
    return result
