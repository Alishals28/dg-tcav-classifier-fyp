import numpy as np
import pytest

from src.metrics import bootstrap_intervals, classification_metrics, pairwise_metrics


def test_perfect_predictions():
    y = np.arange(4)
    p = np.eye(4)
    result = classification_metrics(y, p)
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_roc_auc_ovr"):
        assert result[k] == 1
    assert result["brier_multiclass"] == 0
    intervals = bootstrap_intervals(y, p, samples=10)
    assert intervals["macro_f1"] == {"lower": 1.0, "upper": 1.0}


def test_all_cn_predictions():
    p = np.tile([1.0, 0, 0, 0], (4, 1))
    result = classification_metrics(np.arange(4), p)
    assert result["accuracy"] == 0.25
    assert result["balanced_accuracy"] == 0.25
    assert result["macro_f1"] == pytest.approx(0.1)
    assert result["macro_roc_auc_ovr"] == 0.5


def test_absent_class_auc_is_undefined():
    result = classification_metrics(np.array([0, 1]), np.eye(4)[:2])
    assert result["per_class"]["AD"]["roc_auc"] is None
    assert result["macro_roc_auc_ovr"] is None
    assert result["balanced_accuracy"] is None


def test_pairwise_definitions_differ():
    p = np.array([[0.3, 0.5, 0.1, 0.1], [0.1, 0.5, 0.1, 0.3]])
    result = pairwise_metrics(np.array([0, 3]), p)["AD_vs_CN"]
    assert result["four_class_subset_accuracy"] == 0
    assert result["forced_pair_accuracy"] == 1


def test_bad_probabilities_rejected():
    with pytest.raises(ValueError):
        classification_metrics([0], [[1, 1, 0, 0]])
