import pandas as pd
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from src.engine import make_loader, run_epoch


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_weighted_epoch_loss_is_independent_of_batch_partition(batch_size):
    logits = torch.tensor(
        [[2.0, 0, 0, 0], [0.0, 2, 0, 0], [1.0, 1, 2, 0], [0.0, 2, 0, 1], [1.0, 2, 0, 0]]
    )
    labels = torch.tensor([0, 1, 2, 3, 0])
    weights = torch.tensor([0.5, 1.0, 2.0, 3.0])
    rows = [
        {
            "image": logits[i],
            "label": labels[i],
            "subject_id": str(i),
            "image_id": str(i),
            "phase": "synthetic",
        }
        for i in range(5)
    ]
    loader = torch.utils.data.DataLoader(rows, batch_size=batch_size)
    loss = torch.nn.CrossEntropyLoss(weight=weights)
    metrics, predictions = run_epoch(
        torch.nn.Identity(), loader, torch.device("cpu"), loss
    )
    assert metrics["loss"] == pytest.approx(loss(logits, labels).item(), rel=1e-6)
    assert len(predictions) == 5


def test_balanced_sampler_uses_inverse_class_frequency():
    labels = [0, 0, 0, 1, 2, 3]
    index = pd.DataFrame(
        {
            "split": ["train"] * len(labels),
            "label": labels,
            "subject_id": [str(i) for i in range(len(labels))],
        }
    )
    config = {
        "training": {"batch_size": 2, "num_workers": 0, "balanced_sampler": True},
        "evaluation": {"batch_size": 2, "num_workers": 0},
        "augmentation": {"enabled": False},
        "data": {},
    }
    loader = make_loader(index, "train", config, torch.Generator().manual_seed(4))
    assert isinstance(loader.sampler, WeightedRandomSampler)
    assert loader.sampler.weights.tolist() == pytest.approx(
        [1 / 3, 1 / 3, 1 / 3, 1, 1, 1]
    )
