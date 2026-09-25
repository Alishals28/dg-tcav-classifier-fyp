import pytest
import torch

from src.engine import run_epoch


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
