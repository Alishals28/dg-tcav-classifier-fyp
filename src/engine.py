"""Shared training/evaluation loop, weighted-loss accounting, and data-loader setup."""

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import MRIDataset
from .metrics import classification_metrics
from .transforms import build_transform


def choose_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(name)


def class_weights(index):
    """N/(K*n_c), calculated from training records only."""
    labels = index.loc[index.split == "train", "label"].to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=4)
    if np.any(counts == 0):
        raise ValueError("Every class must be represented in training")
    return torch.tensor(len(labels) / (4 * counts), dtype=torch.float32)


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(index, split, config, generator=None):
    settings = config["training" if split == "train" else "evaluation"]
    transform = build_transform(config["augmentation"]) if split == "train" else None
    dataset = MRIDataset(index, split, config["data"], transform)
    return DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=split == "train",
        num_workers=settings["num_workers"],
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def run_epoch(
    model, loader, device, criterion, optimizer=None, scaler=None, clip_norm=0.0
):
    """Train or evaluate one epoch and retain subject-aligned prediction records.

    Weighted CE's mean denominator is the sum of target weights, not batch size.
    Epoch loss accumulates that same denominator across batches.
    """
    training = optimizer is not None
    model.train(training)
    numerator = denominator = 0.0
    labels, probabilities, records = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp = bool(scaler is not None and scaler.is_enabled())
        with (
            torch.set_grad_enabled(training),
            torch.autocast(device_type=device.type, enabled=amp),
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite loss; stop and inspect input/model values"
            )
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), clip_norm, error_if_nonfinite=True
                )
            scaler.step(optimizer)
            scaler.update()
        weight_sum = (
            float(criterion.weight[targets].sum().item())
            if criterion.weight is not None
            else len(targets)
        )
        numerator += loss.item() * weight_sum
        denominator += weight_sum
        prob = logits.detach().float().softmax(1).cpu().numpy()
        target = targets.cpu().numpy()
        labels.extend(target.tolist())
        probabilities.extend(prob.tolist())
        for i in range(len(target)):
            records.append(
                {
                    "subject_id": batch["subject_id"][i],
                    "image_id": batch["image_id"][i],
                    "phase": batch["phase"][i],
                    "true_label": int(target[i]),
                    "predicted_label": int(prob[i].argmax()),
                    **{
                        f"prob_{name}": float(prob[i, j])
                        for j, name in enumerate(("cn", "emci", "lmci", "ad"))
                    },
                }
            )
    result = classification_metrics(labels, probabilities)
    result["loss"] = numerator / denominator
    return result, records
