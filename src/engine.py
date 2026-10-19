"""Data loaders, class weights and the epoch loop shared by training and evaluation."""

import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

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
    sampler = None
    if split == "train" and config["training"].get("balanced_sampler", False):
        labels = np.asarray([record["label"] for record in dataset.records], dtype=int)
        counts = np.bincount(labels, minlength=4)
        if np.any(counts == 0):
            raise ValueError("Balanced sampling requires every class in training")
        sample_weights = torch.as_tensor(1.0 / counts[labels], dtype=torch.double)
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=split == "train" and sampler is None,
        sampler=sampler,
        num_workers=settings["num_workers"],
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def gpu_augment(images: torch.Tensor, settings: dict) -> torch.Tensor:
    """Batched random affine + intensity jitter on the GPU; zero background stays zero.

    Per sample: rotations up to +-degrees about each axis, isotropic scale 1 +- scales,
    translation up to +-translation_voxels, applied with probability p; then a
    foreground intensity scale 1 + N(0, intensity_std) and shift N(0, intensity_shift).
    """
    n = images.shape[0]
    device = images.device
    p = settings.get("p", 0.8)
    active = (torch.rand(n, device=device) < p).float().view(n, 1, 1)
    angles = (torch.rand(n, 3, device=device) * 2 - 1) * math.radians(settings.get("degrees", 5)) * active[:, :, 0]
    cos, sin = angles.cos(), angles.sin()
    one, zero = torch.ones(n, device=device), torch.zeros(n, device=device)
    rx = torch.stack([one, zero, zero, zero, cos[:, 0], -sin[:, 0], zero, sin[:, 0], cos[:, 0]], 1).view(n, 3, 3)
    ry = torch.stack([cos[:, 1], zero, sin[:, 1], zero, one, zero, -sin[:, 1], zero, cos[:, 1]], 1).view(n, 3, 3)
    rz = torch.stack([cos[:, 2], -sin[:, 2], zero, sin[:, 2], cos[:, 2], zero, zero, zero, one], 1).view(n, 3, 3)
    scale = 1 + (torch.rand(n, 1, 1, device=device) * 2 - 1) * settings.get("scales", 0.05) * active
    matrix = rz @ ry @ rx * scale
    size = images.shape[-1]
    shift = (torch.rand(n, 3, 1, device=device) * 2 - 1) * settings.get("translation_voxels", 3) * 2 / size * active
    theta = torch.cat([matrix, shift], 2)
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    images = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    foreground = (images != 0).float()
    gain = 1 + torch.randn(n, 1, 1, 1, 1, device=device) * settings.get("intensity_std", 0.1)
    offset = torch.randn(n, 1, 1, 1, 1, device=device) * settings.get("intensity_shift", 0.1)
    return images * gain + offset * foreground


def run_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer=None,
    scaler=None,
    clip_norm=0.0,
    desc=None,
    ema=None,
    augment=None,
):
    """Return epoch metrics and predictions with their subject IDs.

    Weighted cross-entropy divides by the sum of target-class weights.
    Accumulate that denominator across batches when reporting epoch loss.
    """
    training = optimizer is not None
    model.train(training)
    numerator = denominator = 0.0
    labels, probabilities, records = [], [], []
    iterator = loader
    if desc and tqdm is not None:
        iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        if training and augment:
            with torch.no_grad():
                images = gpu_augment(images, augment)
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
            # Clip the actual gradients, after undoing AMP's loss scaling.
            scaler.unscale_(optimizer)
            if clip_norm > 0:
                # Under AMP, early steps can overflow; scaler.step skips them and
                # lowers the scale. Without AMP a non-finite norm means divergence.
                nn.utils.clip_grad_norm_(
                    model.parameters(), clip_norm, error_if_nonfinite=not amp
                )
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update_parameters(model)
        weight_sum = (
            float(criterion.weight[targets].sum().item())
            if criterion.weight is not None
            else len(targets)
        )
        numerator += loss.item() * weight_sum
        denominator += weight_sum
        if desc and tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{numerator / denominator:.4f}"})
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
