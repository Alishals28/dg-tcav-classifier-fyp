"""Data loaders, class weights, logit adjustment, ordinal/L2 regularization, and epoch loop."""

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .dataset import MRIDataset
from .metrics import classification_metrics, fit_prior_calibration, pairwise_metrics
from .transforms import build_transform


def choose_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(name)


def class_weights(index, power=1.0):
    """(N/(K*n_c))**power, calculated from training records only."""
    labels = index.loc[index.split == "train", "label"].to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=4)
    if np.any(counts == 0):
        raise ValueError("Every class must be represented in training")
    raw = len(labels) / (4.0 * counts)
    if float(power) != 1.0:
        raw = np.power(raw, float(power))
        raw = raw / raw.mean()
    return torch.tensor(raw, dtype=torch.float32)


def class_priors(index):
    """Training class prior probabilities pi_c = n_c / N."""
    labels = index.loc[index.split == "train", "label"].to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=4)
    if np.any(counts == 0):
        raise ValueError("Every class must be represented in training")
    return torch.tensor(counts / float(len(labels)), dtype=torch.float32)


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


def _ordinal_continuum_penalty(logits, targets):
    """Penalize predictions proportionally to stage distance |k - y| along CN(0)->EMCI(1)->LMCI(2)->AD(3)
    plus monotonic cumulative stage consistency (CN < EMCI < LMCI < AD)."""
    probs = torch.softmax(logits, dim=1)
    classes = torch.arange(logits.shape[1], device=logits.device, dtype=probs.dtype).unsqueeze(0)
    distances = torch.abs(classes - targets.unsqueeze(1).to(probs.dtype)) / 3.0
    emd = (probs * distances).sum(dim=1).mean()
    # Cumulative stage probabilities P(Y >= k) for k = 1, 2, 3
    p_ge_1 = (probs[:, 1] + probs[:, 2] + probs[:, 3]).clamp(1e-6, 1.0 - 1e-6)
    p_ge_2 = (probs[:, 2] + probs[:, 3]).clamp(1e-6, 1.0 - 1e-6)
    p_ge_3 = probs[:, 3].clamp(1e-6, 1.0 - 1e-6)
    t_ge_1 = (targets >= 1).to(probs.dtype)
    t_ge_2 = (targets >= 2).to(probs.dtype)
    t_ge_3 = (targets >= 3).to(probs.dtype)
    bce = (
        nn.functional.binary_cross_entropy(p_ge_1, t_ge_1)
        + nn.functional.binary_cross_entropy(p_ge_2, t_ge_2)
        + nn.functional.binary_cross_entropy(p_ge_3, t_ge_3)
    ) / 3.0
    return 0.5 * emd + 0.5 * bce


def _l2_parameter_penalty(model):
    """Compute mean L2 norm squared across trainable weight tensors (excluding BatchNorm/bias)."""
    l2_sum = torch.tensor(0.0, device=next(model.parameters()).device)
    tensors = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.ndim >= 2 and "bn" not in name:
            l2_sum = l2_sum + torch.mean(param ** 2)
            tensors += 1
    return l2_sum / max(tensors, 1)


def run_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer=None,
    scaler=None,
    clip_norm=0.0,
    desc=None,
    l2_lambda=0.0,
    ordinal_loss_weight=0.0,
    logit_adjustment=None,
):
    """Return epoch metrics (including 3-class, calibrated 4-class, and pairwise) and subject predictions."""
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
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp = bool(scaler is not None and scaler.is_enabled())
        with (
            torch.set_grad_enabled(training),
            torch.autocast(device_type=device.type, enabled=amp),
        ):
            logits = model(images)
            loss_logits = (
                logits + logit_adjustment.to(logits.dtype)
                if (training and logit_adjustment is not None)
                else logits
            )
            ce_loss = criterion(loss_logits, targets)
            loss = ce_loss
            if training and ordinal_loss_weight > 0.0:
                loss = loss + float(ordinal_loss_weight) * _ordinal_continuum_penalty(
                    logits, targets
                )
            if training and l2_lambda > 0.0:
                loss = loss + float(l2_lambda) * _l2_parameter_penalty(model)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite loss; stop and inspect input/model values"
            )
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), clip_norm, error_if_nonfinite=not amp
                )
            scaler.step(optimizer)
            scaler.update()
        weight_sum = (
            float(criterion.weight[targets].sum().item())
            if getattr(criterion, "weight", None) is not None
            else len(targets)
        )
        numerator += ce_loss.item() * weight_sum
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
    pw = pairwise_metrics(labels, probabilities)
    result["pairwise"] = pw
    result["ad_vs_cn_acc"] = pw.get("AD_vs_CN", {}).get("forced_pair_accuracy")
    result["emci_vs_lmci_acc"] = pw.get("EMCI_vs_LMCI", {}).get("forced_pair_accuracy")
    result["mci_vs_cn_acc"] = pw.get("MCI_vs_CN", {}).get("forced_pair_accuracy")
    if not training and len(set(labels)) == 4:
        mults, cal = fit_prior_calibration(labels, probabilities)
        result["calibration_multipliers"] = mults
        result["calibrated_macro_f1"] = cal["macro_f1"]
        result["calibrated_accuracy"] = cal["accuracy"]
        result["calibrated_balanced_accuracy"] = cal["balanced_accuracy"]
    else:
        result["calibration_multipliers"] = [1.0, 1.0, 1.0, 1.0]
        result["calibrated_macro_f1"] = result["macro_f1"]
        result["calibrated_accuracy"] = result["accuracy"]
        result["calibrated_balanced_accuracy"] = result["balanced_accuracy"]
    return result, records
