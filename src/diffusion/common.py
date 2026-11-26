"""Shared helpers for the diffusion pipeline: config, hashing, checkpoints, AMP, GPU profiles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

CLASS_NAMES = ("CN", "MCI", "AD")
CONCEPTS = ("HA", "ECT", "VE", "THD", "GCT", "AA")
TEMPLATE_SHA256 = "dbbd4542823c1e2b12265c9b7e3cf959ee02ff7829bbf05cdcfbecc0013a00fa"
PIPELINE_VERSION = "masked_rigid_affine_centroid_seed42_v4"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, default=str))


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(state, path) -> None:
    """Write via a temporary file so an interrupted save keeps the previous checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


REQUIRED_SECTIONS = (
    "experiment", "data", "classes", "model", "latent", "stage1", "stage2",
    "sampling", "probes", "qc", "profiles",
)


def validate_ldm_config(cfg: dict) -> None:
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"Missing diffusion configuration sections: {missing}")
    classes = cfg["classes"]
    names = classes["names"]
    if list(names) != list(CLASS_NAMES) and len(names) != classes["num_classes"]:
        raise ValueError("classes.names must have num_classes entries")
    if classes["null_index"] != classes["num_classes"]:
        raise ValueError("The CFG null index must equal the class count")
    shape = cfg["data"]["expected_shape"]
    if len(shape) != 3 or any(type(v) is not int or v < 8 or v % 8 for v in shape):
        raise ValueError("data.expected_shape must be three integers divisible by 8")
    if cfg["data"]["normalization"] != "none":
        raise ValueError("The diffusion contract keeps the stored z-score (normalization: none)")
    for key in ("train", "val"):
        if key not in cfg["data"]["allowed_splits"]:
            raise ValueError("Train and val must be allowed splits")
    if "test" in cfg["data"]["allowed_splits"]:
        raise ValueError("The test partition must never be an allowed diffusion split")
    t = cfg["model"]["unet"]
    if cfg["model"]["unet"]["num_class_embeds"] != classes["num_classes"] + 1:
        raise ValueError("unet.num_class_embeds must equal num_classes + 1")
    if t["cross_attention_dim"] != cfg["model"]["anatomy"]["token_dim"]:
        raise ValueError("Cross-attention width must equal the anatomy token width")
    cfgs = cfg["stage2"]["cfg_dropout"]
    if not (0 <= cfgs["class_and_context"] < 1 and 0 <= cfgs["morphology_only"] < 1):
        raise ValueError("CFG dropout probabilities must be in [0, 1)")
    for name, prof in cfg["profiles"].items():
        for stage in ("stage1", "stage2"):
            if prof[stage]["microbatch"] < 1 or prof[stage]["accumulation"] < 1:
                raise ValueError(f"Invalid batch settings in profile {name}")
        if prof["precision"] not in {"fp16", "bf16", "fp32"}:
            raise ValueError(f"Invalid precision in profile {name}")


def load_ldm_config(path, profile: str | None = None, overrides: dict | None = None) -> dict:
    """Load YAML with optional ``extends`` inheritance, then select a GPU profile."""
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(load_ldm_config(path.parent / parent), cfg)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    validate_ldm_config(cfg)
    name = profile or cfg["experiment"].get("profile")
    if name is not None:
        if name not in cfg["profiles"]:
            raise ValueError(f"Unknown GPU profile {name!r}; choose from {sorted(cfg['profiles'])}")
        cfg["active_profile"] = {"name": name, **cfg["profiles"][name]}
    return cfg


def choose_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def amp_settings(precision: str, device: torch.device):
    """Return (autocast context factory, GradScaler or None) for a profile precision."""
    if device.type != "cuda" or precision == "fp32":
        return (lambda: nullcontext()), None
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            precision = "fp16"
        else:
            return (lambda: torch.autocast("cuda", dtype=torch.bfloat16)), None
    scaler = torch.amp.GradScaler("cuda")
    return (lambda: torch.autocast("cuda", dtype=torch.float16)), scaler


def lr_at(update: int, total: int, peak: float, floor: float, warmup: int, start_frac=0.1) -> float:
    """Linear warm-up from ``start_frac * peak``, then cosine decay to ``floor``."""
    if update < warmup:
        return peak * (start_frac + (1 - start_frac) * update / max(warmup, 1))
    progress = min(1.0, (update - warmup) / max(total - warmup, 1))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def rng_state() -> dict:
    import random

    import numpy as np

    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    import random

    import numpy as np

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def preflight_report(device: torch.device, updates: int, seconds: float, losses: list[float]) -> dict:
    """Summarize memory/throughput; admit a profile only below 85 % reserved memory."""
    report = {"updates": updates, "seconds": seconds,
              "updates_per_second": updates / max(seconds, 1e-9),
              "all_losses_finite": all(math.isfinite(v) for v in losses), "device": str(device)}
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        reserved = torch.cuda.max_memory_reserved(device)
        report.update(
            gpu=torch.cuda.get_device_name(device),
            max_memory_allocated=torch.cuda.max_memory_allocated(device),
            max_memory_reserved=reserved, total_memory=total,
            reserved_fraction=reserved / total,
            admitted=bool(reserved / total < 0.85 and report["all_losses_finite"]),
        )
    else:
        report["admitted"] = bool(report["all_losses_finite"])
        report["note"] = "CPU run; GPU admission not evaluated"
    return report


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
