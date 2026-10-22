"""YAML configuration loading with inheritance and basic validation."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge an override into a configuration dictionary."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path, _seen: set | None = None) -> dict[str, Any]:
    """Load YAML, resolve an optional parent config, and validate key settings."""
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen.copy()
    if path in seen:
        raise ValueError(f"Circular configuration inheritance: {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")

    parent_name = config.pop("extends", None)
    if parent_name:
        parent = load_config(path.parent / parent_name, seen)
        config = _deep_merge(parent, config)

    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    required_sections = {
        "experiment",
        "data",
        "model",
        "training",
        "evaluation",
        "augmentation",
    }
    missing = sorted(required_sections - config.keys())
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    for section, key in (
        ("experiment", "smoke_test"),
        ("data", "preprocessing_confirmed"),
        ("model", "pretrained"),
        ("training", "weighted_loss"),
        ("training", "mixed_precision"),
        ("augmentation", "enabled"),
    ):
        if type(config[section][key]) is not bool:
            raise ValueError(
                f"{section}.{key} must be a YAML boolean, not a quoted string"
            )
    for section, key, minimum in (
        ("training", "epochs", 1),
        ("training", "batch_size", 1),
        ("training", "num_workers", 0),
        ("evaluation", "batch_size", 1),
        ("evaluation", "num_workers", 0),
        ("evaluation", "bootstrap_samples", 0),
        ("training", "early_stopping_patience", 1),
        ("model", "base_channels", 1),
        ("experiment", "seed", 0),
    ):
        value = config[section][key]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{section}.{key} must be an integer >= {minimum}")
    sizes = config["data"]["expected_split_sizes"]
    if set(sizes) != {"train", "val", "test"} or any(
        n is not None and (type(n) is not int or n < 1) for n in sizes.values()
    ):
        raise ValueError(
            "expected_split_sizes values must be positive integers or null"
        )

    shape = config["data"].get("expected_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(int(v) <= 0 for v in shape)
    ):
        raise ValueError("data.expected_shape must contain three positive integers")
    if any(type(v) is not int for v in shape):
        raise ValueError("Shape entries must be integers")
    spacing = config["data"]["expected_spacing"]
    if len(spacing) != 3 or any(v <= 0 for v in spacing):
        raise ValueError("Spacing must contain three positive values")
    if config["evaluation"]["bootstrap_samples"] < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    if config["data"]["spacing_tolerance"] < 0:
        raise ValueError("spacing_tolerance must be nonnegative")
    if any(
        v < 0
        for k, v in config["augmentation"].items()
        if k not in {"enabled", "blur_sigma"}
    ):
        raise ValueError("Augmentation magnitudes must be nonnegative")
    blur_sigma = config["augmentation"].get("blur_sigma", (0.0, 1.5))
    if len(blur_sigma) != 2 or blur_sigma[0] < 0 or blur_sigma[1] < blur_sigma[0]:
        raise ValueError("augmentation.blur_sigma must be an increasing nonnegative pair")
    if config["model"].get("num_classes") != 4:
        raise ValueError("This project is defined as a four-class classifier")
    architecture = config["model"].get("architecture", "ResNet18")
    if architecture != "ResNet18":
        raise ValueError(f"Unsupported model architecture: {architecture}. Supported: ResNet18")
    if config["training"].get("optimizer", "adamw").lower() not in {"adam", "adamw"}:
        raise ValueError("training.optimizer must be adam or adamw")
    if "balanced_sampler" in config["training"] and type(
        config["training"]["balanced_sampler"]
    ) is not bool:
        raise ValueError("training.balanced_sampler must be a YAML boolean")
    if int(config["training"].get("epochs", 0)) <= 0:
        raise ValueError("training.epochs must be positive")
    for section in ("training", "evaluation"):
        if config[section]["batch_size"] < 1 or config[section]["num_workers"] < 0:
            raise ValueError(f"Invalid batch size or worker count in {section}")
    t = config["training"]
    if not 0 <= t["min_learning_rate"] <= t["learning_rate"] or t["learning_rate"] <= 0:
        raise ValueError("Invalid learning rate range")
    if t["selection_metric"] != "macro_f1":
        raise ValueError("Model selection is fixed to validation macro_f1")
    if (
        t["early_stopping_patience"] < 1
        or t["weight_decay"] < 0
        or t["gradient_clip_norm"] < 0
    ):
        raise ValueError("Invalid training parameter")
    if config["data"]["normalization"] not in {"none", "minmax", "zscore"}:
        raise ValueError("Unsupported normalization")
    if config["data"].get("reference_grid_mode", "exact") not in {
        "exact",
        "same_fov",
    }:
        raise ValueError("reference_grid_mode must be exact or same_fov")
    if config["data"].get("reference_fov_tolerance_mm", 0.1) < 0:
        raise ValueError("reference_fov_tolerance_mm must be nonnegative")
    if config["model"]["shortcut"] not in {"A", "B"}:
        raise ValueError("shortcut must be A or B")
    if config["model"]["pretrained"] and not config["model"]["pretrained_checkpoint"]:
        raise ValueError("Pretrained training requires a checkpoint path")
    checkpoint_sha256 = config["model"].get("pretrained_checkpoint_sha256")
    if checkpoint_sha256 is not None and not re.fullmatch(
        r"[0-9a-fA-F]{64}", checkpoint_sha256
    ):
        raise ValueError("pretrained_checkpoint_sha256 must be a 64-character hex digest")


def require_preprocessing(config: dict[str, Any]) -> None:
    """Require preprocessing confirmation and the agreed input grid for real-data runs."""
    if config["experiment"].get("smoke_test", False):
        return
    data = config["data"]
    if not data.get("preprocessing_confirmed") or not data.get("reference_image"):
        raise ValueError(
            "Preprocessing handoff missing: confirm QC and set data.reference_image. "
            "Do not use raw scans or set smoke_test to bypass this for real training."
        )
    if config["model"]["base_channels"] != 64:
        raise ValueError(
            "Scientific runs require the full-width model (base_channels=64)"
        )
    # The declared shape and spacing are checked against every image and the
    # reference image by dataset.load_volume. Keep the gate focused on requiring
    # an explicitly confirmed preprocessing handoff instead of hard-coding one
    # team's grid here; approved preprocessing pipelines can use other grids.
