"""YAML configuration loading with inheritance and basic validation."""

from __future__ import annotations

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
        type(n) is not int or n < 1 for n in sizes.values()
    ):
        raise ValueError(
            "expected_split_sizes must contain positive integer train/val/test counts"
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
    if any(v < 0 for k, v in config["augmentation"].items() if k != "enabled"):
        raise ValueError("Augmentation magnitudes must be nonnegative")
    if config["model"].get("num_classes") != 4:
        raise ValueError("This project is defined as a four-class classifier")
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
    if config["model"]["shortcut"] not in {"A", "B"}:
        raise ValueError("shortcut must be A or B")
    if config["model"]["pretrained"] and not config["model"]["pretrained_checkpoint"]:
        raise ValueError("Pretrained training requires a checkpoint path")


def require_preprocessing(config: dict[str, Any]) -> None:
    """Fail before scientific training unless the preprocessing handoff is explicit."""
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
    if data["expected_shape"] != [91, 109, 91] or data["expected_spacing"] != [
        2.0,
        2.0,
        2.0,
    ]:
        raise ValueError("Scientific input contract is 91x109x91 at 2 mm spacing")
