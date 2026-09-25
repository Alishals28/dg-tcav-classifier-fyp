"""Strict, reported transfer of the MedicalNet ResNet-18 backbone."""

import json
from pathlib import Path

import torch

from .dataset import sha256_file


def load_medicalnet(model, path, report_path=None):
    """Load all backbone tensors; allow only obsolete batch counters to be absent.

    Segmentation/classification heads are ignored. A partial backbone is a failed
    load, not a successful transfer-learning experiment. Only use trusted weights.
    """
    if model.base_channels != 64 or model.shortcut != "A":
        raise ValueError(
            "Supported resnet_18[_23dataset].pth requires width 64 and shortcut A"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError("MedicalNet checkpoint must contain a state dictionary")
    cleaned = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key in cleaned:
            raise ValueError(f"Duplicate checkpoint key after prefix removal: {key}")
        cleaned[key] = value
    target = {k: v for k, v in model.state_dict().items() if not k.startswith("fc.")}
    compatible = {
        k: v
        for k, v in cleaned.items()
        if k in target and isinstance(v, torch.Tensor) and v.shape == target[k].shape
    }
    missing = [
        k
        for k in target
        if k not in compatible and not k.endswith("num_batches_tracked")
    ]
    ignored = [k for k in cleaned if k.startswith(("fc.", "conv_seg."))]
    unexpected = [k for k in cleaned if k not in target and k not in ignored]
    nonfinite = [
        k
        for k, v in compatible.items()
        if v.is_floating_point() and not torch.isfinite(v).all()
    ]
    report = {
        "checkpoint_sha256": sha256_file(path),
        "loaded_keys": sorted(compatible),
        "missing_or_mismatched": missing,
        "ignored_head_keys": ignored,
        "unexpected_keys": unexpected,
        "nonfinite_keys": nonfinite,
        "success": not missing and not unexpected and not nonfinite,
        "backbone_tensor_fraction": len(compatible) / len(target),
    }
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2))
    if not report["success"]:
        raise ValueError(
            f"MedicalNet backbone incompatible: {len(missing)} missing/mismatched, "
            f"{len(unexpected)} unexpected keys. See loading report."
        )
    model.load_state_dict(compatible, strict=False)
    return report
