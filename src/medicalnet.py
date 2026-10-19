"""Load MedicalNet backbone weights and report checkpoint compatibility."""

import json
import re
from pathlib import Path

import torch

from .dataset import sha256_file


def load_medicalnet(model, path, report_path=None, expected_sha256=None):
    """Load the backbone, allowing legacy BatchNorm batch counters to be absent.

    Ignore the original head; reject missing or incompatible backbone tensors.
    The supplied checkpoint should come from a trusted source.
    """
    if model.base_channels != 64 or model.shortcut != "A":
        raise ValueError(
            "Supported resnet_18[_23dataset].pth requires width 64 and shortcut A"
        )
    checkpoint_sha256 = sha256_file(path)
    if expected_sha256 is not None:
        expected_sha256 = expected_sha256.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("expected_sha256 must be a 64-character SHA-256 hex digest")
        if checkpoint_sha256 != expected_sha256:
            raise ValueError("MedicalNet checkpoint SHA-256 does not match expected value")
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
    target_all = {
        k: v for k, v in model.state_dict().items() if not k.startswith("fc.")
    }
    target = {k: v for k, v in target_all.items() if not k.endswith("num_batches_tracked")}
    compatible = {
        k: v
        for k, v in cleaned.items()
        if k in target_all
        and isinstance(v, torch.Tensor)
        and v.shape == target_all[k].shape
    }
    missing = [
        k
        for k in target
        if k not in compatible and not k.endswith("num_batches_tracked")
    ]
    ignored = [k for k in cleaned if k.startswith(("fc.", "conv_seg."))]
    unexpected = [k for k in cleaned if k not in target_all and k not in ignored]
    nonfinite = [
        k
        for k, v in compatible.items()
        if v.is_floating_point() and not torch.isfinite(v).all()
    ]
    report = {
        "checkpoint_sha256": checkpoint_sha256,
        "expected_sha256": expected_sha256,
        "sha256_match": checkpoint_sha256 == expected_sha256
        if expected_sha256 is not None
        else None,
        "loaded_keys": sorted(compatible),
        "missing_or_mismatched": missing,
        "ignored_head_keys": ignored,
        "unexpected_keys": unexpected,
        "nonfinite_keys": nonfinite,
        "success": not missing and not unexpected and not nonfinite,
        "backbone_tensor_fraction": sum(key in compatible for key in target) / len(target),
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
