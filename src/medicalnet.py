"""Load MedicalNet backbone weights for ResNet18 and HybridInceptionResNet3D."""

import json
import re
from pathlib import Path

import torch

from .dataset import sha256_file
from .model import HybridInceptionResNet3D


def load_medicalnet(model, path, report_path=None, expected_sha256=None):
    """Load MedicalNet weights into ResNet18 (full backbone) or HybridInceptionResNet3D (stem + layers 1-3).

    Ignores the segmentation head (`conv_seg`) and any intentional hybrid/attention modules
    (`spatial_pool`, `layer4` in HybridInceptionResNet3D), while rejecting unexpected or
    corrupted ResNet backbone tensors.
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

    is_hybrid = isinstance(model, HybridInceptionResNet3D)
    excluded_prefixes = ("fc.", "spatial_pool.", "cbam.", "layer4.") if is_hybrid else ("fc.", "spatial_pool.", "cbam.")

    target_all = {
        k: v
        for k, v in model.state_dict().items()
        if not k.startswith(excluded_prefixes)
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
    ignored_prefixes = ("fc.", "conv_seg.", "layer4.") if is_hybrid else ("fc.", "conv_seg.")
    ignored = [k for k in cleaned if k.startswith(ignored_prefixes)]
    unexpected = [k for k in cleaned if k not in target_all and k not in ignored]
    nonfinite = [
        k
        for k, v in compatible.items()
        if v.is_floating_point() and not torch.isfinite(v).all()
    ]
    report = {
        "architecture": type(model).__name__,
        "hybrid_stem_transfer": is_hybrid,
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
        "backbone_tensor_fraction": sum(key in compatible for key in target) / len(target)
        if target
        else 0.0,
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
