"""Reproducibility helpers and environment reporting."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Some CUDA pooling kernels lack deterministic backward implementations.
        # Surface that limitation rather than promising cross-hardware identity.
        torch.use_deterministic_algorithms(True, warn_only=True)


def collect_environment() -> dict[str, Any]:
    """Collect versions and hardware information needed to reproduce a run."""
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git_sha,
        "packages": {
            name: version(name)
            for name in [
                "numpy",
                "pandas",
                "nibabel",
                "scikit-learn",
                "torchio",
                "PyYAML",
                "matplotlib",
            ]
        },
        "pid": os.getpid(),
    }


def save_environment(path: str | Path) -> None:
    Path(path).write_text(json.dumps(collect_environment(), indent=2), encoding="utf-8")
