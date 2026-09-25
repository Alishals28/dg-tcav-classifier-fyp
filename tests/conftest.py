from pathlib import Path

import pytest
import torch

from scripts.make_synthetic import make_synthetic
from src.config import load_config

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    cfg = load_config(ROOT / "configs/smoke_test.yaml")
    root = make_synthetic(tmp_path / "synthetic")
    cfg["data"].update(
        manifest_path=str(root / "manifest.csv"),
        splits_path=str(root / "splits.json"),
        volume_dir=str(root / "volumes"),
    )
    cfg["experiment"]["output_dir"] = str(tmp_path / "runs")
    return cfg
