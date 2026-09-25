import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import load_config, require_preprocessing
from src.evaluate import evaluate
from src.export_activations import export_features
from src.train import train


def test_end_to_end_and_resume(config, tmp_path):
    uninterrupted = train(config)
    full = torch.load(uninterrupted / "last.pt", weights_only=True)
    interrupted = train(config, stop_after_epoch=1)
    assert not (interrupted / "test_predictions.csv").exists()
    resumed = train(config, resume=interrupted / "last.pt")
    checkpoint = torch.load(resumed / "last.pt", weights_only=True)
    assert checkpoint["epoch"] == 2
    for key in full["model_state_dict"]:
        torch.testing.assert_close(
            full["model_state_dict"][key],
            checkpoint["model_state_dict"][key],
            rtol=0,
            atol=0,
        )
    output = evaluate(config, resumed / "best.pt", split="test")
    metrics = json.loads((output / "test_metrics.json").read_text())
    assert metrics["synthetic_smoke_test"]
    assert len(pd.read_csv(output / "test_predictions.csv")) == 4
    for name in ("confusion_matrix.png", "roc_curves.png", "classification_report.csv"):
        assert (output / name).stat().st_size > 0
    export = export_features(config, resumed / "best.pt", tmp_path / "features")
    assert np.load(export / "pooled_features.npy").shape == (4, 64)
    # Evaluation cannot silently overwrite an earlier result.
    with pytest.raises(FileExistsError):
        evaluate(config, resumed / "best.pt", split="test")


def test_missing_preprocessing_blocks_real_training(config):
    config["experiment"]["smoke_test"] = False
    with pytest.raises(ValueError, match="handoff"):
        require_preprocessing(config)


def test_config_cycle(tmp_path):
    path = tmp_path / "cycle.yaml"
    path.write_text("extends: cycle.yaml")
    with pytest.raises(ValueError, match="Circular"):
        load_config(path)


def test_boolean_string_cannot_bypass_preprocessing_gate(tmp_path):
    base = Path(__file__).resolve().parents[1] / "configs/base.yaml"
    path = tmp_path / "bad.yaml"
    path.write_text(f'extends: {base}\nexperiment:\n  smoke_test: "false"\n')
    with pytest.raises(ValueError, match="boolean"):
        load_config(path)


def test_checkpoint_rejects_changed_normalization(config, tmp_path):
    run = train(config, stop_after_epoch=1)
    changed = deepcopy(config)
    changed["data"]["normalization"] = "zscore"
    with pytest.raises(ValueError, match="normalization"):
        evaluate(changed, run / "best.pt", output=tmp_path / "bad_eval")
    with pytest.raises(ValueError, match="config"):
        train(changed, resume=run / "last.pt")
