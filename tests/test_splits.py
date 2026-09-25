import json
from pathlib import Path

import pandas as pd
import pytest

from src.dataset import build_index
from src.engine import class_weights


def test_index_and_training_weights(config):
    index = build_index(config["data"])
    assert index.groupby("split").size().to_dict() == {"train": 8, "val": 4, "test": 4}
    assert class_weights(index).tolist() == [1.0] * 4
    index.loc[index.split == "test", "label"] = 0
    assert class_weights(index).tolist() == [1.0] * 4


@pytest.mark.parametrize(
    "problem", ["overlap", "duplicate", "missing_key", "unknown_subject", "wrong_count"]
)
def test_bad_splits_fail(config, problem):
    path = Path(config["data"]["splits_path"])
    splits = json.loads(path.read_text())
    if problem == "overlap":
        splits["val"][0] = splits["train"][0]
    elif problem == "duplicate":
        splits["train"][1] = splits["train"][0]
    elif problem == "missing_key":
        del splits["test"]
    elif problem == "wrong_count":
        splits["test"].pop()
    else:
        splits["test"][0] = "unknown"
    path.write_text(json.dumps(splits))
    with pytest.raises(ValueError):
        build_index(config["data"])


@pytest.mark.parametrize(
    "column,value",
    [("diagnosis", "UNKNOWN"), ("subject_id", ""), ("image_id", "invalid")],
)
def test_bad_manifest_values(config, column, value):
    path = config["data"]["manifest_path"]
    frame = pd.read_csv(path, dtype=str)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        build_index(config["data"])


def test_duplicate_manifest_subject(config):
    path = config["data"]["manifest_path"]
    frame = pd.read_csv(path, dtype=str)
    frame.loc[1, "subject_id"] = frame.loc[0, "subject_id"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate subject"):
        build_index(config["data"])
