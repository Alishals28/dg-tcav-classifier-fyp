"""Evaluate a saved checkpoint on validation or held-out test subjects."""

import argparse
from pathlib import Path

import pandas as pd
import torch

from .config import load_config, require_preprocessing
from .constants import LABEL_TO_INDEX
from .dataset import audit_dataset, build_index, sha256_file
from .engine import choose_device, make_loader, run_epoch
from .metrics import bootstrap_intervals, classification_metrics, pairwise_metrics
from .model import build_model
from .plots import plot_evaluation
from .train import write_json


def load_fitted_model(checkpoint_path, config, device):
    """Load a fitted model after checking architecture, preprocessing and dataset identity."""
    require_preprocessing(config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state["label_mapping"] != LABEL_TO_INDEX:
        raise ValueError("Checkpoint label mapping does not match this project")
    trained = state["config"]
    for key in ("num_classes", "base_channels", "shortcut"):
        if config["model"][key] != trained["model"][key]:
            raise ValueError(f"Model configuration differs from checkpoint: {key}")
    for key in ("expected_shape", "expected_spacing", "orientation", "normalization"):
        if config["data"][key] != trained["data"][key]:
            raise ValueError(
                f"Preprocessing configuration differs from checkpoint: {key}"
            )
    if config["experiment"]["smoke_test"] != trained["experiment"]["smoke_test"]:
        raise ValueError("Cannot relabel a smoke checkpoint as a scientific experiment")
    for path_key, hash_key in (
        ("manifest_path", "manifest_sha256"),
        ("splits_path", "splits_sha256"),
    ):
        if sha256_file(config["data"][path_key]) != state["data_audit"][hash_key]:
            raise ValueError(f"Dataset metadata changed: {path_key}")
    reference = config["data"].get("reference_image")
    if (sha256_file(reference) if reference else None) != state["data_audit"][
        "reference_sha256"
    ]:
        raise ValueError("Reference image changed since training")
    model = build_model(trained["model"]).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return model, state


def evaluate(config, checkpoint, split="test", output=None):
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    device = choose_device(config["training"]["device"])
    model, state = load_fitted_model(checkpoint, config, device)
    output = Path(output) if output else Path(checkpoint).parent / f"{split}_evaluation"
    output.mkdir(parents=True, exist_ok=False)
    index = build_index(config["data"], resolve_splits=(split,))
    audit = audit_dataset(index, config["data"], splits=(split,))
    write_json(output / "data_audit.json", audit)
    if not audit["passed"]:
        raise ValueError("Evaluation data audit failed")
    weights = state["class_weights"]
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, device=device) if weights is not None else None
    )
    loader = make_loader(index, split, config)
    metrics, records = run_epoch(model, loader, device, criterion)
    predictions = pd.DataFrame(records)
    labels = predictions.true_label.to_numpy(dtype=int)
    probabilities = predictions[
        [f"prob_{name.lower()}" for name in LABEL_TO_INDEX]
    ].to_numpy()
    metrics["bootstrap_95ci"] = bootstrap_intervals(
        labels,
        probabilities,
        config["evaluation"]["bootstrap_samples"],
        config["experiment"]["seed"],
    )
    metrics["pairwise"] = pairwise_metrics(labels, probabilities)
    metrics["per_phase"] = {}
    for phase, group in predictions.groupby("phase"):
        metrics["per_phase"][phase] = classification_metrics(
            group.true_label.to_numpy(dtype=int),
            group[[f"prob_{name.lower()}" for name in LABEL_TO_INDEX]].to_numpy(),
        )
    metrics["checkpoint_sha256"] = sha256_file(checkpoint)
    metrics["checkpoint_epoch"] = state["epoch"]
    metrics["synthetic_smoke_test"] = config["experiment"]["smoke_test"]
    predictions.to_csv(output / f"{split}_predictions.csv", index=False)
    pd.DataFrame(metrics["per_class"]).T.to_csv(
        output / "classification_report.csv", index_label="class"
    )
    write_json(output / f"{split}_metrics.json", metrics)
    plot_evaluation(labels, probabilities, metrics, output)
    print(
        f"{split}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}. Outputs: {output}"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_final_test:
        parser.error(
            "Test evaluation requires --confirm-final-test after freezing model choices"
        )
    evaluate(load_config(args.config), args.checkpoint, args.split, args.output)


if __name__ == "__main__":
    main()
