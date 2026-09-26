"""Train on Cohort A and select a checkpoint using validation macro-F1."""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .config import load_config, require_preprocessing
from .constants import LABEL_TO_INDEX
from .dataset import audit_dataset, build_index
from .engine import choose_device, class_weights, make_loader, run_epoch
from .medicalnet import load_medicalnet
from .model import build_model
from .plots import plot_history
from .reproducibility import save_environment, set_seed


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def save_checkpoint(path, state):
    """Write to a temporary file so an interrupted save leaves the previous checkpoint intact."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def rng_state(generator):
    """Capture random states needed to resume at an epoch boundary."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ],
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
    }


def restore_rng(state, generator):
    """Restore the saved random streams, including training-batch shuffling."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    generator.set_state(state["loader"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def train(config, resume=None, stop_after_epoch=None):
    """Run training and validation, returning the run's output directory."""
    if stop_after_epoch is not None and stop_after_epoch < 1:
        raise ValueError("stop_after_epoch must be positive")
    require_preprocessing(config)
    seed = config["experiment"]["seed"]
    set_seed(seed)
    device = choose_device(config["training"]["device"])
    generator = torch.Generator().manual_seed(seed)
    if resume:
        resume = Path(resume).resolve()
        if resume.name != "last.pt":
            raise ValueError("Resume requires last.pt, not a selected best checkpoint")
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["config"] != config:
            raise ValueError(
                "Resume config must match exactly; changing the experiment requires a new run"
            )
        run_dir = resume.parent
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = (
            Path(config["experiment"]["output_dir"])
            / f"{config['experiment']['name']}_seed{seed}_{stamp}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        state = None
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        save_environment(run_dir / "environment.json")

    # Check full split membership while keeping test images out of training.
    index = build_index(config["data"], resolve_splits=("train", "val"))
    audit = audit_dataset(index, config["data"], splits=("train", "val"))
    if state is not None and audit != state["data_audit"]:
        raise ValueError(
            "Dataset contents or split definitions changed since the checkpoint"
        )
    write_json(run_dir / "data_audit.json", audit)
    if not audit["passed"]:
        raise ValueError(f"Dataset audit failed; inspect {run_dir / 'data_audit.json'}")
    index.to_csv(run_dir / "resolved_index.csv", index=False)
    train_loader = make_loader(index, "train", config, generator)
    val_loader = make_loader(
        index, "val", config, torch.Generator().manual_seed(seed + 1)
    )
    weights = class_weights(index) if config["training"]["weighted_loss"] else None
    model = build_model(config["model"])
    if state is None and config["model"]["pretrained"]:
        load_medicalnet(
            model,
            config["model"]["pretrained_checkpoint"],
            run_dir / "medicalnet_load_report.json",
        )
    model.to(device)
    write_json(
        run_dir / "model_summary.json",
        {
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "class_weights": weights.tolist() if weights is not None else None,
            "label_mapping": LABEL_TO_INDEX,
        },
    )
    settings = config["training"]
    criterion = torch.nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["epochs"], eta_min=settings["min_learning_rate"]
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=settings["mixed_precision"] and device.type == "cuda"
    )
    start, best, bad_epochs, history = 0, -1.0, 0, []
    # Restore RNG last: constructing the model above consumes random numbers.
    if state is not None:
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start, best, bad_epochs = (
            state["epoch"],
            state["best_validation_metric"],
            state["bad_epochs"],
        )
        history = state["history"]
        restore_rng(state["rng_state"], generator)
        if (
            start >= settings["epochs"]
            or bad_epochs >= settings["early_stopping_patience"]
        ):
            raise ValueError("This run has already completed or early-stopped")

    for epoch in range(start + 1, settings["epochs"] + 1):
        import time

        started = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        training, _ = run_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            scaler,
            settings["gradient_clip_norm"],
        )
        validation, predictions = run_epoch(model, val_loader, device, criterion)
        score = validation["macro_f1"]
        improved = score > best
        # A tied score keeps the earlier checkpoint and counts toward patience.
        best, bad_epochs = (score, 0) if improved else (best, bad_epochs + 1)
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": lr,
            "seconds": time.perf_counter() - started,
            **{
                f"train_{k}": training[k]
                for k in ("loss", "accuracy", "macro_f1", "balanced_accuracy")
            },
            **{
                f"val_{k}": validation[k]
                for k in ("loss", "accuracy", "macro_f1", "balanced_accuracy")
            },
        }
        history.append(row)
        snapshot = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_validation_metric": best,
            "bad_epochs": bad_epochs,
            "config": config,
            "label_mapping": LABEL_TO_INDEX,
            "class_weights": weights.tolist() if weights is not None else None,
            "history": history,
            "rng_state": rng_state(generator),
            "data_audit": audit,
        }
        save_checkpoint(run_dir / "last.pt", snapshot)
        # Keep the selected model as well as the latest state needed for resume.
        if improved:
            save_checkpoint(run_dir / "best.pt", snapshot)
            pd.DataFrame(predictions).to_csv(
                run_dir / "validation_predictions.csv", index=False
            )
            write_json(run_dir / "validation_metrics.json", validation)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(
            f"Epoch {epoch}/{settings['epochs']} loss={training['loss']:.4f} "
            f"val_macro_f1={score:.4f} best={best:.4f} seconds={row['seconds']:.1f}",
            flush=True,
        )
        if bad_epochs >= settings["early_stopping_patience"] or (
            stop_after_epoch and epoch >= stop_after_epoch
        ):
            break
    plot_history(pd.DataFrame(history), run_dir / "learning_curves.png")
    print(f"Run outputs: {run_dir}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume", help="Existing last.pt; config must remain unchanged"
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Engineering interruption test; preserves total cosine schedule",
    )
    args = parser.parse_args()
    train(load_config(args.config), args.resume, args.stop_after_epoch)


if __name__ == "__main__":
    main()
