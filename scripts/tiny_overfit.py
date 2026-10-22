"""Check whether the model can memorize a small, balanced subset of training scans."""

import argparse
import warnings
from copy import deepcopy
from pathlib import Path

import torch

warnings.filterwarnings(
    "ignore",
    message=".*(max_pool3d_with_indices_backward_cuda|avg_pool3d_backward_cuda).*",
    category=UserWarning,
)

from src.config import load_config, require_preprocessing
from src.dataset import build_index
from src.engine import choose_device, make_loader, run_epoch
from src.medicalnet import load_medicalnet
from src.model import build_model
from src.reproducibility import set_seed
from src.train import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--output", default="outputs/tiny_overfit")
    args = parser.parse_args()
    if args.steps < 1 or args.per_class < 1:
        parser.error("steps and per-class must be positive")
    cfg = deepcopy(load_config(args.config))
    require_preprocessing(cfg)
    set_seed(cfg["experiment"]["seed"])
    cfg["augmentation"]["enabled"] = False
    cfg["training"]["num_workers"] = 0
    cfg["training"]["balanced_sampler"] = False
    index = build_index(cfg["data"], ("train",))
    subset = (
        index[index.split == "train"]
        .groupby("label", group_keys=False)
        .head(args.per_class)
    )
    if len(subset) != 4 * args.per_class:
        raise ValueError("Not enough subjects for a balanced tiny subset")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out / "subjects.csv", index=False)
    device = choose_device(cfg["training"]["device"])
    model = build_model(cfg["model"], cfg["data"]["expected_shape"]).to(device)
    if cfg["model"]["pretrained"]:
        load_medicalnet(
            model,
            cfg["model"]["pretrained_checkpoint"],
            out / "medicalnet_load_report.json",
            cfg["model"].get("pretrained_checkpoint_sha256"),
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"] if cfg["model"]["pretrained"] else 1e-3,
        weight_decay=cfg["training"]["weight_decay"] if cfg["model"]["pretrained"] else 0.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    criterion = torch.nn.CrossEntropyLoss()
    loader = make_loader(subset, "train", cfg)
    results = []
    clip_norm = cfg["training"].get("gradient_clip_norm", 1.0)
    for epoch in range(args.steps):
        training, _ = run_epoch(
            model,
            loader,
            device,
            criterion,
            optimizer,
            scaler,
            clip_norm=clip_norm,
        )
        results.append(
            {
                "epoch": epoch + 1,
                "accuracy": training["accuracy"],
                "loss": training["loss"],
            }
        )
        print(results[-1], flush=True)
        if training["accuracy"] >= 0.95:
            break
    write_json(
        out / "result.json",
        {
            "passed": results[-1]["accuracy"] >= 0.95,
            "history": results,
            "note": "Train-only debugging; no generalization claim",
        },
    )
    if results[-1]["accuracy"] < 0.95:
        raise SystemExit(
            "Overfit check did not reach 95%; inspect data and optimization before main runs"
        )


if __name__ == "__main__":
    main()
