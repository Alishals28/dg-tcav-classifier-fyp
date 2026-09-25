"""Export pooled layer-4 representations with subject metadata, not TCAV scores."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .dataset import MRIDataset, build_index, sha256_file
from .engine import choose_device
from .evaluate import load_fitted_model
from .train import write_json


def export_features(config, checkpoint, output, split="val", spatial=False):
    device = choose_device(config["training"]["device"])
    model, _ = load_fitted_model(checkpoint, config, device)
    index = build_index(config["data"], resolve_splits=(split,))
    # Exports are deterministic, even when selecting training subjects.
    loader = torch.utils.data.DataLoader(
        MRIDataset(index, split, config["data"]),
        batch_size=config["evaluation"]["batch_size"],
        num_workers=config["evaluation"]["num_workers"],
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    features, rows = [], []
    with torch.no_grad():
        for batch in loader:
            activation = model.get_layer4_activations(batch["image"].to(device))
            features.append(model.avgpool(activation).flatten(1).cpu().numpy())
            for i, subject in enumerate(batch["subject_id"]):
                row = {
                    "row": len(rows),
                    "subject_id": subject,
                    "image_id": batch["image_id"][i],
                    "label": int(batch["label"][i]),
                    "split": split,
                }
                if spatial:
                    filename = f"layer4_{len(rows):05d}.npy"
                    np.save(
                        output / filename,
                        activation[i].cpu().numpy(),
                        allow_pickle=False,
                    )
                    row["spatial_file"] = filename
                rows.append(row)
    np.save(
        output / "pooled_features.npy", np.concatenate(features), allow_pickle=False
    )
    pd.DataFrame(rows).to_csv(output / "subjects.csv", index=False)
    write_json(
        output / "provenance.json",
        {
            "checkpoint_sha256": sha256_file(checkpoint),
            "split": split,
            "layer": "layer4",
            "rows": len(rows),
            "feature_dim": model.fc.in_features,
            "warning": "Exported arrays have no gradients. TCAV requires live model autograd and a justified bottleneck.",
        },
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument(
        "--spatial",
        action="store_true",
        help="Also save larger per-subject layer-4 arrays",
    )
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_final_test:
        parser.error("Test export requires --confirm-final-test")
    print(
        export_features(
            load_config(args.config),
            args.checkpoint,
            args.output,
            args.split,
            args.spatial,
        )
    )


if __name__ == "__main__":
    main()
