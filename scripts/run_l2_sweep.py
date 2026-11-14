"""Run the supervisor's L2 regularization (lambda) and Dropout sweep on Google Colab.

Usage:
    python -m scripts.run_l2_sweep --config configs/resnet18_medicalnet_regularized.yaml
    python -m scripts.run_l2_sweep --config configs/hybrid_inception_resnet3d.yaml
"""

import argparse
from copy import deepcopy

from src.config import load_config
from src.train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/resnet18_medicalnet_regularized.yaml",
        help="Base YAML config to sweep",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.001, 0.005, 0.01, 0.05],
        help="Weight decay / L2 regularization values to sweep",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability (default: 0.5)",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    for wd in args.lambdas:
        cfg = deepcopy(base_cfg)
        tag = str(wd).replace(".", "p")
        cfg["experiment"]["name"] = f"{base_cfg['experiment']['name']}_wd{tag}"
        cfg["model"]["dropout"] = args.dropout
        cfg["augmentation"]["enabled"] = False
        cfg["training"]["weight_decay"] = float(wd)
        print(f"=== Launching run: {cfg['experiment']['name']} (dropout={args.dropout}, weight_decay={wd}) ===")
        train(cfg)


if __name__ == "__main__":
    main()
