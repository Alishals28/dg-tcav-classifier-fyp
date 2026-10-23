"""Run Phase 5 Controlled Experiment Matrix on Google Colab or target runtime.

Usage:
    python -m scripts.run_phase5_matrix --experiment b
    python -m scripts.run_phase5_matrix --experiment c
    python -m scripts.run_phase5_matrix --experiment d
    python -m scripts.run_phase5_matrix --experiment e
    python -m scripts.run_phase5_matrix --experiment f
    python -m scripts.run_phase5_matrix --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.train import train

EXPERIMENT_CONFIGS = {
    "b": "configs/colab_exp_b_natural_sampling.yaml",
    "c": "configs/colab_exp_c_weighted_loss.yaml",
    "d": "configs/colab_exp_d_mild_augmentation.yaml",
    "e": "configs/colab_exp_e_lower_lr_5e5.yaml",
    "f": "configs/colab_exp_f_combined.yaml",
}


def run_experiment(exp_id: str) -> Path:
    exp_id = exp_id.lower()
    if exp_id not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown experiment: {exp_id}. Choose from: {sorted(EXPERIMENT_CONFIGS.keys())}"
        )
    cfg_path = EXPERIMENT_CONFIGS[exp_id]
    print(f"\n{'='*70}")
    print(f"Starting Phase 5 Experiment {exp_id.upper()}: {cfg_path}")
    print(f"{'='*70}\n")
    config = load_config(cfg_path)
    run_dir = train(config)
    print(f"\n[DONE] Experiment {exp_id.upper()} completed. Output saved to: {run_dir}\n")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 Experiment Matrix Runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiment",
        "-e",
        choices=["b", "c", "d", "e", "f"],
        help="Run a single planned intervention (b, c, d, e, or f)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all planned interventions (b -> c -> d -> e -> f) sequentially",
    )
    args = parser.parse_args()

    if args.all:
        for exp_id in ["b", "c", "d", "e", "f"]:
            try:
                run_experiment(exp_id)
            except Exception as err:
                print(f"[ERROR] Experiment {exp_id.upper()} failed: {err}", file=sys.stderr)
                raise
    else:
        run_experiment(args.experiment)


if __name__ == "__main__":
    main()
