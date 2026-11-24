"""Freeze the Stage 1 AE, fit Train-only latent ``offset``/``scale`` and cache scaled latents.

``z_scaled = (mu - offset) * scale`` with ``scale = 1 / max(std, 1e-6)``. The
statistics are accumulated in float64 over Train posterior means only and are reused
verbatim for validation and inference (decode with ``z / scale + offset``).

    python -m src.diffusion.cache_latents --config configs/diffusion_ldm_128.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import choose_device, load_ldm_config, sha256_file, sha256_json, write_json
from .dataset_diffusion import DiffusionVolumeDataset
from .index_diffusion import flat_records
from .models_ldm import build_models
from .train_stage1_ae import get_index


def latent_cache_dir(cfg, output_dir=None):
    if output_dir is None:
        return Path(cfg["latent"]["cache_dir"])
    return Path(output_dir) / "latent_cache"


def load_latent_stats(path):
    stats = json.loads(Path(path).read_text())
    for key in ("offset", "scale", "std"):
        if not np.isfinite(stats[key]):
            raise ValueError("Nonfinite latent statistics")
    return stats


@torch.no_grad()
def encode_records(ae, records, cfg, device):
    """Yield (record, posterior-mean latent as float32 [4,16,16,16])."""
    ds = DiffusionVolumeDataset(records, cfg["data"])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["stage1"]["num_workers"])
    for i, batch in enumerate(loader):
        mu, _ = ae.encode(batch["image"].to(device))
        z = mu[0].float().cpu().numpy()
        if not np.isfinite(z).all():
            raise ValueError(f"Nonfinite posterior mean for {records[i]['image_id']}")
        yield records[i], z


def cache_latents(cfg, output_dir=None, index=None, device=None, ae_checkpoint=None, log=print):
    exp = cfg["experiment"]
    device = device or choose_device(exp["device"])
    base = Path(output_dir or exp["output_dir"])
    ckpt_path = Path(ae_checkpoint or base / "stage1_ae" / "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ae, _, _, _ = build_models(cfg["model"], cfg["classes"]["num_classes"], use_checkpoint=False)
    ae.load_state_dict(ckpt["ae"])
    ae.to(device).eval().requires_grad_(False)

    index = get_index(cfg, index)
    records = flat_records(index)  # train + val only; test subjects cannot appear
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    cache_dir = latent_cache_dir(cfg, output_dir)

    train_latents = [z for _, z in encode_records(ae, train, cfg, device)]
    count = sum(z.size for z in train_latents)
    total = sum(float(z.astype(np.float64).sum()) for z in train_latents)
    offset = total / count
    sq = sum(float(((z.astype(np.float64) - offset) ** 2).sum()) for z in train_latents)
    std = float(np.sqrt(sq / count))
    if not np.isfinite(std) or std < cfg["latent"]["min_scale"]:
        raise ValueError(f"Degenerate cached latent distribution (std={std})")
    scale = 1.0 / max(std, cfg["latent"]["min_scale"])

    rows = []

    def write(record, z):
        scaled = ((z.astype(np.float64) - offset) * scale).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("Nonfinite scaled latent")
        path = cache_dir / record["split"] / f"{record['subject_id']}_{record['image_id']}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, scaled, allow_pickle=False)
        rows.append({"subject_id": record["subject_id"], "image_id": record["image_id"],
                     "split": record["split"], "label": record["label"], "file": str(path)})
        return scaled

    scaled_train = [write(r, z) for r, z in zip(train, train_latents)]
    del train_latents
    for r, z in encode_records(ae, val, cfg, device):
        write(r, z)
    stats = {
        "offset": offset, "std": std, "scale": scale, "fit_split": "train",
        "latent_values": count, "train_scans": len(train),
        "scaled_train_mean": float(np.mean([s.mean() for s in scaled_train])),
        "ae_checkpoint_sha256": sha256_file(ckpt_path),
        "train_records_sha256": sha256_json([(r["subject_id"], r["image_id"], r["volume_sha256"]) for r in train]),
        "master_split_sha256": index["master_split_sha256"],
        "latent_shape": list(scaled_train[0].shape),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_json(cache_dir / cfg["latent"]["stats_name"], stats)
    write_json(base / "stage1_ae" / cfg["latent"]["stats_name"], stats)
    import pandas as pd

    pd.DataFrame(rows).to_csv(cache_dir / "latent_index.csv", index=False)
    log(json.dumps({k: stats[k] for k in ("offset", "std", "scale", "train_scans")}))
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--profile", choices=["t4", "p100", "a100"])
    p.add_argument("--output-dir")
    p.add_argument("--ae-checkpoint")
    a = p.parse_args()
    cache_latents(load_ldm_config(a.config, a.profile), a.output_dir, ae_checkpoint=a.ae_checkpoint)


if __name__ == "__main__":
    main()
