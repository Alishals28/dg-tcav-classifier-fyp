"""Stage 2: conditional 3D latent U-Net (CN/MCI/AD + optional anatomy token) on cached latents.

EMA 0.999 (after 500 updates), CFG dropout (10 % class+context, 15 % morphology-only),
AMP, gradient accumulation and resumable checkpoints.

    python -m src.diffusion.train_stage2_ldm --config configs/diffusion_ldm_128.yaml --profile t4
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.reproducibility import set_seed

from .cache_latents import latent_cache_dir, load_latent_stats
from .common import (
    amp_settings,
    atomic_torch_save,
    choose_device,
    load_ldm_config,
    lr_at,
    preflight_report,
    restore_rng_state,
    rng_state,
    sha256_json,
    write_json,
)
from .dataset_diffusion import (
    LatentDataset,
    MorphologyEncoder,
    SubjectRoundSampler,
    read_morphometry,
)
from .index_diffusion import flat_records
from .models_ldm import EMA, build_models, diffusion_loss
from .train_stage1_ae import get_index


@torch.no_grad()
def validate(unet, anatomy, alphas, loader, cfg, device):
    """Fixed-timestep, fixed-seed epsilon MSE (no CFG dropout) over validation latents."""
    unet.eval()
    anatomy.eval()
    s2 = cfg["stage2"]
    g = torch.Generator(device="cpu").manual_seed(s2["val_seed"])
    losses = []
    for batch in loader:
        z = batch["latent"].to(device)
        for t in s2["val_timesteps"]:
            noise = torch.randn(z.shape, generator=g).to(device)
            steps = torch.full((z.shape[0],), t, dtype=torch.long, device=device)
            losses.append(float(diffusion_loss(
                unet, anatomy, alphas, z, batch["label"].to(device), batch["morph"].to(device),
                cfg["classes"]["num_classes"], 0.0, 0.0, steps=steps, noise=noise)))
    unet.train()
    anatomy.train()
    return {"val_eps_mse": float(np.mean(losses)), "val_batches": len(losses)}


def run_stage2(cfg, output_dir=None, index=None, resume=True, max_updates=None,
               device=None, preflight=False, log=print):
    exp, s2 = cfg["experiment"], cfg["stage2"]
    prof = cfg["active_profile"]
    set_seed(exp["seed"])
    device = device or choose_device(exp["device"])
    base = Path(output_dir or exp["output_dir"])
    out_dir = base / "stage2_ldm"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = get_index(cfg, index)
    records = flat_records(index)
    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "val"]
    cache_dir = latent_cache_dir(cfg, output_dir)
    stats = load_latent_stats(cache_dir / cfg["latent"]["stats_name"])
    expected_train = sha256_json([(r["subject_id"], r["image_id"], r["volume_sha256"]) for r in train_recs])
    if stats["train_records_sha256"] != expected_train:
        raise ValueError("Latent cache was built from a different Train partition")

    table = read_morphometry(cfg["data"].get("morphometry_csv"))
    encoder = MorphologyEncoder.fit(table, records) if table else MorphologyEncoder()
    micro, accum = prof["stage2"]["microbatch"], prof["stage2"]["accumulation"]
    train_ds = LatentDataset(train_recs, cache_dir, table, encoder)
    val_ds = LatentDataset(val_recs, cache_dir, table, encoder)
    sampler = SubjectRoundSampler(train_recs, exp["seed"])
    train_loader = DataLoader(train_ds, batch_size=micro, sampler=sampler,
                              num_workers=s2["num_workers"], generator=torch.Generator().manual_seed(exp["seed"]))
    val_loader = DataLoader(val_ds, batch_size=max(micro, 1), shuffle=False, num_workers=s2["num_workers"])

    _, unet, anatomy, scheduler = build_models(cfg["model"], cfg["classes"]["num_classes"])
    unet.to(device)
    anatomy.to(device)
    alphas = scheduler.alphas_cumprod.to(device)
    params = list(unet.parameters()) + list(anatomy.parameters())
    optimizer = torch.optim.AdamW(params, lr=s2["lr"], betas=tuple(s2["betas"]),
                                  eps=s2["eps"], weight_decay=s2["weight_decay"])
    ema = EMA([unet, anatomy], s2["ema_decay"], s2["ema_start_updates"])
    eval_unet, eval_anatomy = copy.deepcopy(unet), copy.deepcopy(anatomy)
    autocast, scaler = amp_settings(prof["precision"], device)
    per_epoch = -(-len(train_loader) // accum)
    total_updates = s2["epochs"] * per_epoch
    provenance = {
        "config_sha256": sha256_json({k: v for k, v in cfg.items() if k != "active_profile"}),
        "latent_stats": {k: stats[k] for k in ("offset", "scale", "ae_checkpoint_sha256", "train_records_sha256")},
        "master_split_sha256": index["master_split_sha256"],
        "scheduler": cfg["model"]["scheduler"],
    }
    state = {"epoch": 0, "updates": 0, "best": float("inf"), "bad_checks": 0, "overflows": 0}
    last = out_dir / "last.pt"
    if resume and last.is_file() and not preflight:
        ckpt = torch.load(last, map_location=device, weights_only=False)
        if ckpt["provenance"] != provenance:
            raise ValueError("Checkpoint provenance (config/latents/split hashes) does not match")
        unet.load_state_dict(ckpt["unet"])
        anatomy.load_state_dict(ckpt["anatomy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        ema.load_state_dict(ckpt["ema"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        state.update(ckpt["state"])
        restore_rng_state(ckpt["rng"])
        log(f"Resumed at epoch {state['epoch']} update {state['updates']}")

    def save(path, extra=None):
        ema_state = ema.state_dict()
        atomic_torch_save({
            "unet": unet.state_dict(), "anatomy": anatomy.state_dict(),
            "ema": ema_state, "ema_unet": ema_state[0], "ema_anatomy": ema_state[1],
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "state": dict(state), "rng": rng_state(), "provenance": provenance,
            "model_config": cfg["model"], "latent_stats": stats,
            "morphology": encoder.state_dict(), **(extra or {}),
        }, path)

    limit = max_updates if max_updates is not None else (21 if preflight else None)
    losses, started = [], time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    unet.train()
    anatomy.train()
    num_classes = cfg["classes"]["num_classes"]
    done = False
    while state["epoch"] < s2["epochs"] and not done:
        sampler.set_epoch(state["epoch"])
        batches = iter(train_loader)
        while True:
            group = []
            for _ in range(accum):
                try:
                    group.append(next(batches))
                except StopIteration:
                    break
            if not group:
                break
            lr = lr_at(state["updates"], total_updates, s2["lr"], s2["min_lr"], s2["warmup_updates"])
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for batch in group:
                z = batch["latent"].to(device)
                with autocast():
                    loss = diffusion_loss(
                        unet, anatomy, alphas, z, batch["label"].to(device), batch["morph"].to(device),
                        num_classes, s2["cfg_dropout"]["class_and_context"],
                        s2["cfg_dropout"]["morphology_only"])
                if not torch.isfinite(loss):
                    state["overflows"] += 1
                    continue
                scaled = loss / len(group)
                (scaler.scale(scaled) if scaler is not None else scaled).backward()
                group_loss += float(scaled.detach())
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, s2["grad_clip_norm"])
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            state["updates"] += 1
            ema.update(state["updates"])
            losses.append(group_loss)
            if limit is not None and state["updates"] >= limit:
                done = True
                break
        state["epoch"] += 1
        row = {"epoch": state["epoch"], "updates": state["updates"],
               "train_loss": float(np.mean(losses[-per_epoch:])), "lr": lr}
        if state["epoch"] % s2["val_every_epochs"] == 0 or state["epoch"] == s2["epochs"] or done:
            ema.copy_to([eval_unet, eval_anatomy])
            v = validate(eval_unet, eval_anatomy, alphas, val_loader, cfg, device)
            row.update(v)
            if v["val_eps_mse"] < state["best"]:
                state["best"], state["bad_checks"] = v["val_eps_mse"], 0
                save(out_dir / "best.pt", {"val": v})
            else:
                state["bad_checks"] += 1
                if state["bad_checks"] >= s2["early_stopping_patience"]:
                    done = True
        log(json.dumps(row))
        with (out_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if not preflight:
            save(last)
    if preflight:
        report = preflight_report(device, state["updates"], time.time() - started, losses)
        write_json(out_dir / "preflight.json", report)
        log(json.dumps(report, indent=2))
        return report
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--profile", choices=["t4", "p100", "a100"])
    p.add_argument("--output-dir")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--max-updates", type=int)
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    cfg = load_ldm_config(a.config, a.profile)
    run_stage2(cfg, a.output_dir, resume=not a.no_resume, max_updates=a.max_updates, preflight=a.preflight)


if __name__ == "__main__":
    main()
