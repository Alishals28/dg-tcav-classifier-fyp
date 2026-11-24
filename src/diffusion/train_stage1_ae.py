"""Stage 1: train the 3D AutoencoderKL (128^3 -> 4 x 16^3) on Train scans.

fp16/bf16 AMP, gradient accumulation, resumable checkpoints, validation PSNR/SSIM.

    python -m src.diffusion.train_stage1_ae --config configs/diffusion_ldm_128.yaml --profile t4
    python -m src.diffusion.train_stage1_ae --config ... --validate-only
    python -m src.diffusion.train_stage1_ae --config ... --preflight
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.reproducibility import set_seed

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
    DiffusionVolumeDataset,
    SubjectRoundSampler,
    load_mask,
    load_scan,
)
from .index_diffusion import audit_inputs, flat_records, load_index
from .models_ldm import autoencoder_loss, build_models
from .probe_qc import fit_intensity_scale, paired_metrics


def get_index(cfg, index=None):
    if index is not None:
        return index
    path = Path(cfg["data"]["index_path"])
    return load_index(path) if path.is_file() else audit_inputs(cfg)[0]


def fit_or_load_scale(cfg, records, out_dir):
    """Fit the Train-only QC intensity quantiles once and reuse them."""
    path = Path(out_dir) / "intensity_scale.json"
    if path.is_file():
        data = json.loads(path.read_text())
        return data["low"], data["high"]
    qc = cfg["qc"]
    train = [r for r in records if r["split"] == "train"][: qc["intensity_scale_volumes"]]
    pairs = [(load_scan(r["volume_path"], cfg["data"])[0],
              load_mask(r["mask_path"], cfg["data"]["expected_shape"])) for r in train]
    low, high = fit_intensity_scale(pairs, tuple(qc["intensity_quantiles"]), seed=cfg["experiment"]["seed"])
    write_json(path, {"low": low, "high": high, "volumes": len(pairs)})
    return low, high


@torch.no_grad()
def validate(ae, loader, cfg, scale, device, max_volumes):
    """Weighted L1 plus PSNR/SSIM of ``decode(mu)`` against the input (brain region)."""
    ae.eval()
    s1 = cfg["stage1"]
    l1s, psnrs, ssims = [], [], []
    for n, batch in enumerate(loader):
        if n >= max_volumes:
            break
        x, brain = batch["image"].to(device), batch["brain"].to(device)
        mu, _ = ae.encode(x)
        recon = ae.decode(mu).float()
        w = 1 + s1["brain_weight"] * brain
        l1s.append(float(((recon - x).abs() * w).sum() / w.sum()))
        ref = x[0, 0].cpu().numpy()
        out = recon[0, 0].cpu().numpy()
        region = brain[0, 0].cpu().numpy() > 0
        p, s = paired_metrics(ref, out, region, *scale)
        psnrs.append(p)
        ssims.append(s)
    ae.train()
    finite = [p for p in psnrs if np.isfinite(p)]
    return {
        "weighted_l1": float(np.mean(l1s)), "psnr": float(np.mean(finite)) if finite else float("inf"),
        "ssim": float(np.mean(ssims)), "volumes": len(l1s),
    }


def run_stage1(cfg, output_dir=None, index=None, resume=True, max_updates=None,
               device=None, preflight=False, validate_only=False, log=print):
    exp, s1 = cfg["experiment"], cfg["stage1"]
    prof = cfg["active_profile"]
    set_seed(exp["seed"])
    device = device or choose_device(exp["device"])
    out_dir = Path(output_dir or cfg["experiment"]["output_dir"]) / "stage1_ae"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = get_index(cfg, index)
    records = flat_records(index)
    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = sorted((r for r in records if r["split"] == "val"), key=lambda r: (r["subject_id"], r["image_id"]))
    scale = fit_or_load_scale(cfg, records, out_dir)

    micro, accum = prof["stage1"]["microbatch"], prof["stage1"]["accumulation"]
    train_ds = DiffusionVolumeDataset(train_recs, cfg["data"])
    val_ds = DiffusionVolumeDataset(val_recs, cfg["data"])
    sampler = SubjectRoundSampler(train_recs, exp["seed"])
    gen = torch.Generator().manual_seed(exp["seed"])
    train_loader = DataLoader(train_ds, batch_size=micro, sampler=sampler, drop_last=False,
                              num_workers=s1["num_workers"], generator=gen,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=s1["num_workers"])

    ae, _, _, _ = build_models(cfg["model"], cfg["classes"]["num_classes"])
    ae.to(device)
    provenance = {
        "config_sha256": sha256_json({k: v for k, v in cfg.items() if k != "active_profile"}),
        "master_split_sha256": index["master_split_sha256"],
        "manifest_sha256": index["manifest_sha256"],
        "qc_index_sha256": index["qc_index_sha256"],
        "train_records_sha256": sha256_json([(r["subject_id"], r["image_id"], r["volume_sha256"]) for r in train_recs]),
    }
    if validate_only:
        ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        ae.load_state_dict(ckpt["ae"])
        report = validate(ae, val_loader, cfg, scale, device, len(val_ds))
        report["gates"] = {"psnr_min": cfg["qc"]["psnr_min"], "ssim_min": cfg["qc"]["ssim_min"],
                           "psnr_pass": report["psnr"] >= cfg["qc"]["psnr_min"],
                           "ssim_pass": report["ssim"] >= cfg["qc"]["ssim_min"]}
        report["provenance"] = provenance
        write_json(out_dir / "validation_report.json", report)
        log(json.dumps(report, indent=2))
        return report

    optimizer = torch.optim.AdamW(ae.parameters(), lr=s1["lr"], betas=tuple(s1["betas"]),
                                  eps=s1["eps"], weight_decay=s1["weight_decay"])
    autocast, scaler = amp_settings(prof["precision"], device)
    per_epoch = -(-len(train_loader) // accum)
    total_updates = s1["epochs"] * per_epoch
    state = {"epoch": 0, "updates": 0, "best": float("inf"), "bad_checks": 0, "overflows": 0}
    last = out_dir / "last.pt"
    if resume and last.is_file() and not preflight:
        ckpt = torch.load(last, map_location=device, weights_only=False)
        if ckpt["provenance"] != provenance:
            raise ValueError("Checkpoint provenance (config/split/index hashes) does not match")
        ae.load_state_dict(ckpt["ae"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        state.update(ckpt["state"])
        restore_rng_state(ckpt["rng"])
        log(f"Resumed at epoch {state['epoch']} update {state['updates']}")

    def save(path, extra=None):
        atomic_torch_save({
            "ae": ae.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "state": dict(state), "rng": rng_state(), "provenance": provenance,
            "scale": scale, "config": cfg["model"], **(extra or {}),
        }, path)

    limit = max_updates if max_updates is not None else (21 if preflight else None)
    losses, started = [], time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    ae.train()
    done = False
    while state["epoch"] < s1["epochs"] and not done:
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
            lr = lr_at(state["updates"], total_updates, s1["lr"], s1["min_lr"], s1["warmup_updates"])
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for batch in group:
                x, brain = batch["image"].to(device), batch["brain"].to(device)
                with autocast():
                    loss = autoencoder_loss(ae, x, brain, s1["edge_weight"], s1["kl_weight"], s1["brain_weight"])
                if not torch.isfinite(loss):
                    state["overflows"] += 1
                    continue
                scaled = loss / len(group)
                (scaler.scale(scaled) if scaler is not None else scaled).backward()
                group_loss += float(scaled.detach())
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ae.parameters(), s1["grad_clip_norm"])
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            state["updates"] += 1
            losses.append(group_loss)
            if limit is not None and state["updates"] >= limit:
                done = True
                break
        state["epoch"] += 1
        row = {"epoch": state["epoch"], "updates": state["updates"], "train_loss": float(np.mean(losses[-per_epoch:])), "lr": lr}
        if state["epoch"] % s1["val_every_epochs"] == 0 or state["epoch"] == s1["epochs"]:
            v = validate(ae, val_loader, cfg, scale, device, s1["val_max_volumes"])
            row.update({f"val_{k}": val for k, val in v.items()})
            if v["weighted_l1"] < state["best"]:
                state["best"], state["bad_checks"] = v["weighted_l1"], 0
                save(out_dir / "best.pt", {"val": v})
            else:
                state["bad_checks"] += 1
                if state["bad_checks"] >= s1["early_stopping_patience"]:
                    done = True
        log(json.dumps(row))
        with (out_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if not preflight:
            save(last)
    if not (out_dir / "best.pt").is_file() and not preflight:
        save(out_dir / "best.pt")
    if preflight:
        validate(ae, val_loader, cfg, scale, device, 1)  # include validation decoding in the peak
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
    p.add_argument("--preflight", action="store_true", help="1 + 20 updates, record peak memory")
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    cfg = load_ldm_config(a.config, a.profile)
    run_stage1(cfg, a.output_dir, resume=not a.no_resume, max_updates=a.max_updates,
               preflight=a.preflight, validate_only=a.validate_only)


if __name__ == "__main__":
    main()
