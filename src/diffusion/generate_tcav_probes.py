"""Generate quality-gated counterfactual TCAV probes (6 concepts + PBM neutrals).

For each group the generator draws Train-CN anchors (one visit per subject per pass,
seed 42), runs the 50-step masked DDIM edit inside the atlas support, applies the
Section 5 QC gates, and publishes only accepted unique volumes:

    probes/<group>/<probe_id>.nii.gz  (+ <probe_id>_mask.nii.gz, <probe_id>.json)

Rejected candidates are only journaled. The journal makes runs resumable. The final
``probe_manifest.csv`` is TCAV-ready: ``export`` builds a view that
``src/export_activations.py`` can read directly.

    python -m src.diffusion.generate_tcav_probes run --config configs/diffusion_ldm_128.yaml
    python -m src.diffusion.generate_tcav_probes export --config ... --classifier-config CFG.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml

from src.reproducibility import set_seed

from .build_atlas_masks import load_support_masks
from .common import (
    CONCEPTS,
    choose_device,
    load_ldm_config,
    sha256_file,
    sha256_json,
    write_json,
)
from .dataset_diffusion import (
    MorphologyEncoder,
    load_mask,
    load_scan,
    read_morphometry,
)
from .index_diffusion import assert_no_test_leak, flat_records, load_master_split
from .models_ldm import build_edit_support, build_models, edit
from .probe_qc import QCThresholds, coarse_tissue_map, evaluate_candidate
from .train_stage1_ae import get_index

ATROPHY = {"HA", "ECT", "GCT", "AA"}
DILATION = {"VE", "THD"}


def load_generator(cfg, base, device, ae_checkpoint=None, ldm_checkpoint=None):
    """Load the frozen AE, EMA U-Net/anatomy and latent statistics."""
    base = Path(base)
    ae_path = Path(ae_checkpoint or base / "stage1_ae" / "best.pt")
    ldm_path = Path(ldm_checkpoint or base / "stage2_ldm" / "best.pt")
    ae_ck = torch.load(ae_path, map_location=device, weights_only=False)
    ldm_ck = torch.load(ldm_path, map_location=device, weights_only=False)
    ae, unet, anatomy, scheduler = build_models(cfg["model"], cfg["classes"]["num_classes"], use_checkpoint=False)
    ae.load_state_dict(ae_ck["ae"])
    unet.load_state_dict(ldm_ck["ema_unet"])
    anatomy.load_state_dict(ldm_ck["ema_anatomy"])
    for m in (ae, unet, anatomy):
        m.to(device).eval().requires_grad_(False)
    stats = ldm_ck["latent_stats"]
    if stats["ae_checkpoint_sha256"] != sha256_file(ae_path):
        raise ValueError("The U-Net was trained on latents from a different autoencoder")
    return {
        "ae": ae, "unet": unet, "anatomy": anatomy, "alphas": scheduler.alphas_cumprod.to(device),
        "stats": stats, "encoder": MorphologyEncoder(**ldm_ck.get("morphology", {})),
        "ae_sha256": sha256_file(ae_path), "ldm_sha256": sha256_file(ldm_path),
    }


def anchor_stream(records, seed):
    """Endless (record, pass) stream: one visit per subject per pass, visits rotate."""
    import random

    rng = random.Random(seed)
    visits = {}
    for r in sorted(records, key=lambda r: (r["subject_id"], r["image_id"])):
        visits.setdefault(r["subject_id"], []).append(r)
    subjects = sorted(visits)
    rng.shuffle(subjects)
    p = 0
    while True:
        for s in subjects:
            yield visits[s][p % len(visits[s])], p
        p += 1


def candidate_seed(run_id, group, image_id, variant):
    digest = hashlib.sha256(f"{run_id}|{group}|{image_id}|{variant}".encode()).hexdigest()
    return digest[:16], int(digest[16:24], 16) % (2 ** 31 - 1)


def dose_values(table, records, percentiles):
    """Train-CN percentile targets per concept for the targeted morphology mode."""
    out = {}
    for c in CONCEPTS:
        vals = [table[r["image_id"]][c] for r in records if r["split"] == "train" and r["label"] == 0
                and r["image_id"] in table and np.isfinite(table[r["image_id"]][c])]
        if len(vals) >= 5:
            q = percentiles["atrophy"] if c in ATROPHY else percentiles["dilation"]
            out[c] = float(np.percentile(vals, q))
    return out


def morph_vector(cfg, mode, group, anchor, table, encoder, doses):
    """Return (12-vector or None if unavailable, dose description)."""
    if mode == "class_only":
        return np.zeros(2 * len(CONCEPTS), dtype=np.float32), "class_only"
    values = dict(table.get(anchor["image_id"], {}))
    if not values or any(not np.isfinite(values.get(c, np.nan)) for c in CONCEPTS):
        return None, "measurement_unavailable"
    if group != "PBM":
        if group not in doses:
            return None, "measurement_unavailable"
        values[group] = doses[group]
    return encoder.encode(values), (f"{group}@p{cfg['probes']['dose_percentile']['atrophy' if group in ATROPHY else 'dilation']}"
                                   if group != "PBM" else "retain_anchor")


PSNR_CAP_DB = 100.0  # exact exterior equality gives infinite PSNR; strict JSON cannot store inf


def clean_metric(value):
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if value == float("inf"):
            return PSNR_CAP_DB
        return value if np.isfinite(value) else None
    return value


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def dump_line(entry):
    return json.dumps(entry, default=_json_default) + "\n"


def read_journal(path):
    rows = []
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_probe(array, anchor_path, probe_dir, probe_id, mask_src):
    """Publish volume + companion mask atomically (temp file, then rename)."""
    probe_dir = Path(probe_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    source = nib.load(str(anchor_path))
    header = source.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(np.asarray(array, dtype=np.float32), source.affine, header)
    image.set_qform(source.get_qform(), code=int(source.header["qform_code"]))
    image.set_sform(source.get_sform(), code=int(source.header["sform_code"]))
    vol, mask = probe_dir / f"{probe_id}.nii.gz", probe_dir / f"{probe_id}_mask.nii.gz"
    tmp_vol, tmp_mask = probe_dir / f".{probe_id}.tmp.nii.gz", probe_dir / f".{probe_id}_mask.tmp.nii.gz"
    nib.save(image, str(tmp_vol))
    shutil.copyfile(mask_src, tmp_mask)
    os.replace(tmp_vol, vol)
    os.replace(tmp_mask, mask)
    return vol, mask


def generate_probes(cfg, output_dir=None, index=None, atlas_dir=None, groups=None, target=None,
                    device=None, measure_fn=None, tissue_fn=None, log=print, run_id=None,
                    ae_checkpoint=None, ldm_checkpoint=None):
    exp, pc, sc, qc = cfg["experiment"], cfg["probes"], cfg["sampling"], cfg["qc"]
    set_seed(exp["seed"])
    device = device or choose_device(exp["device"])
    base = Path(output_dir or exp["output_dir"])
    out_root = Path(pc["output_dir"] if output_dir is None else base / "probes")
    out_root.mkdir(parents=True, exist_ok=True)
    groups = list(groups or pc["groups"])
    target = target or pc["per_concept_target"]
    index = get_index(cfg, index)
    records = flat_records(index)
    anchors = [r for r in records if r["split"] == pc["anchor_split"]
               and r["label"] == cfg["classes"]["names"].index(pc["anchor_class"])]
    if not anchors:
        raise ValueError("No eligible Train-CN anchors")
    if pc["anchor_split"] != "train":
        raise ValueError("Probe anchors must come from the train split")
    masks_dir = Path(atlas_dir or cfg["atlas"]["output_dir"])
    cores = load_support_masks(masks_dir, cfg["data"]["expected_shape"])
    atlas_prov = masks_dir / "provenance.json"
    atlas_sha = sha256_file(atlas_prov) if atlas_prov.is_file() else None

    gen = load_generator(cfg, base, device, ae_checkpoint, ldm_checkpoint)
    stats = gen["stats"]
    table = read_morphometry(cfg["data"].get("morphometry_csv"))
    mode = pc["morphology_mode"]
    if mode == "targeted" and not table:
        raise ValueError("morphology_mode=targeted needs data.morphometry_csv")
    doses = dose_values(table, records, pc["dose_percentile"]) if table else {}
    scale_file = json.loads((base / "stage1_ae" / "intensity_scale.json").read_text())
    scale = (scale_file["low"], scale_file["high"])
    thresholds = QCThresholds.from_config(qc)
    tz = qc["tissue_thresholds_z"]
    segment = tissue_fn or (lambda v, b: coarse_tissue_map(v, b, tz["csf_max"], tz["gm_max"]))
    run_id = run_id or sha256_json({"cfg": exp["name"], "seed": exp["seed"], "ae": gen["ae_sha256"],
                                    "ldm": gen["ldm_sha256"]})[:12]
    names = cfg["classes"]["names"]
    spacing = cfg["data"]["expected_spacing"]
    journal_path = out_root / "journal.jsonl"
    journal = read_journal(journal_path)
    if any(j.get("run_id") != run_id for j in journal):
        raise ValueError("Journal belongs to a different run; use a fresh output directory")
    seen_hashes = {j["content_sha256"] for j in journal if j["accepted"]}
    {r["volume_sha256"] for r in anchors}

    summary = {}
    loaded = {}
    for group in groups:
        done = [j for j in journal if j["group"] == group]
        accepted = sum(j["accepted"] for j in done)
        attempts = len(done)
        budget = pc["max_attempt_factor"] * target
        stream = anchor_stream(anchors, exp["seed"] + (sum(map(ord, group)) % 997))
        # Skip anchors already consumed by an earlier (resumed) run.
        for _ in range(attempts):
            next(stream)
        log(f"[{group}] resume accepted={accepted} attempts={attempts} target={target}")
        while accepted < target and attempts < budget:
            for _ in range(min(pc["batch_size"], budget - attempts)):
                if accepted >= target:
                    break
                anchor_rec, pass_idx = next(stream)
                variant = attempts
                cid, seed = candidate_seed(run_id, group, anchor_rec["image_id"], variant)
                attempts += 1
                entry = {"run_id": run_id, "candidate_id": cid, "group": group, "variant": variant,
                         "seed": seed, "anchor_image_id": anchor_rec["image_id"],
                         "anchor_subject_id": anchor_rec["subject_id"], "accepted": False,
                         "reasons": [], "content_sha256": None}
                vec, dose = morph_vector(cfg, mode, group, anchor_rec, table, gen["encoder"], doses)
                if vec is None:
                    entry["reasons"] = ["measurement_unavailable"]
                    journal.append(entry)
                    with journal_path.open("a") as h:
                        h.write(dump_line(entry))
                    continue
                key = anchor_rec["image_id"]
                if key not in loaded:
                    if len(loaded) > 8:
                        loaded.clear()
                    vol, _ = load_scan(anchor_rec["volume_path"], cfg["data"])
                    loaded[key] = (vol, load_mask(anchor_rec["mask_path"], cfg["data"]["expected_shape"]))
                anchor_np, brain = loaded[key]
                concept = group if group != "PBM" else CONCEPTS[variant % len(CONCEPTS)]
                support, alpha = build_edit_support(cores[concept], brain, spacing, sc["pad_mm"],
                                                    sc["domain_dilation_voxels"])
                ctx = names.index(pc["neutral_context"] if group == "PBM" or mode == "targeted"
                                  else pc["concept_context"])
                with torch.inference_mode():
                    composed, proposal = edit(
                        gen["ae"], gen["unet"], gen["anatomy"], gen["alphas"],
                        torch.from_numpy(anchor_np)[None, None].to(device),
                        torch.from_numpy(support)[None, None].to(device),
                        torch.from_numpy(alpha)[None, None].to(device),
                        torch.tensor([ctx], device=device),
                        torch.from_numpy(vec)[None].to(device),
                        stats["offset"], stats["scale"], seed, sc["t_max"], sc["guidance"],
                        cfg["classes"]["num_classes"], sc["steps"], return_proposal=True)
                cand = composed[0, 0].float().cpu().numpy()
                prop = proposal[0, 0].float().cpu().numpy()
                content = hashlib.sha256(np.ascontiguousarray(cand).tobytes()).hexdigest()
                tfn = lambda v, b=brain: segment(v, b)  # noqa: E731
                result = evaluate_candidate(
                    anchor_np, cand, support, brain, scale, thresholds, tfn, group,
                    proposal=prop, measure_fn=measure_fn, seen_hashes=seen_hashes,
                    source_hash=hashlib.sha256(np.ascontiguousarray(anchor_np.astype(np.float32)).tobytes()).hexdigest(),
                    candidate_hash=content)
                entry.update(content_sha256=content, concept=concept, dose=dose, class_context=names[ctx],
                             metrics={k: clean_metric(v) for k, v in result.metrics.items()
                                      if not isinstance(v, dict)},
                             gates={k: bool(v) for k, v in result.gates.items()}, reasons=list(result.reasons),
                             acceptance_level=result.acceptance_level, pass_index=pass_idx)
                if result.accepted:
                    probe_id = f"{group}_{cid}"
                    vol_path, mask_path = save_probe(cand, anchor_rec["volume_path"],
                                                     out_root / group, probe_id, anchor_rec["mask_path"])
                    reload = np.asarray(nib.load(str(vol_path)).dataobj, dtype=np.float32)
                    exterior = ~support
                    if not (np.array_equal(reload, cand) and
                            np.array_equal(reload[exterior].view(np.uint32),
                                           anchor_np.astype(np.float32)[exterior].view(np.uint32))):
                        vol_path.unlink()
                        mask_path.unlink()
                        entry["reasons"].append("reload_mismatch")
                    else:
                        entry.update(accepted=True, probe_id=probe_id, path=str(vol_path),
                                     mask_path=str(mask_path), sha256=sha256_file(vol_path),
                                     mask_sha256=sha256_file(mask_path))
                        write_json(out_root / group / f"{probe_id}.json", entry)
                        seen_hashes.add(content)
                        accepted += 1
                journal.append(entry)
                with journal_path.open("a") as h:
                    h.write(dump_line(entry))
            log(f"[{group}] accepted={accepted}/{target} attempts={attempts}")
        summary[group] = {"accepted": accepted, "attempts": attempts, "shortfall": max(0, target - accepted)}

    return finalize(cfg, out_root, journal, summary, index, gen, atlas_sha, run_id, target)


def finalize(cfg, out_root, journal, summary, index, gen, atlas_sha, run_id, target):
    """Write the TCAV-ready manifest, private source map and generation summary."""
    accepted = [j for j in journal if j["accepted"]]
    splits, _ = load_master_split(cfg["data"]["master_split_path"])
    assert_no_test_leak([{"subject_id": j["anchor_subject_id"]} for j in accepted], splits["test"])
    rows, private = [], []
    for j in accepted:
        m = j["metrics"]
        rows.append({
            "probe_id": j["probe_id"], "group": j["group"], "concept": j["concept"],
            "is_neutral": j["group"] == "PBM", "path": j["path"], "mask_path": j["mask_path"],
            "sha256": j["sha256"], "mask_sha256": j["mask_sha256"],
            "source_split": "train", "anchor_group": hashlib.sha256(j["anchor_subject_id"].encode()).hexdigest()[:12],
            "class_context": j["class_context"], "dose": j["dose"], "seed": j["seed"],
            "acceptance_level": j["acceptance_level"], "run_id": run_id,
            "psnr_exterior": m.get("psnr_exterior"), "ssim_exterior": m.get("ssim_exterior"),
            "offtarget_tissue_drift": m.get("offtarget_tissue_drift"),
            "proxy_csf_delta": m.get("proxy_csf_delta"),
            "ae_sha256": gen["ae_sha256"], "ldm_sha256": gen["ldm_sha256"],
            "template_sha256": cfg["data"]["template_sha256"], "atlas_provenance_sha256": atlas_sha,
            "master_split_sha256": index["master_split_sha256"], "qc_index_sha256": index["qc_index_sha256"],
        })
        private.append({"probe_id": j["probe_id"], "subject_id": j["anchor_subject_id"],
                        "image_id": j["anchor_image_id"]})
    pd.DataFrame(rows).to_csv(out_root / "probe_manifest.csv", index=False)
    pd.DataFrame(private).to_csv(out_root / "probe_source_map_private.csv", index=False)
    per_group = {g: dict(v) for g, v in summary.items()}
    for g in per_group:
        gj = [j for j in journal if j["group"] == g]
        per_group[g]["distinct_source_subjects"] = len({j["anchor_subject_id"] for j in gj if j["accepted"]})
        per_group[g]["unique_hashes"] = len({j["content_sha256"] for j in gj if j["accepted"]})
        per_group[g]["rejections"] = dict(Counter(r for j in gj if not j["accepted"] for r in j["reasons"]))
    report = {
        "run_id": run_id, "target_per_group": target, "groups": per_group,
        "total_accepted": len(accepted),
        "quota_met": all(v["accepted"] >= target for v in per_group.values()),
        "exploratory_only": any(j["acceptance_level"] != "clinical" for j in accepted),
        "note": "Accepted counts are unique content hashes; shortfalls are published, not hidden.",
    }
    write_json(out_root / "generation_summary.json", report)
    return report


def export_for_tcav(manifest_csv, export_dir, classifier_config, config_out=None):
    """Create a classifier-style view of the probes readable by ``src.export_activations``."""
    manifest = pd.read_csv(manifest_csv, dtype=str, keep_default_na=False)
    if manifest.empty:
        raise ValueError("No accepted probes to export")
    export_dir = Path(export_dir)
    vols = export_dir / "volumes"
    vols.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in manifest.iterrows():
        for src, name in ((r["path"], f"{r['probe_id']}_preprocessed_128.nii.gz"),
                          (r["mask_path"], f"{r['probe_id']}_preprocessed_128_mask.nii.gz")):
            dst = vols / name
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
        rows.append({"subject_id": r["probe_id"], "image_id": str(9_000_000 + i), "diagnosis": "CN",
                     "phase": "PROBE", "visit": "synthetic", "group": r["group"],
                     "concept": r["concept"], "is_neutral": r["is_neutral"],
                     "anchor_group": r["anchor_group"], "acceptance_level": r["acceptance_level"]})
    pd.DataFrame(rows).to_csv(export_dir / "probe_classifier_manifest.csv", index=False)
    write_json(export_dir / "probe_splits.json",
               {"train": [r["subject_id"] for r in rows], "val": [], "test": []})
    config_out = Path(config_out or export_dir / "probe_export_config.yaml")
    config_out.write_text(yaml.safe_dump({
        "extends": str(Path(classifier_config).resolve()),
        "experiment": {"name": "dgtcav_probe_export", "output_dir": str(export_dir / "runs")},
        "data": {
            "manifest_path": str(export_dir / "probe_classifier_manifest.csv"),
            "splits_path": str(export_dir / "probe_splits.json"),
            "volume_dir": str(vols), "file_pattern": "{subject_id}_preprocessed_128.nii.gz",
            "require_all_classes": False, "expected_split_sizes": {"train": None, "val": None, "test": None},
        },
    }, sort_keys=False))
    return config_out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--profile", choices=["t4", "p100", "a100"])
    run.add_argument("--output-dir")
    run.add_argument("--atlas-dir")
    run.add_argument("--groups", nargs="*")
    run.add_argument("--target", type=int)
    ex = sub.add_parser("export")
    ex.add_argument("--config", required=True)
    ex.add_argument("--classifier-config", required=True)
    ex.add_argument("--output-dir")
    a = p.parse_args()
    cfg = load_ldm_config(a.config, getattr(a, "profile", None))
    if a.command == "run":
        report = generate_probes(cfg, a.output_dir, atlas_dir=a.atlas_dir, groups=a.groups, target=a.target)
        print(json.dumps(report, indent=2))
        return
    base = Path(a.output_dir) / "probes" if a.output_dir else Path(cfg["probes"]["output_dir"])
    export = Path(a.output_dir) / "probe_export" if a.output_dir else Path(cfg["probes"]["export_dir"])
    print(export_for_tcav(base / "probe_manifest.csv", export, a.classifier_config))


if __name__ == "__main__":
    main()
