"""Tests for the conditional 3D latent diffusion and TCAV probe pipeline.

Software-level checks on tiny synthetic 32^3 data (CPU): shape propagation, zero
test-subject leakage, exterior voxel invariance after 50-step masked DDIM editing,
paired QC metrics, training/resume, latent caching, probe generation and export.
Full-resolution (128^3) shapes are verified on the meta device.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import load_config
from src.dataset import MRIDataset, build_index
from src.diffusion import index_diffusion as idx
from src.diffusion.build_atlas_masks import (
    REGIONS,
    TEMPLATE_SHA256,
    build_masks,
    load_support_masks,
)
from src.diffusion.cache_latents import cache_latents, load_latent_stats
from src.diffusion.common import (
    CONCEPTS,
    load_ldm_config,
    lr_at,
    preflight_report,
    sha256_file,
)
from src.diffusion.dataset_diffusion import (
    DiffusionVolumeDataset,
    LatentDataset,
    MorphologyEncoder,
    SubjectRoundSampler,
)
from src.diffusion.generate_tcav_probes import export_for_tcav, generate_probes
from src.diffusion.models_ldm import (
    EMA,
    autoencoder_loss,
    build_edit_support,
    build_models,
    diffusion_loss,
    edit,
)
from src.diffusion.probe_qc import (
    QCThresholds,
    calibrate_frechet_threshold,
    coarse_tissue_map,
    evaluate_candidate,
    fit_intensity_scale,
    frechet_distance,
    paired_metrics,
    preservation,
)
from src.diffusion.train_stage1_ae import run_stage1
from src.diffusion.train_stage2_ldm import run_stage2

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/diffusion_ldm_128.yaml"
N = 32  # synthetic grid edge (divisible by 8)
SPACING = 2.0

TINY_MODEL = {
    "autoencoder": {"channels": [8, 16, 16, 16], "num_res_blocks": [1, 1, 1, 1],
                    "use_checkpoint": False},
    "unet": {"channels": [16, 32, 32], "num_res_blocks": [1, 1, 1],
             "num_head_channels": [16, 16, 16]},
}


# ----------------------------------------------------------------- fixtures
def _affine():
    return np.diag([SPACING, SPACING, SPACING, 1.0])


def _write_scan(path, rng, offset=0.0):
    array = np.full((N,) * 3, -0.4, dtype=np.float32)
    array[2:-2, 2:-2, 2:-2] = rng.normal(0.6 + offset, 0.2, (N - 4,) * 3).astype(np.float32)
    image = nib.Nifti1Image(array, _affine())
    image.header.set_xyzt_units("mm")
    image.set_qform(_affine(), code=1)
    image.set_sform(_affine(), code=1)
    nib.save(image, str(path))
    brain = np.zeros((N,) * 3, dtype=np.uint8)
    brain[2:-2, 2:-2, 2:-2] = 1
    mask = nib.Nifti1Image(brain, _affine())
    mask.header.set_xyzt_units("mm")
    nib.save(mask, str(path).replace(".nii.gz", "_mask.nii.gz"))


def _core_boxes():
    """Six small atlas cores placed on the low-x side so a large exterior remains."""
    boxes = {}
    for i, concept in enumerate(CONCEPTS):
        core = np.zeros((N,) * 3, dtype=bool)
        y = 6 + 4 * (i % 3)
        z = 6 + 8 * (i // 3)
        core[4:8, y:y + 3, z:z + 3] = True
        boxes[concept] = core
    return boxes


def _write_atlas_masks(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for concept, core in _core_boxes().items():
        image = nib.Nifti1Image(core.astype(np.uint8), _affine())
        image.header.set_xyzt_units("mm")
        nib.save(image, str(directory / f"{concept}_support.nii.gz"))
    return directory


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Synthetic cohort: 12 train + 6 val subjects with files; 6 test subjects WITHOUT files."""
    root = tmp_path_factory.mktemp("ldm_world")
    vols = root / "volumes"
    vols.mkdir()
    rng = np.random.default_rng(0)
    plan = {"train": 12, "val": 6, "test": 6}
    diag = ["CN", "EMCI", "AD", "LMCI"]
    splits = {"train": [], "val": [], "test": []}
    manifest, qc = [], []
    image_id = 1000
    for split, count in plan.items():
        for i in range(count):
            subject = f"{split[:2]}_S_{i:04d}"
            splits[split].append(subject)
            for visit in ("scmri", "m12"):
                image_id += 1
                label = diag[(i + (visit == "m12")) % 4]
                if visit == "scmri" and i % 4 == 0:
                    label = "CN"  # guarantee CN anchors at baseline
                manifest.append({
                    "subject_id": subject, "image_id": image_id, "phase": "ADNI2/GO",
                    "visit": visit, "field_strength": 3.0, "series_description": "MPRAGE",
                    "image_date": "2012-01-01", "baseline_diagnosis": diag[i % 4],
                    "visit_diagnosis": label,
                })
                stem = f"{subject}/{image_id}_preprocessed_128"
                qc.append({
                    "subject_id": subject, "image_id": image_id, "pipeline_version": idx.VERSION,
                    "template_sha256": TEMPLATE_SHA256, "status": "SUCCESS",
                    "volume_path": f"{stem}.nii.gz", "mask_path": f"{stem}_mask.nii.gz",
                })
                if split != "test":  # test files are deliberately absent from disk
                    (vols / subject).mkdir(exist_ok=True)
                    _write_scan(vols / f"{stem}.nii.gz", rng, offset=0.01 * image_id % 0.3)
    pd.DataFrame(manifest).to_csv(root / "manifest.csv", index=False)
    pd.DataFrame(qc).to_csv(root / "qc.csv", index=False)
    (root / "splits.json").write_text(json.dumps(splits))
    return {"root": root, "volumes": vols, "splits": splits,
            "manifest": root / "manifest.csv", "qc": root / "qc.csv",
            "split_path": root / "splits.json", "atlas": _write_atlas_masks(root / "atlas")}


def make_cfg(world, out, **qc_over):
    cfg = load_ldm_config(CONFIG, "t4", overrides={
        "experiment": {"output_dir": str(out), "device": "cpu"},
        "data": {
            "master_split_path": str(world["split_path"]),
            "longitudinal_manifest_path": str(world["manifest"]),
            "qc_index_path": str(world["qc"]), "volume_root": str(world["volumes"]),
            "index_path": str(out / "index.json"),
            "expected_shape": [N, N, N], "expected_spacing": [SPACING] * 3,
            "reference_image": None, "expected_master_test_subjects": None,
        },
        "model": TINY_MODEL,
        "latent": {"cache_dir": str(out / "latent_cache")},
        "stage1": {"epochs": 2, "val_every_epochs": 1, "warmup_updates": 1, "num_workers": 0,
                   "val_max_volumes": 3, "early_stopping_patience": 5},
        "stage2": {"epochs": 2, "val_every_epochs": 1, "warmup_updates": 1, "num_workers": 0,
                   "ema_start_updates": 1, "val_timesteps": [100, 500]},
        "sampling": {"t_max": 589, "pad_mm": 2.5, "domain_dilation_voxels": 1},
        "probes": {"per_concept_target": 2, "batch_size": 2, "output_dir": str(out / "probes"),
                   "export_dir": str(out / "export")},
        "qc": {"intensity_scale_volumes": 6, **qc_over},
        "profiles": {"t4": {"precision": "fp16", "stage1": {"microbatch": 2, "accumulation": 2},
                            "stage2": {"microbatch": 2, "accumulation": 2}}},
    })
    return cfg


@pytest.fixture(scope="module")
def pipeline(world, tmp_path_factory):
    """Run index -> Stage 1 -> latent cache -> Stage 2 -> probes once for many assertions."""
    out = tmp_path_factory.mktemp("ldm_run")
    cfg = make_cfg(world, out, proxy_min_delta=-10.0, pbm_max_abs_delta=10.0)
    index, report = idx.audit_inputs(cfg)
    run_stage1(cfg, out, index=index, log=lambda *_: None)
    stats = cache_latents(cfg, out, index=index, log=lambda *_: None)
    run_stage2(cfg, out, index=index, log=lambda *_: None)
    summary = generate_probes(cfg, out, index=index, atlas_dir=world["atlas"],
                              measure_fn=lambda a, c, s: (True, {"stub": 1.0}), log=lambda *_: None)
    return {"cfg": cfg, "out": out, "index": index, "stats": stats, "summary": summary,
            "report": report}


# --------------------------------------------------------------- config tests
@pytest.mark.parametrize("profile", ["t4", "p100", "a100"])
def test_config_profiles_and_contract(profile):
    cfg = load_ldm_config(CONFIG, profile)
    assert cfg["classes"]["names"] == ["CN", "MCI", "AD"]
    assert cfg["classes"]["null_index"] == cfg["classes"]["num_classes"] == 3
    assert cfg["model"]["unet"]["num_class_embeds"] == 4
    assert cfg["data"]["expected_shape"] == [128, 128, 128]
    assert cfg["data"]["normalization"] == "none"
    assert "test" not in cfg["data"]["allowed_splits"]
    assert cfg["data"]["template_sha256"] == TEMPLATE_SHA256 == sha256_file(ROOT / "template/mni152.nii.gz")
    assert cfg["probes"]["per_concept_target"] >= 200
    assert cfg["probes"]["groups"] == [*CONCEPTS, "PBM"]
    assert len(cfg["probes"]["groups"]) * cfg["probes"]["per_concept_target"] >= 1400
    assert cfg["stage2"]["ema_decay"] == 0.999
    assert cfg["stage2"]["cfg_dropout"] == {"class_and_context": 0.10, "morphology_only": 0.15}
    assert cfg["active_profile"]["name"] == profile
    assert dict(cfg["atlas"]["regions"]) == {k: list(v) for k, v in REGIONS.items()}


def test_config_rejects_bad_settings():
    with pytest.raises(ValueError, match="Unknown GPU profile"):
        load_ldm_config(CONFIG, "v100")
    with pytest.raises(ValueError, match="null index"):
        load_ldm_config(CONFIG, overrides={"classes": {"null_index": 4}})
    with pytest.raises(ValueError, match="test partition"):
        load_ldm_config(CONFIG, overrides={"data": {"allowed_splits": ["train", "val", "test"]}})
    with pytest.raises(ValueError, match="divisible by 8"):
        load_ldm_config(CONFIG, overrides={"data": {"expected_shape": [100, 128, 128]}})


def test_lr_schedule_warmup_and_cosine():
    peak, floor = 1e-4, 1e-6
    assert lr_at(0, 100, peak, floor, 10) == pytest.approx(0.1 * peak)
    assert lr_at(10, 100, peak, floor, 10) == pytest.approx(peak)
    assert lr_at(100, 100, peak, floor, 10) == pytest.approx(floor)


# ------------------------------------------------------- shape propagation
def test_full_resolution_shape_propagation_on_meta_device():
    with torch.device("meta"):
        ae, unet, anatomy, scheduler = build_models()
        x = torch.empty(1, 1, 128, 128, 128)
        mu, sigma = ae.encode(x)
        assert mu.shape == sigma.shape == (1, 4, 16, 16, 16)
        assert ae.decode(mu).shape == x.shape
        out = unet(torch.empty(1, 4, 16, 16, 16), torch.zeros(1, dtype=torch.long),
                   context=anatomy(torch.empty(1, 12)).unsqueeze(1),
                   class_labels=torch.zeros(1, dtype=torch.long))
        assert out.shape == (1, 4, 16, 16, 16)
    assert scheduler.num_train_timesteps == 1000
    assert unet.class_embedding.num_embeddings == 4  # CN, MCI, AD + CFG null index 3


def test_512x_spatial_and_128x_value_compression():
    assert 128 ** 3 // 16 ** 3 == 512
    assert 128 ** 3 // (4 * 16 ** 3) == 128


def tiny_models():
    return build_models(TINY_MODEL, 3, use_checkpoint=False)


def test_tiny_models_shapes_and_losses_backward():
    torch.manual_seed(0)
    ae, unet, anatomy, scheduler = tiny_models()
    x = torch.randn(2, 1, N, N, N)
    brain = (torch.rand_like(x) > 0.3).float()
    assert ae.encode(x)[0].shape == (2, 4, 4, 4, 4)
    loss = autoencoder_loss(ae, x, brain)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in ae.encoder.parameters())
    z = torch.randn(3, 4, 4, 4, 4)
    d = diffusion_loss(unet, anatomy, scheduler.alphas_cumprod, z, torch.tensor([0, 1, 2]),
                       torch.randn(3, 12))
    assert torch.isfinite(d)
    d.backward()
    assert anatomy.weight.grad is not None
    assert anatomy.bias is None


def test_cfg_dropout_rates_and_null_index():
    seen = {}

    class Probe(torch.nn.Module):
        def forward(self, x, t, context=None, class_labels=None):
            seen["labels"], seen["context"] = class_labels, context
            return torch.zeros_like(x)

    anatomy = torch.nn.Linear(12, 64, bias=False)
    torch.manual_seed(1)
    n = 20000
    diffusion_loss(Probe(), anatomy, torch.linspace(0.99, 0.01, 1000), torch.zeros(n, 4, 1, 1, 1),
                   torch.zeros(n, dtype=torch.long), torch.ones(n, 12))
    null = (seen["labels"] == 3).float().mean().item()
    assert null == pytest.approx(0.10, abs=0.01)
    zero_ctx = (seen["context"].abs().sum(dim=(1, 2)) == 0).float().mean().item()
    assert zero_ctx == pytest.approx(1 - 0.9 * 0.85, abs=0.015)  # 10 % joint + 15 % of the rest
    morph_only = zero_ctx - null
    assert morph_only == pytest.approx(0.9 * 0.15, abs=0.015)


def test_ema_starts_after_warmup_and_tracks():
    layer = torch.nn.Linear(2, 2)
    ema = EMA([layer], decay=0.9, start=3)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    ema.update(1)
    assert torch.allclose(ema.shadow[0]["weight"], torch.ones(2, 2))  # copied before the start
    with torch.no_grad():
        layer.weight.fill_(2.0)
    ema.update(5)
    assert torch.allclose(ema.shadow[0]["weight"], torch.full((2, 2), 1.1))


# ---------------------------------------------------- masked DDIM edit tests
def _edit_inputs(seed=0):
    torch.manual_seed(seed)
    ae, unet, anatomy, scheduler = tiny_models()
    for m in (ae, unet, anatomy):
        m.eval()
    anchor = torch.randn(1, 1, N, N, N)
    core = _core_boxes()["HA"]
    brain = np.zeros((N,) * 3, dtype=bool)
    brain[2:-2, 2:-2, 2:-2] = True
    support, alpha = build_edit_support(core, brain, (SPACING,) * 3, 2.5, 1)
    return (ae, unet, anatomy, scheduler.alphas_cumprod, anchor,
            torch.from_numpy(support)[None, None], torch.from_numpy(alpha)[None, None])


def test_edit_support_core_taper_and_domain():
    brain = np.zeros((N,) * 3, dtype=bool)
    brain[2:-2, 2:-2, 2:-2] = True
    core = _core_boxes()["VE"]
    support, alpha = build_edit_support(core, brain, (SPACING,) * 3, 2.5, 0)
    assert (alpha[core] == 1).all() and support[core].all()
    assert alpha.max() == 1 and alpha.min() == 0
    assert (alpha[support & ~core] < 1).all() and (alpha[support & ~core] > 0).all()
    assert not support[~brain & ~core].any()  # nothing outside the domain
    with pytest.raises(ValueError):
        build_edit_support(np.zeros_like(core), brain, (SPACING,) * 3)


def test_masked_ddim_edit_keeps_exterior_bit_identical():
    ae, unet, anatomy, alphas, anchor, support, alpha = _edit_inputs()
    calls = []
    unet.register_forward_hook(lambda *_: calls.append(1))
    out, proposal = edit(ae, unet, anatomy, alphas, anchor, support, alpha, torch.tensor([2]),
                         torch.zeros(1, 12), 0.0, 1.0, seed=7, return_proposal=True)
    assert len(calls) == 100  # 50 DDIM steps x (unconditional + conditional) passes
    exterior = ~support
    assert torch.equal(out[exterior].view(torch.int32), anchor[exterior].view(torch.int32))
    assert (out[support] != anchor[support]).any()
    assert not torch.equal(proposal[exterior], anchor[exterior])  # raw decode alone is not invariant
    assert torch.isfinite(out).all()
    again = edit(ae, unet, anatomy, alphas, anchor, support, alpha, torch.tensor([2]),
                 torch.zeros(1, 12), 0.0, 1.0, seed=7)
    other = edit(ae, unet, anatomy, alphas, anchor, support, alpha, torch.tensor([2]),
                 torch.zeros(1, 12), 0.0, 1.0, seed=8)
    assert torch.equal(out, again)
    assert not torch.equal(out, other)
    assert torch.equal(other[exterior], anchor[exterior])


def test_edit_taper_blends_and_core_uses_proposal():
    ae, unet, anatomy, alphas, anchor, support, alpha = _edit_inputs(1)
    out, proposal = edit(ae, unet, anatomy, alphas, anchor, support, alpha, torch.tensor([1]),
                         torch.zeros(1, 12), 0.0, 1.0, seed=3, return_proposal=True)
    expected = anchor + alpha * (proposal - anchor)
    assert torch.allclose(out[support], expected[support], atol=1e-6)


@pytest.mark.parametrize("problem", ["t_max", "empty", "alpha", "shape"])
def test_edit_validates_inputs(problem):
    ae, unet, anatomy, alphas, anchor, support, alpha = _edit_inputs()
    kwargs = dict(label=torch.tensor([0]), morph=torch.zeros(1, 12), latent_offset=0.0,
                  latent_scale=1.0, seed=0)
    t_max = 589
    if problem == "t_max":
        t_max = 10
    elif problem == "empty":
        support = torch.zeros_like(support)
    elif problem == "alpha":
        alpha = alpha + 2
    elif problem == "shape":
        alpha = alpha[..., :-1]
    with pytest.raises(ValueError):
        edit(ae, unet, anatomy, alphas, anchor, support, alpha, t_max=t_max, **kwargs)


# --------------------------------------------------------- paired QC metrics
def test_paired_psnr_ssim_known_values():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(24, 24, 24))
    region = np.ones_like(a, dtype=bool)
    psnr, ssim = paired_metrics(a, a, region, -3, 3)
    assert psnr == float("inf") and ssim == pytest.approx(1.0)
    b = a + 0.05
    psnr_b, ssim_b = paired_metrics(a, b, region, -3, 3)
    assert psnr_b == pytest.approx(-10 * np.log10((0.05 / 6) ** 2), abs=0.05)
    assert 0.9 < ssim_b < 1.0
    noisy = a + rng.normal(scale=1.0, size=a.shape)
    p2, s2 = paired_metrics(a, noisy, region, -3, 3)
    assert p2 < psnr_b and s2 < ssim_b


@pytest.mark.parametrize("bad", ["shape", "nan", "scale", "empty", "small"])
def test_paired_metrics_rejects_invalid_input(bad):
    a = np.zeros((16, 16, 16))
    region = np.ones_like(a, dtype=bool)
    low, high = -1.0, 1.0
    b = a.copy()
    if bad == "shape":
        b = np.zeros((16, 16, 8))
    elif bad == "nan":
        b[0, 0, 0] = np.nan
    elif bad == "scale":
        high = low
    elif bad == "empty":
        region = np.zeros_like(region)
    elif bad == "small":
        region = np.zeros_like(region)
        region[:4, :4, :4] = True
    with pytest.raises(ValueError):
        paired_metrics(a, b, region, low, high)


def test_preservation_detects_exterior_change_and_tissue_drift():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(16, 16, 16)).astype(np.float32)
    support = np.zeros(ref.shape, bool)
    support[2:6, 2:6, 2:6] = True
    domain = np.ones(ref.shape, bool)
    tissue = rng.integers(1, 4, ref.shape).astype(np.uint8)
    cand = ref.copy()
    cand[support] += 1
    exact, drift = preservation(ref, cand, support, domain, tissue, tissue)
    assert exact and drift == 0
    cand[10, 10, 10] = np.nextafter(cand[10, 10, 10], np.float32(9))
    exact, _ = preservation(ref, cand, support, domain, tissue, tissue)
    assert not exact
    changed = tissue.copy()
    changed[~support & domain] = 2
    _, drift = preservation(ref, ref, support, domain, tissue, changed)
    assert drift > 0
    with pytest.raises(ValueError):
        preservation(ref, ref, support, domain, tissue, np.full(ref.shape, 7, np.uint8))


def test_frechet_distance_known_shift_and_calibration():
    rng = np.random.default_rng(2)
    real = rng.normal(size=(400, 4))
    assert frechet_distance(real, real) == pytest.approx(0.0, abs=1e-8)
    shifted = real + np.array([1.0, 0, 0, 0])
    assert frechet_distance(real, shifted) == pytest.approx(1.0, abs=1e-6)
    threshold, n = calibrate_frechet_threshold(rng.normal(size=(200, 4)), n=32, resamples=30)
    assert n == 32 and 0 < threshold < 5
    with pytest.raises(ValueError):
        frechet_distance(real, real[:10])


def _qc_case():
    rng = np.random.default_rng(3)
    anchor = rng.normal(size=(24, 24, 24)).astype(np.float32)
    brain = np.ones(anchor.shape, bool)
    support = np.zeros(anchor.shape, bool)
    support[2:6, 2:6, 2:6] = True
    cand = anchor.copy()
    cand[support] += 0.5
    return anchor, cand, support, brain


def test_evaluate_candidate_gates():
    anchor, cand, support, brain = _qc_case()
    th = QCThresholds()
    tissue = lambda v: coarse_tissue_map(v, brain)  # noqa: E731
    res = evaluate_candidate(anchor, cand, support, brain, (-3, 3), th, tissue, "HA",
                             measure_fn=lambda a, c, s: (True, {}), candidate_hash="x", source_hash="y")
    assert res.accepted and res.acceptance_level == "clinical"
    assert res.metrics["psnr_exterior"] == float("inf")
    # Altering one exterior voxel must fail the bitwise gate.
    bad = cand.copy()
    bad[20, 20, 20] += 1e-3
    res = evaluate_candidate(anchor, bad, support, brain, (-3, 3), th, tissue, "HA",
                             measure_fn=lambda a, c, s: (True, {}))
    assert not res.accepted and "exterior_bitwise" in res.reasons
    # Duplicate content and failed morphology are rejected with reasons.
    res = evaluate_candidate(anchor, cand, support, brain, (-3, 3), th, tissue, "HA",
                             measure_fn=lambda a, c, s: (False, {}), candidate_hash="h",
                             seen_hashes={"h"})
    assert not res.accepted and {"unique_content", "target_morphology"} <= set(res.reasons)
    nan = cand.copy()
    nan[0, 0, 0] = np.nan
    res = evaluate_candidate(anchor, nan, support, brain, (-3, 3), th, tissue, "HA")
    assert not res.accepted and res.reasons == ["nonfinite_or_wrong_shape"]


def test_proxy_gate_is_exploratory_and_directional():
    anchor, cand, support, brain = _qc_case()
    anchor = np.full_like(anchor, 0.5)  # all tissue, no CSF-like voxels
    cand = anchor.copy()
    cand[support] = -1.0  # becomes CSF-like
    th = QCThresholds(proxy_min_delta=0.002)
    tissue = lambda v: coarse_tissue_map(v, brain)  # noqa: E731
    pos = evaluate_candidate(anchor, cand, support, brain, (-3, 3), th, tissue, "VE")
    assert pos.gates["target_morphology"] and pos.accepted
    assert pos.acceptance_level == "exploratory"
    assert pos.metrics["measurement_status"] == "measurement_unavailable"
    neutral = evaluate_candidate(anchor, cand, support, brain, (-3, 3), th, tissue, "PBM")
    assert not neutral.gates["target_morphology"]  # PBM must not change the CSF fraction
    unchanged = evaluate_candidate(anchor, anchor + 0, support, brain, (-3, 3), th, tissue, "VE",
                                   candidate_hash="a", source_hash="a")
    assert not unchanged.accepted  # verbatim copy of the source


def test_intensity_scale_and_tissue_map():
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(16, 16, 16))
    mask = np.ones_like(vol, bool)
    low, high = fit_intensity_scale([(vol, mask)] * 2)
    assert low < 0 < high
    labels = coarse_tissue_map(np.array([[[-1.0, 0.3, 2.0]]]), np.ones((1, 1, 3), bool))
    assert labels.ravel().tolist() == [3, 1, 2]
    with pytest.raises(ValueError):
        fit_intensity_scale([])


# ------------------------------------------------------------ atlas masks
def _scan_on_template_grid(path, template, bad=False):
    qt = template.get_qform()
    affine = qt.copy()
    affine[:3, :3] = qt[:3, :3] @ np.diag(np.asarray(template.shape) / 128)
    if bad:
        affine[0, 3] += 5.0
    image = nib.Nifti1Image(np.zeros((128,) * 3, np.float32), affine)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, str(path))


def _synthetic_cerebra(path):
    labels = sorted({i for ids in REGIONS.values() for i in ids})
    data = np.zeros((160, 200, 170), dtype=np.int16)
    for k, label in enumerate(labels):
        x, y, z = (k % 4) * 30 + 10, ((k // 4) % 4) * 36 + 12, (k // 16) * 60 + 40
        data[x:x + 10, y:y + 10, z:z + 10] = label
    affine = np.eye(4)
    affine[:3, 3] = (-78, -112, -72)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return labels


def test_atlas_masks_on_verified_grid(tmp_path):
    template_path = ROOT / "template/mni152.nii.gz"
    template = nib.load(str(template_path))
    scan = tmp_path / "scan.nii.gz"
    _scan_on_template_grid(scan, template)
    atlas = tmp_path / "atlas.nii.gz"
    labels = _synthetic_cerebra(atlas)
    digest = sha256_file(atlas)
    meta = build_masks(atlas, template_path, scan, tmp_path / "out", digest)
    assert set(meta["voxels"]) == set(CONCEPTS)
    assert all(v > 0 for v in meta["voxels"].values())
    assert meta["voxels"]["GCT"] > meta["voxels"]["HA"] > 0  # 12 labels vs 2
    masks = load_support_masks(tmp_path / "out")
    assert all(m.shape == (128, 128, 128) for m in masks.values())
    assert not (masks["HA"] & masks["AA"]).any()
    assert len(labels) == 22
    assert json.loads((tmp_path / "out/provenance.json").read_text())["template_sha256"] == TEMPLATE_SHA256
    with pytest.raises(FileExistsError):
        build_masks(atlas, template_path, scan, tmp_path / "out", digest)


def test_atlas_masks_reject_bad_inputs(tmp_path):
    template_path = ROOT / "template/mni152.nii.gz"
    template = nib.load(str(template_path))
    scan, bad_scan = tmp_path / "scan.nii.gz", tmp_path / "bad.nii.gz"
    _scan_on_template_grid(scan, template)
    _scan_on_template_grid(bad_scan, template, bad=True)
    atlas = tmp_path / "atlas.nii.gz"
    _synthetic_cerebra(atlas)
    digest = sha256_file(atlas)
    with pytest.raises(ValueError, match="atlas digest"):
        build_masks(atlas, template_path, scan, tmp_path / "o1", "0" * 64)
    with pytest.raises(ValueError, match="atlas digest"):
        build_masks(atlas, template_path, scan, tmp_path / "o1b", None)
    with pytest.raises(ValueError, match="pinned preprocessing contract"):
        build_masks(atlas, template_path, bad_scan, tmp_path / "o2", digest)
    other_template = tmp_path / "other.nii.gz"
    shutil.copy(scan, other_template)
    with pytest.raises(ValueError, match="template version"):
        build_masks(atlas, other_template, scan, tmp_path / "o3", digest)
    # An atlas that lacks a required label is rejected.
    data = np.asarray(nib.load(str(atlas)).dataobj).copy()
    data[data == 48] = 0
    partial = tmp_path / "partial.nii.gz"
    nib.save(nib.Nifti1Image(data, nib.load(str(atlas)).affine), str(partial))
    with pytest.raises(ValueError, match="absent"):
        build_masks(partial, template_path, scan, tmp_path / "o4", sha256_file(partial))


# ------------------------------------------- index: zero test-subject leakage
def _generate(world, **kw):
    args = dict(master_split=world["split_path"], longitudinal_manifest=world["manifest"],
                qc_index=world["qc"], volume_root=world["volumes"])
    args.update(kw)
    return idx.generate_index(**args)


def test_index_contains_only_train_and_val_and_blocks_all_test_subjects(world):
    index = _generate(world)
    assert set(index["partitions"]) == {"train", "val"}
    test_ids = set(world["splits"]["test"])
    for split in ("train", "val"):
        owners = {r["subject_id"] for r in index["partitions"][split]}
        assert owners <= set(world["splits"][split])
        assert not owners & test_ids
    assert len(index["partitions"]["train"]) == 24 and len(index["partitions"]["val"]) == 12
    assert index["test_denylist_size"] == 6
    assert {r["label"] for r in index["partitions"]["train"]} == {0, 1, 2}
    assert all(r["source_label"] in idx.LABELS for r in index["partitions"]["val"])
    summary = idx.index_summary(index)
    assert summary["train"]["subjects"] == 12 and summary["val"]["subjects"] == 6
    idx.assert_no_test_leak(index["partitions"], world["splits"]["test"])
    leaked = {"train": [{"subject_id": world["splits"]["test"][0]}]}
    with pytest.raises(ValueError, match="Test subject"):
        idx.assert_no_test_leak(leaked, world["splits"]["test"])
    with pytest.raises(ValueError, match="train and val"):
        idx.load_index(_write(world, {"partitions": {"train": [], "test": []}}))


def _write(world, content):
    path = world["root"] / "tmp_index.json"
    path.write_text(json.dumps(content))
    return path


def test_test_scans_are_never_opened_or_hashed(world, monkeypatch):
    touched = []
    original = idx.file_digest

    def spy(path):
        touched.append(str(path))
        return original(path)

    monkeypatch.setattr(idx, "file_digest", spy)
    _generate(world)
    assert touched
    assert not [p for p in touched if "te_S_" in p]


def test_index_rejects_test_path_aliasing_and_bad_split(world, tmp_path):
    rows = list(csv.DictReader(world["qc"].open()))
    dev = next(r for r in rows if r["subject_id"].startswith("tr_S_"))
    test = next(r for r in rows if r["subject_id"].startswith("te_S_"))
    test["volume_path"] = dev["volume_path"]  # a test row pointing at a development scan
    alias = tmp_path / "alias.csv"
    with alias.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="aliases a test-owned"):
        _generate(world, qc_index=alias)
    with pytest.raises(ValueError, match="Expected 247"):
        _generate(world, expected_test_subjects=247)
    overlapping = json.loads(world["split_path"].read_text())
    overlapping["val"][0] = overlapping["train"][0]
    bad = tmp_path / "overlap.json"
    bad.write_text(json.dumps(overlapping))
    with pytest.raises(ValueError, match="overlapping"):
        _generate(world, master_split=bad)


def test_index_excludes_ineligible_qc_rows_with_reasons(world, tmp_path):
    frame = pd.read_csv(world["qc"], dtype=str)
    frame.loc[0, "status"] = "FAILED"
    frame.loc[1, "pipeline_version"] = "other_version"
    frame.loc[2, "template_sha256"] = "0" * 64
    frame = frame.drop(index=3)
    path = tmp_path / "qc.csv"
    frame.to_csv(path, index=False)
    index = _generate(world, qc_index=path)
    reasons = idx.index_summary(index)["excluded_reasons"]
    assert reasons == {"qc_status_not_success": 1, "pipeline_version_mismatch": 1,
                       "template_digest_mismatch": 1, "missing_qc_row": 1}
    assert len(index["partitions"]["train"]) == 20


def test_index_rejects_duplicate_mri_bytes(world, tmp_path):
    rows = list(csv.DictReader(world["qc"].open()))
    dev = [r for r in rows if r["subject_id"].startswith("tr_S_")]
    shutil.copy(world["volumes"] / dev[0]["volume_path"], world["volumes"] / dev[1]["volume_path"])
    try:
        with pytest.raises(ValueError, match="Repeated MRI bytes"):
            _generate(world)
    finally:  # restore a distinct scan so later tests are unaffected
        _write_scan(world["volumes"] / dev[1]["volume_path"], np.random.default_rng(99))


def test_qc_index_builder_for_existing_baseline_scans(tmp_path):
    base = tmp_path / "baseline.csv"
    pd.DataFrame({"subject_id": ["a", "b", "c"], "image_id": ["I11", "12", "13"],
                  "diagnosis": ["CN", "AD", "CN"], "phase": ["x"] * 3}).to_csv(base, index=False)
    proc = tmp_path / "proc.csv"
    pd.DataFrame({"subject_id": ["a", "a", "b"], "status": ["FAILED", "SUCCESS", "FAILED"]}).to_csv(proc, index=False)
    frame = idx.build_qc_index(base, proc, tmp_path, tmp_path / "qc.csv")
    assert frame.set_index("subject_id").status.to_dict() == {"a": "SUCCESS", "b": "FAILED", "c": "MISSING"}
    assert frame.volume_path.iloc[0] == "a_preprocessed_128.nii.gz"
    assert frame.mask_path.iloc[0] == "a_preprocessed_128_mask.nii.gz"
    assert frame.image_id.tolist() == ["11", "12", "13"]


# ------------------------------------------------------------------ datasets
def test_dataset_shapes_dtypes_and_zscore_contract(world):
    index = _generate(world)
    ds = DiffusionVolumeDataset(index["partitions"]["train"], make_cfg(world, world["root"] / "x")["data"])
    item = ds[0]
    assert item["image"].shape == item["brain"].shape == (1, N, N, N)
    assert item["image"].dtype == torch.float32
    assert item["image"].min() < 0  # background stays negative: no min-max or clipping
    assert item["brain"].unique().tolist() == [0.0, 1.0]
    assert item["morph"].shape == (12,) and item["morph"].abs().sum() == 0
    assert item["label"] in (0, 1, 2) and item["split"] == "train"


def test_dataset_refuses_test_records(world):
    index = _generate(world)
    record = dict(index["partitions"]["train"][0], split="test")
    with pytest.raises(ValueError, match="forbidden split"):
        DiffusionVolumeDataset([record], make_cfg(world, world["root"] / "x")["data"])
    with pytest.raises(ValueError, match="forbidden split"):
        LatentDataset([record], world["root"])
    with pytest.raises(ValueError):
        DiffusionVolumeDataset([], {})


def test_subject_round_sampler_one_visit_per_subject_per_round(world):
    records = _generate(world)["partitions"]["train"]
    sampler = SubjectRoundSampler(records, seed=1)
    order = list(sampler)
    assert len(order) == len(records)
    subjects = [records[i]["subject_id"] for i in order]
    first_round = subjects[:12]
    assert len(set(first_round)) == 12  # each subject once per round
    visits = {}
    for i in order:
        visits.setdefault(records[i]["subject_id"], []).append(records[i]["image_id"])
    assert all(len(v) == 2 and v[0] != v[1] for v in visits.values())  # visits rotate
    sampler.set_epoch(1)
    assert list(sampler) != order


def test_morphology_encoder_robust_standardization():
    table = {str(i): {c: float(i) for c in CONCEPTS} for i in range(1, 21)}
    records = [{"image_id": str(i), "split": "train", "label": 0} for i in range(1, 21)]
    enc = MorphologyEncoder.fit(table, records)
    vec = enc.encode({"HA": 10.5, "ECT": None, "VE": 1000.0})
    assert vec.shape == (12,)
    assert vec[0] == pytest.approx(0.0, abs=0.01) and vec[6] == 1
    assert vec[1] == 0 and vec[7] == 0  # unavailable -> zero value, flag false
    assert vec[2] == 5.0  # clipped at +5
    assert enc.encode(None).sum() == 0


# ----------------------------------------------------- end-to-end pipeline
def test_audit_report(pipeline):
    rep = pipeline["report"]
    assert rep["passed"] and rep["summary"]["train"]["visits"] == 24
    assert rep["expected_visits_vs_accepted"]["train"]["expected"] == 4232


def test_stage1_checkpoints_validation_and_resume(pipeline, world):
    cfg, out = pipeline["cfg"], pipeline["out"]
    stage1 = out / "stage1_ae"
    assert (stage1 / "best.pt").is_file() and (stage1 / "last.pt").is_file()
    rows = [json.loads(line) for line in (stage1 / "metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [1, 2]
    assert all(np.isfinite(r["train_loss"]) and np.isfinite(r["val_ssim"]) for r in rows)
    ck = torch.load(stage1 / "last.pt", weights_only=False)
    assert ck["state"]["epoch"] == 2 and ck["state"]["updates"] >= 2
    assert {"optimizer", "rng", "provenance", "scale"} <= set(ck)
    report = run_stage1(cfg, out, index=pipeline["index"], validate_only=True, log=lambda *_: None)
    assert {"psnr", "ssim", "weighted_l1", "gates"} <= set(report)
    # Resuming a finished run performs no additional epochs.
    state = run_stage1(cfg, out, index=pipeline["index"], log=lambda *_: None)
    assert state["epoch"] == 2
    # A checkpoint from a different config is refused.
    other = make_cfg(world, out)
    other["stage1"]["lr"] = 5e-5
    with pytest.raises(ValueError, match="provenance"):
        run_stage1(other, out, index=pipeline["index"], log=lambda *_: None)


def test_stage1_resume_continues_midway(world, tmp_path):
    cfg = make_cfg(world, tmp_path)
    index, _ = idx.audit_inputs(cfg)
    first = run_stage1(cfg, tmp_path, index=index, max_updates=2, log=lambda *_: None)
    assert first["updates"] == 2
    final = run_stage1(cfg, tmp_path, index=index, log=lambda *_: None)
    assert final["epoch"] == 2 and final["updates"] > 2


def test_latent_cache_shapes_and_train_only_float64_stats(pipeline):
    stats, cfg, index = pipeline["stats"], pipeline["cfg"], pipeline["index"]
    cache = Path(cfg["latent"]["cache_dir"])
    train = index["partitions"]["train"]
    arrays = [np.load(cache / "train" / f"{r['subject_id']}_{r['image_id']}.npy") for r in train]
    assert arrays[0].shape == (4, 4, 4, 4) and arrays[0].dtype == np.float32  # [4,16,16,16] at 128^3
    assert stats["fit_split"] == "train" and stats["train_scans"] == len(train)
    assert np.mean([a.astype(np.float64).mean() for a in arrays]) == pytest.approx(0, abs=1e-5)
    assert np.concatenate([a.ravel() for a in arrays]).std() == pytest.approx(1.0, abs=1e-3)
    assert stats["scale"] == pytest.approx(1 / stats["std"])
    assert len(list((cache / "val").glob("*.npy"))) == len(index["partitions"]["val"])
    loaded = load_latent_stats(cache / "latent_stats.json")
    assert loaded["offset"] == stats["offset"]
    test_ids = set(json.loads(Path(cfg["data"]["master_split_path"]).read_text())["test"])
    cached_subjects = {p.name.rsplit("_", 1)[0] for p in cache.rglob("*.npy")}
    assert cached_subjects and not cached_subjects & test_ids


def test_stage2_checkpoint_has_ema_and_null_class(pipeline):
    ck = torch.load(pipeline["out"] / "stage2_ldm/best.pt", weights_only=False)
    assert {"unet", "anatomy", "ema_unet", "ema_anatomy", "latent_stats", "optimizer"} <= set(ck)
    assert ck["latent_stats"]["ae_checkpoint_sha256"] == sha256_file(pipeline["out"] / "stage1_ae/best.pt")
    assert ck["unet"]["class_embedding.weight"].shape[0] == 4
    assert not torch.equal(ck["unet"]["class_embedding.weight"], ck["ema_unet"]["class_embedding.weight"])
    rows = [json.loads(line) for line in (pipeline["out"] / "stage2_ldm/metrics.jsonl").read_text().splitlines()]
    assert all(np.isfinite(r["val_eps_mse"]) for r in rows)


def test_preflight_reports_admission(world, tmp_path):
    cfg = make_cfg(world, tmp_path)
    cfg["stage1"]["epochs"] = 10  # 6 updates per epoch -> the 1 + 20 update preflight fits
    index, _ = idx.audit_inputs(cfg)
    report = run_stage1(cfg, tmp_path, index=index, preflight=True, log=lambda *_: None)
    assert report["admitted"] and report["updates"] == 21 and report["all_losses_finite"]
    assert not (tmp_path / "stage1_ae/last.pt").exists()
    fake = preflight_report(torch.device("cpu"), 1, 1.0, [float("nan")])
    assert not fake["admitted"]


def test_probe_generation_quota_integrity_and_leakage(pipeline, world):
    summary, out = pipeline["summary"], pipeline["out"]
    assert summary["quota_met"] and summary["total_accepted"] == 14
    assert set(summary["groups"]) == {*CONCEPTS, "PBM"}
    manifest = pd.read_csv(out / "probes/probe_manifest.csv")
    assert len(manifest) == 14 and manifest.sha256.is_unique and manifest.probe_id.is_unique
    assert manifest.groupby("group").size().eq(2).all()
    assert set(manifest.source_split) == {"train"}
    test_ids = set(world["splits"]["test"])
    private = pd.read_csv(out / "probes/probe_source_map_private.csv")
    assert not set(private.subject_id) & test_ids
    assert set(private.subject_id) <= set(world["splits"]["train"])
    assert "subject_id" not in manifest.columns  # participant identity stays in the private map
    assert manifest.is_neutral.sum() == 2
    assert {"psnr_exterior", "ssim_exterior", "ae_sha256", "template_sha256"} <= set(manifest.columns)
    assert (manifest.psnr_exterior >= 28.5).all() and (manifest.ssim_exterior >= 0.88).all()
    for _, row in manifest.iterrows():
        assert Path(row["path"]).is_file() and Path(row["mask_path"]).is_file()
        assert sha256_file(row["path"]) == row["sha256"]
    assert summary["groups"]["HA"]["distinct_source_subjects"] == 2


def test_probe_exterior_is_bitwise_anchor_after_reload(pipeline):
    out, cfg = pipeline["out"], pipeline["cfg"]
    manifest = pd.read_csv(out / "probes/probe_manifest.csv")
    private = pd.read_csv(out / "probes/probe_source_map_private.csv").set_index("probe_id")
    records = {(r["subject_id"], r["image_id"]): r for r in pipeline["index"]["partitions"]["train"]}
    cores = _core_boxes()
    for _, row in manifest.iterrows():
        src = private.loc[row.probe_id]
        record = records[(src.subject_id, str(src.image_id))]
        anchor = np.asarray(nib.load(record["volume_path"]).dataobj, dtype=np.float32)
        probe = np.asarray(nib.load(row["path"]).dataobj, dtype=np.float32)
        assert probe.shape == anchor.shape and np.isfinite(probe).all()
        concept = row["concept"]
        support, _ = build_edit_support(cores[concept], np.asarray(nib.load(record["mask_path"]).dataobj) > 0,
                                        (SPACING,) * 3, cfg["sampling"]["pad_mm"],
                                        cfg["sampling"]["domain_dilation_voxels"])
        assert np.array_equal(probe[~support].view(np.uint32), anchor[~support].view(np.uint32))
        assert not np.array_equal(probe, anchor)
        assert nib.load(row["path"]).affine.tolist() == nib.load(record["volume_path"]).affine.tolist()


def test_probe_resume_adds_no_duplicates(pipeline, world):
    cfg, out = pipeline["cfg"], pipeline["out"]
    before = pd.read_csv(out / "probes/probe_manifest.csv")
    journal_lines = len((out / "probes/journal.jsonl").read_text().splitlines())
    summary = generate_probes(cfg, out, index=pipeline["index"], atlas_dir=world["atlas"],
                              measure_fn=lambda a, c, s: (True, {}), log=lambda *_: None)
    after = pd.read_csv(out / "probes/probe_manifest.csv")
    assert summary["total_accepted"] == 14 and len(after) == len(before)
    assert len((out / "probes/journal.jsonl").read_text().splitlines()) == journal_lines
    assert before.sha256.tolist() == after.sha256.tolist()


def test_probe_shortfall_is_reported_not_hidden(pipeline, world, tmp_path):
    cfg = make_cfg(world, tmp_path)
    cfg["probes"]["max_attempt_factor"] = 1
    for sub in ("stage1_ae", "stage2_ldm"):
        shutil.copytree(pipeline["out"] / sub, tmp_path / sub)
    summary = generate_probes(cfg, tmp_path, index=pipeline["index"], atlas_dir=world["atlas"],
                              groups=["HA", "PBM"], measure_fn=lambda a, c, s: (False, {}),
                              log=lambda *_: None)
    assert not summary["quota_met"] and summary["total_accepted"] == 0
    assert summary["groups"]["HA"]["shortfall"] == 2
    assert summary["groups"]["HA"]["rejections"]["target_morphology"] == 2
    assert not list((tmp_path / "probes/HA").glob("*.nii.gz"))  # rejected volumes are never saved


def test_default_proxy_acceptance_is_labeled_exploratory(world, tmp_path, pipeline):
    cfg = make_cfg(world, tmp_path, proxy_min_delta=-10.0, pbm_max_abs_delta=10.0)
    for sub in ("stage1_ae", "stage2_ldm"):
        shutil.copytree(pipeline["out"] / sub, tmp_path / sub)
    summary = generate_probes(cfg, tmp_path, index=pipeline["index"], atlas_dir=world["atlas"],
                              groups=["AA"], target=1, log=lambda *_: None)
    assert summary["exploratory_only"] and summary["total_accepted"] == 1
    row = pd.read_csv(tmp_path / "probes/probe_manifest.csv").iloc[0]
    assert row.acceptance_level == "exploratory"


def test_probe_export_is_compatible_with_classifier_loader(pipeline):
    out = pipeline["out"]
    cfg_path = export_for_tcav(out / "probes/probe_manifest.csv", out / "export",
                               ROOT / "configs/preprocessed_128.yaml")
    export_cfg = load_config(cfg_path)
    assert export_cfg["model"]["num_classes"] == 4  # the frozen classifier config is untouched
    export_cfg["data"].update(expected_shape=[N, N, N], expected_spacing=[SPACING] * 3,
                              reference_image=None, orientation="RAS")
    index = build_index(export_cfg["data"], resolve_splits=("train",))
    assert (index.split == "train").sum() == 14 and index.path[index.split == "train"].str.len().gt(0).all()
    item = MRIDataset(index, "train", export_cfg["data"])[0]
    assert item["image"].shape == (1, N, N, N)
    frame = pd.read_csv(out / "export/probe_classifier_manifest.csv", dtype=str)
    assert frame.image_id.is_unique and set(frame.diagnosis) == {"CN"}
    assert set(frame.group) == {*CONCEPTS, "PBM"}
    assert json.loads((out / "export/probe_splits.json").read_text())["test"] == []
