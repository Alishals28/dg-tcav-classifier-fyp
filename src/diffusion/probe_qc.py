"""Paired voxel QC, non-target tissue checks, Frechet distance and the Section 5 gate runner.

Appendix B.3 of the blueprint plus ``evaluate_candidate``, which orchestrates the
integrity, paired PSNR/SSIM, exterior invariance, off-target drift and target
morphology gates. Thresholds come from ``configs/diffusion_ldm_128.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter


# ------------------------------------------------------------------- paired QC
def paired_metrics(reference, candidate, region, train_low, train_high):
    """Return paired PSNR/SSIM on a fixed train-defined intensity scale."""
    reference, candidate = (np.asarray(a, dtype=np.float64) for a in (reference, candidate))
    region = np.asarray(region, dtype=bool)
    if reference.ndim != 3 or reference.shape != candidate.shape or region.shape != reference.shape:
        raise ValueError("Paired QC requires matching scalar 3-D grids")
    if not all(np.isfinite(a).all() for a in (reference, candidate, train_low, train_high)):
        raise ValueError("Nonfinite input to image QC")
    if not region.any() or train_high <= train_low:
        raise ValueError("Empty QC region or invalid training intensity scale")
    span = train_high - train_low
    x = np.clip((reference - train_low) / span, 0, 1)
    y = np.clip((candidate - train_low) / span, 0, 1)
    mse = np.mean((x[region] - y[region]) ** 2)

    def local(a):
        return gaussian_filter(a, sigma=1.5, truncate=5 / 1.5, mode="reflect")

    mx, my = local(x), local(y)
    vx = np.maximum(local(x * x) - mx * mx, 0)
    vy = np.maximum(local(y * y) - my * my, 0)
    xy = local(x * y) - mx * my
    ssim = ((2 * mx * my + 0.01 ** 2) * (2 * xy + 0.03 ** 2)
            / ((mx * mx + my * my + 0.01 ** 2) * (vx + vy + 0.03 ** 2)))
    interior = minimum_filter(region, size=11, mode="constant", cval=0)
    if not interior.any():
        raise ValueError("Need at least one 11-cube interior center")
    psnr = float("inf") if mse == 0 else float(-10 * np.log10(mse))
    return psnr, float(ssim[interior].mean())


def preservation(reference, candidate, support, measurement_domain, tissue_a, tissue_b):
    """Exact exterior equality (bitwise float32) and normalized GM/WM/CSF count drift."""
    if not (reference.shape == candidate.shape == support.shape
            == measurement_domain.shape == tissue_a.shape == tissue_b.shape):
        raise ValueError("Voxel and segment grids differ")
    support = np.asarray(support, dtype=bool)
    exterior = ~support
    region = exterior & np.asarray(measurement_domain, dtype=bool)
    if not exterior.any() or not region.any():
        raise ValueError("No non-target brain domain")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Nonfinite voxel")
    # Provided arrays use a frozen coarse segmentation label map on this grid.
    if not np.isin(tissue_a[region], [0, 1, 2, 3]).all() or not np.isin(tissue_b[region], [0, 1, 2, 3]).all():
        raise ValueError("Use 0=other, 1=GM, 2=WM, 3=CSF")
    counts_a = np.array([(tissue_a[region] == label).sum() for label in (1, 2, 3)])
    counts_b = np.array([(tissue_b[region] == label).sum() for label in (1, 2, 3)])
    if not counts_a.sum():
        raise ValueError("No measurable reference tissue outside the support")
    drift = abs(counts_b - counts_a).sum() / max(counts_a.sum(), 1)
    a = np.asarray(reference[exterior], dtype=np.float32).view(np.uint32)
    b = np.asarray(candidate[exterior], dtype=np.float32).view(np.uint32)
    exact = np.array_equal(a, b)
    return exact, float(drift)


def frechet_distance(real_features, generated_features):
    """Squared Gaussian Frechet distance for matched-size feature sets."""
    real, generated = (np.asarray(a, dtype=np.float64) for a in (real_features, generated_features))
    if real.ndim != 2 or generated.shape != real.shape or len(real) < 2:
        raise ValueError("Provide equally sized 2D feature sets")
    if not (np.isfinite(real).all() and np.isfinite(generated).all()):
        raise ValueError("Nonfinite feature set")
    cov_a = np.atleast_2d(np.cov(real, rowvar=False, ddof=1))
    cov_b = np.atleast_2d(np.cov(generated, rowvar=False, ddof=1))
    val, vec = np.linalg.eigh(cov_a)
    root_a = np.einsum("ik,jk,k->ij", vec, vec, np.sqrt(np.maximum(val, 0)))
    middle = np.einsum("ij,jk,kl->il", root_a, cov_b, root_a)
    trace_root = np.sqrt(np.maximum(np.linalg.eigvalsh(0.5 * (middle + middle.T)), 0)).sum()
    delta = real.mean(0) - generated.mean(0)
    value = np.sum(delta ** 2) + np.trace(cov_a) + np.trace(cov_b) - 2 * trace_root
    if not np.isfinite(value):
        raise ValueError("Nonfinite Frechet distance")
    return max(0.0, float(value))


def calibrate_frechet_threshold(features, n=64, resamples=200, quantile=0.95, seed=42):
    """95th percentile of real-real Frechet distances between disjoint equal-size subsets."""
    features = np.asarray(features, dtype=np.float64)
    n = min(n, len(features) // 2)
    if n < 2:
        raise ValueError("Too few real feature vectors to calibrate the Frechet gate")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        pick = rng.permutation(len(features))[: 2 * n]
        values.append(frechet_distance(features[pick[:n]], features[pick[n:]]))
    return float(np.quantile(values, quantile)), n


# ------------------------------------------------------- intensity and tissue maps
def fit_intensity_scale(volumes_and_masks, quantiles=(0.005, 0.995), max_voxels_per_volume=20000, seed=42):
    """Train-fit low/high intensity quantiles over equal-weighted brain samples."""
    rng = np.random.default_rng(seed)
    samples = []
    for volume, mask in volumes_and_masks:
        values = np.asarray(volume)[np.asarray(mask, dtype=bool)]
        if values.size == 0:
            continue
        if values.size > max_voxels_per_volume:
            values = rng.choice(values, max_voxels_per_volume, replace=False)
        samples.append(values.astype(np.float64))
    if not samples:
        raise ValueError("No brain voxels to fit the QC intensity scale")
    low, high = np.quantile(np.concatenate(samples), quantiles)
    if not high > low:
        raise ValueError("Degenerate QC intensity scale")
    return float(low), float(high)


def coarse_tissue_map(volume, brain, csf_max=-0.1, gm_max=0.7):
    """Coarse intensity tissue labels (0 other, 1 GM, 2 WM, 3 CSF) inside the brain mask.

    A frozen, deterministic fallback for software QC. Release gates should replace it
    with an independent segmenter (e.g. SynthSeg) via ``tissue_fn``.
    """
    volume = np.asarray(volume)
    labels = np.zeros(volume.shape, dtype=np.uint8)
    brain = np.asarray(brain, dtype=bool)
    labels[brain & (volume <= csf_max)] = 3
    labels[brain & (volume > csf_max) & (volume <= gm_max)] = 1
    labels[brain & (volume > gm_max)] = 2
    return labels


def csf_fraction(volume, region, tissue_fn):
    """Fraction of ``region`` voxels labeled CSF by ``tissue_fn(volume)`` (proxy measurement)."""
    labels = tissue_fn(volume)
    region = np.asarray(region, dtype=bool)
    return float((labels[region] == 3).mean()) if region.any() else float("nan")


# ------------------------------------------------------------------ gate runner
@dataclass
class QCThresholds:
    psnr_min: float = 28.5
    ssim_min: float = 0.88
    offtarget_drift_max: float = 0.015
    proxy_min_delta: float = 0.002
    pbm_max_abs_delta: float = 0.01

    @classmethod
    def from_config(cls, qc_cfg):
        return cls(**{k: qc_cfg[k] for k in cls.__dataclass_fields__ if k in qc_cfg})


@dataclass
class QCResult:
    accepted: bool
    gates: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    acceptance_level: str = "exploratory"


def evaluate_candidate(anchor, candidate, support, brain, intensity_scale, thresholds,
                       tissue_fn, group, proposal=None, measure_fn=None, seen_hashes=None,
                       source_hash=None, candidate_hash=None):
    """Run the Section 5 gates for one candidate against its anchor.

    ``group`` is a concept code or ``"PBM"``. ``measure_fn(anchor, candidate, support)``
    returns an independently validated ``(passes: bool, details: dict)`` morphology
    verdict; when absent the directional CSF-fraction proxy is used and the result is
    labeled ``exploratory`` (``measurement_unavailable`` for clinical release).
    """
    out = QCResult(accepted=False)
    support = np.asarray(support, dtype=bool)
    brain = np.asarray(brain, dtype=bool)
    low, high = intensity_scale

    finite = bool(np.isfinite(candidate).all()) and candidate.shape == anchor.shape
    out.gates["integrity"] = finite
    if not finite:
        out.reasons.append("nonfinite_or_wrong_shape")
        return out
    unique = (source_hash is None or candidate_hash != source_hash) and (
        seen_hashes is None or candidate_hash not in seen_hashes)
    out.gates["unique_content"] = bool(unique)
    if not unique:
        out.reasons.append("duplicate_content")

    region = brain & ~support
    psnr, ssim = paired_metrics(anchor, candidate, region, low, high)
    out.metrics.update(psnr_exterior=psnr, ssim_exterior=ssim)
    out.gates["psnr"] = psnr >= thresholds.psnr_min
    out.gates["ssim"] = ssim >= thresholds.ssim_min
    if proposal is not None:
        p_raw, s_raw = paired_metrics(anchor, proposal, region, low, high)
        out.metrics.update(psnr_exterior_precomposite=p_raw, ssim_exterior_precomposite=s_raw)
    inside = support & brain
    out.metrics["psnr_support"] = (
        paired_metrics(anchor, candidate, inside, low, high)[0] if _has_interior(inside) else None
    )

    tissue_a, tissue_b = tissue_fn(anchor), tissue_fn(candidate)
    exact, drift = preservation(anchor, candidate, support, brain, tissue_a, tissue_b)
    out.metrics["offtarget_tissue_drift"] = drift
    out.gates["exterior_bitwise"] = bool(exact)
    out.gates["offtarget_drift"] = drift < thresholds.offtarget_drift_max

    if measure_fn is not None:
        passes, details = measure_fn(anchor, candidate, support)
        out.metrics["morphology"] = details
        out.gates["target_morphology"] = bool(passes)
        out.acceptance_level = "clinical"
    else:
        before = csf_fraction(anchor, support & brain, tissue_fn)
        after = csf_fraction(candidate, support & brain, tissue_fn)
        delta = after - before
        out.metrics.update(proxy_csf_before=before, proxy_csf_after=after, proxy_csf_delta=delta)
        if group == "PBM":
            out.gates["target_morphology"] = bool(abs(delta) <= thresholds.pbm_max_abs_delta)
        else:
            out.gates["target_morphology"] = bool(delta >= thresholds.proxy_min_delta)
        out.acceptance_level = "exploratory"
        out.metrics["measurement_status"] = "measurement_unavailable"

    out.reasons += [k for k, v in out.gates.items() if not v and k not in out.reasons]
    out.accepted = all(out.gates.values())
    return out


def _has_interior(mask):
    return bool(minimum_filter(np.asarray(mask, dtype=bool), size=11, mode="constant", cval=0).any())
