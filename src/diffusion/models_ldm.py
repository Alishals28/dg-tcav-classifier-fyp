"""MONAI AutoencoderKL (128^3 -> 16^3), conditional U-Net, losses and masked DDIM editing.

Appendix B.2 of the blueprint. All reductions run in float32. The decoder output is
linear (no sigmoid/tanh/min-max) so the stored z-score domain is preserved.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from scipy.ndimage import binary_dilation, distance_transform_edt
from torch import nn
from torch.nn import functional as F

DEFAULT_MODEL_CFG = {
    "autoencoder": {"channels": [16, 32, 64, 96], "num_res_blocks": [1, 1, 2, 2],
                    "latent_channels": 4, "norm_num_groups": 8, "use_checkpoint": True},
    "unet": {"channels": [64, 96, 128], "num_res_blocks": [2, 2, 2],
             "attention_levels": [False, True, True], "num_head_channels": [32, 32, 32],
             "norm_num_groups": 8, "cross_attention_dim": 64, "transformer_num_layers": 1,
             "upcast_attention": True, "use_flash_attention": False},
    "anatomy": {"num_measurements": 6, "token_dim": 64},
    "scheduler": {"num_train_timesteps": 1000, "schedule": "scaled_linear_beta",
                  "beta_start": 0.0015, "beta_end": 0.0195},
}


def build_models(model_cfg=None, num_classes=3, use_checkpoint=None):
    """Return ``(ae, unet, anatomy, scheduler)``. ``model_cfg`` may override defaults."""
    cfg = deepcopy(DEFAULT_MODEL_CFG)
    for section, values in (model_cfg or {}).items():
        cfg.setdefault(section, {}).update(values)
    a, u, m, s = cfg["autoencoder"], cfg["unet"], cfg["anatomy"], cfg["scheduler"]
    levels = len(a["channels"])
    ae = AutoencoderKL(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=tuple(a["channels"]), num_res_blocks=tuple(a["num_res_blocks"]),
        latent_channels=a["latent_channels"], attention_levels=(False,) * levels,
        norm_num_groups=a["norm_num_groups"],
        with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
        use_checkpoint=a["use_checkpoint"] if use_checkpoint is None else use_checkpoint,
        use_convtranspose=False,
    )
    unet = DiffusionModelUNet(
        spatial_dims=3, in_channels=a["latent_channels"], out_channels=a["latent_channels"],
        channels=tuple(u["channels"]), num_res_blocks=tuple(u["num_res_blocks"]),
        attention_levels=tuple(u["attention_levels"]),
        num_head_channels=tuple(u["num_head_channels"]),
        norm_num_groups=u["norm_num_groups"], with_conditioning=True,
        cross_attention_dim=u["cross_attention_dim"],
        transformer_num_layers=u["transformer_num_layers"],
        num_class_embeds=num_classes + 1, upcast_attention=u["upcast_attention"],
        use_flash_attention=u["use_flash_attention"],
    )
    # Six standardized measurements + six availability flags -> one context token.
    anatomy = nn.Linear(2 * m["num_measurements"], m["token_dim"], bias=False)
    scheduler = DDPMScheduler(
        num_train_timesteps=s["num_train_timesteps"], schedule=s["schedule"],
        beta_start=s["beta_start"], beta_end=s["beta_end"],
        prediction_type="epsilon", clip_sample=False,
    )
    return ae, unet, anatomy, scheduler


# ---------------------------------------------------------------- latent scaling
def scale_latent(mu, offset, scale):
    return (mu.float() - offset) * scale


def unscale_latent(z, offset, scale):
    return z / scale + offset


# ------------------------------------------------------------------------ losses
def autoencoder_loss(ae, x, brain, edge_weight=0.1, kl_weight=1e-5, brain_weight=2.0):
    """weighted_L1 + 0.1 * edge_L1 + 1e-5 * KL_mean (float32 reductions)."""
    mu, sigma = ae.encode(x)
    recon = ae.decode(ae.sampling(mu, sigma)).float()
    x, brain, mu, sigma = x.float(), brain.float(), mu.float(), sigma.float()
    vox = (recon - x).abs()
    weights = 1 + brain_weight * brain
    fidelity = (vox * weights).sum() / weights.sum()
    edge = sum((recon.diff(dim=d) - x.diff(dim=d)).abs().mean() for d in (2, 3, 4)) / 3
    var = sigma.square().clamp_min(1e-12)
    kl = 0.5 * (mu.square() + var - var.log() - 1).mean()
    return fidelity + edge_weight * edge + kl_weight * kl


def diffusion_loss(unet, anatomy, alphas, scaled_z, labels, morph, num_classes=3,
                   p_class=0.10, p_morph=0.15, steps=None, noise=None):
    """Epsilon MSE with learned-unconditional CFG dropout.

    ``p_class`` masks the class label and the measurements together; ``p_morph``
    additionally masks only the measurements (explicit class-only control).
    """
    batch = scaled_z.shape[0]
    device = scaled_z.device
    if steps is None:
        steps = torch.randint(len(alphas), (batch,), device=device)
    alpha = alphas.to(device=device, dtype=torch.float32)[steps].view(batch, 1, 1, 1, 1)
    eps = torch.randn_like(scaled_z) if noise is None else noise
    noisy = alpha.sqrt() * scaled_z + (1 - alpha).sqrt() * eps
    drop_label = torch.rand(batch, device=device) < p_class
    drop_measurements = drop_label | (torch.rand(batch, device=device) < p_morph)
    labels = labels.masked_fill(drop_label, num_classes)
    morph = morph.masked_fill(drop_measurements[:, None], 0)
    pred = unet(noisy, steps, context=anatomy(morph).unsqueeze(1), class_labels=labels)
    return F.mse_loss(pred.float(), eps.float())


class EMA:
    """Exponential moving average over one or more modules (EMA starts after warm-up)."""

    def __init__(self, modules, decay=0.999, start=500):
        self.modules = list(modules)
        self.decay, self.start = decay, start
        self.shadow = [{k: v.detach().clone().float() for k, v in m.state_dict().items()}
                       for m in self.modules]

    @torch.no_grad()
    def update(self, updates_done):
        for m, shadow in zip(self.modules, self.shadow):
            for k, v in m.state_dict().items():
                v = v.detach().float()
                if updates_done < self.start or not v.is_floating_point():
                    shadow[k].copy_(v)
                else:
                    shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def state_dict(self):
        return [{k: v.clone() for k, v in s.items()} for s in self.shadow]

    def load_state_dict(self, state):
        for shadow, saved in zip(self.shadow, state):
            for k in shadow:
                shadow[k].copy_(saved[k])

    def copy_to(self, modules):
        for m, shadow in zip(modules, self.shadow):
            m.load_state_dict({k: v.to(next(m.parameters()).dtype) for k, v in shadow.items()})


# ---------------------------------------------------------------- edit support
def build_edit_support(core, brain, spacing, pad_mm=2.5, dilation_voxels=2):
    """Boolean ``support`` and float32 ``alpha_voxel`` (1 in the core, linear taper over the pad).

    ``distance_transform_edt(~core, sampling=spacing)`` measures the physical distance
    (mm) from the core. The pad stays inside the dilated intracranial domain
    (brain mask plus a CSF margin) or the core itself.
    """
    core = np.asarray(core, dtype=bool)
    if not core.any():
        raise ValueError("Empty atlas core")
    distance = distance_transform_edt(~core, sampling=tuple(float(s) for s in spacing))
    alpha = np.clip(1 - distance / pad_mm, 0, 1).astype(np.float32)
    alpha[core] = 1.0
    domain = binary_dilation(np.asarray(brain, dtype=bool), iterations=dilation_voxels) if dilation_voxels else np.asarray(brain, dtype=bool)
    domain = domain | core
    alpha = np.where(domain, alpha, 0).astype(np.float32)
    return alpha > 0, alpha


# ------------------------------------------------------------------ DDIM editor
@torch.inference_mode()
def edit(ae, unet, anatomy, alphas, anchor, support, alpha_voxel, label,
         morph, latent_offset, latent_scale, seed, t_max=589, guidance=2.0,
         num_classes=3, steps=50, return_proposal=False):
    """50-step masked DDIM edit. Canonical intensity on the model path; exterior == anchor.

    Returns ``torch.where(support, anchor + alpha_voxel * (proposal - anchor), anchor)``,
    so every voxel outside ``support`` is bit-identical to the anchor.
    """
    if not steps - 1 <= t_max < len(alphas):
        raise ValueError(f"Need {steps} distinct timesteps in the training range")
    if anchor.shape != support.shape or anchor.shape != alpha_voxel.shape:
        raise ValueError("Image and mask shapes must agree")
    if support.dtype != torch.bool or not bool(support.any()):
        raise ValueError("An edit requires nonempty Boolean support")
    if bool((~torch.isfinite(alpha_voxel) | (alpha_voxel < 0) | (alpha_voxel > 1)).any()):
        raise ValueError("Blend weights must be finite and in [0, 1]")
    z = (ae.encode(anchor)[0].float() - latent_offset) * latent_scale
    mask = F.max_pool3d(support.float(), kernel_size=8, stride=8) > 0
    if mask.shape[2:] != z.shape[2:]:
        raise ValueError("Support grid and latent shape disagree")
    device = z.device
    abar = alphas.to(device=device, dtype=torch.float32)
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(z.shape, device=device, dtype=z.dtype, generator=g)
    timesteps = torch.linspace(t_max, 0, steps).round().to(torch.int64).tolist()
    if len(set(timesteps)) != steps:
        raise ValueError("Timesteps are not distinct")
    context = anatomy(morph).unsqueeze(1)
    null_context = torch.zeros_like(context)
    null_label = torch.full_like(label, num_classes)

    def anchor_at(a):
        return a.sqrt() * z + (1 - a).sqrt() * noise

    sample = anchor_at(abar[timesteps[0]])
    for j, now in enumerate(timesteps):
        current_alpha = abar[now]
        next_alpha = abar[timesteps[j + 1]] if j + 1 < len(timesteps) else abar.new_tensor(1)
        timestep = torch.full_like(label, now)
        eps_uncond = unet(sample, timestep, context=null_context, class_labels=null_label)
        eps_cond = unet(sample, timestep, context=context, class_labels=label)
        eps = eps_uncond.float() + guidance * (eps_cond.float() - eps_uncond.float())
        clean = (sample - (1 - current_alpha).sqrt() * eps) / current_alpha.sqrt()
        update = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * eps
        sample = torch.where(mask, update, anchor_at(next_alpha))

    proposal = ae.decode(sample / latent_scale + latent_offset).float()
    if not bool(torch.isfinite(proposal).all()):
        raise ValueError("Decoder produced a nonfinite candidate")
    composed = torch.where(support, anchor + alpha_voxel * (proposal - anchor), anchor)
    return (composed, proposal) if return_proposal else composed
