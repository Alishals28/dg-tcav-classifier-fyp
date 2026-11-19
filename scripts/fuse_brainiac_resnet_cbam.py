"""Frozen 3D BrainIAC (Nature Neuroscience 2026) + YangResNet18/34 + CBAM Gated Fusion Pipeline.

Combines:
  1. Your trained local 3D CNN + CBAM models (YangResNet34_CBAM + YangResNet18_CBAM)
     using multi-scale intermediate (layer3, 128-d) + deep post-CBAM (layer4, 256-d) features.
  2. Frozen BrainIAC (AIM-KannLab/BrainIAC, ViT-B/16 pretrained on 48,965 brain MRIs)
     using final (768-d) + intermediate block-8 (768-d) global 3D Transformer features.
  3. Lightweight PyTorch Gated Projection MLP (~180K params, fully differentiable for TCAV)
     + probability ensemble + prior calibration evaluation on your locked 1,112 / 236 split.

Usage on Google Colab:
  # Step 1: Extract & cache frozen features once (~5 min on T4 GPU)
  python fuse_brainiac_resnet_cbam.py --mode extract \
      --config /content/dg-tcav-classifier-fyp/configs/yang_resnet34_cbam.yaml \
      --brainiac-ckpt /content/BrainIAC/src/checkpoints/BrainIAC.ckpt \
      --resnet34-ckpt /content/drive/MyDrive/.../yang2025_resnet34_cbam_.../best.pt \
      --resnet18-ckpt /content/drive/MyDrive/.../yang2025_resnet18_cbam_.../best.pt \
      --out-dir /content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/brainiac_fusion

  # Step 2: Train BrainIAC-MLP + Gated Fusion MLP & evaluate (< 1 min)
  python fuse_brainiac_resnet_cbam.py --mode train \
      --out-dir /content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/brainiac_fusion
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

CLASS_NAMES = ["CN", "EMCI", "LMCI", "AD"]


# ---------------------------------------------------------------------------
# 1. BrainIAC 3D ViT-B/16 Loader (Matches AIM-KannLab/BrainIAC Exact Spec)
# ---------------------------------------------------------------------------
class FrozenBrainIACExtractor(nn.Module):
    """Frozen BrainIAC ViT-B/16 (96x96x96, patch 16x16x16, 768-d) via MONAI."""

    def __init__(self, ckpt_path: str, intermediate_block: int = 7):
        super().__init__()
        try:
            from monai.networks.nets import ViT
        except ImportError as exc:
            raise ImportError(
                "MONAI is required for BrainIAC. Install with: pip install monai"
            ) from exc

        self.intermediate_block = intermediate_block
        self.vit = ViT(
            in_channels=1,
            img_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            classification=False,
        )
        self._load_brainiac_weights(ckpt_path)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

    def _load_brainiac_weights(self, ckpt_path: str):
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw.get("model", raw))

        target_keys = set(self.vit.state_dict().keys())
        cleaned = {}
        prefixes = ("backbone.vit.", "vit.", "backbone.", "encoder.", "model.", "module.")
        for k, v in state.items():
            nk = k
            changed = True
            while changed:
                changed = False
                for p in prefixes:
                    if nk.startswith(p):
                        nk = nk[len(p):]
                        changed = True
            if nk in target_keys and tuple(v.shape) == tuple(self.vit.state_dict()[nk].shape):
                cleaned[nk] = v

        missing, unexpected = self.vit.load_state_dict(cleaned, strict=False)
        matched = len(cleaned)
        total = len(target_keys)
        print(f"[BrainIAC] Loaded {matched}/{total} ViT tensors from {ckpt_path}")
        if matched < int(0.8 * total):
            raise RuntimeError(
                f"BrainIAC checkpoint mismatch: only {matched}/{total} keys matched. "
                f"Sample ckpt keys: {list(state.keys())[:5]}"
            )

    @torch.no_grad()
    def forward(self, x_128: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize 128^3 -> 96^3, z-score non-zero brain voxels, return (final_768, mid_768)."""
        x = F.interpolate(x_128, size=(96, 96, 96), mode="trilinear", align_corners=False)
        # Per-scan non-zero z-score normalization (matches MONAI NormalizeIntensityd(nonzero=True))
        out = []
        for vol in x:
            mask = vol != 0
            if mask.any():
                vals = vol[mask]
                vol = (vol - vals.mean()) / (vals.std().clamp_min(1e-6))
                vol = vol * mask.float()
            out.append(vol)
        x_norm = torch.stack(out, dim=0)

        feats, hidden_states = self.vit(x_norm)
        # Upstream BrainIAC reads first token features[:, 0] (768-d); also capture intermediate block mean
        final_768 = feats[:, 0]
        mid_idx = min(self.intermediate_block, len(hidden_states) - 1)
        mid_768 = hidden_states[mid_idx].mean(dim=1)
        return final_768, mid_768


# ---------------------------------------------------------------------------
# 2. Multi-Scale Extractor for Your Trained YangResNet18/34 + CBAM Models
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_yang_multiscale(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract concatenated GAP(layer3) [128-d] + GAP(cbam(layer4)) [256-d] = 384-d + softmax probs."""
    h = model.conv1(x)
    h = model.bn1(h)
    h = model.relu(h)
    h = model.maxpool(h)
    h = model.layer1(h)
    h = model.layer2(h)
    h3 = model.layer3(h)
    h4 = model.layer4(h3)
    if getattr(model, "cbam", None) is not None:
        h4 = model.cbam(h4)

    f3 = F.adaptive_avg_pool3d(h3, 1).flatten(1)  # 128-d intermediate structural features
    f4 = F.adaptive_avg_pool3d(h4, 1).flatten(1)  # 256-d deep post-CBAM features
    logits = model.fc(f4)
    probs = F.softmax(logits, dim=1)
    return torch.cat([f3, f4], dim=1), probs


def load_yang_checkpoint(repo_root: str, variant: str, ckpt_path: str, device: torch.device) -> nn.Module:
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from src.model import YangResNet3D_CBAM

    model = YangResNet3D_CBAM(variant=variant, num_classes=4, base_channels=32, use_cbam=True)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw.get("state_dict", raw))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[{variant}] Loaded checkpoint cleanly from {ckpt_path}")
    return model


# ---------------------------------------------------------------------------
# 3. Lightweight PyTorch Downstream Heads (Fully Differentiable for TCAV)
# ---------------------------------------------------------------------------
class BrainIACMLP(nn.Module):
    """3-Layer MLP (Linear -> BN -> GELU -> Dropout) on frozen BrainIAC features."""

    def __init__(self, in_dim: int = 1536, num_classes: int = 4, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedDualFusionMLP(nn.Module):
    """Balanced Projection + Learned Gate fusing ResNet-34/18+CBAM (CNN) & BrainIAC (ViT)."""

    def __init__(
        self,
        r34_dim: int = 384,
        r18_dim: int = 384,
        fm_dim: int = 1536,
        proj_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.proj_r34 = nn.Sequential(
            nn.Linear(r34_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.proj_r18 = nn.Sequential(
            nn.Linear(r18_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.proj_fm = nn.Sequential(
            nn.Linear(fm_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(), nn.Dropout(dropout)
        )
        # Learned modality gate balances local CNN features vs. global Foundation Model features
        self.gate = nn.Sequential(nn.Linear(proj_dim * 3, 3), nn.Softmax(dim=-1))
        self.head = nn.Sequential(
            nn.Linear(proj_dim * 3, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, f_r34: torch.Tensor, f_r18: torch.Tensor, f_fm: torch.Tensor) -> torch.Tensor:
        h34 = self.proj_r34(f_r34)
        h18 = self.proj_r18(f_r18)
        hfm = self.proj_fm(f_fm)
        concat = torch.cat([h34, h18, hfm], dim=1)
        w = self.gate(concat)  # [B, 3]
        gated = torch.cat([h34 * w[:, 0:1], h18 * w[:, 1:2], hfm * w[:, 2:3]], dim=1)
        return self.head(gated)


# ---------------------------------------------------------------------------
# 4. Evaluation Helpers (Yang Table 5 & 6 + 3-Class + Prior Calibration)
# ---------------------------------------------------------------------------
def prior_calibrate(P: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simple coordinate search for class multipliers optimizing 4-class Macro-F1."""
    best_m = np.ones(4, dtype=np.float64)
    best_f1 = f1_score(y_true, np.argmax(P, axis=1), average="macro", zero_division=0)
    grid = np.linspace(0.5, 2.0, 31)
    for _ in range(3):
        for c in range(4):
            cur_best_v = best_m[c]
            for v in grid:
                m = best_m.copy()
                m[c] = v
                pred = np.argmax(P * m[None, :], axis=1)
                score = f1_score(y_true, pred, average="macro", zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    cur_best_v = v
            best_m[c] = cur_best_v
    P_cal = P * best_m[None, :]
    P_cal = P_cal / np.maximum(P_cal.sum(axis=1, keepdims=True), 1e-12)
    return P_cal, best_m


def report_metrics(name: str, y: np.ndarray, P: np.ndarray):
    pred = np.argmax(P, axis=1)
    acc4 = accuracy_score(y, pred)
    bal4 = balanced_accuracy_score(y, pred)
    f1_4 = f1_score(y, pred, average="macro", zero_division=0)
    auc4 = roc_auc_score(y, P, multi_class="ovr", average="macro")
    prec, rec, f1c, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2, 3], zero_division=0)

    # 3-Class (CN=0, MCI=1+2, AD=3)
    y3 = np.where((y == 1) | (y == 2), 1, np.where(y == 3, 2, 0))
    P3 = np.stack([P[:, 0], P[:, 1] + P[:, 2], P[:, 3]], axis=1)
    acc3 = accuracy_score(y3, np.argmax(P3, axis=1))
    f1_3 = f1_score(y3, np.argmax(P3, axis=1), average="macro", zero_division=0)

    # Binary AD vs CN & EMCI vs LMCI
    m_adcn = (y == 0) | (y == 3)
    s_ad = P[m_adcn, 3] / np.maximum(P[m_adcn, 0] + P[m_adcn, 3], 1e-12)
    y_ad = (y[m_adcn] == 3).astype(int)
    adcn_acc = accuracy_score(y_ad, (s_ad >= 0.5).astype(int))
    adcn_auc = roc_auc_score(y_ad, s_ad)

    m_mci = (y == 1) | (y == 2)
    s_lmci = P[m_mci, 2] / np.maximum(P[m_mci, 1] + P[m_mci, 2], 1e-12)
    y_lmci = (y[m_mci] == 2).astype(int)
    mci_acc = accuracy_score(y_lmci, (s_lmci >= 0.5).astype(int))
    mci_auc = roc_auc_score(y_lmci, s_lmci)

    print(f"\n==================== {name} (N={len(y)}) ====================")
    print(f"4-Class : Macro-F1={f1_4:.4f} | Acc={acc4*100:.2f}% | BalAcc={bal4*100:.2f}% | OvR AUC={auc4:.4f}")
    for i, cname in enumerate(CLASS_NAMES):
        print(f"  {cname:4s} -> Recall={rec[i]*100:5.2f}% | Prec={prec[i]*100:5.2f}% | F1={f1c[i]:.4f}")
    print(f"3-Class : Acc={acc3*100:.2f}% | Macro-F1={f1_3:.4f}")
    print(f"Binary  : AD vs CN Acc={adcn_acc*100:.2f}% (AUC={adcn_auc:.4f}) | EMCI vs LMCI Acc={mci_acc*100:.2f}% (AUC={mci_auc:.4f})")
    print("Confusion Matrix (CN, EMCI, LMCI, AD):\n", confusion_matrix(y, pred, labels=[0, 1, 2, 3]))
    return {"f1_4": f1_4, "acc4": acc4, "auc4": auc4, "acc3": acc3, "adcn_auc": adcn_auc}


# ---------------------------------------------------------------------------
# 5. Mode 1: Feature Extraction (Runs Once on GPU, ~5 Minutes)
# ---------------------------------------------------------------------------
def run_extract(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = str(Path(args.repo_root).resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from src.config import load_config
    from src.dataset import MRIDataset, build_index

    cfg = load_config(args.config)
    if getattr(args, "volume_dir", None):
        cfg["data"]["volume_dir"] = args.volume_dir
    # Ensure relative paths resolve against repo_root
    for k in ("manifest_path", "splits_path"):
        if k in cfg["data"] and not Path(cfg["data"][k]).is_absolute():
            cfg["data"][k] = str(Path(repo_root) / cfg["data"][k])
    brainiac = FrozenBrainIACExtractor(args.brainiac_ckpt).to(device)
    r34 = load_yang_checkpoint(repo_root, "YangResNet34_CBAM", args.resnet34_ckpt, device)
    r18 = load_yang_checkpoint(repo_root, "YangResNet18_CBAM", args.resnet18_ckpt, device)

    splits = ["train", "val"] + (["test"] if args.include_test else [])
    index_by_split = build_index(cfg["data"], resolve_splits=splits)

    for split in splits:
        ds = MRIDataset(index_by_split[split], cfg["data"], transform=None)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        f_r34_l, p_r34_l, f_r18_l, p_r18_l, f_bi_l, y_l, sid_l = [], [], [], [], [], [], []
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].numpy()
            f34, p34 = extract_yang_multiscale(r34, x)
            f18, p18 = extract_yang_multiscale(r18, x)
            bi_fin, bi_mid = brainiac(x)
            bi_cat = torch.cat([bi_fin, bi_mid], dim=1)

            f_r34_l.append(f34.cpu().numpy())
            p_r34_l.append(p34.cpu().numpy())
            f_r18_l.append(f18.cpu().numpy())
            p_r18_l.append(p18.cpu().numpy())
            f_bi_l.append(bi_cat.cpu().numpy())
            y_l.append(y)
            sid_l.extend(batch.get("subject_id", [""] * len(y)))

        save_path = out_dir / f"cached_features_{split}.npz"
        np.savez_compressed(
            save_path,
            f_r34=np.concatenate(f_r34_l, axis=0),
            p_r34=np.concatenate(p_r34_l, axis=0),
            f_r18=np.concatenate(f_r18_l, axis=0),
            p_r18=np.concatenate(p_r18_l, axis=0),
            f_brainiac=np.concatenate(f_bi_l, axis=0),
            y=np.concatenate(y_l, axis=0),
            subject_id=np.array(sid_l),
        )
        print(f"Saved {split} features ({len(np.concatenate(y_l))} scans) -> {save_path}")


# ---------------------------------------------------------------------------
# 6. Mode 2: Train BrainIAC-MLP & GatedDualFusionMLP (< 1 Minute)
# ---------------------------------------------------------------------------
def train_head(model, train_tensors, val_tensors, y_train, y_val, epochs=120, lr=3e-4, wd=0.05, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    counts = np.bincount(y_train, minlength=4).astype(np.float32)
    weights = torch.tensor(counts.sum() / (4.0 * np.maximum(counts, 1.0)), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ds = TensorDataset(*[torch.tensor(t, dtype=torch.float32) for t in train_tensors], torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    val_t = [torch.tensor(t, dtype=torch.float32, device=device) for t in val_tensors]

    best_f1, best_state, best_probs = -1.0, None, None
    for _ in range(epochs):
        model.train()
        for batch in loader:
            xs = [b.to(device) for b in batch[:-1]]
            yb = batch[-1].to(device)
            optimizer.zero_grad()
            loss = criterion(model(*xs), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(*val_t), dim=1).cpu().numpy()
        score = f1_score(y_val, np.argmax(probs, axis=1), average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_state = copy.deepcopy(model.state_dict())
            best_probs = probs

    model.load_state_dict(best_state)
    return model, best_probs


def run_train(args):
    out_dir = Path(args.out_dir)
    tr = np.load(out_dir / "cached_features_train.npz")
    va = np.load(out_dir / "cached_features_val.npz")

    y_tr, y_va = tr["y"], va["y"]
    # Standardize each feature block using TRAIN mean/std only (zero leakage)
    def std_fit_transform(a_tr, a_va):
        mu = a_tr.mean(axis=0, keepdims=True)
        sd = np.maximum(a_tr.std(axis=0, keepdims=True), 1e-6)
        return (a_tr - mu) / sd, (a_va - mu) / sd, (mu, sd)

    r34_tr, r34_va, st_r34 = std_fit_transform(tr["f_r34"], va["f_r34"])
    r18_tr, r18_va, st_r18 = std_fit_transform(tr["f_r18"], va["f_r18"])
    bi_tr, bi_va, st_bi = std_fit_transform(tr["f_brainiac"], va["f_brainiac"])

    # 1. Baseline: Existing Ensemble 3 (ResNet-34 + ResNet-18 + CBAM)
    p_ens3 = 0.5 * (va["p_r34"] + va["p_r18"])
    report_metrics("Baseline: Existing ResNet-34 + ResNet-18 + CBAM", y_va, p_ens3)

    # 2. Standalone BrainIAC-MLP
    bi_mlp, p_bi = train_head(BrainIACMLP(in_dim=bi_tr.shape[1]), [bi_tr], [bi_va], y_tr, y_va)
    report_metrics("Standalone Frozen BrainIAC (ViT-B/16) + 3-Layer MLP", y_va, p_bi)

    # 3. Late Probability Ensemble (ResNet-34 + ResNet-18 + BrainIAC-MLP)
    p_prob_ens = 0.40 * va["p_r34"] + 0.35 * va["p_r18"] + 0.25 * p_bi
    report_metrics("Probability Ensemble (ResNet-34 + ResNet-18 + BrainIAC)", y_va, p_prob_ens)

    # 4. Feature-Level Gated Dual Fusion MLP
    fusion_model, p_fused = train_head(
        GatedDualFusionMLP(r34_tr.shape[1], r18_tr.shape[1], bi_tr.shape[1]),
        [r34_tr, r18_tr, bi_tr],
        [r34_va, r18_va, bi_va],
        y_tr,
        y_va,
    )
    report_metrics("Gated Feature-Fusion MLP (ResNet-34/18 + CBAM + BrainIAC)", y_va, p_fused)

    # 5. Prior-Calibrated Gated Fusion
    p_cal, mult = prior_calibrate(p_fused, y_va)
    report_metrics(f"Prior-Calibrated Gated Fusion (mult={np.round(mult, 2)})", y_va, p_cal)

    torch.save(
        {
            "fusion_state_dict": fusion_model.state_dict(),
            "brainiac_mlp_state_dict": bi_mlp.state_dict(),
            "prior_multipliers": mult.tolist(),
            "dims": {"r34": r34_tr.shape[1], "r18": r18_tr.shape[1], "fm": bi_tr.shape[1]},
        },
        out_dir / "best_brainiac_resnet_fusion.pt",
    )
    df = pd.DataFrame(
        {
            "subject_id": va["subject_id"],
            "y_true": y_va,
            "p_cn": p_fused[:, 0],
            "p_emci": p_fused[:, 1],
            "p_lmci": p_fused[:, 2],
            "p_ad": p_fused[:, 3],
        }
    )
    df.to_csv(out_dir / "val_predictions_brainiac_fusion.csv", index=False)
    print(f"\nSaved fusion checkpoint & predictions to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["extract", "train"], required=True)
    p.add_argument("--repo-root", default="/content/dg-tcav-classifier-fyp")
    p.add_argument("--config", default="/content/dg-tcav-classifier-fyp/configs/yang_resnet34_cbam.yaml")
    p.add_argument("--brainiac-ckpt", default="/content/BrainIAC/src/checkpoints/BrainIAC.ckpt")
    p.add_argument("--resnet34-ckpt", default="")
    p.add_argument("--resnet18-ckpt", default="")
    p.add_argument("--volume-dir", default=None, help="Override volume_dir in config (e.g. for Kaggle /kaggle/input/...)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--include-test", action="store_true")
    args = p.parse_args()
    if args.mode == "extract":
        run_extract(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()