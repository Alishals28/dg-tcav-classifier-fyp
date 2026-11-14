# DG-TCAV 3D MRI Classifier

Four-class Alzheimer's disease staging (`CN`, `EMCI`, `LMCI`, `AD`) on `128×128×128` MNI152-aligned T1-weighted structural MRI scans from ADNI (`N = 1,593` unique baseline subjects: `1,112` train, `236` validation, `245` locked test).

## Key Features & Supervisor Upgrades Included

1. **Three Supported 3D Architectures (`src/model.py`) with 512-D TCAV Bottleneck**:
   - `ResNet18`: Full MedicalNet 3D ResNet-18 (`resnet_18_23dataset.pth`) with configurable `dropout` (default `0.5` in regularized profile) and spatial attention pooling.
   - `HybridInceptionResNet3D`: **MedicalNet-pretrained 3D ResNet stem (`conv1`..`layer3`) + Multi-Scale 3D Inception-ResNet `layer4`** ($1^3$, $3^3$, dilated $5^3$ RF, and MaxPool branches with residual skip connections). Loads 100% of `conv1`..`layer3` weights from `resnet_18_23dataset.pth`.
   - `Inception3D`: Low-compute multi-scale 3D Inception network (~3.1M parameters, 10× smaller than ResNet-18).
2. **Medial-Temporal 3D Spatial Attention Pooling (`pooling: roi_attention | attention | gap`)**:
   - Prevents flat $16 	imes 16 	imes 16$ Global Average Pooling from washing out localized hippocampal/entorhinal atrophy with 97% normal background brain tissue.
3. **Head Warm-Up & Layer-Wise Differential Learning Rates (`src/train.py`)**:
   - `warmup_epochs: 5` freezes `conv1`..`layer3` while warming up the classification head and `layer4`, then unfreezes the stem at `backbone_lr_scale: 0.2` (5× lower learning rate than the head).
4. **L2 Regularization Sweep (`weight_decay` & `l2_lambda`), Label Smoothing, & Ordinal Continuum Penalty (`src/engine.py`)**:
   - Supports decoupled `weight_decay`, explicit L2 weight norm penalty `l2_lambda`, `label_smoothing`, and ordinal continuum loss (`ordinal_loss_weight`) along $	ext{CN} 	o 	ext{EMCI} 	o 	ext{LMCI} 	o 	ext{AD}$, with `augmentation.enabled: false`.
5. **Companion Brain Masking (`apply_companion_mask: true`) & Explicit Pairwise Sub-Task Logging**:
   - Applies `{subject_id}_preprocessed_128_mask.nii.gz` when present and logs `AD_vs_CN` and `EMCI_vs_LMCI` accuracy at every epoch.

## Clean Repository Structure

```text
├── configs/
│   ├── base.yaml                              # Base defaults & 1,593-subject data contract
│   ├── resnet18_medicalnet.yaml               # Locked Candidate C reference (0.4459 Val F1 / 48.31% Acc)
│   ├── resnet18_medicalnet_regularized.yaml   # MedicalNet ResNet-18 + Dropout 0.5 + L2 + ROI Attention + Warmup
│   ├── hybrid_inception_resnet3d.yaml         # Hybrid MedicalNet Stem + 3D Inception-ResNet + Dropout 0.5 + L2
│   ├── inception3d.yaml                       # Low-compute Multi-Scale 3D Inception (~3.1M params)
│   └── smoke_test.yaml                        # Fast synthetic CPU profile for pytest
├── data/preprocessed_128_classifier_inputs/   # Audited 1,593-subject manifest & subject-level splits
├── docs/                                      # Runbook, code walkthrough & supervisor meeting report
├── notebooks/train.ipynb                      # Clean 6-cell Google Colab launcher
├── scripts/
│   ├── run_l2_sweep.py                        # Automated L2 regularization (lambda) sweep runner
│   ├── prepare_preprocessed_manifest.py       # Reconcile manifest from preprocessing logs
│   ├── validate_dataset.py                    # Audit NIfTI geometry, affines & SHA-256 hashes
│   ├── inspect_volume.py                      # Orthogonal 3-plane slice viewer
│   ├── tiny_overfit.py                        # 8-scan memorization sanity check
│   └── make_synthetic.py                      # Synthetic 16^3 dataset generator for unit tests
├── src/                                       # Core package
├── template/mni152.nii.gz                     # Reference MNI152 template
└── tests/                                     # Automated pytest suite
```
