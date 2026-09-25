# DG-TCAV: Cohort A MRI classifier

Four-class CN / EMCI / LMCI / AD classification with a MedicalNet-compatible 3D
ResNet-18. Classifier only: no skull stripping, registration, diffusion, or TCAV
scores. No patient scans or real experimental results are included.

## Status and input dependency

The code implements fixed-split loading, training, resume, evaluation, plots, and
feature export. Synthetic tests verify software behavior, not clinical accuracy
or the actual pretrained checkpoint. Real training requires the preprocessing
teammate's QC-approved volumes and the exact reference image used for their grid.
Native-space images are never silently resized or registered by this repository.

## Installation and software check

Python 3.10+, from the repository root. Tests do not require a GPU.

```bash
pip install -r requirements.txt
pip install -e . --no-deps
python -m pytest
python -m scripts.make_synthetic
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m src.train --config configs/smoke_test.yaml
```

On Windows, omit the environment-variable prefix or set it through PowerShell.
Synthetic generation refuses to overwrite a directory; use a new `--output` and
update the config to regenerate. Copy the run path printed by training below.

```bash
python -m src.evaluate --config configs/smoke_test.yaml --checkpoint outputs/<run>/best.pt --split val
python -m src.export_activations --config configs/smoke_test.yaml --checkpoint outputs/<run>/best.pt --output outputs/smoke_features
python -m scripts.tiny_overfit --config configs/smoke_test.yaml
```

Synthetic mode uses 16³ artificial volumes and reduced width. It is labeled in
configs/results and must never be used for ADNI performance claims.

## Preprocessing handoff (not the classifier member's task)

Required: original manifest CSV (subject_id, image_id, diagnosis, phase), original
split JSON (train, val, test), one processed NIfTI per record, exact reference
NIfTI, and preprocessing software/version/QC/failure records. Preserve image IDs
through processing, including when output filenames contain only subject IDs.

Planned counts: train=1134, val=242, test=247. Labels: CN=0, EMCI=1, LMCI=2, AD=3.
No new split or silent exclusions are made. Resolve failed images with the team;
any justified cohort revision requires documented manifest/split/config changes.

Edit the paths and settings in `configs/kaggle.yaml` after receiving the handoff:

```yaml
data:
  manifest_path: /kaggle/input/YOUR_DATASET/cohort_a_baseline_manifest.csv
  splits_path: /kaggle/input/YOUR_DATASET/cohort_a_splits.json
  volume_dir: /kaggle/input/YOUR_DATASET/preprocessed
  file_pattern: "{subject_id}.nii.gz"
  reference_image: /kaggle/input/YOUR_DATASET/reference.nii.gz
  orientation: RAS
  normalization: minmax
  preprocessing_confirmed: true
```

These names/orientation are examples, not verified properties of delivered files.
Set confirmation only after receiving QC. Match orientation to the agreed template.
Inputs must be 91×109×91, 2 mm spacing, with explicit mm units and affine matching
the exact reference grid. These checks do not prove skull stripping or anatomical
registration. A screenshot alone cannot establish spacing.

Filename templates also support `{subject_id}_I{image_id}.nii.gz` or
`{subject_id}/*.nii.gz`; zero or multiple matches fail. Labels come from the CSV,
not filenames. A generic subject filename cannot itself prove the correct baseline
scan was selected: that requires upstream image-ID provenance.

Normalization is per image, never fit across cohorts. Options: minmax (foreground
[0,1]), zscore (foreground standardized), none (already normalized as agreed).
Nonzero voxels define this foreground, not a brain segmentation. Zero background
is preserved before augmentation. Avoid unintended double normalization.

## Kaggle launcher

Use `notebooks/train.ipynb`. Keep authorized ADNI inputs private; do not commit
MRI, clinical tables, subject manifests, credentials, or large checkpoints to this
public repository. Attach only data you are authorized to use.

```bash
git clone --branch classifier-implementation https://github.com/Alishals28/dg-tcav-classifier-fyp.git
cd dg-tcav-classifier-fyp
pip install -r requirements.txt
pip install -e . --no-deps
python -m scripts.validate_dataset --config configs/kaggle.yaml
python -m scripts.tiny_overfit --config configs/kaggle.yaml --output /kaggle/working/tiny_overfit
python -m src.train --config configs/kaggle.yaml
```

After merge, use main. Pin/record the commit for experiments and never pull code
during a run. Installation may need internet; scans and trusted pretrained weights
are attached inputs. Keep Kaggle's GPU-compatible PyTorch when it meets requirements.

Before full training, copy kaggle.yaml to a real-data smoke config with epochs=2,
a separate experiment name, smoke_test=false, and full width. Do not use the
synthetic config for real data. Measure epoch time/GPU memory; batch size 4 is an
initial setting, not a guaranteed fit. No GPU quota/runtime is promised.

The audit normally opens train/validation only. `--include-test-qc` permits image
integrity/geometry checks without model predictions. Training validates all split
metadata but never resolves or loads test images.

Resume using the original resolved config and last.pt:

```bash
python -m src.train --config /kaggle/working/outputs/<run>/config.yaml --resume /kaggle/working/outputs/<run>/last.pt
```

Preserve/download the entire run directory before session expiry. Exact resume
requires unchanged paths/config/data. Do not change the cosine horizon or resume
from best.pt. Hyperparameter changes require a new experiment.
Random generators/workers are seeded. Unsupported deterministic CUDA operations
emit warnings; exact reproducibility across hardware/software is not guaranteed.

## Scientific protocol

- MedicalNet ResNet-18: shortcut A, two blocks per stage, dilation 2/4 in layers
  3/4, replacing its segmentation decoder with pooling and a four-logit head.
- Supported official weights: resnet_18.pth or resnet_18_23dataset.pth, width 64,
  shortcut A. Obtain trusted files from [MedicalNet](https://github.com/Tencent/MedicalNet).
  All backbone tensors must match, except legacy BatchNorm batch counters. Head
  tensors are ignored explicitly; unknown/backbone mismatches fail. A filename
  alone does not establish origin. Checkpoints are loaded with weights_only=True.
- Weighted cross-entropy uses N_train/(4*n_class_train). No weighted sampler is
  also applied. Epoch loss uses the target-weight denominator, not batch size.
- AdamW: lr=1e-4, decay=1e-4. Cosine annealing to 1e-6 over up to 50 epochs.
  Early stopping patience=10; best checkpoint uses validation macro-F1. No implicit
  warm-up/freezing schedule. All layers and BatchNorm are fine-tuned; small-batch
  behavior must be checked. Single-device training.
- Augmentation is off for the initial comparison. augmented.yaml enables small
  TorchIO affine transforms in physical mm and Gaussian noise on training only.
  No flips, arbitrary axis swaps, cropping, or elastic deformation.
- random_baseline.yaml and kaggle.yaml hold loss/topology/augmentation constant,
  changing initialization only. Random/unweighted versus pretrained/weighted
  changes two factors and cannot isolate transfer learning's benefit.
- Select settings on validation, then freeze the protocol before test access.
  Additional seeds/ablations depend on compute, not promised accuracy targets.

## Held-out evaluation

```bash
python -m src.evaluate --config configs/kaggle.yaml --checkpoint /kaggle/working/outputs/<run>/best.pt --split test --confirm-final-test
```

Default split is val. The test flag is a deliberate reminder, not a technical
guarantee of one-time use. Results refuse overwrites. Never use test outcomes for
tuning or choosing the most favorable seed.

Reports include accuracy, balanced accuracy, macro precision/recall/F1, macro OvR
AUC, per-class metrics, multiclass Brier score, per-phase results, confusion matrices,
ROC/PR plots, and subject-aligned probabilities. Missing-class AUC is null, not zero.
Macro-F1 includes all four labels (zero division=0); balanced accuracy is null if
any class is absent. Interpret per-phase metrics with their supports. With all
classes present, balanced accuracy and macro recall are identical.

95% CIs are class-stratified subject bootstraps for this fixed fitted model,
conditional on observed class counts. They do not measure retraining/seed variability.
Binary summaries separate four-class predictions on a subset from forced pairwise
decisions; neither is a dedicated binary-trained classifier.

## Outputs and TCAV handoff

Runs save config, environment/package versions, data audit/input hashes, resolved
index, model summary, history, learning curves, best.pt, last.pt, and best validation
predictions/metrics. Pretrained runs also save a loading report. Evaluation has its
own directory with checkpoint hash, reports, probabilities, and figures.

```bash
python -m src.export_activations --config configs/kaggle.yaml --checkpoint /kaggle/working/outputs/<run>/best.pt --split val --output /kaggle/working/features
```

Exports pooled_features.npy ([N,512] for full width), subjects.csv, provenance.json.
`--spatial` also exports per-subject layer-4 maps. Live model methods retain
autograd; exported arrays do not. This is feature access, not CAV fitting or TCAV.

**TCAV caution:** logits are linear in the final pooled vector, so their gradient
there is constant across subjects. The same issue applies immediately before
pooling. Standard sign-based logit TCAV at this point can be degenerate. The TCAV
stage must justify an earlier nonlinear bottleneck or another explicit design;
the exported 512-vector alone is not claimed to yield informative TCAV scores.

## Code map

| Files | Responsibility |
|---|---|
| config.py, constants.py | Settings and class mapping |
| dataset.py, transforms.py | Fixed-split indexing, checks, normalization, augmentation |
| model.py, medicalnet.py | Backbone/head and strict pretrained loading |
| engine.py, train.py | Epoch loop, selection, resume |
| metrics.py, plots.py, evaluate.py | Results and figures |
| export_activations.py | Features with subject alignment |
| scripts/, tests/ | Audit, inspection, overfit and regression checks |

See [code walkthrough](docs/CODE_WALKTHROUGH.md), [verification evidence](docs/VERIFICATION.md),
and [upstream attribution](THIRD_PARTY_NOTICES.md).
