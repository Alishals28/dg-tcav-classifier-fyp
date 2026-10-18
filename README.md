# DG-TCAV classifier

Four-class classification of baseline ADNI T1 MRI scans: CN, EMCI, LMCI, and AD.
The repository supports a MedicalNet-compatible 3D ResNet-18 and a four-block
dilated `ClassifierCNN`, both with four-class heads. Real-data profiles use the project's
preprocessed 128×128×128 NIfTI scans.
MRI preprocessing and diffusion generation are separate project components.

This repository contains code and documentation, not the private scans, subject
manifests, pretrained weights, or training outputs. Keep those files in Google
Drive and out of Git. The current scan inventory and run guidance are in the
[Colab runbook](docs/HANDOFF_AND_RUNBOOK.md); historical checks and limitations
are recorded in [verification](docs/VERIFICATION.md).

## Install

Use Python 3.10 or newer. In Colab, preserve the runtime's compatible PyTorch
installation if it already includes CUDA support.

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

## Data contract

| Input | Requirement |
|---|---|
| Classifier manifest | Tracked at `data/preprocessed_128_classifier_inputs/classifier_manifest.csv` |
| Split JSON | Tracked at `data/preprocessed_128_classifier_inputs/classifier_splits.json` |
| Volume | `{subject_id}_preprocessed_128.nii.gz`, shape 128×128×128 |
| Spacing | Approximately 1.192616×1.474926×1.238708 mm |
| Orientation | RAS in the template and sample headers |
| Reference | Tracked at `template/mni152.nii.gz` |
| Intensities | Already normalized; use `normalization: none` |

The prepared manifest, splits, and reference template are bundled directly with
the repository. Only the large preprocessed MRI scans and training outputs reside
on Google Drive.

The checked sample headers have the expected shape, spacing, orientation, and
field-of-view lengths. Their affine origin is zero, unlike the MNI reference
origin. The configured `same_fov` check validates orientation and physical
extent, not world-coordinate origin. Preprocessing QC is a separate requirement;
header compatibility alone does not establish anatomical quality. See the
[runbook](docs/HANDOFF_AND_RUNBOOK.md) for this distinction and the atlas-alignment
limitation.

## Colab Workflow

Mount Google Drive in Colab and clone this repository. The preprocessed scans
must be available at `/content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/preprocessed_128/`.

If you ever need to regenerate `classifier_inputs` from a revised preprocessing report:
```bash
ROOT=/content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS
python -m scripts.prepare_preprocessed_manifest \
  --baseline-manifest "$ROOT/classifier_setup/cohort_a_baseline_manifest.csv" \
  --splits "$ROOT/classifier_setup/cohort_a_splits.json" \
  --preprocessing-manifest "$ROOT/classifier_setup/preprocessing_manifest.csv" \
  --output-dir "data/preprocessed_128_classifier_inputs"
```

The output directory contains `classifier_manifest.csv`,
`classifier_splits.json`, and `preprocessing_summary.json`. Review the summary,
especially the final-success count, retry count, missing-status subjects, and
class counts by split. The training split must contain all four classes.

## Audit and train on Colab

Reading scans straight from the mounted Drive is slow, so first copy them to the
Colab VM disk (repeat after every runtime reset):

```bash
mkdir -p /content/preprocessed_128
rsync -a --include='*_preprocessed_128.nii.gz' --exclude='*' \
  /content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/preprocessed_128/ \
  /content/preprocessed_128/
```

`configs/colab_classifier_cnn_128.yaml` reads scans from that copy and writes
outputs to Drive. It selects the ClassifierCNN profile, uses batch size 4, and
enables mixed precision. Confirm that batch size fits the selected GPU before
starting the full run. Resume a run with the exact config it started with.
First audit the actual files and run the small overfit check:

```bash
python -m scripts.validate_dataset --config configs/colab_classifier_cnn_128.yaml
python -m scripts.tiny_overfit \
  --config configs/colab_classifier_cnn_128.yaml \
  --output /content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs/tiny_overfit
```

If those checks pass, run a short trial and then the planned training run:

```bash
python -m src.train --config configs/colab_classifier_cnn_128.yaml --stop-after-epoch 2
python -m src.train --config configs/colab_classifier_cnn_128.yaml
```

This profile uses Adam, instance normalization, Gaussian blur, affine
augmentation, and balanced sampling. Weighted cross-entropy is disabled to avoid
correcting class imbalance twice. These choices are experiment settings, not a
guarantee of stable optimization or improved accuracy.

Training uses train and validation images only. Choose the model using validation
results, freeze the choices, and then evaluate the held-out test set explicitly:

```bash
python -m src.evaluate \
  --config configs/colab_classifier_cnn_128.yaml \
  --checkpoint PATH_TO_RUN/best.pt --split test --confirm-final-test
```

Export validation activations for later analysis:

```bash
python -m src.export_activations \
  --config configs/colab_classifier_cnn_128.yaml \
  --checkpoint PATH_TO_RUN/best.pt --split val \
  --output /content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs/features
```

Feature export does not define a TCAV bottleneck or scoring procedure. Those
research choices require separate justification; see the [code walkthrough](docs/CODE_WALKTHROUGH.md).

## Synthetic software profile

The synthetic profile uses artificial 16³ volumes and a reduced model width. It
checks software behavior only; it is not evidence of diagnostic performance.

```bash
python -m scripts.make_synthetic
python -m src.train --config configs/smoke_test.yaml
python -m scripts.tiny_overfit --config configs/smoke_test.yaml
```

Run the automated suite after installing development requirements with
`pip install -r requirements-dev.txt` and `python -m pytest`.

## Repository map

- `src/`: configuration, dataset, model, training, evaluation, and feature export.
- `scripts/`: preprocessing-manifest preparation, dataset audit, volume inspection,
  and synthetic-data utilities.
- `configs/`: current CNN profiles, retained ResNet comparison profiles, and the
  isolated synthetic profile.
- `tests/`: automated unit and synthetic workflow checks.
- `docs/`: runbook, implementation walkthrough, and verification record.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for MedicalNet attribution.
