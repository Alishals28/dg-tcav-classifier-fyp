# DG-TCAV classifier

Four-class classification of baseline ADNI T1 MRI: CN, EMCI, LMCI, and AD.
The model uses a MedicalNet-compatible 3D ResNet-18 backbone with a new
classification head. Preprocessing and the diffusion model are separate work.

The implementation has passed CPU tests on synthetic data. The manifest and fixed
subject splits have been checked, while real-data training still requires the
preprocessing outputs, the actual MedicalNet checkpoint, and a short Kaggle GPU
run. See the [data handoff and runbook](docs/HANDOFF_AND_RUNBOOK.md).

## Setup

Use Python 3.10 or newer. Run these commands from the repository root:

```bash
git clone https://github.com/Alishals28/dg-tcav-classifier-fyp.git
cd dg-tcav-classifier-fyp
pip install -r requirements.txt
pip install -e . --no-deps
python -m pytest
```

Kaggle users can start with [notebooks/train.ipynb](notebooks/train.ipynb).
Attach the preprocessed scans and pretrained weights as private inputs. Keep
patient data, subject manifests, credentials, and checkpoints out of this public
repository. Installation may need internet access; retain Kaggle's GPU-compatible
PyTorch installation if it meets the requirements.

## Required data

The loader joins the baseline manifest to the existing subject split JSON.

| Input | Required contents |
|---|---|
| Manifest CSV | subject_id, image_id, diagnosis, phase |
| Split JSON | train: 1,134 subjects; val: 242; test: 247 |
| Processed scans | One NIfTI per manifest record, 91×109×91 at 2 mm spacing |
| Reference NIfTI | Exact template grid used by the preprocessing pipeline |
| Preprocessing records | Image-ID mapping, software versions, QC and failed-scan list |

The label mapping is CN=0, EMCI=1, LMCI=2, AD=3. The original split is preserved.
Missing scans, duplicate subjects, overlapping splits, and ambiguous file matches
stop the run. Any agreed exclusions must be recorded in revised manifests,
splits, and expected counts.

The preprocessing member supplies skull-stripped, MNI-registered volumes.
The loader checks shape, spacing, explicit mm units, orientation, affine and
voxel values. Anatomical alignment and brain extraction still need visual QC.

Update `configs/kaggle.yaml` when the files are available:

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

Use the delivered filenames and reference orientation; the values above are
examples. Set `preprocessing_confirmed` only after receiving QC approval.
The reference affine must match the scan grid. Paths are relative to the working
directory unless absolute; `extends` paths are relative to the YAML file.

Other supported patterns include `{subject_id}_I{image_id}.nii.gz` and
`{subject_id}/*.nii.gz`. Exactly one file must match. Labels come from the CSV.
Image IDs accept an optional `I` prefix. A subject-only filename relies on the
preprocessing records to establish which baseline scan it represents.

Choose `minmax` for foreground scaling to [0,1], `zscore` for foreground
standardization, or `none` for images already normalized as agreed. Each image is
normalized separately using nonzero voxels, with zero background preserved.
This foreground mask is not a brain segmentation.

## Train on Kaggle

Audit the inputs and run the small training-set overfit check first:

```bash
python -m scripts.validate_dataset --config configs/kaggle.yaml
python -m scripts.tiny_overfit --config configs/kaggle.yaml --output /kaggle/working/tiny_overfit
```

Next, run the provided two-epoch, full-width trial. Check memory use, output files,
MedicalNet loading and epoch time before the full run. Batch size 4 is a starting
point and may need adjustment.

```bash
python -m src.train --config configs/kaggle_trial.yaml
```

```bash
python -m src.train --config configs/kaggle.yaml
```

The main settings are weighted cross-entropy, AdamW at 1e-4, cosine decay,
50 epochs maximum, and early stopping after 10 epochs without improved validation
macro-F1. The [code walkthrough](docs/CODE_WALKTHROUGH.md) explains the loss,
checkpoint selection, and experiment comparisons.

Training opens only train/validation images. The audit's `--include-test-qc`
option includes test image integrity checks without generating predictions.
Record the Git commit used for each run and keep the code fixed while it runs.

To resume, use that run's saved configuration and `last.pt`:

```bash
python -m src.train --config /kaggle/working/outputs/<run>/config.yaml --resume /kaggle/working/outputs/<run>/last.pt
```

Replace `<run>` with the directory printed by training. Resume requires unchanged
configuration, paths and data. Changed settings or an extended schedule require a
new experiment. Save the whole run directory before the Kaggle session ends.

## Evaluate and export

After selecting the configuration on validation data, evaluate the held-out test
set:

```bash
python -m src.evaluate --config configs/kaggle.yaml --checkpoint /kaggle/working/outputs/<run>/best.pt --split test --confirm-final-test
```

Evaluation defaults to validation. The test flag records an explicit choice; it
does not enforce one-time access. Test results must not guide tuning or seed
selection. Existing evaluation directories are not overwritten.

Results include aggregate/per-class/per-phase metrics, bootstrap intervals,
subject predictions and probabilities, confusion matrices, and ROC/PR curves.
The walkthrough explains undefined metrics and the binary-subset summaries.

For the TCAV stage:

```bash
python -m src.export_activations --config configs/kaggle.yaml --checkpoint /kaggle/working/outputs/<run>/best.pt --split val --output /kaggle/working/features
```

This saves `pooled_features.npy` ([N,512]), `subjects.csv` and `provenance.json`.
Add `--spatial` to save layer-4 maps. The code supports feature access; bottleneck
selection and TCAV scoring remain separate. In particular, the final linear head
has constant logit gradients at the pooled features, as explained in the walkthrough.

## Test without MRI data

```bash
python -m scripts.make_synthetic
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m src.train --config configs/smoke_test.yaml
python -m scripts.tiny_overfit --config configs/smoke_test.yaml
```

On Windows, set those environment variables in PowerShell or omit the prefix.
The synthetic config uses 16³ artificial images and reduced model width. Its
results test the code, not disease classification. Generation refuses to overwrite
its directory; use a new `--output` and update the config to regenerate.

The same evaluation/export commands work with `configs/smoke_test.yaml` and the
printed checkpoint path. Use validation for a routine software check.

## Files

| Location | Purpose |
|---|---|
| `configs/` | Base, Kaggle, random-initialization, augmentation and synthetic settings |
| `src/dataset.py`, `transforms.py` | Indexing, image checks, normalization and augmentation |
| `src/model.py`, `medicalnet.py` | Architecture and pretrained-weight loading |
| `src/engine.py`, `train.py` | Epoch loop, checkpoints and resume |
| `src/evaluate.py`, `metrics.py`, `plots.py` | Evaluation and figures |
| `src/export_activations.py` | Feature export with subject mapping |
| `src/config.py`, `constants.py`, `reproducibility.py` | Settings, labels and run metadata |
| `scripts/`, `tests/` | Data audit, inspection, synthetic checks and automated tests |

Each run saves its configuration, package versions, input hashes, resolved index,
class weights, model summary, history, learning curves, best/last checkpoints, and
best validation predictions/metrics. Pretrained runs add a weight-loading report.
Evaluation outputs include the checkpoint hash.

[Code walkthrough](docs/CODE_WALKTHROUGH.md) ·
[Data handoff and runbook](docs/HANDOFF_AND_RUNBOOK.md) ·
[Test results](docs/VERIFICATION.md) ·
[MedicalNet attribution](THIRD_PARTY_NOTICES.md)
