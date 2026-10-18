# Colab data handoff and runbook

The real-data classifier profile consumes baseline ADNI T1 scans preprocessed to
128×128×128. Preprocessing runs separately. Use Google Drive for data, weights,
and outputs; keep them out of the public code repository.

## Current input facts

The repository's local inspection covered seven sample volumes and companion
masks. Their headers report 128³ dimensions, spacing approximately
1.192616×1.474926×1.238708 mm, millimetre units, and RAS orientation. The MNI152
reference reports 207×256×215 dimensions, 0.737463 mm isotropic spacing, and RAS
orientation. The sample and template field-of-view lengths agree within header
precision.

The saved output affine origin is (0, 0, 0), while the template origin is about
(-75.763, -110.763, -71.763) mm. The `same_fov` check verifies orientation and
extent; it does not establish identical MNI world coordinates. The user has
confirmed anatomical alignment by preprocessing QC. For atlas overlays or
world-coordinate analysis, use outputs with a verified affine mapping and verify
the atlas against those outputs. The classifier reads volumes, not companion
masks. Header checks do not replace visual QC.

## Drive layout

Mount Drive in Colab and use this root:

```text
/content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/
├── classifier_setup/
│   ├── cohort_a_baseline_manifest.csv
│   ├── cohort_a_splits.json
│   └── preprocessing_manifest.csv
├── preprocessed_128/           # scan volumes and masks
├── mni152.nii.gz
├── resnet_18_23dataset.pth     # only needed for the ResNet profile
└── classifier_outputs/
```

The actual scan files must match `{subject_id}_preprocessed_128.nii.gz`; masks
follow the corresponding `_mask.nii.gz` pattern. The reference must be accessible
at the configured path. The checkpoint is only needed for a ResNet profile with
pretrained weights enabled. Confirm required files in the mounted Drive before
running the audit.

## Build the labeled subset

The baseline manifest provides diagnosis and scan metadata. The original split
JSON assigns subjects to train, validation, and test. The preprocessing report
determines which scans are eligible. The preparation script uses the last report
row for each subject as its final status; this is the project's chosen retry
policy because the report has no attempt timestamp. It keeps subjects whose final
status is `SUCCESS`, while preserving their original labels and split membership.
It also reports subjects that failed or have no preprocessing status.

Run in Colab after cloning the repository and mounting Drive:

```bash
python -m scripts.prepare_preprocessed_manifest \
  --baseline-manifest /content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_setup/cohort_a_baseline_manifest.csv \
  --splits /content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_setup/cohort_a_splits.json \
  --preprocessing-manifest /content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_setup/preprocessing_manifest.csv \
  --output-dir data/preprocessed_128_classifier_inputs_refresh
```

The output directory must not already exist. It contains a classifier manifest,
filtered split JSON, and preprocessing summary. Check the summary and confirm all
four labels occur in training. The run configs use
`data/preprocessed_128_classifier_inputs`, so point them at the reviewed refresh
directory or replace the existing inputs only after backing them up. As of the
currently inspected report, last-row selection yields 768 successful subjects
split as 535 train, 112 validation, and 121 test. Re-run the script for each
updated preprocessing report; do not assume these counts remain current.

## Audit and run

Use `configs/colab_classifier_cnn_128.yaml`. It reads scans from a copy on the
Colab VM disk because reading from mounted Drive every epoch is slow; outputs
still go to Drive. Copy the scans after each runtime reset:

```bash
mkdir -p /content/preprocessed_128
rsync -a --include='*_preprocessed_128.nii.gz' --exclude='*' \
  /content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/preprocessed_128/ \
  /content/preprocessed_128/
```

The profile uses batch size 4 with mixed precision as a starting point. Verify
that this fits and record epoch time on the selected Colab GPU before starting a
long run. Resume a run with the exact config it started with.

```bash
python -m scripts.validate_dataset --config configs/colab_classifier_cnn_128.yaml
python -m scripts.tiny_overfit \
  --config configs/colab_classifier_cnn_128.yaml \
  --output /content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs/tiny_overfit
python -m src.train --config configs/colab_classifier_cnn_128.yaml --stop-after-epoch 2
python -m src.train --config configs/colab_classifier_cnn_128.yaml
```

Training uses train and validation images only. Select the model using validation
results, freeze choices, then evaluate the held-out test split:

```bash
python -m src.evaluate \
  --config configs/colab_classifier_cnn_128.yaml \
  --checkpoint PATH_TO_RUN/best.pt --split test --confirm-final-test
```

Training and validation results do not replace scan QC. Keep run outputs and the
Git commit identifier together. If more subjects finish preprocessing, regenerate
the classifier inputs and start a new run rather than resuming a checkpoint
trained on a different subject set. Synthetic checks do not measure diagnostic
performance.

The older `colab_local_128.yaml` and `kaggle_128.yaml` profiles remain for
ResNet comparison runs. The Kaggle kernel launcher now selects the
`ClassifierCNN` profile and discovers the attached scan folder and `.nii` or
`.nii.gz` suffix at runtime.
