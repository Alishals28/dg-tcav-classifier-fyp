# Data handoff and Kaggle runbook

This page records what is already available, what the preprocessing member must
deliver, and the order in which the classifier should be run. Update it if the
agreed preprocessing contract changes.

## Current inputs

The shared Drive folder `02_Cohort_A_Classifier_Baseline` was inspected on
26 September 2026.

| Item | Verified information |
|---|---|
| `cohort_a_baseline_manifest.csv` | 1,623 rows; unique subject and image IDs; required columns present |
| `cohort_a_splits.json` | `train`, `val`, and `test` keys; 1,134, 242, and 247 subjects respectively |
| Split membership | No overlap; the split IDs exactly cover the manifest |
| Baseline scans | Subject folders contain `MPRAGE.nii.gz` and sidecar JSON files |
| Pretrained weights | Not found in the shared Drive folder |

The observed diagnosis counts are:

| Split | CN | EMCI | LMCI | AD | Total |
|---|---:|---:|---:|---:|---:|
| Train | 429 | 318 | 223 | 164 | 1,134 |
| Validation | 92 | 68 | 47 | 35 | 242 |
| Test | 93 | 69 | 49 | 36 | 247 |

One baseline NIfTI was checked as a format sample. It was RAS-oriented with shape
170×256×256 and spacing 1.2×1.0×1.0 mm. This is a converted native-resolution
scan rather than the classifier's required 91×109×91, 2 mm grid. One sample cannot
establish the properties of every scan or whether skull stripping was applied.

## Preprocessing handoff

The preprocessing member should add the following to the existing shared project
folder:

```text
preprocessed/
    one model-ready NIfTI for each of the 1,623 manifest rows
reference.nii.gz
preprocessing_qc/
    scan-to-image-ID mapping, failed-scan list, QC summary and software versions
```

Before the classifier member accepts the handoff, confirm:

1. Skull stripping and MNI-152 registration were completed and visually checked.
2. Every output is 91×109×91 at 2 mm isotropic spacing on the same reference affine.
3. The orientation and filename pattern are stated explicitly.
4. Each output remains linked to the manifest `subject_id` and baseline `image_id`.
5. Exclusions or failed scans are documented. A cohort change requires matching
   updates to the manifest, split file, expected counts and experiment record.
6. The intensity-processing steps are stated so the classifier does not apply an
   unintended second normalization.

The classifier checks files and NIfTI headers. It cannot determine from metadata
alone that brain extraction or anatomical registration is correct.

## Kaggle inputs

Create a private Kaggle Dataset from the existing shared project data after the
preprocessed folder and reference image are ready. Kaggle assigns the mounted path;
the expected configuration currently uses:

```text
/kaggle/input/dg-tcav-cohort-a/
    cohort_a_baseline_manifest.csv
    cohort_a_splits.json
    reference.nii.gz
    preprocessed/
```

Attach the trusted MedicalNet `resnet_18_23dataset.pth` file as another private
Kaggle Dataset. Do not commit scans, manifests, weights, credentials or experiment
checkpoints to GitHub.

Update `configs/kaggle.yaml` to match the paths and delivered filenames. For flat
subject filenames, use `{subject_id}.nii.gz`. If the preprocessing member retains
one folder per subject, a pattern such as `{subject_id}/MPRAGE_preprocessed.nii.gz`
is also supported. Set `preprocessing_confirmed: true` only after the six handoff
checks above are resolved.

## Execution order

Run these commands from the cloned repository in Kaggle.

1. Confirm the environment and software tests:

   ```bash
   python -m pytest
   ```

2. Audit training and validation inputs:

   ```bash
   python -m scripts.validate_dataset --config configs/kaggle.yaml
   ```

   Add `--include-test-qc` only when checking test image integrity. This does not
   produce model predictions.

3. Check that the model can memorize eight training scans:

   ```bash
   python -m scripts.tiny_overfit --config configs/kaggle.yaml --output /kaggle/working/tiny_overfit
   ```

4. Run the two-epoch full-width trial:

   ```bash
   python -m src.train --config configs/kaggle_trial.yaml
   ```

   Review GPU memory, epoch time, the MedicalNet loading report, learning curves
   and output files. Resolve failures before starting the longer experiment.

5. Run the planned MedicalNet experiment:

   ```bash
   python -m src.train --config configs/kaggle.yaml
   ```

6. Run the random-initialization comparison with the same loss and augmentation:

   ```bash
   python -m src.train --config configs/random_baseline.yaml
   ```

7. Choose the configuration using validation results. Record the chosen run and
   checkpoint before accessing test predictions.

8. Evaluate the frozen checkpoint on the test split:

   ```bash
   python -m src.evaluate --config configs/kaggle.yaml --checkpoint /kaggle/working/outputs/EXACT_RUN/best.pt --split test --confirm-final-test
   ```

Download the entire output directory before the Kaggle session expires. The run
contains the resolved configuration, environment, input hashes, checkpoints,
history, validation results and figures needed to explain the experiment.

## What each command establishes

| Command | What a successful result means | What it does not establish |
|---|---|---|
| `pytest` | Tested code paths behave as expected on artificial data | Clinical validity or GPU behavior |
| `validate_dataset` | Files match the declared split and NIfTI grid | Correct-looking skull stripping or registration |
| `tiny_overfit` | Data, model, loss and optimizer can fit a tiny training sample | Generalization performance |
| Two-epoch trial | Real-data GPU execution and checkpoint writing work | Final model quality |
| Full training | A checkpoint was selected using validation macro-F1 | Unbiased test performance |
| Final evaluation | Metrics for the previously frozen checkpoint | Permission to tune using the test result |
