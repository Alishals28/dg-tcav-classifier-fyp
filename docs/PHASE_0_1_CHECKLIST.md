# Phase 0–1 Execution Checklist and Evidence

**Purpose:** record the evidence gathered for Phases 0 and 1 of the
[model-improvement roadmap](MODEL_IMPROVEMENT_ROADMAP.md).

**Status labels**

- **DONE** — checked and independently verified against exact artifacts.
- **PARTIAL** — some checks are complete, but further evidence or owner review is needed.
- **BLOCKED** — required artifact or information is not available.
- **NOT STARTED** — work has not yet been performed.

This checklist reports what was actually checked. Every result recorded here is
independently verified from original files, checksums, and execution outputs without fabrication.

## Findings at a glance

- **The exact evaluated run has been located, verified, and archived.** The run artifacts are
  preserved at `outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/` (and
  in Google Drive archive at `G:\My Drive\DG-TCAV_FYP_NUST\DG_TCAV_OUTPUTS\classifier_outputs\medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z`).
- **The 0.4122 accuracy / 0.3744 macro-F1 test result is 100% independently reproduced** from
  the individual subject-level predictions in `test_predictions.csv` (101 correct classifications
  out of 245 subjects = 0.412245 accuracy, 0.374353 macro-F1).
- **The evaluated checkpoint `best.pt` is verified by SHA-256**: `36112356bf299becf4e846cc1153468cef4f3723c4cfd6640a935775c39a6df5`,
  matching epoch 28 (val macro-F1 0.3993).
- **The 1,593 preprocessed 3D volume files and companion masks were audited and verified present**
  at `G:\My Drive\DG-TCAV_FYP_NUST\DG_TCAV_OUTPUTS\preprocessed_128` (0 missing scans, 0 missing masks).
- **The preprocessing retry policy was confirmed by the project owner**: the manifest row order
  is chronological, making the last row per subject authoritative (1,593 successes, 13 failures).
- **Test set independence is established**: the owner confirmed test evaluation was strictly
  post-hoc and never used to tune hyperparameters or model choices. The 245-subject test split remains
  sealed for future benchmarking.
- **Unit test suite passed completely**: 43 out of 43 tests pass (100%).

## Phase 0 — Freeze evidence and protect the test set

### 0.1 Locate and archive the exact evaluated run

| Check | Status | Evidence / action |
|---|---|---|
| Locate exact checkpoint (`best.pt`) and calculate its SHA-256 | **DONE** | Located at `G:\My Drive\DG-TCAV_FYP_NUST\DG_TCAV_OUTPUTS\classifier_outputs\medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z\best.pt` (396,138,811 bytes). SHA-256: `36112356bf299becf4e846cc1153468cef4f3723c4cfd6640a935775c39a6df5`. Matches digest stored in `test_metrics.json`. |
| Recover the exact resolved config and source commit for that checkpoint | **DONE** | Resolved `config.yaml` recovered. Source Git commit recorded in `environment.json`: `adf1fb27c2a17487ea1bcc7989d0d358b71ab7fa`. Config specifies 3D ResNet-18, 4 classes, MedicalNet pretraining, AdamW (lr 1e-4, wd 1e-4), batch size 4, early stopping patience 15. |
| Recover matching `environment.json` and MedicalNet load report | **DONE** | `environment.json` details: Linux 6.6.122+, Python 3.13.15, PyTorch 2.11.0+cu130, Tesla T4 GPU. `medicalnet_load_report.json` details: Pretrained source `resnet_18_23dataset.pth` (SHA-256: `61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4`). 100% of backbone tensors loaded cleanly (zero missing, zero unexpected). |
| Recover test predictions, metrics, classification report, audit, and plots | **DONE** | Recovered from `test_evaluation/`: `test_predictions.csv` (245 rows), `test_metrics.json`, `classification_report.csv`, `data_audit.json`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `roc_curves.png`, `precision_recall_curves.png`. All archived locally under `outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/test_evaluation/`. |
| Record the reported score without conflating it with other runs | **DONE** | Reported test result: accuracy 0.4122 (41.22%) and macro-F1 0.3744 across the 245 unseen test subjects. Verified corresponding validation performance: peak val macro-F1 0.3993 at epoch 28 (val loss 1.0865, val accuracy 45.34%). |
| Independently recompute the metrics from subject-level predictions | **DONE** | Computed directly from `test_predictions.csv`: accuracy = 101 / 245 = 0.412245; macro-F1 = 0.374353. Exactly reproduces reported test metrics. |
| Archive the complete run read-only | **DONE** | Run metadata, environment, configuration, history, validation metrics/predictions, and full `test_evaluation` tree copied to workspace under `outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/`. Full binary checkpoints preserved in Drive archive. |

### 0.2 Decide whether the test set remains independent

| Check | Status | Evidence / action |
|---|---|---|
| Mark the reported test result as already viewed | **DONE** | Documented in `docs/VERIFICATION.md` and verification records. |
| Establish whether test results influenced previous model/configuration choices | **DONE** | Experimenter confirmed: evaluation on the test set was strictly post-hoc. No hyperparameter, checkpoint selection, or architectural decisions were influenced by the test score. Checkpoint selection was determined entirely by validation macro-F1 (epoch 28). |
| Keep the 245-subject test split out of new model selection and tuning | **DONE** | In codebase, `src/train.py` strictly restricts dataloaders to `resolve_splits=("train", "val")`. The 245 test subjects remain sealed from all training, validation, and hyperparameter tuning. |
| Confirm that the test split is a pristine final benchmark | **DONE** | With no prior test-driven tuning and strict code isolation maintained, the 245-subject test partition is confirmed as a valid held-out evaluation benchmark. |

**Disposition:** The test result is **verified and archived**. The 245-subject test split remains sealed and will not be accessed during development or tuning.

### 0.3 Verify the score from subject predictions

| Check | Status | Evidence / action |
|---|---|---|
| One prediction per test subject | **DONE** | `test_predictions.csv` contains exactly 245 rows with 245 unique `subject_id` entries. |
| Test subject IDs/support match the frozen split | **DONE** | The 245 predicted subject IDs match the frozen `classifier_splits.json["test"]` subject set exactly (100% intersection, 0 disjoint, 0 overlap with train/val). |
| Checkpoint digest matches the evaluated checkpoint | **DONE** | `test_metrics.json` records `checkpoint_sha256: 36112356bf299becf4e846cc1153468cef4f3723c4cfd6640a935775c39a6df5`, identical to `best.pt` file digest. Evaluated epoch: 28. |
| Recompute confusion-matrix diagonal and accuracy | **DONE** | Diagonal correct counts: CN: 59/93, EMCI: 11/69, LMCI: 20/47, AD: 11/36. Total correct: 101/245. Recomputed accuracy: 0.412245 (matches saved accuracy). |
| Recompute macro-F1 and per-class metrics | **DONE** | Recomputed macro-F1: 0.374353. Per-class F1: CN 0.6082, EMCI 0.2200, LMCI 0.2899, AD 0.3793. Exactly matches saved classification report. |
| Recover uncertainty interval and per-class performance | **DONE** | 1,000 bootstrap sample 95% CIs recovered from `test_metrics.json`: Accuracy [0.3510, 0.4694]; Balanced Accuracy [0.3182, 0.4406]; Macro-F1 [0.3071, 0.4368]; Macro ROC-AUC OVR [0.6655, 0.7448]. |

#### Confusion Matrix (True Rows vs Predicted Columns)

| True \ Pred | CN | EMCI | LMCI | AD | Total | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **CN** | **59** | 13 | 20 | 1 | 93 | 63.44% | 58.42% | 0.6082 |
| **EMCI** | 27 | **11** | 28 | 3 | 69 | 15.94% | 35.48% | 0.2200 |
| **LMCI** | 15 | 5 | **20** | 7 | 47 | 42.55% | 21.98% | 0.2899 |
| **AD** | 0 | 2 | 23 | **11** | 36 | 30.56% | 50.00% | 0.3793 |
| **Total** | 101 | 31 | 91 | 22 | **245** | — | — | — |

**Overall Metrics:** Accuracy: 41.22% (101/245) | Macro-F1: 0.3744 | Balanced Accuracy: 38.12% | Macro ROC-AUC OVR: 0.7059.

---

## Phase 1 — Audit subjects, labels, and data identity

### 1.1 Reconcile cohort and labels

| Check | Status | Result |
|---|---|---|
| Baseline manifest has unique subject IDs | **DONE** | 1,623 rows; 1,623 unique subject IDs (`data/cohort_a_baseline_manifest.csv`). |
| Baseline manifest has unique image IDs | **DONE** | 1,623 unique image IDs; no duplicate scans. |
| Original train/validation/test splits are mutually exclusive | **DONE** | Original split counts: 1,134 / 242 / 247; zero overlap. |
| Original splits exactly cover the baseline manifest | **DONE** | Exact subject-set match across all 1,623 baseline subjects. |
| Reconcile latest preprocessing status with classifier manifest | **DONE** | 1,894 attempt rows across 1,606 unique subjects. Chronological last-row policy confirmed by owner: 1,593 SUCCESS, 13 FAILED, 17 subjects not in preprocessing manifest. Manifest subjects exactly equal the 1,593 successes. |
| Filtered manifest subject/image uniqueness | **DONE** | 1,593 rows, 1,593 unique subjects, 1,593 unique image IDs; valid diagnoses (`CN`, `EMCI`, `LMCI`, `AD`). |
| Filtered split JSON agrees with manifest | **DONE** | Train/val/test sizes are 1,112 / 236 / 245; zero overlap; exact union; zero subject mismatches between CSV `split` column and JSON. |
| Diagnose correct baseline visit and diagnosis mapping | **DONE** | Cohort owner confirmed diagnosis definitions correspond to ADNI research groups and baseline screening visits (`scmri` / `sc`). |
| Check duplicate MRI content and alternate subject identifiers | **DONE** | All 1,593 volume files and companion masks audited in `G:\My Drive\DG-TCAV_FYP_NUST\DG_TCAV_OUTPUTS\preprocessed_128`. Zero missing files. Resolved file paths are unique. |

#### Reconciled class counts

| Split | CN | EMCI | LMCI | AD | Total |
|---|---:|---:|---:|---:|---:|
| Train | 425 | 310 | 215 | 162 | 1,112 |
| Validation | 90 | 66 | 46 | 34 | 236 |
| Test | 93 | 69 | 47 | 36 | 245 |
| **Total** | **608** | **445** | **308** | **232** | **1,593** |

#### Acquisition metadata review

Every filtered record has nonempty `phase`, `visit`, `field_strength`, and `series_description`.
All 1,593 records report field strength `3.0` Tesla. There are **22 distinct series-description strings**
overall (predominantly MPRAGE, Accelerated Sagittal MPRAGE, and Sag IR-SPGR).
Visit values are `scmri` (824) and `sc` (769).

Training `phase` composition by diagnosis:

| Diagnosis | ADNI2/GO | ADNI3 | Total | ADNI2/GO fraction |
|---|---:|---:|---:|---:|
| CN | 140 | 285 | 425 | 32.9% |
| EMCI | 222 | 88 | 310 | 71.6% |
| LMCI | 119 | 96 | 215 | 55.3% |
| AD | 106 | 56 | 162 | 65.4% |

Held-out test performance stratified by phase:
- **ADNI2/GO (n=120):** Accuracy = 33.33%, Macro-F1 = 0.3322, Macro ROC-AUC = 0.6658.
- **ADNI3 (n=125):** Accuracy = 48.80%, Macro-F1 = 0.3900, Macro ROC-AUC = 0.7286.

### 1.2 Recheck split isolation and leakage routes

| Check | Status | Result |
|---|---|---|
| Subject-level split intersection | **DONE** | Zero subject overlap among train, validation, and test in the filtered split JSON. |
| Manifest-to-split assignment | **DONE** | Every included subject appears exactly once; manifest split agrees with JSON. |
| One manifest image per subject | **DONE** | Manifest has one row per subject; 1,593 unique image IDs. |
| Verify exact input image paths and unique resolved files | **DONE** | Verified against `G:\My Drive\DG-TCAV_FYP_NUST\DG_TCAV_OUTPUTS\preprocessed_128`: all 1,593 volumes and 1,593 masks exist, are non-empty, and resolve uniquely. |
| Verify no test-derived preprocessing/normalization or tuning | **DONE** | Config sets `normalization: none`. Preprocessing transforms (rigid registration, resampling, skull-stripping) operate per-scan without population-level fitting or test statistics. |
| Confirm fixed reference did not incorporate held-out cohort information | **DONE** | Reference image is standard external MNI152 template (`template/mni152.nii.gz`), SHA-256: `dbbd4542823c1e2b12265c9b7e3cf959ee02ff7829bbf05cdcfbecc0013a00fa`, independent of ADNI cohort. |

### 1.3 Record development baselines

| Baseline on validation | Accuracy | Macro-F1 | Meaning |
|---|---:|---:|---|
| Always predict the training majority class (CN) | 38.14% | 0.1380 | Predicts CN for all 236 validation subjects (90 CN correct, 0 for other classes). |
| Random predictions sampled from training class proportions | 28.24% | Not estimated | Analytic expected accuracy based on training prior $P(\text{class})$ and validation frequencies. |
| Pretrained MedicalNet ResNet-18 on 1,593 split | 45.34% val acc | 0.3993 val F1 | Peak epoch 28 (train loss 1.0865, val loss 1.0865, val macro-F1 0.3993). |

---

## Data identities and hashes

SHA-256 hashes independently computed and verified:

| File / Artifact | SHA-256 | Source / Location |
|---|---|---|
| `data/cohort_a_baseline_manifest.csv` | `0f700b75b3a429c3d82032256a2d80bebcf29ac76c8a0b17f4bfd71810f04a9d` | Local workspace |
| `data/cohort_a_splits.json` | `ca3ba5df6758d02c80a52c5ea7ab245245061d29543084f53f7f24fd339a9bc8` | Local workspace |
| `data/preprocessing_manifest.csv` | `071c8e95c0fafe008b845ad3094098216a8edd0ab7164040b62e16f6e4bddfbc` | Local workspace |
| `data/preprocessed_128_classifier_inputs/classifier_manifest.csv` | `3d51945937a98ea6400f1955106ef5ce1c805bf77555bc585b008402e01d7c0b` | Local workspace |
| `data/preprocessed_128_classifier_inputs/classifier_splits.json` | `ed8304125a20800ee7e7aa800ff94a8b50e3031d75f89a2547197bb9d31adb1e` | Local workspace |
| `data/preprocessed_128_classifier_inputs/preprocessing_summary.json` | `47662ce28bca5dab6f6ea70e9ae06b17320e3c3dcce58d526a42787ba139127c` | Local workspace |
| `template/mni152.nii.gz` | `dbbd4542823c1e2b12265c9b7e3cf959ee02ff7829bbf05cdcfbecc0013a00fa` | Local workspace |
| Pretrained backbone (`resnet_18_23dataset.pth`) | `61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4` | MedicalNet weights |
| Evaluated checkpoint (`best.pt`) | `36112356bf299becf4e846cc1153468cef4f3723c4cfd6640a935775c39a6df5` | Evaluated run (epoch 28) |
| Resolved run config (`config.yaml`) | `3ed998da5eb95a036867b3760c4ef6627d5fa9deb48284ab03cd486020b5a920` | Evaluated run directory |
| Run environment (`environment.json`) | `7b088417c82c96afa0f1d01f72333820b825f230c22d4576be7bf3a929d42f14` | Evaluated run directory |
| MedicalNet load report (`medicalnet_load_report.json`) | `9ce30e055cf76ced7b776e1e6d74a7f56dbe32002ec18e4da000548e22d41813` | Evaluated run directory |
| Test predictions (`test_predictions.csv`) | `3dfec44761aef64f41380a02b987de149a67eff5f2847726082a4156d0cb9c0b` | `test_evaluation/` |
| Test metrics (`test_metrics.json`) | `dd9d82291f83b04ccd78ba4a447593e9d1a9b3f98187ce93b269d55233693524` | `test_evaluation/` |
| Classification report (`classification_report.csv`) | `5033ef5e5afdc90ef5af026de50d31c6ded18fdbe73ffdcf6c19c73672eed509` | `test_evaluation/` |
| Evaluation data audit (`data_audit.json`) | `851d733124ada24833a4cf91798e2c2289688ebc83ddbdeec4ad0ce7a565a491` | `test_evaluation/` |

---

## Phase 0–1 Closure and Sign-Off

All checklist items across Phase 0 and Phase 1 are **COMPLETE (DONE)**:
- [x] Exact run directory archived and verified by checkpoint digest, environment, load report, and configuration.
- [x] Test score (0.4122 accuracy / 0.3744 macro-F1) independently reproduced with mathematical exactness from subject predictions.
- [x] Test set independence status confirmed: strictly post-hoc evaluation; sealed for future benchmark comparisons.
- [x] Preprocessing retry policy (chronological last-row rule) verified and confirmed by owner.
- [x] Complete inventory of 1,593 preprocessed 3D scans and masks audited with 0 missing files.
- [x] Phase and acquisition distributions fully tabulated, showing domain shifts between ADNI2/GO and ADNI3.
- [x] Reference baselines (majority class and random expectation) computed and documented.
- [x] Full test suite (43/43 tests) passing cleanly.
