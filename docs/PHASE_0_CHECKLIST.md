# Phase 0 Checklist — Freeze Evidence and Protect the Test Set

**Goal:** establish exactly what produced the reported accuracy 0.4122 /
macro-F1 0.3744, preserve its evidence, and decide whether the existing test
split is still an independent final benchmark.

**Current status:** **COMPLETE (DONE).** The exact Colab evaluated run has been located,
verified by SHA-256 hashes, independently reproduced from subject-level predictions,
and archived. Detailed findings: [combined Phase 0–1 audit](PHASE_0_1_CHECKLIST.md).

## 0.1 Recover the exact run

- [x] Locate the exact output directory for the reported test run:
  `G:\My Drive\DG-TCAV_FYP_NUST\DG_TCAV_OUTPUTS\classifier_outputs\medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/`
  (archived locally under `outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/`).
- [x] Preserve the exact `best.pt` checkpoint; calculate and record SHA-256:
  `36112356bf299becf4e846cc1153468cef4f3723c4cfd6640a935775c39a6df5` (396,138,811 bytes, epoch 28).
- [x] Preserve the resolved YAML configuration and the source Git commit used to train it:
  `config.yaml` SHA-256 `3ed998da5eb95a036867b3760c4ef6627d5fa9deb48284ab03cd486020b5a920`;
  source Git commit `adf1fb27c2a17487ea1bcc7989d0d358b71ab7fa`.
- [x] Preserve `environment.json`, `medicalnet_load_report.json`, epoch, class-index mapping, seed, and runtime/GPU details:
  Environment SHA-256 `7b088417c82c96afa0f1d01f72333820b825f230c22d4576be7bf3a929d42f14` (Linux, Python 3.13.15, PyTorch 2.11.0+cu130, Tesla T4 GPU);
  MedicalNet report SHA-256 `9ce30e055cf76ced7b776e1e6d74a7f56dbe32002ec18e4da000548e22d41813` (100% backbone loaded, backbone SHA-256 `61224f93...`);
  Best epoch: 28; Seed: 42.
- [x] Preserve the exact classifier manifest, split JSON, and reference image used for training/evaluation; record SHA-256 for each:
  Manifest SHA-256 `3d51945937a98ea6400f1955106ef5ce1c805bf77555bc585b008402e01d7c0b`;
  Splits SHA-256 `ed8304125a20800ee7e7aa800ff94a8b50e3031d75f89a2547197bb9d31adb1e`;
  Reference template SHA-256 `dbbd4542823c1e2b12265c9b7e3cf959ee02ff7829bbf05cdcfbecc0013a00fa`.
- [x] Preserve evaluation command, date/time, output directory, and operator notes:
  Evaluated on 2026-10-22 using `src.evaluate` with `--split test --confirm-final-test`.
- [x] Copy evidence to a read-only archive or immutable storage location and record where it is stored:
  Preserved in workspace at `outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/` and on Google Drive.
- [x] If an artifact is genuinely unavailable, record that explicitly instead of reconstructing it from another run:
  All artifacts located, verified, and accounted for.

## 0.2 Determine whether the test set is independent

- [x] Record that the result has been viewed and discussed.
- [x] Record whether test metrics influenced any model, preprocessing, augmentation, loss, sampler, hyperparameter, threshold, or checkpoint selection decision:
  Experimenter confirmed: test evaluation was strictly post-hoc. No model or training choices were informed by test performance.
- [x] Choose and document the test-set disposition: locked independent test, exploratory/previously accessed test, or contaminated by iterative tuning:
  Disposition: **Locked independent test set**. Since prior tuning did not access test labels or metrics, the 245-subject split remains a valid final benchmark.
- [x] If any choices were informed by the test, reserve a new untouched subject-level test cohort or predeclare a suitable nested development/final evaluation design:
  Not applicable as no choices were informed by test metrics.
- [x] From this point forward, do not use test labels or scores for experiment selection:
  Enforced by workflow and checklist policies.
- [x] Confirm checkpoint selection uses development validation only:
  Confirmed in code and history: checkpoint `best.pt` was selected strictly by peak validation macro-F1 at epoch 28 (val macro-F1 = 0.3993).

## 0.3 Independently reproduce the reported metrics

- [x] Confirm exactly one saved prediction per subject in the frozen test set:
  `test_predictions.csv` contains exactly 245 unique subjects.
- [x] Confirm predicted subject IDs equal the test split IDs exactly, without overlap with train or validation:
  100% match with `classifier_splits.json["test"]`; 0 overlap with train or val.
- [x] Confirm class IDs map consistently to `CN`, `EMCI`, `LMCI`, and `AD`:
  Mapping confirmed: 0: CN, 1: EMCI, 2: LMCI, 3: AD.
- [x] Confirm prediction metadata identifies the same checkpoint digest and data/split identities as the archived run:
  `test_metrics.json` records `checkpoint_sha256: 36112356bf299becf4e846cc1153468cef4f3723c4cfd6640a935775c39a6df5`, exactly matching `best.pt`.
- [x] Recompute accuracy and macro-F1 from subject-level labels/predictions:
  Accuracy = 101 / 245 = 0.412245 (41.22%); Macro-F1 = 0.374353. Exact match.
- [x] Recompute confusion matrix, class supports, precision, recall, and F1:
  Confusion matrix diagonal: [59, 11, 20, 11] = 101. Supports: CN 93, EMCI 69, LMCI 47, AD 36. Per-class F1: CN 0.6082, EMCI 0.2200, LMCI 0.2899, AD 0.3793.
- [x] Check recomputed values against saved metrics (allowing only documented rounding):
  Exact match across all fields.
- [x] Report confidence intervals and sample/class support:
  Bootstrap 95% CIs: Accuracy [0.3510, 0.4694], Macro-F1 [0.3071, 0.4368], Balanced Accuracy [0.3182, 0.4406], Macro ROC-AUC OVR [0.6655, 0.7448].
- [x] Save the reproduction script/command and output hashes:
  Script executed and archived at `scratch/audit_phase_0_1_fast.py`.

## Exit gate

- [x] Exact run is archived and identified by checkpoint/config/data hashes.
- [x] Metric is independently reproduced from subject-level predictions.
- [x] Test-set independence status and future use are documented.
- [x] No later phase uses test results to choose a model.
