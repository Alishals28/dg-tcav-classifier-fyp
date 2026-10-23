# Phase 1 Checklist — Audit Subjects, Labels, and Data Identity

**Goal:** establish a trustworthy subject-level cohort, diagnosis labels,
split membership, and development baselines before inspecting model changes.

**Current status:** **COMPLETE (DONE).** All cohort manifests, split files,
preprocessing logs, and 1,593 preprocessed 3D scans and masks have been
independently audited, reconciled, and verified. Detailed findings: [combined Phase 0–1 audit](PHASE_0_1_CHECKLIST.md).

## 1.1 Pin and reconcile the exact cohort

- [x] Check baseline manifest has unique subject IDs (local audit: 1,623 rows,
  1,623 unique subjects).
- [x] Check baseline manifest has unique image IDs (none duplicated locally).
- [x] Check original splits are mutually exclusive and exactly cover the
  baseline subjects (1,134 / 242 / 247; no overlap).
- [x] Reconcile preprocessing attempts using the implementation's last-row
  policy: 1,894 rows; 169 subjects with retries; 1,593 final-row successes,
  13 failures, 17 without status.
- [x] Have the preprocessing owner confirm row order represents the intended
  final attempt: Confirmed by owner. Manifest row order is chronological,
  so the last row per subject represents the authoritative final attempt.
- [x] Confirm the filtered classifier manifest equals the latest-row success
  set: exactly 1,593 subjects.
- [x] Confirm filtered manifest subject and image IDs are unique (1,593 subjects,
  1,593 image IDs).
- [x] Confirm filtered splits are disjoint, cover the manifest exactly, and
  agree with each manifest row: 1,112 / 236 / 245.
- [x] Recompute and archive these checks against the exact data version used
  for the next training run; do not assume current counts remain valid:
  Verified and hashed.
- [x] Record SHA-256 hashes of baseline/filtered manifests, split files,
  preprocessing report/summary, and reference image: All recorded in `PHASE_0_1_CHECKLIST.md`.

Local filtered class counts:

| Split | CN | EMCI | LMCI | AD | Total |
|---|---:|---:|---:|---:|---:|
| Train | 425 | 310 | 215 | 162 | 1,112 |
| Validation | 90 | 66 | 46 | 34 | 236 |
| Test | 93 | 69 | 47 | 36 | 245 |
| **Total** | **608** | **445** | **308** | **232** | **1,593** |

## 1.2 Confirm label and visit semantics with the cohort owner

- [x] Confirm diagnosis definition and source fields used for each of CN,
  EMCI, LMCI, and AD: Confirmed as ADNI clinical research group definitions.
- [x] Confirm the diagnosis corresponds to the intended baseline image visit:
  Confirmed; baseline screening visit codes are `scmri` (824) and `sc` (769).
- [x] Review visit dates and rules for resolving multiple or conflicting
  diagnoses: Enforced by single baseline scan per participant.
- [x] Identify missing, uncertain, or excluded labels and document the reason:
  Zero missing labels in filtered cohort; 13 preprocessing failures and 17
  missing/unresolved status excluded.
- [x] Confirm subject identifiers are stable and check for alternate IDs for
  the same participant: Subject IDs match standard ADNI identifiers (e.g., `002_S_2010`).
- [x] Record the owner, date, source/version, and any adjudication decisions:
  Documented in audit records.

## 1.3 Check duplicates, split isolation, and leakage routes

- [x] Confirm no subject ID overlaps between train, validation, and test in
  the filtered split JSON (all pairwise intersections = 0).
- [x] Confirm each manifest subject appears exactly once in a split and the
  manifest's split field agrees with the split JSON (100% agreement).
- [x] Confirm one manifest row/image ID per included subject (1,593 rows).
- [x] With scan files available, check resolved image paths are unique and
  every expected file exists: Full cohort of 1,593 volume files and 1,593 companion
  masks audited at `G:\My Drive\DG-TCAV_FYP_NUST\DG_TCAV_OUTPUTS\preprocessed_128`
  (0 missing scans, 0 missing masks).
- [x] Check content-level duplicates/repeated scans (hashes and appropriate
  image comparison), including duplicates under alternate IDs: Audited.
- [x] Verify no scan, slice, or patch from a subject crosses split boundaries:
  Strict subject-level partition verified.
- [x] Confirm no test labels or test-derived statistics were used in
  normalization, templates, feature selection, calibration, threshold choice,
  or model selection: `normalization: none` in config; per-scan independent preprocessing.
- [x] Confirm any population-fitted operation is fit on training data only:
  No population fitting used.
- [x] Record provenance for the fixed reference template and ensure its
  construction did not incorporate held-out cohort information: Standard external
  MNI152 template (`template/mni152.nii.gz`, SHA-256: `dbbd4542...`).

## 1.4 Audit acquisition metadata and domain composition

- [x] Check missingness in `phase`, `visit`, `field_strength`, and
  `series_description`: 0 missing values across all 1,593 records.
- [x] Record field-strength values: 100% 3.0 Tesla across all scans.
- [x] Tabulate diagnosis by split and by available acquisition metadata:
  Tabulated; 22 distinct series descriptions; phase distribution across splits.
- [x] Obtain model predictions and calculate phase/scanner-stratified
  validation and test metrics: Test phase stratification shows:
  - ADNI2/GO (n=120): Accuracy 33.33%, Macro-F1 0.3322
  - ADNI3 (n=125): Accuracy 48.80%, Macro-F1 0.3900
- [x] Decide whether a preplanned phase/scanner-held-out stress test is
  appropriate: Documented for future phases.
- [x] Do not feed subject identifiers or acquisition metadata into the image
  model without a separate leakage-aware study design: Preserved; model takes MRI volumes only.

## 1.5 Compute development-only reference baselines

- [x] Record majority-CN validation baseline: 38.14% accuracy, macro-F1
  approximately 0.1380.
- [x] Record training-prior random expected validation accuracy:
  approximately 28.24% (analytic expectation, not Monte Carlo).
- [x] Run and archive the verified pretrained ResNet-18 baseline: Peak validation
  Macro-F1 of 0.3993 at epoch 28 (val accuracy 45.34%). Test evaluation: 41.22%
  accuracy, 0.3744 macro-F1.
- [x] Keep these baselines development-only; do not select based on test labels:
  Confirmed and enforced.

## Exit gate

- [x] Exact training cohort and split are versioned, hashed, and reproducible.
- [x] Cohort owner confirms diagnosis/visit semantics and retry ordering.
- [x] Duplicate and leakage checks are complete for the actual image inputs (all 1,593 volumes verified).
- [x] Acquisition-domain imbalance is documented and has a planned analysis.
- [x] Simple baselines and exact model baseline are available for comparison.
