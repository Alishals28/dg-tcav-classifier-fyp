# Evidence-First Roadmap for Improving the Four-Class MRI Classifier

## Purpose and limits

This roadmap gives a reproducible path for investigating whether the current
four-class ADNI classifier can improve. The requested **0.84+ test accuracy is
an aspirational target, not a result this plan can promise**. Model performance
depends on the data, label quality, preprocessing, split difficulty, and training
variation. Do not claim the target has been achieved until it is measured on a
locked, subject-independent test set with complete artifacts.

This repository implements a MedicalNet-compatible 3D ResNet-18 with a
four-class head. The current Colab profile uses a pretrained checkpoint,
balanced sampling, no weighted loss, batch size 4, mixed precision, and no
augmentation. It selects a checkpoint by validation macro-F1. The data loader
supports class-weighted cross-entropy as an alternative to balanced sampling.
Do not enable both balancing methods together in an experiment unless there is
a specific, separately justified study question.

The workspace artifacts inspected previously are not sufficient to reproduce
the newly reported 41.22% test accuracy / 0.3744 macro-F1: the available
`outputs/dg-tcav-classifier-full/` record is a from-scratch run on an older
535/112/121 split, and the test-evaluation directory referred to in
`docs/VERIFICATION.md` was not present in that inspected output tree. Start by
locating and preserving the exact artifacts for the run that produced the
reported score. Do not infer its per-class errors from the older run.

## What success means

Use these as separate claims:

1. **Reproducible training:** another run with the same code, data identity,
   configuration, checkpoint, and seed produces consistent results within
   expected hardware nondeterminism.
2. **Improved development performance:** the selected model improves on
   predeclared validation or cross-validation metrics, without severe loss of
   recall in any diagnosis class.
3. **Final held-out performance:** the frozen model is evaluated once on a test
   cohort not used to choose the model, thresholds, preprocessing, or
   hyperparameters.
4. **External generalization:** the frozen model is evaluated on a genuinely
   independent cohort whose diagnostic labels and preprocessing can be
   harmonized without looking at its predictions.

Treat 0.84 accuracy as a **stretch goal for a predeclared evaluation**, not as a
pass/fail promise. Keep macro-F1, balanced accuracy, class-wise recall, and the
confusion matrix alongside accuracy. The study team must decide clinically
meaningful minimum class performance before final evaluation; this document
does not invent such cutoffs.

The paper in
[`reference/docs/Multiclass-3dResNetEnsemble.pdf`](../reference/docs/Multiclass-3dResNetEnsemble.pdf)
reports 94.51% accuracy for its single 3D ResNet-18 and 95.97% for its weighted
ResNet-18/34/50 ensemble in its four-class experiment. Its CBAM models,
preprocessing, split and experimental protocol differ from this project. Its
numbers are context, not an expected outcome or a directly comparable target.
The paper describes a random 70/30 sample split; do not assume that is equivalent
to a subject-grouped independent split. Likewise, the per-class F1 ranges and
the proposed “3–5% external drop” in informal notes are not established
acceptance criteria unless independently sourced and justified.

## Phase 0 — Freeze evidence and protect the test set

**Do this before another model run.**

Execution status and checked local evidence are tracked in
[`PHASE_0_1_CHECKLIST.md`](PHASE_0_1_CHECKLIST.md). Phase 0 cannot be closed
until the exact run artifacts for the reported test result are recovered.
Separate actionable checklists for all phases are indexed in
[`PHASE_CHECKLISTS.md`](PHASE_CHECKLISTS.md).

### 0.1 Locate and archive the exact evaluated run

For the run that produced 0.4122 accuracy / 0.3744 macro-F1, preserve a
read-only copy of:

- `best.pt` and its SHA-256 digest.
- The exact resolved YAML config and the Git commit or source archive used.
- `environment.json`, `medicalnet_load_report.json`, and the checkpoint's
  recorded epoch and label mapping.
- The exact classifier manifest, split JSON, reference image, and their
  SHA-256 hashes.
- `test_evaluation/test_metrics.json`,
  `test_evaluation/test_predictions.csv`,
  `test_evaluation/classification_report.csv`, `data_audit.json`, and plots.
- The command used to evaluate, date, runtime/GPU, and operator notes.

The current evaluator writes these outputs and records the checkpoint digest,
bootstrap intervals, per-class results, per-phase results, and pairwise
summaries. Use a fresh output directory because the evaluator refuses to
overwrite an existing one.

If the prediction file or exact checkpoint/config cannot be recovered, record
the metric as **reported but not independently reproduced**. Do not backfill
missing evidence with metrics from another run.

### 0.2 Decide whether the test set is still independent

The reported test score has been viewed. Keep that split closed for the
experiments below. If test results have already influenced architecture,
augmentation, loss, hyperparameters, threshold choice, or model selection, the
split is no longer a pristine final benchmark. It can still be reported
transparently as exploratory evidence, but reserve a new untouched cohort or a
predeclared nested subject-level cross-validation protocol for a final
performance claim.

Never select the best checkpoint by test accuracy. In this project, checkpoint
selection is by **validation macro-F1**.

### 0.3 Verify the reported score from subject predictions

Recompute accuracy and macro-F1 from the saved test predictions, and verify
that:

- There is exactly one prediction per test subject and no subject occurs in
  another split.
- The number and class supports match the frozen split JSON.
- The reported checkpoint digest matches the checkpoint evaluated.
- Class IDs map consistently to `CN`, `EMCI`, `LMCI`, and `AD`.
- The confusion-matrix diagonal total divided by the number of test subjects
  reproduces accuracy, subject to rounding.

For a 245-subject test set, 41.22% is approximately 101 correct predictions.
The current 245-subject split's largest class has 93 subjects, so a constant
majority-class prediction baseline is about 38.0% accuracy. This is only a
simple reference point, not a sufficient benchmark. The test set is small
enough that confidence intervals will be wide; report the evaluator's
subject-bootstrap interval and do not overinterpret small percentage-point
differences. Bootstrap uncertainty for one fitted model does not capture
variation across training seeds.

## Phase 1 — Audit subjects, labels, and data identity

Do not try to “train through” a data-integrity problem.

### 1.1 Reconcile cohort and labels

Use the manifests and split JSON as the auditable record. Confirm the current
cohort totals, class counts by split, one baseline image per subject, and
zero subject overlap across train, validation, and test. The locally
reconciled manifest previously contained 1,593 subjects split 1,112/236/245;
regenerate and verify counts from the exact data version being trained rather
than assuming these counts are permanent.

Inspect label definitions and diagnosis timing with the ADNI/cohort owner:

- Confirm each diagnosis corresponds to the image's intended baseline visit.
- Check for conflicting diagnoses, uncertain/missing labels, and rules used to
  map source diagnoses to CN/EMCI/LMCI/AD.
- Check duplicate image IDs, duplicate files, repeated scans, and subjects
  represented under alternate identifiers.
- Review class counts and diagnosis proportions by `phase`, field strength,
  scanner/series metadata when available.

The manifest includes `phase`, `visit`, and `field_strength`; these are useful
for auditing. Do not feed identifiers or metadata into the image classifier
without a separate, leakage-aware study design.

The completed local audit found that training diagnosis proportions vary by
ADNI phase; see the execution checklist for counts. Investigate this as a
potential source of domain confounding, not as proof of leakage.

### 1.2 Recheck split isolation and leakage routes

The project code checks subject-level split membership, duplicate manifest
subjects, and unique image paths. Verify the exact training data still satisfies
those checks and that no downstream preprocessing step used test labels or
test-derived statistics.

Use training data only to fit any cohort-level normalization, template,
feature selection, calibration, or threshold. A fixed, external reference
template can be applied uniformly only if its use is documented and does not
incorporate held-out cohort information. Per-image normalization is distinct
from fitting population statistics.

### 1.3 Record simple baselines

On the development split only, compute and save:

- Majority-class accuracy and macro-F1.
- A stratified random predictor baseline.
- The current pretrained ResNet-18 configuration.

These baselines identify whether a proposed change adds useful information.
Do not compare models using different test cohorts or silently changed labels.

## Phase 2 — Verify actual spatial preprocessing and image quality

This is a **high-priority validity check**, but do not assume the affine-origin
discrepancy is necessarily the cause of the model's errors. The current
`same_fov` code checks orientation and physical field-of-view extent; it does
not prove that the images have matching MNI world coordinates or anatomical
alignment.

### 2.1 Verify spatial provenance

Ask the preprocessing owner for the exact transformation/resampling procedure,
target grid, reference image and version, interpolation methods, and any
qform/sform handling. Determine whether the saved output affine is intended to
represent true world coordinates or only an internal voxel grid.

Do **not** fix a header-origin mismatch by changing qform/sform metadata alone.
Changing metadata without resampling the voxel data can make the headers look
consistent while leaving anatomy misregistered.

### 2.2 Perform all-volume checks plus visual QC

For every included image, validate finite voxels, shape, spacing, units,
orientation, and a non-singular affine. Then compare voxel-to-world mappings
against the agreed target grid. If the pipeline intentionally uses a
non-MNI grid, verify that every image uses the same anatomical voxel
correspondence instead of describing it as world-coordinate MNI alignment.

Create blinded overlays/slice montages of each image against the reference and
review a prespecified, stratified sample covering all diagnosis classes,
ADNI phases, field strengths, and known QC edge cases. Include landmarks or
atlas overlays only when their coordinate mapping is verified. Record
accept/reject criteria and exceptions before training. Visual QC complements
automated checks; it does not replace them.

If a true alignment issue is confirmed, correct the upstream preprocessing
pipeline, regenerate every scan using one documented target grid, rebuild the
manifest, rerun audits, and start a **new** experiment. Do not mix old and
reprocessed volumes or resume checkpoints across different data identities.

## Phase 3 — Reproduce a clean baseline

Use a new run directory and the current
`configs/colab_medicalnet_resnet18_128.yaml` as the baseline profile.

1. Pin the repository commit and dependency/runtime versions.
2. Set and record the trusted MedicalNet checkpoint SHA-256 in
   `model.pretrained_checkpoint_sha256`; inspect the generated loading report
   to confirm the intended backbone tensors loaded and the classifier head was
   handled as expected.
3. Run the dataset audit and inspect errors before training:

   ```bash
   python -m scripts.validate_dataset \
     --config configs/colab_medicalnet_resnet18_128.yaml
   ```

4. Run the tiny-overfit check as an implementation sanity check. It tests that
   the code can fit a very small sample; it is **not** a test of generalization:

   ```bash
   python -m scripts.tiny_overfit \
     --config configs/colab_medicalnet_resnet18_128.yaml \
     --output /content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs/tiny_overfit_baseline
   ```

5. Run the documented short trial, check memory, loss finiteness, checkpoint
   loading, and output files, then train the complete baseline with a new
   `experiment.name` and fixed seed:

   ```bash
   python -m src.train \
     --config configs/colab_medicalnet_resnet18_128.yaml \
     --stop-after-epoch 2
   python -m src.train \
     --config configs/colab_medicalnet_resnet18_128.yaml
   ```

   The two commands above start separate runs unless the second explicitly
   resumes the first run's `last.pt`. Do not assume the trial automatically
   continues training.

6. Inspect `data_audit.json`, `environment.json`, `model_summary.json`,
   `medicalnet_load_report.json`, `history.csv`, and
   `validation_predictions.csv`. Confirm the manifest/split hashes match the
   intended cohort and compare training versus validation curves.

Run the baseline before changing the sampler, augmentation, optimizer, or
architecture. Without a reproducible reference run, the effect of a later
change cannot be attributed.

## Phase 4 — Diagnose learning and generalization on development data

Use validation predictions and training history, not the test set, to identify
the problem.

1. Inspect confusion matrix and per-class precision, recall, F1, and support.
   Identify whether EMCI/LMCI collapse into each other, whether CN or AD are
   overpredicted, and whether severe cross-stage errors occur.
2. Compare train and validation metrics over epochs. A growing train/validation
   gap suggests overfitting; poor performance on both suggests underfitting,
   optimization trouble, label noise, or weak signal. A noisy validation curve
   alone does not establish which cause applies.
3. Stratify validation errors by ADNI phase, field strength, series, and other
   available acquisition metadata. Look for site/scanner shortcuts or a
   distribution shift between train and validation.
4. Review false positives and false negatives with blinded image QC. Any image
   inspection used to modify preprocessing must happen without consulting the
   locked test results.
5. Compare to the simple development baselines from Phase 1. An intervention
   must improve the chosen validation metric consistently, not just produce a
   more attractive training curve.

## Phase 5 — Run a small, controlled experiment matrix

Create separate configuration files that `extends` the current Colab profile.
Give each experiment a unique `experiment.name`, seed, and output directory.
Never edit an old run's saved config or checkpoint.

Change one factor at a time first. Use the same manifest, split, pretrained
weights, epoch limit, selection metric, and evaluation procedure for fair
comparisons.

| Experiment | Configuration change | Question |
|---|---|---|
| A. Reproduced baseline | Existing profile: balanced sampler on, weighted loss off, augmentation off | Does the result reproduce with verified artifacts? |
| B. Natural sampling | `balanced_sampler: false`, `weighted_loss: false` | Is the sampler helping, or increasing training variance? |
| C. Weighted loss | `balanced_sampler: false`, `weighted_loss: true` | Does training-only class weighting improve minority-class validation metrics? |
| D. Mild augmentation | Keep baseline sampler/loss; enable only conservative affine and/or blur | Does realistic variation reduce overfitting without harming validation recall? |
| E. Lower learning rate | Compare a small predeclared set around the current `1e-4` rate, one value per run | Is full fine-tuning too aggressive for the pretrained representation? |
| F. Combined candidate | Combine only changes that independently helped validation | Do individually useful changes remain useful together? |

The project transform uses TorchIO affine, blur, and optional noise, but
augmentation is disabled in the current MedicalNet profile. Start with mild,
anatomically plausible transforms already represented by the project's
configuration (small degrees, translations in millimetres, and scale changes).
Inspect transformed volumes before a full run. Avoid left/right flips,
aggressive warping, cropping that removes relevant anatomy, or intensity
changes that erase class information. Augmentation is a hypothesis to test,
not an automatic improvement.

The current sampler draws with replacement and balances classes approximately
per epoch. Weighted cross-entropy uses weights derived from training labels.
Validation and test loaders must preserve natural class frequencies. Compare
natural sampling, sampler-only, and weighted-loss-only; do not stack sampler
and weighted loss for the primary comparison.

Keep the experiment count small and predeclare the primary selection metric
(the repository currently fixes selection to validation macro-F1). Track
accuracy, balanced accuracy, class-wise recall/F1, and validation loss as
secondary diagnostics. Avoid repeatedly tuning on the same small validation
set; every additional search can overfit it.

### Fine-tuning changes that need code support

The current training loop updates all model parameters; it does not implement
frozen-backbone or staged-unfreezing schedules. If you want to test head-only
warm-up followed by unfreezing, implement it as a separate, tested change:
confirm which tensors are trainable in each stage, ensure optimizer parameter
groups are correct, preserve resume behavior, and record the schedule in the
checkpoint. Do not claim staged fine-tuning was tested by merely changing the
YAML. First compare lower learning rates with the existing supported
all-parameter training path.

## Phase 6 — Measure variability and prevent validation overfit

One train/validation split and one seed are not enough to establish robust
improvement. For the most promising one or two settings:

- Train at multiple fixed seeds on the same development split and report every
  seed, not only the best run.
- Prefer subject-level stratified cross-validation on the development cohort
  if compute permits. All preprocessing, feature selection, calibration, and
  tuning must be performed inside each training fold. Keep the final test set
  outside the folds.
- If acquisition domain is a concern, add a preplanned phase/scanner-held-out
  analysis. Interpret it as a domain-shift stress test; do not replace
  subject-level isolation with scan-level splitting.
- Report mean and spread across runs/folds as well as subject-bootstrap
  confidence intervals. The evaluator's bootstrap interval captures
  uncertainty conditional on one fitted model, not run-to-run variation.
- Keep a table of config hash, commit, data hash, seed, best epoch, validation
  metrics, and checkpoint SHA-256 for every run.

Choose the final candidate using only the predeclared development procedure.
Do not cherry-pick an unusually strong seed.

## Phase 7 — Freeze and evaluate once on the final test set

Before opening the test results, freeze:

- model architecture and checkpoint-selection rule;
- preprocessing version and image inclusion rules;
- sampler/loss, augmentation, optimizer, seed strategy, and training schedule;
- class mapping and metric definitions;
- any calibration or decision thresholds, fitted only on development data.

Then evaluate the chosen checkpoint on the frozen test split once. For example:

```bash
python -m src.evaluate \
  --config configs/colab_medicalnet_resnet18_128.yaml \
  --checkpoint PATH_TO_FINAL_RUN/best.pt \
  --split test \
  --confirm-final-test \
  --output /content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs/final_test_evaluation
```

Use a new output directory. Verify its `test_metrics.json` and
`test_predictions.csv` against the checkpoint digest, label mapping, split
identity, and test audit. Report accuracy, macro-F1, balanced accuracy,
confusion matrix, per-class precision/recall/F1/support, per-class ROC-AUC
where defined, and confidence intervals. State the sample count and class
supports. Do not change the model in response to this test result and then
continue calling that same test an independent final evaluation.

If the final test result is below 0.84, report the actual result. Do not adjust
labels, remove difficult subjects post hoc, tune thresholds on test predictions,
or choose a different checkpoint based on test metrics to force the target.
Plan a new independent evaluation for any subsequent model iteration.

## Phase 8 — External generalization, only after internal validity

External evaluation is a separate study phase, not a guaranteed last few
percentage points of validation. Before using OASIS, AIBL, or another cohort:

1. Confirm it has comparable four-class diagnostic labels. Cohorts may not
   define EMCI and LMCI in a way that maps to this project's ADNI labels.
   Do not invent a mapping after viewing model predictions.
2. Confirm permissions, inclusion criteria, scan modality, preprocessing,
   acquisition differences, and participant-level independence.
3. Freeze the trained model and preprocessing. Do not fine-tune, recalibrate,
   or adjust thresholds on the external test cohort if calling the result
   zero-shot generalization.
4. Report class-wise metrics, uncertainty, and domain differences. A “no more
   than 3–5% drop” is not a universal standard; predefine an acceptable change
   based on the intended use and uncertainty before evaluation.

If no genuinely compatible external cohort is available, state that external
generalization is untested.

## Phase 9 — Explanations and clinical interpretation

Do not use attractive attribution maps as evidence that the classifier learned
valid disease mechanisms. First establish stable, leakage-aware predictive
performance.

- Review Grad-CAM or other attribution maps for correct tensor/grid handling,
  stability under small input perturbations, and sensitivity to model
  randomization. Compare true and false predictions and use blinded reviewers.
- Test for shortcut dependence on scanner/site, skull-stripping boundaries,
  background, registration artifacts, and demographic proxies.
- Treat TCAV as a separate analysis. Define concepts and negative controls
  before analysis, use live model gradients where required, and test concept
  sensitivity across seeds and subjects. Saved `.npy` exports alone do not
  implement a TCAV scoring workflow.
- Describe these methods as model-behavior analyses, not clinical validation
  or proof of causal neuroanatomical reasoning.

## Project execution checklist

### Gate 1 — Evidence

- [ ] Exact run/checkpoint/config for 0.4122 / 0.3744 recovered.
- [ ] Test predictions, per-class report, confusion matrix, data audit, and
      hashes saved.
- [ ] Test split classified as locked, exploratory, or contaminated by prior
      tuning.

### Gate 2 — Data and preprocessing

- [ ] Cohort and class counts regenerated from the exact manifest.
- [ ] Subject overlap, duplicates, diagnosis/visit mapping, and split leakage
      checked.
- [ ] Spatial provenance and affine interpretation confirmed with the
      preprocessing owner.
- [ ] Automated checks and prespecified visual QC completed; issues corrected
      upstream, not by editing headers.

### Gate 3 — Modeling

- [ ] MedicalNet load report and checkpoint SHA verified.
- [ ] Clean baseline reproduced in a new run directory.
- [ ] Validation error analysis completed.
- [ ] Sampler/loss/augmentation/learning-rate comparisons run as separate,
      documented experiments.
- [ ] Promising candidates repeated across seeds or development folds.

### Gate 4 — Final claims

- [ ] Candidate and analysis plan frozen before final test evaluation.
- [ ] One final test evaluation saved with full metrics and confidence
      intervals.
- [ ] 0.84+ claimed only if measured under that frozen protocol; otherwise the
      observed value is reported without adjustment.
- [ ] External generalization and explainability claims limited to analyses
      actually performed.

## Repository-specific references

- [Colab data handoff and runbook](HANDOFF_AND_RUNBOOK.md)
- [Verification record](VERIFICATION.md)
- [Code walkthrough](CODE_WALKTHROUGH.md)
- [`src/dataset.py`](../src/dataset.py): split validation, image-grid checks,
  normalization, and audit.
- [`src/model.py`](../src/model.py): ResNet-18 and activation interface.
- [`src/engine.py`](../src/engine.py): balanced sampler, class weights, and
  shared train/evaluation loop.
- [`src/train.py`](../src/train.py): validation macro-F1 selection and
  checkpointing.
- [`src/evaluate.py`](../src/evaluate.py): frozen-checkpoint evaluation and
  saved metrics/predictions.
