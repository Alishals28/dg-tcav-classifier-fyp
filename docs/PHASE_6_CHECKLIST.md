# Phase 6 Checklist — Measure Variability and Prevent Validation Overfit

**Goal:** determine whether the best development configuration is robust
across seeds/folds and not just a favorable run.

**Current status:** **NOT STARTED.** Begin after Phase 5 has identified at most
one or two promising candidates using the predeclared development procedure.

## 6.1 Repeat promising candidates

- [ ] Select at most one or two candidates using validation-only rules before
  starting repeats.
- [ ] Train each at multiple fixed seeds on the same development split.
- [ ] Report every seed; do not report only the best-performing run.
- [ ] If compute permits, use subject-level stratified cross-validation on the
  development cohort.
- [ ] Keep the final test cohort outside all folds and tuning.
- [ ] Within each fold, fit any normalization, feature selection, calibration,
  and threshold only using that fold's training data.
- [ ] If justified by Phase 1/4, run a preplanned phase/scanner-held-out
  stress test while preserving subject-level separation.

## 6.2 Summarize uncertainty and run provenance

- [ ] Report mean and spread across seeds/folds for all primary metrics.
- [ ] Report subject-bootstrap confidence intervals separately; explain that
  they are conditional on one fitted model and do not replace run-to-run
  variation.
- [ ] Include class-wise performance and support for every repeat/fold.
- [ ] Maintain a table with config hash, commit, data hash, seed/fold, best
  epoch, validation metrics, and checkpoint SHA-256 for every run.
- [ ] Investigate material divergence across seeds rather than cherry-picking.

## Exit gate

- [ ] Candidate choice follows a documented development-only rule.
- [ ] Repeats/folds and their variability are fully reported.
- [ ] No single unusually strong seed is presented as representative without
  qualification.
