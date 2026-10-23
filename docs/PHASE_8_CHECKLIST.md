# Phase 8 Checklist — External Generalization

**Goal:** assess transport to an independent cohort only after internal data
and evaluation validity have been established.

**Current status:** **NOT STARTED.** No compatible external evaluation is
documented. If no eligible cohort is available, report external
generalization as untested.

## 8.1 Validate cohort suitability before accessing model predictions

- [ ] Identify a genuinely independent cohort and confirm participant-level
  non-overlap with training, development, and internal test cohorts.
- [ ] Confirm permissions, ethics/data-use requirements, and inclusion rules.
- [ ] Confirm MRI modality/protocol and preprocessing can be applied
  consistently.
- [ ] Confirm diagnostic label definitions are comparable, including whether
  EMCI and LMCI have valid equivalents.
- [ ] Predefine any label harmonization with clinical/cohort owners before
  inspecting model predictions.
- [ ] Document differences in site, scanner, acquisition, demographics, and
  population.
- [ ] Predefine metrics, uncertainty, subgroup analyses, and acceptable
  performance change for intended use; do not adopt a universal 3–5% drop.

## 8.2 Freeze and evaluate zero-shot

- [ ] Freeze model checkpoint, preprocessing, label mapping, and thresholds
  before external evaluation.
- [ ] Do not fine-tune, recalibrate, or tune thresholds on the external test
  cohort if claiming zero-shot generalization.
- [ ] Verify cohort independence and image/label eligibility.
- [ ] Report class-wise metrics, supports, confidence intervals, and domain
  differences.
- [ ] Archive predictions, metrics, cohort/protocol identity, and hashes.
- [ ] Clearly distinguish zero-shot evaluation from adapted/fine-tuned
  performance.

## Exit gate

- [ ] External result is independently reproducible and fully qualified, or
  report external generalization as untested.
