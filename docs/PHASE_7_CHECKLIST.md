# Phase 7 Checklist — Final Test Evaluation

**Goal:** estimate final held-out performance once, after freezing the
complete model and analysis protocol.

**Current status:** **NOT STARTED for a newly frozen final evaluation.** The
reported test result has already been viewed, and its prior use for tuning is
unknown. See [Phase 0](PHASE_0_CHECKLIST.md). If that test informed choices,
use a new untouched test cohort/design for a final claim.

## 7.1 Freeze before opening test results

- [ ] Record final candidate and checkpoint SHA-256.
- [ ] Freeze architecture and validation-based checkpoint-selection rule.
- [ ] Freeze preprocessing version, image inclusion rules, and data identity.
- [ ] Freeze sampler/loss, augmentation, optimizer, seed strategy, and
  training schedule.
- [ ] Freeze class mapping, metrics, confidence interval procedure, and any
  calibration/thresholds; fit calibration/thresholds on development data only.
- [ ] Confirm Phase 0 test disposition and ensure model selection did not use
  this test result.
- [ ] Record protocol, date, responsible analyst, and command before evaluation.

## 7.2 Run one final evaluation

- [ ] Use the frozen checkpoint and config, a fresh output directory, and
  explicit final-test confirmation:

  ```bash
  python -m src.evaluate \
    --config configs/colab_medicalnet_resnet18_128.yaml \
    --checkpoint PATH_TO_FINAL_RUN/best.pt \
    --split test \
    --confirm-final-test \
    --output /content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs/final_test_evaluation
  ```

- [ ] Verify checkpoint digest, manifest/split identity, audit, class order,
  and one prediction per test subject.
- [ ] Independently recompute saved metrics from predictions.
- [ ] Report accuracy, macro-F1, balanced accuracy, confusion matrix,
  per-class precision/recall/F1/support, ROC-AUC where defined, uncertainty,
  total sample count, and class supports.
- [ ] Archive predictions, metrics, report, audit, plots, command, and hashes.

## 7.3 Report without post-hoc tuning

- [ ] Report the measured result even if it is below 0.84.
- [ ] Do not change labels, exclude difficult subjects after seeing results,
  tune thresholds on test predictions, or select another checkpoint from test
  metrics.
- [ ] If the test result informs a later model iteration, label it exploratory
  and plan a new independent final evaluation.

## Exit gate

- [ ] Evaluation exactly follows the frozen analysis plan and complete
  artifacts are saved.
- [ ] Claims are limited to the cohort, split, and metrics actually tested.
