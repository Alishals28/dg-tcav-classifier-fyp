# Phase Checklists

Use these phase-specific checklists alongside the
[model-improvement roadmap](MODEL_IMPROVEMENT_ROADMAP.md). Work through phases
in order unless a later phase is explicitly a separate, preplanned analysis.
Do not treat a checked item as evidence unless its linked or saved artifact
exists.

## Checklist files

| Phase | Focus | Checklist |
|---|---|---|
| 0 | Freeze evidence and protect the test set | [Phase 0 checklist](PHASE_0_CHECKLIST.md) |
| 1 | Audit subjects, labels, and data identity | [Phase 1 checklist](PHASE_1_CHECKLIST.md) |
| 2 | Verify spatial preprocessing and image quality | [Phase 2 checklist](PHASE_2_CHECKLIST.md) |
| 3 | Reproduce a clean baseline | [Phase 3 checklist](PHASE_3_CHECKLIST.md) |
| 4 | Diagnose learning and generalization on development data | [Phase 4 checklist](PHASE_4_CHECKLIST.md) |
| 5 | Run a small, controlled experiment matrix | [Phase 5 checklist](PHASE_5_CHECKLIST.md) |
| 6 | Measure variability and prevent validation overfit | [Phase 6 checklist](PHASE_6_CHECKLIST.md) |
| 7 | Freeze and evaluate once on the final test set | [Phase 7 checklist](PHASE_7_CHECKLIST.md) |
| 8 | Evaluate external generalization | [Phase 8 checklist](PHASE_8_CHECKLIST.md) |
| 9 | Assess explanations and clinical interpretation | [Phase 9 checklist](PHASE_9_CHECKLIST.md) |

## Status convention

- **DONE** — evidence is present and the check has been completed.
- **PARTIAL** — some evidence exists, but the check is not fully satisfied.
- **BLOCKED** — required data, artifact, or owner decision is unavailable.
- **NOT STARTED** — work remains to be performed.

The 0.84+ accuracy objective is aspirational, not a promised outcome or a
phase-completion criterion. Report the measured result under the frozen
protocol, whatever it is.

The original combined audit record remains available at
[Phase 0–1 execution evidence](PHASE_0_1_CHECKLIST.md); it contains cohort
counts, hashes, baseline calculations, and the evidence limitations recorded
for this workspace.
