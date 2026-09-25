# Implementation verification — 2026-09-25

This is software verification on artificial data, not a report of ADNI results.

## Checks actually performed

- Editable package installation succeeded; `pip check` reported no broken requirements.
- `python -m pytest`: 41 tests passed on CPU. No tests skipped.
- Ruff checks/formatting, Python compile checks, and git whitespace checks passed.
- Notebook JSON passed nbformat schema validation.
- Synthetic CLI training completed two epochs and wrote best/last checkpoints,
  history, validation predictions, configuration, and learning curves.
- Integration tests exercised evaluation metrics, figures, predictions, feature
  export, refused result overwrites, and checkpoint/config mismatch rejection.
- Epoch-boundary CPU resume reproduced uninterrupted model-state tensors exactly
  in the controlled two-epoch test (num_workers=0).
- Tiny-overfit CLI on eight artificial training examples reached 100% training
  accuracy in eval mode after 11 epochs. This is a debugging result only.
- Full-width model at input [1,1,91,109,91] completed forward/backward on CPU,
  returned [1,4] logits and [1,512] pooled features, with finite stem gradients.
  Model parameter count: 32,988,228.
- Backbone layer-4 outputs matched Tencent/MedicalNet's implementation exactly
  for the same weights and a controlled 16³ input. Reference source commit:
  `20f76aaab5cac8056eaf50b79ed97c09dbfbd3bd`. This validates architecture behavior,
  not loading of the real distributed pretrained checkpoint.
- An incomplete real-training config correctly stopped at the preprocessing gate.

Test categories: partition coverage/overlap/duplicates; invalid labels/IDs;
missing/ambiguous volumes; shape, spacing, units, orientation, affine, NaN/Inf,
empty/constant rejection; normalization; train-only augmentation; feature/gradient
access; strict synthetic MedicalNet-state loading; known-answer metrics;
weighted loss accounting; training, resume, evaluation, and export.

## Environment used

Python 3.12.14; PyTorch 2.14.0+cpu; NumPy 2.3.5; pandas 2.2.3;
NiBabel 5.4.2; scikit-learn 1.8.0; TorchIO 0.23.1; PyYAML 6.0.3;
Matplotlib 3.10.8; pytest 8.4.2. No CUDA device was available here.
Every real run also writes its own installed package versions and Git commit.

## Not yet verified — required before results can be reported

1. Receipt/QC of all real preprocessed Cohort A volumes and their exact template grid.
2. Audit against the actual manifest and fixed split JSON, preserving image-ID provenance.
3. Full loading of the actual trusted MedicalNet ResNet-18 checkpoint, with report/hash.
4. Tiny-overfit on model-ready training examples (no test subjects).
5. Real-data GPU smoke run: CUDA/AMP behavior, memory, BatchNorm behavior, I/O and epoch time.
6. Full experiments, frozen model selection, and held-out evaluation.

The Kaggle notebook has been schema-validated, not executed inside Kaggle.
CUDA determinism and exact cross-device/cross-version resume are not guaranteed.
No ADNI scans were used in the tests and no clinical performance claim is made.
