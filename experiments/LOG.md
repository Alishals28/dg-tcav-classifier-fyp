# Experiment log: DG-TCAV 3D MRI classifier (branch exp/autoresearch-20261019)

Goal: maximise validation macro-F1 (mean of 3 seeds), target mean >= 0.50 with every class F1 >= 0.30.
Protocol: fixed split only (manifest SHA-256 44d9c943..., splits 2dcf5947...; train 535, val 112, test 121).
The test set is locked and never evaluated here. Model selection is on validation only.

Noise rule: an experiment is accepted only if its 3-seed mean macro-F1 beats the current baseline mean by more
than max(seed std of either). Below that it is "inconclusive" (or "reject" if worse beyond the same margin).
With 112 validation subjects, single-run differences below about 0.03 macro-F1 are noise.

## Results

<!-- table:start -->
| id | change | val macro-F1 mean±std | bal-acc mean±std | worst-class F1 (class) | seeds | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 001 | ResNet-18 random init, batch 8, patience 20 (else as before: weighted CE, AdamW 1e-4 cosine/50 ep, AMP, no aug) | 0.374 ± 0.010 | 0.382 ± 0.005 | 0.239 (LMCI) | 3 | accept |
| 002 | MedicalNet-pretrained ResNet-18, Adeel's settings, repointed to the fixed split | 0.381 ± 0.033 | 0.386 ± 0.038 | 0.251 (LMCI) | 3 | inconclusive |
| 003 | ClassifierCNN with Adeel's committed settings (epochs capped 100 -> 40 for budget) | 0.201 ± 0.120 | 0.284 ± 0.078 | 0.029 (LMCI) | 3 | reject |
| 004 | 001 + GPU augmentation (rotation ±8°, scale ±8%, translation ±4 vox, p 0.8; foreground gain/shift 0.1) | 0.410 ± 0.025 | 0.414 ± 0.029 | 0.182 (EMCI) | 3 | accept |
| 005 | 001 + EMA of weights (decay 0.998 from step 0; validation/checkpoints use EMA weights) | 0.146 ± 0.055 | 0.254 ± 0.006 | 0.000 (LMCI) | 3 | reject |
| 006 | 001 with data.input_size 96 (trilinear downsample on CPU in the data loader) | 0.394 ± 0.014 | 0.391 ± 0.009 | 0.248 (LMCI) | 3 | accept |
| 007 | 004 with epochs 50 -> 100 and patience 20 -> 30 | 0.430 ± 0.010 | 0.440 ± 0.012 | 0.296 (EMCI) | 3 | inconclusive |
| 008 | 004 + ordinal Gaussian soft labels during training (sigma 0.6) | 0.434 ± 0.026 | 0.436 ± 0.032 | 0.308 (LMCI) | 3 | accept |
| 009 | 004 + 96^3 input (data.input_size 96) | 0.418 ± 0.017 | 0.424 ± 0.014 | 0.294 (EMCI) | 3 | inconclusive |
| 010 | 004 with MedicalNet resnet_18_23dataset.pth initialisation (only the init differs) | 0.333 ± 0.022 | 0.360 ± 0.026 | 0.120 (EMCI) | 3 | reject |
| 011 | 004 with GroupNorm (min(32, C/4) groups) instead of BatchNorm | 0.215 ± 0.109 | 0.286 ± 0.063 | 0.038 (EMCI) | 3 | reject |
| 012 | 004 with stronger GPU augmentation (±15°, ±15% scale, ±8 vox, intensity 0.2, p 0.9) | 0.385 ± 0.031 | 0.398 ± 0.011 | 0.179 (EMCI) | 3 | inconclusive |
| 013 | 004 with AdamW weight decay 1e-4 -> 0.05 | 0.419 ± 0.024 | 0.425 ± 0.018 | 0.241 (EMCI) | 3 | inconclusive |
| 014 | 004 + EMA of weights, decay 0.998 with warm-up min(0.998, (1+n)/(10+n)); validation/checkpoints use EMA weights (fair retest of 005) | 0.375 ± 0.034 | 0.387 ± 0.018 | 0.177 (EMCI) | 3 | reject |
| 015 | none; 004 settings with seeds 4, 5, 6 | 0.400 ± 0.026 | 0.404 ± 0.038 | 0.221 (EMCI) | 3 | replication |
| 016 | none; 008 settings with seeds 4, 5, 6 | 0.444 ± 0.021 | 0.457 ± 0.042 | 0.282 (EMCI) | 3 | replication |
| 017 | 004 + ordinal soft labels with sigma 1.0 (instead of 0.6) | 0.398 ± 0.021 | 0.438 ± 0.011 | 0.236 (LMCI) | 3 | inconclusive |
<!-- table:end -->

Reference (not a 3-seed result): earlier ResNet-18, random init, batch 2, patience 10, seed 42, two runs on
14 Oct 2026: best val macro-F1 0.296 and 0.333 (identical settings; GPU non-determinism).

## Setup notes and assumptions

- 2026-10-19. Pulled origin/main (Adeel's 4 commits: ClassifierCNN, balanced sampler, Adam/betas options, blur
  augmentation, MedicalNet SHA-256 pinning, tqdm). Branched exp/autoresearch-20261019 from d5e9f61.
- **Missing `data/preprocessed_128_classifier_inputs_refresh/`.** Not in git history, not on this machine, not on
  Drive. Not invented. Adeel must commit it. Evidence of a different protocol: Adeel's Colab ClassifierCNN runs
  (18 Oct, Drive classifier_outputs) used manifest SHA-256 3d519459... and splits ed830412... with
  train/val/test = 1112/236/245. That is a different cohort and split. It must never be mixed into these
  comparisons unless its subject IDs are identical to the fixed 535/112/121 split (they cannot be: the
  counts differ).
- Adeel's MedicalNet config is repointed to the committed manifest/splits in a new file
  (experiments/002_medicalnet_resnet18/config.yaml); teammate configs are not edited.
- MedicalNet weights were not on Drive (the config expects DG_TCAV_OUTPUTS/resnet_18_23dataset.pth). Downloaded the
  official file from Hugging Face TencentMedicalNet/MedicalNet-Resnet18 (MIT), SHA-256 61224f93...; uploaded
  as private Kaggle dataset khizerinam08/medicalnet-resnet18-23dataset.
- Adeel's two Colab ClassifierCNN runs collapsed: train loss stuck at ln 4 = 1.386, validation balanced
  accuracy exactly 0.25 every epoch, about 280 s/epoch on a T4 (about 5x ResNet-18).
- kaggle/kernel.py now defaults to ClassifierCNN, so it is not used. experiments/run.py builds its own
  private kernel per experiment, clones the exact pushed commit, and runs the seeds in parallel on the
  Kaggle T4 x2 machine (one seed per GPU).
- Determinism: src.reproducibility already sets cudnn.deterministic, disables benchmark, and enables
  use_deterministic_algorithms(warn_only=True). 3D pooling backward passes remain non-deterministic on
  CUDA, so identical seeds can still diverge.
- Dead-run rule (training.dead_run_epochs = 6): stop a seed when validation predicts a single class for
  6 consecutive epochs. Logged in dead_run.json and counted as that seed's result at its best epoch.
- Seeds 1, 2, 3 for every experiment.
- Kaggle allows 2 concurrent GPU sessions, so experiments run in pairs. Experiments launched in parallel are
  each judged against the same accepted baseline; accepted changes are combined in a later experiment.
- Pre-existing test failures (also on main): 8 tests in tests/test_dataset.py and tests/test_smoke_training.py
  fail because the synthetic volumes no longer match the reference grid that configs/base.yaml now requires.
  Not caused by this branch; model/engine/metrics tests pass. New options were checked on CPU directly.
- 004 uses GPU batch augmentation (src/engine.py gpu_augment) instead of torchio: CPU affine on 128^3 with
  4 vCPUs shared by two seeds would roughly double epoch time.
- 2026-10-19 19:50 UTC: Kaggle left 005 QUEUED for over 2 h, while a later kernel (008) started at once. A queued
  kernel counts toward the 2-session limit, so 005 also blocked a slot. Deleted the empty 005 kernel and relaunched
  it with the same config and seeds; no outputs existed.
- 2026-10-19 20:35 UTC: second Kaggle account added (mabdullahtahir1, key KAGGLE_ABT_API_KEY): 2 more concurrent
  T4 x2 sessions. It holds its own private copies of the 768 scans and the MedicalNet weights. runner: --account.
  All seeds of one experiment always run on one account (same Kaggle image, same T4 type).
- Data finding: the scans have no zero voxels. Background is a constant z-scored value (-0.57 to -0.83, about 63%
  of each volume), and the union of brain masks spans the full 128^3 cube. So (a) a fixed brain bounding-box crop
  gains nothing and is dropped; (b) gpu_augment pads rotated-in borders with 0 instead of the background value, and
  its "foreground" intensity shift applies to the whole volume. This is a small artefact inside the accepted 004
  bundle, kept as is for comparability; a background-aware version is a candidate experiment.
- 2026-10-19 20:55 UTC, free check (no GPU): averaging the 3 seeds' validation probabilities does not help.
  001 0.374 -> 0.365, 002 0.381 -> 0.383, 004 0.410 -> 0.398, 006 0.394 -> 0.359 (per-seed mean -> ensemble macro-F1).
  Reading: each seed's number comes from its best epoch, chosen on the same 112 validation subjects, so per-seed scores
  are optimistically biased; the fixed-rule ensemble has no such selection. All experiments share this bias, so
  comparisons stay fair, but expect lower absolute numbers on the locked test set. Seed ensembling (idea F) dropped.
- 2026-10-20 02:00 UTC: 008 (ordinal soft labels) confirmed on 6 matched seeds (with 015/016) and accepted; new
  baseline = 008, mean 0.439 ± 0.022 over seeds 1-6 (the table shows the seeds 1-3 entry). Budget: the default
  of 12 full experiments is used up; only in-flight runs (011, 013, 014, 017) plus one final combination on the new
  baseline (018) are run before stopping.
- 2026-10-20 06:30 UTC: 014 (EMA with warm-up) crashed on its first optimiser step in every seed: the warm-up EMA
  averaged all buffers, including BatchNorm's integer num_batches_tracked ("lerp_cuda not implemented for Long").
  My CPU check used a model without integer buffers and missed it. Fixed (integer buffers are copied) and checked on
  a BatchNorm ResNet; 014 rerun with the same config and seeds. No result from the crashed run is used.
- Same run: seeds 1-2 spent 5.1 h before their first step, while seed 3 (started later) reached it in 80 s. 011 showed the
  same pattern (5.8 h wall for 1.1 h of training). So the first read of /kaggle/input is sometimes extremely slow. The
  kernel now copies the scans to local disk once before training and logs the time. This affects I/O only, not results.
