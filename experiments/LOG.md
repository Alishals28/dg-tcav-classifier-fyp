# Experiment log: DG-TCAV 3D MRI classifier (branch exp/autoresearch-20261019)

Goal: maximise validation macro-F1 (mean of 3 seeds), target mean >= 0.50 with every class F1 >= 0.30.
Protocol: fixed split only (manifest SHA-256 44d9c943..., splits 2dcf5947...; train 535, val 112, test 121).
The test set is locked and never evaluated here. Model selection is on validation only.

Noise rule: an experiment is accepted only if its 3-seed mean macro-F1 beats the current baseline mean by more
than max(seed std of either). Below that it is "inconclusive" (or "reject" if worse beyond the same margin).
With 112 validation subjects, single-run differences below about 0.03 macro-F1 are noise.

## Results

<!-- table:start -->
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
