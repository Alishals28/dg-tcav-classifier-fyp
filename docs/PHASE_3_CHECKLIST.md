# Phase 3 Checklist — Reproduce a Clean Baseline

**Goal:** train a trustworthy reference model using the current supported
MedicalNet-compatible 3D ResNet-18 profile before changing training choices.

**Current status:** **COMPLETE (DONE).** The reference MedicalNet-compatible 3D ResNet-18
baseline run (`medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z`) has been
audited, traced to exact source commit `adf1fb2`, resolved config, cohort data identity
(1,593 preprocessed subjects), and pretrained weights. All training history, validation
predictions, checkpoints, and environment logs are archived, verified, and preserved for Phase 4 diagnostics.

---

## 3.1 Freeze the run inputs

- [x] **Complete Phase 1 and Phase 2 exit gates.**
  - **Phase 1 exit gate:** Signed off and complete in [PHASE_1_CHECKLIST.md](PHASE_1_CHECKLIST.md) and [PHASE_0_1_CHECKLIST.md](PHASE_0_1_CHECKLIST.md). The 1,593-subject cohort (1,112 train, 236 val, 245 test) has zero subject overlap, strictly verified chronological retry policy, confirmed baseline screening visit semantics (`scmri`/`sc`), and fixed reference template (`template/mni152.nii.gz`, SHA-256: `dbbd4542...`).
  - **Phase 2 exit gate:** Signed off and complete in [PHASE_2_CHECKLIST.md](PHASE_2_CHECKLIST.md). Preprocessing provenance documented (pipeline `masked_rigid_affine_centroid_seed42_v4`, commit `ec726c61`, target grid origin convention confirmed). Automated geometry/intensity checks audited all 1,593 volumes with 0 errors in [outputs/dataset_audit.json](../outputs/dataset_audit.json). Prespecified visual QC inspected 22 stratified subjects with zero unresolvable alignment defects.

- [x] **Pin repository commit and dependency/runtime versions.**
  - Source commit: `adf1fb27c2a17487ea1bcc7989d0d358b71ab7fa` ("Fix tiny_overfit evaluation on BatchNorm and silence avg_pool3d warning").
  - Runtime environment recorded in [outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/environment.json](../outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/environment.json) (SHA-256: `7b088417c82c96afa0f1d01f72333820b825f230c22d4576be7bf3a929d42f14`):
    - Platform: `Linux-6.6.122+-x86_64-with-glibc2.39` (Google Colab VM)
    - Python: `3.13.15`
    - PyTorch: `2.11.0+cu130` (CUDA 13.0, Tesla T4 GPU, PID 14556)
    - Key dependencies: `numpy 2.1.3`, `pandas 2.2.3`, `nibabel 5.4.2`, `scikit-learn 1.6.1`, `torchio 0.23.1`, `PyYAML 6.0.3`, `matplotlib 3.10.0`.

- [x] **Use a new unique experiment name, output path, and fixed seed.**
  - Experiment name: `medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z`
  - Output path: `/content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z` (local workspace archive: [outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/](../outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/)).
  - Random seed: Fixed to `42` across Python `random`, NumPy `np.random.seed`, and PyTorch `torch.manual_seed` (both CPU and CUDA with deterministic CuDNN flags).

- [x] **Copy the resolved config into the run output; never edit an old run's saved configuration.**
  - Resolved configuration serialized directly upon run initialization to [outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/config.yaml](../outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/config.yaml).
  - Config SHA-256: `3ed998da5eb95a036867b3760c4ef6627d5fa9deb48284ab03cd486020b5a920`. Preserved read-only; no retrospective modifications.

- [x] **Record hashes for config, manifest, split JSON, reference, and pretrained checkpoint.**
  - **Resolved Config:** `3ed998da5eb95a036867b3760c4ef6627d5fa9deb48284ab03cd486020b5a920`
  - **Classifier Manifest (`data/preprocessed_128_classifier_inputs/classifier_manifest.csv`):** `3d51945937a98ea6400f1955106ef5ce1c805bf77555bc585b008402e01d7c0b` (1,593 subjects)
  - **Classifier Splits (`data/preprocessed_128_classifier_inputs/classifier_splits.json`):** `ed8304125a20800ee7e7aa800ff94a8b50e3031d75f89a2547197bb9d31adb1e` (1,112 train, 236 val, 245 test)
  - **Reference Template (`template/mni152.nii.gz`):** `dbbd4542823c1e2b12265c9b7e3cf959ee02ff7829bbf05cdcfbecc0013a00fa`
  - **Pretrained MedicalNet Weights (`resnet_18_23dataset.pth`):** `61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4` (131,992,196 bytes)

- [x] **Set the trusted MedicalNet checkpoint SHA-256 in `model.pretrained_checkpoint_sha256`.**
  - Digest `61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4` verified in `medicalnet_load_report.json` and pinned in [configs/colab_medicalnet_resnet18_128.yaml](../configs/colab_medicalnet_resnet18_128.yaml).

---

## 3.2 Validate data and implementation

- [x] **Run `python -m scripts.validate_dataset --config configs/colab_medicalnet_resnet18_128.yaml` against the intended data.**
  - Cohort dataset validator audited all 1,593 volumes across train, val, and test splits.
  - Audit output preserved in [outputs/dataset_audit.json](../outputs/dataset_audit.json) (SHA-256: `bc19d17f...`) and run test archive [outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/test_evaluation/data_audit.json](../outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/test_evaluation/data_audit.json) (SHA-256: `851d7331...`).
  - Result: `passed: true`, `total_audited_volumes: 1593`, `total_errors: 0`.

- [x] **Review all warnings/errors in `data_audit.json`; resolve or document each before training.**
  - Exactly 0 shape, spacing, affine, or non-finite errors detected across the 1,593 preprocessed NIfTI files.
  - Visual QC from Phase 2 reviewed automated outliers (10 low parenchyma volume subjects due to severe ventricular enlargement in AD/LMCI); all scans confirmed anatomically valid with correct spatial bounding boxes.

- [x] **Confirm all four classes occur in training and validation.**
  - Class distribution across splits:
    - **Train split (1,112 subjects):** CN = 425 (38.22%), EMCI = 310 (27.88%), LMCI = 215 (19.33%), AD = 162 (14.57%)
    - **Validation split (236 subjects):** CN = 90 (38.14%), EMCI = 66 (27.97%), LMCI = 46 (19.49%), AD = 34 (14.41%)
    - **Test split (245 subjects):** CN = 93 (37.96%), EMCI = 69 (28.16%), LMCI = 47 (19.18%), AD = 36 (14.69%)
  - All four classes are present and identically proportioned across all three partitions.

- [x] **Run tiny-overfit sanity check using a fresh output directory:**
  - Ran `python -m scripts.tiny_overfit --config configs/colab_medicalnet_resnet18_128.yaml --output /content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs/tiny_overfit_medicalnet`.
  - Output artifacts preserved at `G:\My Drive\DG-TCAV_FYP_NUST\DG_TCAV_OUTPUTS\classifier_outputs\tiny_overfit_medicalnet` (`subjects.csv`, `medicalnet_load_report.json`).
  - Unit test suite independently verifies model, data pipeline, and training loop via `pytest` (43/43 tests passed, 100%).

- [x] **Confirm the tiny subset can be fit and loss/gradients remain finite. Treat this only as an implementation check, not generalization evidence.**
  - In [notebooks/train.ipynb](../notebooks/train.ipynb) (Cell 17), execution logs confirm rapid convergence on the 8-subject balanced sample:
    - Epoch 1: Loss = 1.7540, Accuracy = 25.0% (random baseline)
    - Epoch 4: Loss = 1.0323, Accuracy = 75.0%
    - Epoch 7: Loss = 0.6087, Accuracy = 87.5%
    - Epoch 10: Loss = 0.4764, Accuracy = 87.5%
    - Epoch 12: Loss = 0.3458, Accuracy = 87.5%
  - Gradients remained non-zero, finite, and well-conditioned under gradient clipping (`clip_norm: 1.0`).

- [x] **Inspect `medicalnet_load_report.json`: intended backbone weights loaded; classifier head mismatch/reinitialization is expected and documented.**
  - Preserved in [outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/medicalnet_load_report.json](../outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/medicalnet_load_report.json) (SHA-256: `9ce30e055cf76ced7b776e1e6d74a7f56dbe32002ec18e4da000548e22d41813`):
    - `backbone_tensor_fraction: 1.0`: All 108 ResNet-18 backbone weight/bias/BatchNorm running tensors loaded cleanly.
    - `missing_or_mismatched: []` (0 missing).
    - `unexpected_keys: []` (0 unexpected).
    - `nonfinite_keys: []` (0 nonfinite).
    - Classification head: The original MedicalNet segmentation head (`conv_seg`) is replaced with a fresh linear layer `Linear(in_features=512, out_features=4)` initialized from scratch for the 4-class ADNI classification target.

- [x] **Confirm model architecture, class order, output shape, and trainable parameters in `model_summary.json`.**
  - Preserved in [outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/model_summary.json](../outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/model_summary.json) (SHA-256: `88a34deef3bf4c330062311aa8717388bb0dbf421bf08d49903038f168dbdcf2`):
    - Trainable parameter count: 32,988,228 (~33.0M)
    - Label mapping: `{"CN": 0, "EMCI": 1, "LMCI": 2, "AD": 3}`
    - Class weights: `null` (unweighted standard cross-entropy loss)
    - Output tensor shape: `(batch_size, 4)`

---

## 3.3 Run and verify the baseline

- [x] **Run a short trial on the target runtime; inspect GPU memory, epoch time, I/O, mixed precision, loss finiteness, and checkpoint load/save.**
  - Documented trial run executed on Colab Tesla T4 GPU using `--stop-after-epoch 2`.
  - **GPU Memory:** Peak VRAM consumption ~3.8 GB, safely within the 16.0 GB Tesla T4 boundary using batch size 4 and Automatic Mixed Precision (AMP `torch.cuda.amp.autocast`).
  - **Epoch speed:** ~160–165 seconds per epoch (~2.2 iterations/sec over 278 training mini-batches per epoch).
  - **I/O throughput:** Scans staged to local VM SSD (`/content/preprocessed_128`), avoiding Drive I/O bottlenecks.
  - **Loss finiteness:** Initial loss commenced at 1.385 (consistent with theoretical uniform guess $-\ln(0.25) \approx 1.3863$), with zero NaN/Inf instances.
  - **Checkpoint persistence:** `best.pt` and `last.pt` serialized properly with weights, optimizer, and scaler states.

- [x] **Ensure the short trial and full run are distinct unless resume is explicitly configured and documented.**
  - Distinct execution verified: Full baseline was trained fresh from epoch 1 under dedicated timestamped folder `medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z` without checkpoint contamination.

- [x] **Train full baseline using the predeclared current profile: balanced sampler on, weighted loss off, augmentation off, validation macro-F1 checkpoint selection.**
  - Predeclared profile strictly applied:
    - Balanced sampler: `balanced_sampler: true` (`WeightedRandomSampler` over inverse class frequencies in training split).
    - Weighted loss: `weighted_loss: false` (standard cross-entropy).
    - Augmentation: `augmentation.enabled: false`.
    - Checkpoint selection: `selection_metric: macro_f1` evaluated on the validation cohort after every epoch.

- [x] **Save training history and validation predictions.**
  - Full 43-epoch training history saved to [outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/history.csv](../outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/history.csv) (SHA-256: `c3744b79...`).
  - Validation predictions for all 236 validation subjects saved to [outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/validation_predictions.csv](../outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/validation_predictions.csv) (SHA-256: `72cdd099...`).

- [x] **Check train/validation curves and early stopping behavior.**
  - Learning curves plotted and preserved in [outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/learning_curves.png](../outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/learning_curves.png).
  - Train loss decreased steadily from 1.385 (epoch 1) down to 0.943 (epoch 43); training accuracy rose to 57.7%.
  - Validation macro-F1 peaked at **0.3993** at **Epoch 28** (val accuracy 43.64%, val loss 1.2754).
  - Early stopping: Configured with `early_stopping_patience: 15`. No validation macro-F1 exceeded 0.3993 during epochs 29–43; early stopping cleanly terminated training after epoch 43 ($28 + 15 = 43$).

- [x] **Preserve `data_audit.json`, `environment.json`, `model_summary.json`, `medicalnet_load_report.json`, `history.csv`, and `validation_predictions.csv`.**
  - All primary artifacts are preserved in local repo storage and Google Drive:
    - [data_audit.json](../outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/test_evaluation/data_audit.json): SHA-256 `851d733124ada24833a4cf91798e2c2289688ebc83ddbdeec4ad0ce7a565a491`
    - [environment.json](../outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/environment.json): SHA-256 `7b088417c82c96afa0f1d01f72333820b825f230c22d4576be7bf3a929d42f14`
    - [model_summary.json](../outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/model_summary.json): SHA-256 `88a34deef3bf4c330062311aa8717388bb0dbf421bf08d49903038f168dbdcf2`
    - [medicalnet_load_report.json](../outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/medicalnet_load_report.json): SHA-256 `9ce30e055cf76ced7b776e1e6d74a7f56dbe32002ec18e4da000548e22d41813`
    - [history.csv](../outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/history.csv): SHA-256 `c3744b79f3bd1f8d63936c67958a9cfe500f4c1faa2fe7494b469602c3afbc4b`
    - [validation_predictions.csv](../outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/validation_predictions.csv): SHA-256 `72cdd09966da7be14e84e161951eefc73b43bb1527cf6de306e710ba15a47c61`
    - [validation_metrics.json](../outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/validation_metrics.json): SHA-256 `d02c08356d646f77aafe029b9d56ca51525fc59e8f6cfd2e85227a9dd9e98803`
    - Checkpoint `best.pt`: SHA-256 `36112356bf299becf4e846cc1153468cef4f3723c4cfd6640a935775c39a6df5` (396,138,811 bytes, epoch 28)

- [x] **Record best epoch, validation macro-F1, all secondary metrics, checkpoint digest, seed, commit, and data/config hashes.**
  - **Best Epoch:** Epoch 28 (of 43 trained epochs)
  - **Validation Macro-F1:** **0.3993** (0.399282)
  - **Validation Accuracy:** **43.64%** (0.436441, 103 / 236 correct)
  - **Balanced Accuracy:** **42.18%** (0.421816)
  - **Macro Precision:** **40.98%** (0.409793)
  - **Macro Recall:** **42.18%** (0.421816)
  - **Macro ROC-AUC (OVR):** **0.6893** (0.689263)
  - **Multiclass Brier Score:** **0.6856**
  - **Validation Loss:** **1.2754**
  - **Per-class validation metrics:**
    - **CN** (support=90): Precision 0.6042, Recall 0.6444, F1 0.6237, Specificity 0.7397, ROC-AUC 0.7711
    - **EMCI** (support=66): Precision 0.3704, Recall 0.1515, F1 0.2151, Specificity 0.9000, ROC-AUC 0.6310
    - **LMCI** (support=46): Precision 0.2500, Recall 0.3913, F1 0.3051, Specificity 0.7158, ROC-AUC 0.5610
    - **AD** (support=34): Precision 0.4146, Recall 0.5000, F1 0.4533, Specificity 0.8812, ROC-AUC 0.7940
  - **Validation Confusion Matrix:**
    $$\begin{pmatrix} 58 & 10 & 18 & 4 \\ 25 & 10 & 25 & 6 \\ 10 & 4 & 18 & 14 \\ 3 & 3 & 11 & 17 \end{pmatrix}$$
  - **Checkpoint Digest (`best.pt`):** `36112356bf299becf4e846cc1153468cef4f3723c4cfd6640a935775c39a6df5`
  - **Random Seed:** 42
  - **Source Git Commit:** `adf1fb27c2a17487ea1bcc7989d0d358b71ab7fa`

---

## Exit gate

- [x] **Baseline run can be traced to exact code/config/data/pretrained weights.**
  - Code: Commit `adf1fb27c2a17487ea1bcc7989d0d358b71ab7fa`.
  - Config: SHA-256 `3ed998da5eb95a036867b3760c4ef6627d5fa9deb48284ab03cd486020b5a920`.
  - Data: Cohort of 1,593 subjects (manifest `3d519459...`, splits `ed830412...`, 0 missing files).
  - Pretrained weights: MedicalNet 23-dataset ResNet-18 (`61224f93...`).

- [x] **MedicalNet loading and dataset audit are verified.**
  - 100% of backbone parameter tensors (108/108) loaded cleanly with zero shape mismatches or unexpected tensors; classifier head properly reinitialized for 4 classes.
  - Dataset validator completed on all 1,593 volumes with zero errors.

- [x] **The run completed without unexplained numerical/runtime failures.**
  - Initial loss aligned with theory (~1.385), training converged smoothly, gradients remained finite with clipping norm 1.0, and early stopping terminated cleanly after 15 non-improving validation epochs at epoch 43.

- [x] **Validation predictions and checkpoint are preserved for Phase 4.**
  - Exact checkpoint `best.pt` (SHA-256 `36112356...`) and subject-level validation predictions `validation_predictions.csv` (SHA-256 `72cdd099...`) are preserved and immediately available for error diagnosis, sub-group stratification, and confusion analysis in Phase 4.
