# Phase 5 Checklist — Controlled Experiment Matrix

**Goal:** evaluate a small number of predeclared interventions, changing one
factor at a time and selecting only on development validation macro-F1.

**Current status:** **IN PROGRESS.** Prerequisite Phases 0, 1, 2, 3, and 4 are fully verified and completed. Experiment controls, configurations, and verification artifacts have been generated and pre-registered with SHA-256 digests. Augmentation pipeline visual QC has been conducted and archived under `docs/visual_qc/augmentation_qc/`. Baseline Run A is verified. Runs B through F have configurations established and are awaiting full training execution / artifact retrieval.

---

## 5.1 Pre-register experiment controls

- [x] **Use identical manifest, split, pretrained weights, epoch limit, checkpoint-selection metric, and evaluation procedure across comparisons.**
  - **Cohort Manifest:** `data/preprocessed_128_classifier_inputs/classifier_manifest.csv` (1,593 subjects, SHA-256: `3d51945937a98ea6400f1955106ef5ce1c805bf77555bc585b008402e01d7c0b`).
  - **Cohort Splits:** `data/preprocessed_128_classifier_inputs/classifier_splits.json` (1,112 train, 236 val, 245 test; SHA-256: `ed8304125a20800ee7e7aa800ff94a8b50e3031d75f89a2547197bb9d31adb1e`).
  - **Reference Template:** `template/mni152.nii.gz` (SHA-256: `dbbd4542823c1e2b12265c9b7e3cf959ee02ff7829bbf05cdcfbecc0013a00fa`).
  - **Pretrained Backbone:** MedicalNet 3D ResNet-18 (`resnet_18_23dataset.pth`, SHA-256: `61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4`).
  - **Training Schedule & Limits:** Identical maximum 100 epochs, AdamW optimizer with Cosine Annealing, early stopping patience 15 epochs, batch size 4, gradient clipping norm 1.0, mixed precision fp16.
  - **Checkpoint Selection:** Fixed strictly to peak validation macro-F1 on the 236-subject validation split.
  - **Evaluation Procedure:** Standardized batch size 4, 1,000 bootstrap resamples on validation set; test set remains locked and untouched.

- [x] **Create a separate config, unique `experiment.name`, fixed seed, and output directory for each experiment.**
  - Seed fixed to `42` across all runs. Output directory: `/content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs`.
  - Configs created and verified:
    1. **Run A (Baseline):** `configs/colab_medicalnet_resnet18_128.yaml` (`name: medicalnet_resnet18_128_colab`, SHA-256: `2eaa4d23...`).
    2. **Run B (Natural Sampling):** `configs/colab_exp_b_natural_sampling.yaml` (`name: medicalnet_resnet18_exp_b_natural_sampling`, SHA-256: `0baba743...`).
    3. **Run C (Weighted Loss):** `configs/colab_exp_c_weighted_loss.yaml` (`name: medicalnet_resnet18_exp_c_weighted_loss`, SHA-256: `0070afe7...`).
    4. **Run D (Mild Augmentation):** `configs/colab_exp_d_mild_augmentation.yaml` (`name: medicalnet_resnet18_exp_d_mild_augmentation`, SHA-256: `74e65173...`).
    5. **Run E (Lower Learning Rate 5e-5):** `configs/colab_exp_e_lower_lr_5e5.yaml` (`name: medicalnet_resnet18_exp_e_lower_lr_5e5`, SHA-256: `95126689...`).
    6. **Run F (Combined Candidate):** `configs/colab_exp_f_combined.yaml` (`name: medicalnet_resnet18_exp_f_combined_candidate`, SHA-256: `d06ea9af...`).

- [x] **Record config/data/code/checkpoint hashes and planned comparisons before looking at results.**
  - **Source Git Commit:** `78f9bdfa359507fcf0741e81a2d09fc5ab047998`.
  - **SHA-256 Registry:**
    - Manifest: `3d51945937a98ea6400f1955106ef5ce1c805bf77555bc585b008402e01d7c0b`
    - Splits: `ed8304125a20800ee7e7aa800ff94a8b50e3031d75f89a2547197bb9d31adb1e`
    - Reference MNI152: `dbbd4542823c1e2b12265c9b7e3cf959ee02ff7829bbf05cdcfbecc0013a00fa`
    - MedicalNet Backbone Checkpoint: `61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4`
    - Baseline Config A: `2eaa4d23ff4acfd1bffa35ad1aa97e3adbd97ff636d277ed689ce0bf2be0f390`
    - Exp B Config: `0baba743948fb2c723ee4f3aa2dbec8bdd427301db48dc0a580287ce5b62228e`
    - Exp C Config: `0070afe73dd3b971295658e58ea22b7eceef5f45b8f0ae76c36e131283afa240`
    - Exp D Config: `74e65173809fb1b87072e2a5700b0977d7ce0532b43b68e9d3ccdfea6048d73a`
    - Exp E Config: `95126689dd18ff6ff2280034136e430bcec9336836caa74280a19f0073d18b74`
    - Exp F Config: `d06ea9af71e3dd2ff6af86b250753470c007ee3a6d502926f7797ab8351c9be7`

- [x] **Define primary selection metric (currently validation macro-F1) and secondary diagnostics: accuracy, balanced accuracy, class recall/F1, and validation loss.**
  - **Primary Selection Metric:** Validation Macro-F1 (Baseline benchmark = **0.3993**).
  - **Predeclared Class-Wise Guardrails (from Phase 4 Exit Gate):**
    - EMCI Recall $\ge \mathbf{0.2500}$ (arresting the 15.15% baseline collapse).
    - Minimum Recall for all 4 classes $\ge \mathbf{0.2500}$.
    - Balanced Accuracy $\ge \mathbf{40.00\%}$.
  - **Secondary Diagnostics:** Validation overall accuracy, multiclass cross-entropy loss, multiclass Brier score, ROC-AUC (OvR), and per-class precision/recall/F1.

- [x] **Keep validation and test loaders at natural class frequencies.**
  - Implemented in `src/engine.py` (`make_loader`): `sampler` is strictly `None` and `shuffle=False` for all validation and evaluation splits.
  - Class frequencies in validation loader are preserved: CN = 38.14% (90), EMCI = 27.97% (66), LMCI = 19.49% (46), AD = 14.41% (34).

- [x] **Limit the number of trials; repeated searches can overfit the same validation set.**
  - Fixed pre-registered matrix limited to exactly 5 planned intervention experiments (B, C, D, E, and combined candidate F). No open-ended random search or iterative hyperparameter sweeping on the validation set.

---

## 5.2 Run one-factor comparisons

- [x] **A — Reproduced baseline:** balanced sampler on, weighted loss off, augmentation off.
  - Verified run: `medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z`.
  - Output artifacts archived at `outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/`.
  - Performance: Validation Macro-F1 = **0.3993**, Validation Accuracy = **43.64%**, Balanced Accuracy = **42.18%**, Validation Loss = **1.2754**, Best Epoch = 28.
  - Class Recalls: CN = 64.44%, EMCI = 15.15%, LMCI = 39.13%, AD = 50.00%.

- [ ] **B — Natural sampling:** balanced sampler off, weighted loss off.
  - *Status:* Configured (`configs/colab_exp_b_natural_sampling.yaml`). Awaiting training execution / artifact retrieval.

- [ ] **C — Weighted loss:** balanced sampler off, weighted loss on; weights derived only from training labels.
  - *Status:* Configured (`configs/colab_exp_c_weighted_loss.yaml`). Training class weights: CN: 0.654, EMCI: 0.897, LMCI: 1.293, AD: 1.716. Awaiting training execution / artifact retrieval.

- [ ] **D — Mild augmentation:** baseline sampler/loss; enable only conservative, anatomically plausible transforms supported by the project.
  - *Status:* Configured (`configs/colab_exp_d_mild_augmentation.yaml`). Awaiting training execution / artifact retrieval.

- [x] **Inspect augmented volumes before full training; avoid left/right flips, aggressive warping, anatomy-removing crops, or destructive intensity shifts.**
  - Transform inspected using real preprocessed scan `002_S_2010_preprocessed_128.nii.gz` via `src/transforms.py`.
  - Transform configuration: TorchIO Compose:
    - `RandomAffine(degrees=5, translation=2mm, scales=0.03, p=0.5)`
    - `RandomBlur(std=(0.0, 1.5), p=0.5)`
    - `RandomNoise(mean=0, std=(0, 0.01), p=0.25)`
  - Verification results:
    - Voxel shape strictly preserved: `(128, 128, 128)`.
    - Chirality strictly preserved ($\det(A) > 0$; zero horizontal flipping).
    - Mean absolute voxel difference: `0.1982`.
    - Brain tissue boundaries and ventricular structures remain intact without truncation.
  - Visual QC comparison montage generated across axial, coronal, and sagittal planes: archived at [docs/visual_qc/augmentation_qc/002_S_2010_augmentation_inspection.png](visual_qc/augmentation_qc/002_S_2010_augmentation_inspection.png).

- [ ] **E — Lower learning rate:** compare a small predeclared set around current `1e-4`, one rate per run.
  - *Status:* Configured (`configs/colab_exp_e_lower_lr_5e5.yaml` at $\eta = 5.0 \times 10^{-5}$). Awaiting training execution / artifact retrieval.

- [ ] **F — Combined candidate:** combine only changes that independently helped validation; verify interaction rather than assuming additive benefit.
  - *Status:* Configured (`configs/colab_exp_f_combined.yaml`). Awaiting training execution / artifact retrieval.

- [x] **Do not combine balanced sampling and weighted loss in primary comparisons.**
  - Enforced by configuration design:
    - Experiment B has sampler off, weighted loss off.
    - Experiment C has sampler off, weighted loss on.
    - Neither experiment stacks balanced sampling with weighted loss.

---

## 5.3 Validate training implementation for any new schedule

- [x] **Do not assume YAML enables staged fine-tuning; current loop updates all model parameters.**
  - Audited `src/train.py` (lines 148–153): optimizer parameter group takes `model.parameters()` in full.
  - Confirmed: all 32,987,588 model parameters are updated simultaneously across all layers. Staged unfreezing is not claimed via YAML.

- [x] **If implementing head warm-up or staged unfreezing, test which tensors are trainable at each stage and optimizer parameter groups.**
  - Confirmed: Phase 5 comparisons use full fine-tuning with conservative learning rates ($\eta = 1.0 \times 10^{-4}$ and $\eta = 5.0 \times 10^{-5}$) rather than staged unfreezing. Parameter updates confirmed across all ResNet-18 residual blocks and final linear classifier.

- [x] **Test resume behavior across stages and ensure schedule is stored in the checkpoint/config.**
  - Verified by unit test `tests/test_smoke_training.py::test_end_to_end_and_resume`.
  - Bit-exact state matching confirmed: `torch.testing.assert_close(full['model_state_dict'][k], checkpoint['model_state_dict'][k], rtol=0, atol=0)`.
  - Checkpoint preserves epoch, optimizer state, Cosine Annealing scheduler state, AMP GradScaler state, RNG streams, resolved config, and data audit.

- [x] **Run focused tests and type/build checks for any code changes before comparing results.**
  - Executed full test suite with project Python 3.13 runtime (`E:\Desktop\git_repos\DG-TCAV\.venv\Scripts\python.exe -m pytest tests/`).
  - Result: **43 / 43 tests passed (100%)** in 29.39s with zero failures or deprecation warnings.

---

## Exit gate

- [ ] Every experiment has complete artifacts and comparable controls.
- [ ] Results are reported for every planned run, including failures.
- [ ] Candidate gains are supported by validation metrics and no unacceptable class-wise regression under predeclared guardrails.
- [ ] Test split has not been used to select or tune any candidate.
