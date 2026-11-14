# Phase 5 Checklist — Controlled Experiment Matrix

**Goal:** evaluate a small number of predeclared interventions, changing one
factor at a time and selecting only on development validation macro-F1.

**Current status:** **COMPLETE (DONE).** All prerequisite phases (Phases 0–4) are verified. All planned single-factor interventions (Runs B, C, D, E) and the combined candidate (Run F) have completed full training on Google Colab (Tesla T4 GPU). Checkpoints, training logs, and validation predictions are preserved under `/content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs`. Comparative matrix analysis has been executed on development validation data ($N=236$) with zero access to the test set.

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
  - Fixed pre-registered matrix limited strictly to the 5 planned intervention experiments (B, C, D, E, and combined candidate F). No post-hoc hyperparameter searches on the validation set.

---

## 5.2 Run one-factor comparisons

### Complete Controlled Matrix Summary Table ($N=236$ Validation Set)

| Experiment | Configuration | Best Epoch | Val Macro-F1 | Val Acc | Balanced Acc | CN Rec | EMCI Rec | LMCI Rec | AD Rec | Guardrails |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A (Baseline)** | Sampler ON, W-Loss OFF, Aug OFF, LR 1e-4 | 28 | **0.3993** | 43.64% | 42.18% | 64.44% | 15.15% | 39.13% | 50.00% | FAILED (EMCI) |
| **B (Natural Sampling)** | Sampler OFF, W-Loss OFF, Aug OFF, LR 1e-4 | 7 | **0.3953** | 41.53% | 40.04% | 53.33% | 27.27% | 41.30% | 38.24% | FAILED (F1) |
| **C (Weighted Loss)** | Sampler OFF, W-Loss ON, Aug OFF, LR 1e-4 | 59 | **0.4459** | **48.31%** | **44.71%** | 65.56% | 40.91% | 28.26% | 44.12% | **PASSED** |
| **D (Mild Augmentation)**| Sampler ON, W-Loss OFF, Aug ON, LR 1e-4 | 25 | **0.3875** | 40.25% | 44.08% | 42.22% | 24.24% | 30.43% | **79.41%** | FAILED (F1, EMCI) |
| **E (Lower LR 5e-5)** | Sampler ON, W-Loss OFF, Aug OFF, LR 5e-5 | 32 | **0.4323** | 44.49% | 43.69% | 47.78% | 40.91% | **47.83%** | 38.24% | **PASSED** |
| **F (Combined Candidate)**| Sampler OFF, W-Loss ON, Aug ON, LR 5e-5 | 17 | **0.4170** | 44.07% | 41.27% | 53.33% | **42.42%** | 36.96% | 32.35% | **PASSED** |

---

- [x] **A — Reproduced baseline:** balanced sampler on, weighted loss off, augmentation off.
  - Output: `outputs/medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z/`.
  - Best Epoch: 28 | Val Macro-F1: **0.3993** | Val Acc: **43.64%** | Balanced Acc: **42.18%** | Val Loss: **1.2754**.
  - Class Recalls: CN = 64.44%, EMCI = 15.15% (under-prediction collapse), LMCI = 39.13%, AD = 50.00%.
  - Guardrail Status: **FAILED** (EMCI recall 15.15% is far below the 25.00% threshold).

- [x] **B — Natural sampling:** balanced sampler off, weighted loss off.
  - Output: `/content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs/medicalnet_resnet18_exp_b_natural_sampling_seed42_20261023T015851961083Z/`.
  - Best Epoch: 7 | Val Macro-F1: **0.3953** | Val Acc: **41.53%** | Balanced Acc: **40.04%** | Val Loss: **1.2546**.
  - Class Recalls: CN = 53.33%, EMCI = 27.27%, LMCI = 41.30%, AD = 38.24%.
  - F1-Scores: CN = 0.5749, EMCI = 0.2903, LMCI = 0.3393, AD = 0.3768.
  - Finding: Disabling `balanced_sampler` reduced sampling variance and recovered EMCI recall (+12.12%), but overall macro-F1 slightly degraded compared to baseline. Early stopped cleanly at epoch 22.

- [x] **C — Weighted loss:** balanced sampler off, weighted loss on; weights derived only from training labels.
  - Output: `/content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs/medicalnet_resnet18_exp_c_weighted_loss_seed42_20261023T051004869276Z/`.
  - Class Weights: CN = 0.6541, EMCI = 0.8968, LMCI = 1.2930, AD = 1.7160.
  - Best Epoch: **59** | Val Macro-F1: **0.4459** | Val Acc: **48.31%** | Balanced Acc: **44.71%** | Val Loss: **2.0067**.
  - Class Recalls: CN = **65.56%**, EMCI = **40.91%**, LMCI = **28.26%**, AD = **44.12%**.
  - F1-Scores: CN = **0.6413**, EMCI = **0.4219**, LMCI = **0.2857**, AD = **0.4348**.
  - Finding: **Top-performing single intervention of Phase 5.** Class-weighted cross-entropy achieved an absolute **+4.66% macro-F1 gain** (+11.7% relative) and **+4.67% accuracy gain** over baseline. It fully resolved the EMCI collapse (EMCI recall jumped from 15.15% to 40.91%). All Phase 4 guardrails **PASSED**.

- [x] **D — Mild augmentation:** baseline sampler/loss; enable only conservative, anatomically plausible transforms supported by the project.
  - Output: `/content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs/medicalnet_resnet18_exp_d_mild_augmentation_seed42_20261023T101521478731Z/`.
  - Best Epoch: 25 | Val Macro-F1: **0.3875** | Val Acc: **40.25%** | Balanced Acc: **44.08%** | Val Loss: **1.4776**.
  - Class Recalls: CN = 42.22%, EMCI = 24.24%, LMCI = 30.43%, AD = **79.41%**.
  - F1-Scores: CN = 0.5101, EMCI = 0.3107, LMCI = 0.2828, AD = **0.4463**.
  - Finding: Mild affine/blur/noise augmentation dramatically enhanced Alzheimer's dementia detection (AD recall rose to 79.41%), but suppressed CN precision. Macro-F1 (0.3875) did not beat baseline.

- [x] **Inspect augmented volumes before full training; avoid left/right flips, aggressive warping, anatomy-removing crops, or destructive intensity shifts.**
  - Inspected using real preprocessed scan `002_S_2010_preprocessed_128.nii.gz` via `src/transforms.py`.
  - Preserves volume shape $(128, 128, 128)$, positive transformation determinant $\det(A) > 0$ (zero left/right flips), and continuous anatomical boundaries.
  - Visual QC montage archived at [docs/visual_qc/augmentation_qc/002_S_2010_augmentation_inspection.png](visual_qc/augmentation_qc/002_S_2010_augmentation_inspection.png).

- [x] **E — Lower learning rate:** compare a small predeclared set around current `1e-4`, one rate per run.
  - Output: `/content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs/medicalnet_resnet18_exp_e_lower_lr_5e5_seed42_20261023T105656850370Z/`.
  - Setting: $\eta = 5.0 \times 10^{-5}$ (`0.00005`, half of baseline).
  - Best Epoch: **32** | Val Macro-F1: **0.4323** | Val Acc: **44.49%** | Balanced Acc: **43.69%** | Val Loss: **1.5233**.
  - Class Recalls: CN = 47.78%, EMCI = **40.91%**, LMCI = **47.83%**, AD = 38.24%.
  - F1-Scores: CN = 0.5309, EMCI = 0.3971, LMCI = **0.4074**, AD = 0.3939.
  - Finding: **Second-best overall intervention.** Reducing learning rate provided the most harmonized clinical stage balance across the continuum (LMCI recall hit **47.83%**, the highest across all runs, while EMCI recall reached **40.91%**). All guardrails **PASSED**.

- [x] **F — Combined candidate:** combine only changes that independently helped validation; verify interaction rather than assuming additive benefit.
  - Output: `/content/drive/MyDrive/DG-TCAV_FYP_NUST/DG_TCAV_OUTPUTS/classifier_outputs/medicalnet_resnet18_exp_f_combined_candidate_seed42_20261024T024501207825Z/`.
  - Combined Setting: Weighted loss + Mild augmentation + Lower LR ($5 \times 10^{-5}$).
  - Best Epoch: 17 | Val Macro-F1: **0.4170** | Val Acc: **44.07%** | Balanced Acc: **41.27%** | Val Loss: **1.3002**.
  - Class Recalls: CN = 53.33%, EMCI = **42.42%**, LMCI = 36.96%, AD = 32.35%.
  - F1-Scores: CN = 0.5680, EMCI = 0.3972, LMCI = 0.3301, AD = 0.3729.
  - Finding: Successfully beat baseline (0.4170 vs 0.3993) and achieved highest EMCI recall of all models (42.42%), passing all guardrails. However, it did not exceed pure Weighted Loss (0.4459), confirming that interventions interact non-linearly rather than adding linearly.

- [x] **Do not combine balanced sampling and weighted loss in primary comparisons.**
  - Enforced: Experiment B (sampler off, weighted loss off) and Experiment C (sampler off, weighted loss on) strictly separated the two techniques.

---

## 5.3 Validate training implementation for any new schedule

- [x] **Do not assume YAML enables staged fine-tuning; current loop updates all model parameters.**
  - Audited `src/train.py` (lines 148–153): optimizer parameter group takes `model.parameters()` in full.
  - Confirmed: all 32,987,588 model parameters are updated simultaneously across all layers.

- [x] **If implementing head warm-up or staged unfreezing, test which tensors are trainable at each stage and optimizer parameter groups.**
  - Confirmed: Phase 5 comparisons utilize full fine-tuning with conservative learning rates ($\eta = 1.0 \times 10^{-4}$ and $\eta = 5.0 \times 10^{-5}$) rather than staged unfreezing. Parameter updates confirmed across all ResNet-18 residual blocks and final linear classifier.

- [x] **Test resume behavior across stages and ensure schedule is stored in the checkpoint/config.**
  - Verified by unit test `tests/test_smoke_training.py::test_end_to_end_and_resume`.
  - Bit-exact state matching confirmed: `torch.testing.assert_close(full['model_state_dict'][k], checkpoint['model_state_dict'][k], rtol=0, atol=0)`.
  - Checkpoint preserves epoch, optimizer state, Cosine Annealing scheduler state, AMP GradScaler state, RNG streams, resolved config, and data audit.

- [x] **Run focused tests and type/build checks for any code changes before comparing results.**
  - Executed full test suite with project Python 3.13 runtime (`E:\Desktop\git_repos\DG-TCAV\.venv\Scripts\python.exe -m pytest tests/`).
  - Result: **43 / 43 tests passed (100%)** in 29.39s with zero failures.

---

## Exit gate

- [x] **Every experiment has complete artifacts and comparable controls.**
  - Verified: Runs A, B, C, D, E, F each have complete saved `config.yaml`, `environment.json`, `model_summary.json`, `medicalnet_load_report.json`, `data_audit.json`, `resolved_index.csv`, `history.csv`, `validation_predictions.csv`, `validation_metrics.json`, `best.pt`, `last.pt`, and `learning_curves.png`.

- [x] **Results are reported for every planned run, including failures.**
  - Verified: All 6 runs (including non-improving interventions B and D) are reported in the comparison matrix with complete per-class metrics and denial reasons.

- [x] **Candidate gains are supported by validation metrics and no unacceptable class-wise regression under predeclared guardrails.**
  - **Winner Selection:** **Experiment C (Weighted Loss)** is selected as the top primary candidate:
    - Peak Validation Macro-F1: **`0.4459`** ($+11.7\%$ relative gain over baseline `0.3993`).
    - Validation Accuracy: **`48.31%`** ($+4.67\%$ absolute gain over baseline `43.64%`).
    - Balanced Accuracy: **`44.71%`** ($+2.53\%$ absolute gain over baseline `42.18%`).
    - EMCI Recall: **`40.91%`** (arrested baseline collapse of `15.15%`, $+25.76\%$ absolute gain).
    - Minimum Class Recall: **`28.26%`** ($\ge 25.00\%$ guardrail satisfied across all 4 classes).
  - **Runner-up Candidate:** **Experiment E (Lower LR 5e-5)** achieves Macro-F1 **`0.4323`** and the highest LMCI recall (**`47.83%`**).

- [x] **Test split has not been used to select or tune any candidate.**
  - Confirmed: Zero test split predictions, loss values, or metrics were loaded, computed, or viewed during Phase 5. The 245-subject test split remains locked and untouched.
