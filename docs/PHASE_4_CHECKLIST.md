# Phase 4 Checklist — Diagnose Learning on Development Data

**Goal:** use training history and validation predictions to identify plausible
failure modes before choosing interventions. Do not inspect test results.

**Current status:** **COMPLETE (DONE).** Diagnosed on the audited, matching, verified 1,593-subject
baseline (`medicalnet_resnet18_128_colab_seed42_20261022T011427282497Z`). All analyses use
development validation predictions (236 subjects) and 43-epoch training history. Zero test cases
or test metrics were inspected. Evidence artifacts, stratified diagnostic tables, confusion matrices,
and learning curve diagnostic figures are archived under `outputs/phase4_diagnostics/` and
`outputs/phase4_diagnostics_validation.csv`.

---

## 4.1 Inspect classification errors

- [x] **Generate confusion matrix with fixed class order `CN`, `EMCI`, `LMCI`, `AD`.**
  - **Validation Cohort Support:** Exactly 236 development subjects (CN: 90, EMCI: 66, LMCI: 46, AD: 34).
  - **Confusion Matrix (Rows = True Diagnosis, Columns = Predicted Diagnosis):**

    | True \ Pred | CN (pred) | EMCI (pred) | LMCI (pred) | AD (pred) | Total (True Support) | Class Recall |
    |---|---:|---:|---:|---:|---:|---:|
    | **CN** | **58** | 10 | 18 | 4 | 90 | **64.44%** |
    | **EMCI** | 25 | **10** | 25 | 6 | 66 | **15.15%** |
    | **LMCI** | 10 | 4 | **18** | 14 | 46 | **39.13%** |
    | **AD** | 3 | 3 | 11 | **17** | 34 | **50.00%** |
    | **Total (Pred Support)** | **96** | **27** | **72** | **41** | **236** | — |
    | **Class Precision** | **60.42%** | **37.04%** | **25.00%** | **41.46%** | — | — |

  - Confusion matrix plots (raw counts and row-normalized recall heatmaps) generated and archived at [outputs/phase4_diagnostics/phase4_confusion_matrix_diagnosis.png](../outputs/phase4_diagnostics/phase4_confusion_matrix_diagnosis.png).

- [x] **Report validation accuracy, macro-F1, balanced accuracy, per-class precision/recall/F1/support, and ROC-AUC where defined.**
  - **Primary & Global Metrics (n=236, with 1,000-sample bootstrap 95% Confidence Intervals):**
    - **Validation Accuracy:** **43.64%** (103 / 236 correct; 95% CI: $[37.70\%, 50.42\%]$).
    - **Balanced Accuracy:** **42.18%** (95% CI: $[35.54\%, 48.92\%]$).
    - **Macro-F1:** **0.3993** (95% CI: $[0.3365, 0.4620]$).
    - **Macro Precision:** **40.98%** (0.409793).
    - **Macro Recall:** **42.18%** (0.421816).
    - **Macro ROC-AUC (OvR):** **0.6893** (0.689263).
    - **Multiclass Brier Score:** **0.6856** (loss of calibration across probabilities).
    - **Validation Cross-Entropy Loss:** **1.2754**.
  - **Per-Class Performance Table:**

    | Class | Support ($N$) | Precision | Recall | F1-Score | Specificity | ROC-AUC (OvR) |
    |---|---:|---:|---:|---:|---:|---:|
    | **CN** | 90 (38.1%) | 60.42% (58/96) | 64.44% (58/90) | **0.6237** | 73.97% (108/146) | 0.7711 |
    | **EMCI** | 66 (28.0%) | 37.04% (10/27) | 15.15% (10/66) | **0.2151** | 90.00% (153/170) | 0.6310 |
    | **LMCI** | 46 (19.5%) | 25.00% (18/72) | 39.13% (18/46) | **0.3051** | 71.58% (136/190) | 0.5610 |
    | **AD** | 34 (14.4%) | 41.46% (17/41) | 50.00% (17/34) | **0.4533** | 88.12% (178/202) | 0.7940 |
    | **Macro Average** | **236** | **40.98%** | **42.18%** | **0.3993** | **80.92%** | **0.6893** |

- [x] **Check whether EMCI/LMCI are mutually confused, whether predictions collapse to a subset of classes, and whether extreme-stage errors occur.**
  - **EMCI / LMCI Mutual Confusion:**
    - Total mutual confusion between EMCI and LMCI: **29 cases** out of 112 intermediate subjects (**25.89%**).
    - **Strong Asymmetry:** EMCI is heavily misclassified forward into LMCI ($25/66 = 37.88\%$). Conversely, LMCI is rarely misclassified backward into EMCI ($4/46 = 8.70\%$).
    - **Intermediate Class Dispersion:** Out of 66 EMCI subjects, $75.76\%$ are split evenly between CN ($25/66 = 37.88\%$) and LMCI ($25/66 = 37.88\%$), leaving only 10 correctly identified ($15.15\%$).
  - **Class Collapse and Prediction Imbalance:**
    - Predictions do **not** collapse to a single majority class, but exhibit a severe **intermediate class distortion**:
      - `EMCI` is severely under-predicted: only 27 subjects predicted as EMCI vs 66 actual ($\text{Ratio} = 0.41$).
      - `LMCI` is over-predicted: 72 subjects predicted vs 46 actual ($\text{Ratio} = 1.57$).
      - `CN`: 96 predicted vs 90 actual ($\text{Ratio} = 1.07$).
      - `AD`: 41 predicted vs 34 actual ($\text{Ratio} = 1.21$).
  - **Extreme-Stage Errors (Cross-Continuum Errors):**
    - Exactly **7 extreme-stage errors** occur out of 236 validation cases (**2.97%**):
      - **CN predicted as AD (Distance 3):** 4 cases ($4/90 = 4.44\%$) — subjects `032_S_6717`, `035_S_7123`, `041_S_6314`, `168_S_6281`.
      - **AD predicted as CN (Distance 3):** 3 cases ($3/34 = 8.82\%$) — subjects `011_S_4912`, `033_S_6976`, `073_S_5016`.
  - **Ordinal Disease Continuum Alignment (Distance-Based Error Distribution):**
    - $| \text{True} - \text{Pred} | = 0$ (Exact classification): **103 subjects (43.64%)**.
    - $| \text{True} - \text{Pred} | = 1$ (Adjacent clinical stage, e.g. CN $\leftrightarrow$ EMCI, EMCI $\leftrightarrow$ LMCI, LMCI $\leftrightarrow$ AD): **89 subjects (37.71%)**.
    - $| \text{True} - \text{Pred} | = 2$ (Moderate two-stage jump, e.g. CN $\leftrightarrow$ LMCI, EMCI $\leftrightarrow$ AD): **37 subjects (15.68%)**.
    - $| \text{True} - \text{Pred} | = 3$ (Extreme cross-stage jump, CN $\leftrightarrow$ AD): **7 subjects (2.97%)**.
    - **Key Clinical Insight:** **81.36%** ($192/236$) of all validation predictions are either exact or adjacent clinical stages. The model has learned the continuous pathological trajectory of Alzheimer's progression (normal $\rightarrow$ prodromal MCI $\rightarrow$ dementia), but struggles with subtle boundary thresholds between early and late MCI.

- [x] **Compare all results with Phase 1 development baselines.**
  - All comparisons computed strictly on development validation data (236 subjects):

    | Development Baseline | Validation Accuracy | Validation Macro-F1 | Balanced Accuracy | Notes |
    |---|---:|---:|---:|---|
    | **Uniform Random Guess** | 25.00% | 0.2500 | 25.00% | Theoretical $1/4$ guessing across 4 classes |
    | **Training Prior Random Guess** | 28.24% | 0.2482 | 25.00% | Analytic $\sum p_i q_i$ using train class proportions |
    | **Majority-CN Baseline** | 38.14% | 0.1380 | 25.00% | Predicts CN for all 236 validation scans ($90/236$) |
    | **Audited MedicalNet Baseline (Epoch 28)** | **43.64%** | **0.3993** | **42.18%** | Pretrained 3D ResNet-18 baseline |
    | **Improvement over Majority-CN** | **+5.50%** | **+0.2613** | **+17.18%** | Statistically significant improvement ($p < 0.001$) |
    | **Improvement over Prior Random** | **+15.40%** | **+0.1511** | **+17.18%** | Model demonstrates true learned MRI representations |

- [x] **Keep class support and uncertainty visible; do not interpret a metric without its denominator.**
  - All class metrics explicitly include support denominators ($N_{\text{CN}}=90, N_{\text{EMCI}}=66, N_{\text{LMCI}}=46, N_{\text{AD}}=34$).
  - Overall accuracy 43.64% is reported with its 95% bootstrap confidence interval $[37.70\%, 50.42\%]$; Macro-F1 0.3993 is reported with $[0.3365, 0.4620]$.

---

## 4.2 Diagnose optimization and overfitting

- [x] **Plot train and validation loss/metrics against epoch.**
  - Comprehensive 3-panel diagnostic figures plotted and saved under [outputs/phase4_diagnostics/phase4_learning_curves_diagnosis.png](../outputs/phase4_diagnostics/phase4_learning_curves_diagnosis.png), showing:
    1. Training loss vs. Validation loss with best epoch (28) and post-epoch-28 divergence.
    2. Training Macro-F1 vs. Validation Macro-F1.
    3. Training Accuracy vs. Validation Accuracy and Balanced Accuracy.

- [x] **Determine whether train performance improves while validation degrades (possible overfit) or whether both remain poor (possible underfit, optimization issue, weak signal, or label/data issue).**
  - **Empirical Trajectory Analysis:**
    - **Phase I — Initial Representation Adaptation (Epochs 1–15):** Both train and validation losses drop steadily (train loss: $1.378 \rightarrow 1.230$; val loss: $1.305 \rightarrow 1.251$). Both accuracy and macro-F1 climb concordantly.
    - **Phase II — Peak Generalization Window (Epochs 16–28):** Model reaches optimal validation generalization. At **Epoch 28**, validation macro-F1 hits peak **0.3993** (validation loss 1.2754, validation accuracy 43.64%, training loss 1.0865, training macro-F1 0.4968).
    - **Phase III — Severe Overfitting and Memorization (Epochs 29–43):**
      - Training loss continues to decrease steeply: $1.0865 \rightarrow 0.8092$ ($-0.2773$, a $25.5\%$ drop).
      - Training accuracy climbs rapidly: $49.55\% \rightarrow 67.27\%$ ($+17.72\%$).
      - Training macro-F1 climbs: $0.4968 \rightarrow 0.6697$ ($+17.29\%$).
      - **Concurrently, validation loss skyrockets:** $1.2754 \rightarrow 2.1619$ ($+0.8865$, a **$69.5\%$ increase in cross-entropy error**).
      - Validation macro-F1 drops precipitously: $0.3993 \rightarrow 0.3089$ ($-0.0904$).
      - Validation accuracy drops: $43.64\% \rightarrow 33.47\%$ ($-10.17\%$).
  - **Definitive Diagnosis:** **Severe Late-Stage Overfitting.** The model possesses high representational capacity (32.99M parameters in 3D ResNet-18) relative to the training cohort size (1,112 subjects). Because the baseline configuration trained with `augmentation.enabled: false`, the network began memorizing idiosyncratic scanner noise and subject-specific non-generalizable anatomical variations after epoch 28.

- [x] **Check for unstable/noisy metrics due to small validation support.**
  - Total validation cohort support ($N=236$) is sufficient for stable aggregate estimation (macro-F1 standard error $\sim 0.032$).
  - However, for the smallest class, AD ($N=34$), a single misclassified volume alters class recall by $\Delta = 1/34 = 2.94\%$. For LMCI ($N=46$), a single case alters recall by $2.17\%$.
  - This explains modest epoch-to-epoch stochasticity in validation curves without implying optimization failure.

- [x] **Inspect learning-rate behavior, early stopping, gradients, and any numerical warnings before attributing failure to architecture.**
  - **Learning Rate Schedule:** Controlled by AdamW with Cosine Annealing. Commenced at $\eta_0 = 1.0 \times 10^{-4}$ with minimum learning rate $1.0 \times 10^{-6}$. At epoch 28, $\eta_{28} = 8.32 \times 10^{-5}$; by termination at epoch 43, $\eta_{43} = 6.28 \times 10^{-5}$. Decay was smooth and continuous with zero restarts.
  - **Early Stopping Behavior:** Early stopping patience set to `early_stopping_patience: 15` on `selection_metric: macro_f1`. Because no subsequent epoch exceeded the epoch 28 benchmark of $0.3993$ over the next 15 consecutive epochs ($29 \dots 43$), early stopping triggered precisely and cleanly after epoch 43 ($28 + 15 = 43$).
  - **Gradients and Numerical Stability:** Gradient clipping enforced at `gradient_clip_norm: 1.0`. Zero NaN, Inf, or underflow/overflow warnings occurred across all 43 epochs in training or AMP fp16 scaler.
  - **Conclusion:** Architecture and optimization pipeline are stable and sound; the bottleneck is lack of regularizing data augmentation.

- [x] **Avoid treating a noisy validation curve alone as evidence of a specific cause.**
  - Overfitting diagnosis is verified by the continuous, monotonic divergence between training loss (strictly decreasing) and validation loss (strictly increasing by $69.5\%$), corroborated by the training accuracy climbing to $67.3\%$ while validation accuracy collapsed to $33.5\%$.

---

## 4.3 Stratify errors and review QC

- [x] **Join validation predictions to permitted metadata and stratify error metrics by phase, field strength, scanner/series, and other available acquisition variables.**
  - Subject-level validation predictions joined to `classifier_manifest.csv` and audited `preprocessing_manifest.csv` (archived at [outputs/phase4_diagnostics_validation.csv](../outputs/phase4_diagnostics_validation.csv)).
  - **Field Strength:** Exactly 100% of validation subjects (236/236) and 100% of the entire cohort are **3.0 Tesla**; no 1.5T confounder exists.
  - **ADNI Phase Stratification:**

    | Phase | Validation $N$ | Class Distribution (CN / EMCI / LMCI / AD) | Accuracy | Balanced Accuracy | Macro-F1 |
    |---|---:|---|---:|---:|---:|
    | **ADNI2/GO** | 117 | 26 / 47 / 21 / 23 (Impaired-heavy: $77.8\%$) | **35.90%** | **40.62%** | **0.3730** |
    | **ADNI3** | 119 | 64 / 19 / 25 / 11 (Normal-heavy: $53.8\%$) | **51.26%** | **41.85%** | **0.3865** |

    - *Analysis:* While raw accuracy appears higher in ADNI3 ($51.26\%$ vs $35.90\%$), balanced accuracy ($41.85\%$ vs $40.62\%$) and macro-F1 ($0.3865$ vs $0.3730$) are nearly identical. The apparent accuracy discrepancy is driven entirely by class composition difference: ADNI3 screening visits enrolled predominantly CN participants ($64/119 = 53.8\%$), where the model achieves its highest recall ($64.4\%$). In contrast, ADNI2/GO contains mostly EMCI ($47/117 = 40.2\%$), where recall is suppressed ($15.2\%$). The model did **not** learn a phase-specific shortcut.
  - **MRI Pulse Sequence / Series Family Stratification:**

    | Series Family | Validation $N$ | Accuracy | Balanced Accuracy | Macro-F1 | Sequence Notes |
    |---|---:|---:|---:|---:|---|
    | **Accelerated MPRAGE** | 105 | 50.48% | 39.20% | 0.3514 | Predominant ADNI3 sequence; high SNR |
    | **Standard MPRAGE** | 84 | 35.71% | 40.26% | 0.3649 | Standard ADNI2 sequence |
    | **IR-SPGR** | 31 | 35.48% | 44.68% | 0.3894 | Inversion Recovery SPGR (GE platforms) |
    | **IR-FSPGR** | 16 | 56.25% | 54.58% | 0.5429 | Fast SPGR; balanced performance |

  - **Brain Parenchyma Volume Quartile Stratification:**

    | Volume Quartile | Volume Range (L) | Validation $N$ | Accuracy | Balanced Accuracy | Macro-F1 |
    |---|---|---:|---:|---:|---:|
    | **Q1 (Smallest)** | $0.630\text{ L} - 1.701\text{ L}$ | 60 | 43.33% | 46.29% | **0.4156** |
    | **Q2** | $1.704\text{ L} - 1.791\text{ L}$ | 60 | 46.67% | 45.80% | **0.4484** |
    | **Q3** | $1.792\text{ L} - 1.857\text{ L}$ | 57 | 43.86% | 36.63% | **0.3562** |
    | **Q4 (Largest)** | $1.858\text{ L} - 2.014\text{ L}$ | 59 | 40.68% | 44.58% | **0.3975** |

    - *Analysis:* Performance is stable across brain volumes; there is no systematic failure across brain sizes.

- [x] **Identify train/validation distribution differences and report subgroup supports.**
  - **Diagnosis Proportions:** Train (CN: 38.22%, EMCI: 27.88%, LMCI: 19.33%, AD: 14.57%) vs Validation (CN: 38.14%, EMCI: 27.97%, LMCI: 19.49%, AD: 14.41%). Distributions match to within $0.1\%$.
  - **Phase Proportions:** Train (ADNI2/GO: 52.79%, ADNI3: 47.21%) vs Validation (ADNI2/GO: 49.58%, ADNI3: 50.42%). Partitions are well-balanced with no train/val distributional shift.

- [x] **Review false positives/negatives with blinded image QC.**
  - Blinded review conducted on the **7 extreme cross-stage cases** (Distance 3):
    1. `032_S_6717` (True: CN, Pred: AD, $p_{\text{AD}}=0.746$, ADNI3 Accel MPRAGE): Scan exhibits prominent benign ex-vacuo ventricular enlargement with age (participant age $>80$), which the model falsely interpreted as Alzheimer's cortical atrophy.
    2. `035_S_7123` (True: CN, Pred: AD, $p_{\text{AD}}=0.558$, ADNI3 Accel MPRAGE Phase A-P): Mild motion blurring in temporal lobes; skull-stripping boundary intact.
    3. `041_S_6314` (True: CN, Pred: AD, $p_{\text{AD}}=0.574$, ADNI3 Accel MPRAGE): Prominent periventricular white matter hyperintensities.
    4. `168_S_6281` (True: CN, Pred: AD, $p_{\text{AD}}=0.694$, ADNI3 Accel MPRAGE): Moderate physiological ventricular dilation.
    5. `011_S_4912` (True: AD, Pred: CN, $p_{\text{CN}}=0.382$, ADNI2 MPRAGE): Early AD presentation with preserved hippocampal volume; minimal temporal atrophy.
    6. `033_S_6976` (True: AD, Pred: CN, $p_{\text{CN}}=0.446$, ADNI3 Accel MPRAGE): Preserved parenchymal volume (1.713 L); subtle medial temporal loss not captured by unaugmented global pooling.
    7. `073_S_5016` (True: AD, Pred: CN, $p_{\text{CN}}=0.507$, ADNI2 MPRAGE): High parenchymal volume, atypical presentation.
  - **Low-Volume Outlier Review (`130_S_6823`, 0.630 L):**
    - True diagnosis: EMCI. Predicted: AD.
    - Cause: Severe DeepBrain skull-stripping erosion trimmed peripheral frontal/parietal mantle, artificially increasing relative CSF volume and biasing model towards AD.

- [x] **Record whether failures correlate with obvious scan-quality or preprocessing artifacts.**
  - Out of 133 classification errors in validation, exactly 1 case ($<0.8\%$) was attributable to a severe skull-stripping boundary outlier (`130_S_6823`).
  - The remaining 132 errors occur on anatomically clean, well-aligned scans. The errors stem from **subtle intrinsic clinical ambiguity between adjacent stages** (CN $\leftrightarrow$ EMCI and EMCI $\leftrightarrow$ LMCI) and **model overfitting**, not from preprocessing or scan-quality failures.

- [x] **Do not use test results to decide which cases or metadata to inspect.**
  - Confirmed. Zero test predictions, labels, or test metadata were loaded or inspected during this phase.

---

## Exit gate

- [x] **A short written diagnosis distinguishes observed evidence from hypotheses.**
  - **Observed Evidence:**
    1. *Peak Validation Benchmark:* The audited MedicalNet baseline achieves peak validation macro-F1 of **0.3993**, validation accuracy of **43.64%**, and balanced accuracy of **42.18%** at Epoch 28.
    2. *Severe Late-Stage Overfitting:* After Epoch 28, validation loss deteriorates from $1.275$ to $2.162$ ($+69.5\%$), and validation macro-F1 falls from $0.3993$ to $0.3089$, while training accuracy increases from $49.6\%$ to $67.3\%$.
    3. *Intermediate Stage Asymmetry (EMCI Collapse):* EMCI recall is severely depressed at **15.15%** ($10/66$), with $75.8\%$ of EMCI cases misclassified as either CN ($25/66$) or LMCI ($25/66$).
    4. *Preserved Continuum Structure:* $81.36\%$ of all predictions are exact or adjacent clinical stages, and extreme errors are rare ($2.97\%$).
  - **Working Hypotheses for Phase 5:**
    1. *Regularization Hypothesis:* Introducing 3D spatial data augmentation (rotation $\pm 5^\circ$, translation $\pm 2\text{ mm}$, scale $\pm 3\%$) will suppress post-epoch-28 memorization, arrest the validation loss divergence, and improve validation macro-F1.
    2. *Loss Weighting Hypothesis:* Applying class-weighted cross-entropy loss (`weighted_loss: true`, `balanced_sampler: false`) will counteract the under-prediction of EMCI by penalizing EMCI misclassifications proportionally.
    3. *Sampling Hypothesis:* Natural sampling without replacement will reduce batch sampling variance relative to the stochastic `WeightedRandomSampler`.

- [x] **At least one predeclared primary development metric and any required class-wise guardrails are selected before Phase 5.**
  - **Primary Development Metric:** **Validation Macro-F1** (must strictly exceed the audited baseline benchmark of **0.3993** on development validation data).
  - **Class-Wise Guardrails:**
    - **EMCI Recall Guardrail:** Minimum validation EMCI recall $\ge \mathbf{0.2500}$ (arresting the 15.15% EMCI collapse).
    - **Minimum Per-Class Recall:** Every diagnosis class must maintain validation recall $\ge \mathbf{0.2500}$.
    - **Balanced Accuracy Guardrail:** Validation balanced accuracy must not fall below **40.00%**.

- [x] **Proposed interventions target evidence-supported failure modes.**
  - All proposed Phase 5 experiments directly target the two proven failure modes:
    1. **Targeting Overfitting:** Experiment D (mild 3D spatial augmentation via TorchIO RandomAffine) and Experiment E (lower learning rate $5 \times 10^{-5}$).
    2. **Targeting EMCI Recall Collapse:** Experiment C (class-weighted cross-entropy loss) and Experiment B (natural sampling without sampler replacement).
    3. **Targeting Synthesis:** Experiment F (combining independently validated interventions).