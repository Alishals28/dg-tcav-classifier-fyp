# DG-TCAV research and design blueprint

23 November 2026 · NUST SEECS FYP · 3D T1-weighted MRI

The diffusion system can be developed against the verified **128 × 128 × 128 NIfTI contract** now. Cohort B uses the same participants and fixed subject partitions as Cohort A. Its raw longitudinal visit archive and its curated manifest are already present, but the complete visit-keyed, QC-passed Cohort B tensor index is still a release gate. The first model conditions on **CN, merged MCI, and AD**. The five reviewed classifier branches still use four-logit heads, so the existing checkpoint interface remains a separate dependency for TCAV. [Upstream preprocessing][P-pipe] · [Cohort B manifest][B-manifest] · [MedicalNet classifier][M-model] · [CBAM metrics][C-metrics]

## 0. Ground-truth pipeline and interface audit

### 0.1 Repository and Drive discovery map

The GitHub source is `Alishams9008/dg-tcav-classifier-fyp`. The existing local checkout points to `Alishals28/dg-tcav-classifier-fyp` and contains older remote refs and a local meeting-prep note. The five tip commits below were matched to the supplied GitHub remote. The path inventory and the scope of content reads are in Appendix A. The preprocessing source is a separate repository identified by the MedicalNet branch's provenance checklist. [GitHub branch list][Branches] · [MedicalNet preprocessing checklist][M-prep]

| Branch | Pinned commit | What it contributes |
| --- | --- | --- |
| `main` | [`adf1fb2`][G-main] | Baseline fixed-split classifier, MedicalNet-style ResNet-18 and plain CNN, manifest/QC scripts, Kaggle and Colab configs, and live model methods behind the activation exporter. The tracked split is the earlier 768-volume subset. |
| `MedicalNet` | [`fef8b8c`][G-med] | Full-width pretrained ResNet-18 experiments, the 1,593-volume manifest, cross-validation and phase-stress manifests, phase checklists, sample preprocessing QC, and evaluation documentation. The validation matrix and phase-stress manifests are distinct from the master split. |
| `Inception-ResNet` | [`b3c54fd`][G-inc] | MedicalNet-initialized stem/stages 1–3 with an Inception-ResNet stage 4, standalone Inception model, ROI-attention pooling option, companion-mask adapter, and L2 sweep. |
| `Resnet-CBAM` | [`7723d4a`][G-cbam] | Yang-style 18/34/50 CBAM backbones, post-CBAM activations, three-class probability aggregation, calibrated ensembles, and BrainIAC fusion. The fusion entry point has its own feature contract. |
| `exp/autoresearch-20261019` | [`f2a0b5b`][G-exp] | Classifier-only Kaggle campaign with 18 experiment directories, ordinal soft-label ablations, augmentation, reduced-input variants, and a validation evidence log. Its 96³ ablation is a classifier resize setting with the 128³ source contract. [Experiment 006 config][E-006] · [Campaign final summary][E-log] |

| Drive directory | Relevant evidence and implications |
| --- | --- |
| [`DG-TCAV_FYP_NUST`][D-root] | `README.md` documents cohort composition and an older classifier input specification. Use it for cohort provenance and the executed code for tensor contracts. |
| [`01_Documentation_and_Presentations`][D-docs] | FYP proposal, complete master document with TeX equations, research document, implementation guide, defense notes, and slides. The detailed proposal and TeX document supply the six positive concepts. |
| [`02_Cohort_A_Classifier_Baseline`][D-a] | Baseline manifest, master subject split JSON, ID exports, QA script, pilot ZIP, and subject-organized baseline NIfTI storage. |
| [`03_Cohort_B_Diffusion_Longitudinal`][D-b] | Visit-level manifest, builder, audit, download chunks, ID exports, expanded ADNI DICOM storage and follow-up ZIP. |
| [`04_Raw_ADNI_Clinical_Metadata`][D-clin] | Key MRI, Structural MRI, ARM, NEUROBAT, assessment diagnosis, ADSP PHC, and provider dictionaries. The longitudinal builder's visit-diagnosis source is DXSUM. |
| [`05_Scripts_and_Pipelines`][D-scripts] | DICOM conversion, cohort/stats audits, deck builders, and pilot preprocessing reports/figures. The production preprocessing source is pinned separately below. |
| [`06_MRI_Scans_Storage`][D-storage] | Pilot DICOM, converted NIfTIs and ZIP. The two folders named for full cohort storage were empty at this snapshot. The operational baseline and follow-up data are under the cohort folders. |
| [`07_Preprocessed_Tensors_MNI152`][D-tensors] and [`08_Model_Checkpoints_and_Logs`][D-checkpoints] | Both folders listed zero children. The operational outputs are under `DG_TCAV_OUTPUTS`. |
| [`DG_TCAV_OUTPUTS`][D-outputs] | Preprocessed scans and companion masks, processing logs and CSV, dataset audit, MedicalNet initialization weights, classifier run configs, checkpoints, metrics and test/validation artifacts. Run-specific configs and hashes matter more than the folder naming. |

### 0.2 Actual preprocessing and classifier contract

Authoritative producer: `adeelHassan123/DG-TCAV-Preprocessing@ec726c6145db53f1a6a71836bff4d63d9eacb2bb`. `run_preprocessing.py` declares `masked_rigid_affine_centroid_seed42_v4`. `preprocess_mri_scan` uses RPS reorientation, N4, DeepBrain skull stripping, masked rigid plus affine registration, an outside-template fraction gate, then whole-grid z-score and 128³ resampling. The mask guides registration and is saved as a companion. The source uses DeepBrain in this commit. The proposal describes HD-BET and a different target resolution as planned methodology. [Upstream runner][P-run] · [`preprocess_mri_scan`][P-pipe] · [`registration`, `resize_image` and `normalize_zscore`][P-ops] · [Proposal §5.3][D-proposal]

The Drive README's classifier section specifies `91×109×91` at 2 mm and min–max normalization. The proposal's diffusion section specifies `96×112×96`. The v4 pipeline output and the five model branches' real-data base settings support `128×128×128` with anisotropic spacing and producer z-score. Use `128³` on every classifier-facing file unless the pinned consumer config explicitly specifies its own input transform. [Drive README, Classifier Training Setup](https://docs.google.com/document/d/19w0EmvZAbvVQZsaif4MAtpWInJVuu8zQ/edit) · [Proposal §5.6][D-proposal] · [Producer target shape][P-config] · [Classifier base config][M-base]

| Property | Verified producer and reader contract |
| --- | --- |
| File and tensor layout | One scalar 3-D `{subject_id}_preprocessed_128.nii.gz` plus `{subject_id}_preprocessed_128_mask.nii.gz` for the baseline CLI. Target `(128,128,128)` in NIfTI array order. Classifier `MRIDataset.__getitem__` adds one channel and batches to `[B,1,128,128,128]`. [Runner `_process_one`][P-run] · [`PipelineConfig`][P-config] · [Classifier `MRIDataset`][M-data] |
| Spacing and geometry | All 1,593 final-success rows record `(1.1926161339506507, 1.474926233291626, 1.238707578741014)` mm. Direct NIfTI header zooms are approximately `(1.1926161,1.4749262,1.2387075)` mm in RAS order. The classifier expects that triplet with `0.03 mm` absolute tolerance and `same_fov` reference mode. The saved qform and sform origins are zero in the three inspected pairs. [Processing manifest][D-prep-report] · [Base data config][M-base] · [Header samples][D-example1] · [Phase 2 checklist §2.1–2.2][M-prep] |
| Dtype | Three inspected NIfTI scan headers store `float64` and their masks store binary `uint8`. `preprocess_mri_scan` writes the normalized/resized SimpleITK image without a scan dtype cast and explicitly casts masks to `sitkUInt8`. The classifier loader casts on read with `get_fdata(dtype=np.float32)`. Use float32 tensors in the diffusion adapter. [Producer `preprocess_mri_scan`][P-pipe] · [`load_volume`][M-data] · [Sample 1][D-example1] · [Sample 2][D-example2] · [Sample 3][D-example3] |
| Intensity | `normalize_zscore` calls `sitk.Normalize` on the complete aligned grid before linear resize. There is no clipping, fixed minimum/maximum, or post-resize standardization in this producer. The inspected float32 scan min/max pairs were approximately `(-0.41245,5.17593)`, `(-0.38013,4.51239)`, `(-0.74203,3.22585)`. Background is generally a negative value after z-scoring, and mask membership cannot be inferred from `x != 0`. The classifier's baseline `normalization: none` preserves these values. [Producer `normalize_zscore` and `resize_image`][P-ops] · [Base config][M-base] · [Sample 1][D-example1] · [Sample 2][D-example2] · [Sample 3][D-example3] |
| Raw-MNI transform | The repo's `template/mni152.nii.gz` has SHA-256 `dbbd4542823c1e2b12265c9b7e3cf959ee02ff7829bbf05cdcfbecc0013a00fa`, shape `207×256×215`, spacing about `0.7374631` mm isotropic, qform origin `(0,0,0)`, and sform origin `(-75.762535,-110.762535,-71.762535)` mm. `resize_image` preserves the SimpleITK origin/direction and multiplies old spacing by old/new dimensions. `same_fov` checks extent and orientation. For atlas mapping, use `A_mni = S_template @ inv(Q_template) @ A_disk` after verifying the pinned header/grid relationship. Keep the original runner's grid for classifier-facing exports and store `A_mni` in a sidecar. [Tracked template][M-template] · [`resize_image`][P-ops] · [Phase 2 checklist §2.1][M-prep] · [`load_volume`][M-data] |

The manifest has 1,894 attempts across 1,606 subjects, all tagged with v4. The final chronological row per subject gives 1,593 successes, 13 failures and 17 of the 1,623 baseline subjects with no row. Successful rows map to 1,112 train, 236 validation and 245 test. The current Drive listing has **1,616 distinct names with paired masks** under `DG_TCAV_OUTPUTS/preprocessed_128`. A listing count and a certified training count answer different questions. Resolve all eligible files by an exact scan-key match, assess missing or duplicate records, and use the successful/preflight-passing intersection during training. [`preprocessing_manifest.csv`][D-prep-report] · [`prepare_preprocessed_manifest.py`][M-prepare] · [Drive outputs][D-preprocessed]

All 1,593 final-success subjects have a visible paired file. Of the remaining pairs on Drive, 7 have a final failure and 16 have no processing-report row. Treat those 23 pairs as ineligible until reconciled. Separately, the expanded raw Cohort B tree contains 4,293 unique follow-up image IDs, all present in the 4,348-ID follow-up export. The remaining 55 require reconciliation against the archive or source before claiming complete ingestion. The Cohort A baseline NIfTI tree has 1,622 subject folders and one NIfTI per folder, a one-subject shortfall against the baseline manifest. These listing counts establish file presence rather than conversion or scan QC. [Processing report][D-prep-report] · [Preprocessed pairs][D-preprocessed] · [Baseline NIfTI tree][D-a-nii] · [Follow-up IDs][B-followups] · [Expanded Cohort B tree][D-b-raw]

The already-written converter `convert_dicom_to_nifti.py` and the pinned preprocessing function solve different steps. Keep `preprocess_mri_scan` unchanged. For Cohort B, join converted inputs to the curated manifest by canonical image ID. Invoke the function with an explicit unique output such as `volumes/<subject_id>/<image_id>_preprocessed_128.nii.gz` and use the same per-scan prefix for the mask. `discover_scans` derives IDs from folder/stem combinations and `_process_one` uses that ID for output and resume, so a baseline subject filename convention cannot serve as the longitudinal join key. Reuse an existing baseline scan only when image ID, pipeline version, template digest and QC match. For each raw image use one input series and preserve the image ID in the index even if the filename lacks it. [Drive converter][D-convert] · [Runner `discover_scans`, `_process_one` and resume][P-run] · [`build_cohort_b.generate_cohort_b`][B-build]

### 0.3 Fixed subject split and three-class conditioning

The canonical split is [`02_Cohort_A_Classifier_Baseline/cohort_a_splits.json`][A-split]. The keys are exactly `train`, `val` and `test`, each a list of subject ID strings. Joining by `subject_id` recomputed zero duplicates across partitions, zero unmapped manifest rows, unique Cohort B image IDs and unique `(subject_id, visit)` pairs. The exact Cohort A baseline `(subject_id,image_id)` pairs all occur in Cohort B. [Cohort A manifest][A-manifest] · [Cohort B manifest][B-manifest] · [Subject split][A-split]

| Split | Master subjects / A visits | B visits | B CN | B merged MCI | B AD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1,134 | 4,232 | 1,424 | 2,176 | 632 |
| Validation | 242 | 863 | 290 | 444 | 129 |
| Test | 247 | 876 | 296 | 427 | 153 |
| Total | 1,623 | 5,971 | 2,010 | 3,047 | 914 |

Counts above come from `visit_diagnosis` with `MCI + EMCI + LMCI` summed within the authoritative subject partition and before preprocessing QC. The builder's `assign_visit_dx` uses the exact baseline label for baseline scans. At follow-up it uses nearest DXSUM assessment by participant within 90 days, with baseline inheritance when unmatched or unmapped. Keep the source label and add a `diagnosis_source` field by reproducing the join. Treat the fallback as a weaker conditioning label and report a sensitivity analysis on assessed visits. Changes in diagnosis across visits do not alter subject membership. [`cohort_b_longitudinal_manifest.csv`][B-manifest] · [`generate_cohort_b/assign_visit_dx`][B-build] · [Assessment diagnosis folder][D-dx]

The tracked 1,593-subject split in MedicalNet, Inception-ResNet and Resnet-CBAM is filtered for classifier QC. Main and the experiment branch each carry the earlier `535/112/121` split. The wider master set of **247 test subject IDs** is the denylist for diffusion, even if a classifier branch evaluated only a subset. If a different frozen classifier holdout is ever adopted, validate that its test IDs remain excluded or refit the generator under that intended holdout before using it for inference claims. [Main filtered split][A-main-split] · [MedicalNet filtered split][A-med-split] · [Master split][A-split] · [`build_index`][M-data]

**Design and enforcement.** Validate master membership, map `CN→0`, `EMCI/LMCI/MCI→1`, `AD→2` by default, and keep the null CFG index equal to the configured class count. Select `train` and `val` by subject **before** accessing files or staging visits. Allow each loader only its own partition and require exact matching status/paths for `(subject_id,image_id)`, no alias path reuse, and no equal content hashes across partitions. Stage 1, Stage 2, normalization/calibration statistics used in training, and probe source subjects all use train IDs. Use validation subjects for model and fidelity gates. Hash the canonical master split and accepted scan index, record every exclusions reason, and assert denylist intersections across the accepted index, latent cache, reference measurements, probe plan and CAV fitting inputs. The master test data are reserved for final frozen classifier TCAV scoring. [Existing `build_index` checks][M-data] · [Proposal §5.2 and §5.8][D-proposal] · [User brief][Brief]

### 0.4 TCAV handoff

| Frozen classifier profile | Classifier path and bottleneck |
| --- | --- |
| Full-width MedicalNet-style ResNet-18 on Main, MedicalNet, or experiment base | `get_layer4_activations` gives `[B,512,16,16,16]` at 128³, from strides `2,2,1,2,1,1` through stem/maxpool/stages and dilated stages 3/4. `features_from_activations` yields `[B,512]` and `forward_head` yields four logits. Main's plain CNN has its own penultimate spatial representation. The experiment's 96³ resize changes its map dimensions. [Main `src/model.py`][Main-model] · [MedicalNet `ResNet18`][M-model] · [Experiment 006][E-006] |
| Inception and hybrid profiles | `HybridInceptionResNet3D` adds a stride-2 Inception stage 4 and produces `[B,512,8,8,8]` at width 64. `InceptionNet3D` uses one more stride-2 block, yielding `[B,8c,4,4,4]` where `c` is the configured base width. The hybrid's config uses companion masking and ROI-attention pooling. [`src/model.py`][I-model] · [Hybrid profile][I-config] |
| Yang CBAM ResNet18/34/50 | Default branch configs use width 32 and a `4³` spatial bottleneck. `get_layer4_activations` explicitly applies CBAM after layer 4. Pooled dims are 256 for 18/34 and 1,024 for 50 with expansion 4. This profile uses a companion brain mask followed by nonzero-foreground z-score in the loader. [`YangResNet3D_CBAM`][C-model] · [`load_volume`][C-data] · [CBAM-18 config][C-18] · [CBAM-50 config][C-50] |

The exporter `export_features` writes detached `pooled_features.npy`, optional per-example `layer4_*.npy`, `subjects.csv` and checkpoint provenance. It resolves the existing labeled classifier index, so a dedicated probe input adapter should call the exact selected checkpoint's `load_volume` and `get_layer4_activations → features_from_activations → forward_head` directly. Match the model's selected pooling across probe fitting and patient scoring. Use `eval()` and a fresh differentiable `h = pooled_features.detach().requires_grad_(True)` for the head gradient. Verify the split forward reproduces the original logits. Refuse a shape, feature-dimension, preprocessing or checkpoint digest mismatch. Preserve companion masks when the checkpoint's adapter uses them. For ventricle edits, validate that the mask handed to the classifier retains the intended intracranial CSF signal. [`export_features`][M-export] · [`ResNet18`][M-model] · [CBAM `load_volume`][C-data] · [Kim et al., §3][TCAV]

All five branches' `LABEL_TO_INDEX` uses CN, EMCI, LMCI, AD. The CBAM branch's `three_class_metrics` sums EMCI and LMCI **probabilities** for reporting. If using the existing four-logit head for a grouped three-class TCAV result, use the three scores `(ℓ_CN, logsumexp(ℓ_EMCI,ℓ_LMCI), ℓ_AD)`. Their group softmax gives the same probability aggregation, and the group score has a defined gradient. Declare that score before evaluation and record separate four-logit and native three-logit protocols. Native three-class TCAV requires a separately trained and validated checkpoint. [`LABEL_TO_INDEX`][M-label] · [`three_class_metrics`][C-metrics] · [Kim et al., §3][TCAV]

The Drive run logs confirm four-class profiles and separately report three-class aggregation for the Yang CBAM runs. For example, the ResNet-18 validation JSON records macro-F1 about 0.4640 on four classes and 0.4878 after grouping MCI. The ResNet-34 file records 0.4483 and 0.4949 respectively. These are validation metrics for the recorded runs. The Drive local–global hybrid record uses a separate model name and should be treated as checkpoint-specific until its exact runnable source and feature path are pinned. The full-resolution shape test reproduced the spatial and pooled shapes and four-logit heads in the table above using randomly initialized models. [ResNet-18 validation metrics][V-cbam18] · [ResNet-34 validation metrics][V-cbam34] · [Local–global run model summary][V-local]

### 0.5 Clinical concepts and cohort targets

The six positive concepts below use the exact names from Proposal §5.5 and the complete TeX document's formal concepts section. Apply the tighter PBM interval from the formal proposal as the initial target, and obtain clinical sign-off on the named measurements before counting the probes as biomarker-positive. The short master research document's candidate table omits Temporal Horn Dilation and marks several thresholds as pending. [Proposal §5.5, pp. 15–16][D-proposal] · [Complete master TeX, formal concept definitions][D-tex] · [Master research document, concept measurement table][D-master]

| Concept | Measurement and proposal threshold | Steering region |
| --- | --- | --- |
| Hippocampal Atrophy (HA) | Bilateral hippocampal volume divided by intracranial volume, at or below the normative 25th percentile. The complete master names the hippocampal formation including CA fields, dentate gyrus and subiculum. | Bilateral CerebrA hippocampus. |
| Entorhinal Cortex Thinning (ECT) | Mean bilateral entorhinal cortical thickness **≤2.80 mm**. | Bilateral entorhinal parcels. |
| Lateral Ventricular Enlargement (VE) | Combined lateral ventricular volume divided by intracranial volume, at or above the normative 75th percentile. | Bilateral lateral ventricle labels. Measure and report inferior-horn effects separately. |
| Temporal Horn Dilation (THD) | Expansion of the inferior lateral ventricular horns bordering the hippocampus. The proposal supplies the location and the complete master TeX defines the bilateral volume/ICV measure and **≥75th percentile** positive rule. | Bilateral inferior lateral ventricle labels. |
| Global Neocortical Thinning (GCT) | Mean thickness across associative frontal, parietal and temporal ribbons **≤2.25 mm** in the proposal. The complete master cites posterior cingulate, precuneus and inferior parietal regions but writes an all-cortex mean formula. Prespecify association-ROI mean as the primary candidate and report the all-cortex mean separately. | Bilateral association parcels enumerated in §3. |
| Amygdalar Atrophy (AA) | Bilateral amygdalar volume divided by intracranial volume, at or below the normative 25th percentile. | Bilateral amygdala. |
| Preserved Brain Morphology (PBM) | Neutral reference with preserved cortical thickness and normative morphology. Proposal targets 40th–60th percentiles for volumes. The complete master describes a 25th–75th percentile range for all six measurements. Use the narrower proposal interval for volumes and preregister the same interval for thickness as a design choice. | Use a balanced mix of the same steering supports with a neutral target. |

Volume thresholds use the declared dimensionless ratio and empirical Train-CN percentile. The complete TeX writes `1000×`-scaled ratios next to numerical thresholds in unscaled-looking units for HA, VE and AA. Retain the percentiles, report the numerator, denominator, units and normative sample, and do not silently apply those ambiguous numerical constants. THD and total ventricle measures can be inherently coupled, depending on segmenter definitions. Measure the main ventricular compartment excluding the temporal horns for cross-concept comparisons and separately report total volume. [Complete TeX, HA/VE/THD/AA subsections][D-tex] · [CerebrA Table 1][Atlas-paper]

The proposal specifies 200 probes per positive concept and 200 PBM controls. This brief sets the quota at **at least 200 accepted volumes per group**, 1,400 minimum across seven groups. Keep attempted and accepted counts separate, count unique output hashes, and report distinct source subjects. Produce mechanism-matched sham and random CAV banks separately from the primary quotas. [Proposal §5.7 and Algorithm 1][D-proposal] · [User brief][Brief]

## 1. Literature and theoretical justification

| Primary source | What DG-TCAV can take from it |
| --- | --- |
| Rombach et al., CVPR 2022, §3 | Separate perceptual compression from iterative denoising and introduce conditioning through cross-attention. Their image experiments do not establish diagnostic or regional morphometry performance on this dataset. [Paper][Rombach] |
| Pinaya et al., 2022, Methods | Whole-brain 3D T1w latent diffusion conditioned on covariates, including brain structure volumes. The authors trained on 31,740 UK Biobank images. That scale and population differ from the 4,232 candidate training visits in this ADNI split. [Paper][Pinaya] · [Cohort B manifest][B-manifest] |
| Khader et al., 2023, Methods and Results | Learned volumetric compression and denoising for medical MRI and CT, with expert assessment of realism, anatomy and interslice coherence. This motivates a blinded review alongside numerical gates. [Paper][Khader] |
| Counterfactual MRI generation for AD effect detection, 2024 | Diagnosis-conditioned LDM and DDPM support subject-linked hypothesis generation in Alzheimer's MRI. A diagnosis condition can shift several structures simultaneously, motivating independently measured anatomical targeting. [Primary preprint][AD-counterfactual] |
| Puglisi et al., 2025, BrLP | Combines longitudinal 3D latent diffusion, disease covariates and uncertainty assessment for forecasting. DG-TCAV uses longitudinal visits to learn anatomy. Forecasting would additionally require a time-conditioned longitudinal evaluation. [Paper][BrLP] |
| Sun et al., 2025–2026, 3D counterfactual benchmark | Evaluate anatomical interventions on ADNI using composition, reversibility, realism, effectiveness and minimality. This motivates target-dose success, leakage-free real comparisons and off-target preservation. [Paper][Sun] |
| Fallah and Healy, 2026, Methods/Results | ADNI diagnosis-conditioned DDIM with a preprocessing ablation and subject-clustered intervals. The model generates 2D coronal slices, so its results do not directly validate 3D shape or a transferable FID threshold. [Paper][Fallah] |
| AnaDiffusion, 2026 preprint | Explicit regional decomposition is a recent alternative if soft steering in a shared generator cannot separate THD from VE or preserve cortical anatomy. Treat this as an unreplicated option for this project. [Preprint][Ana] |
| Kim et al., ICML 2018, §3 | Learn a linear separator in the chosen network representation and use a directional derivative on a class score to quantify sensitivity. Probe realism, classifier influence and disease causality are separate questions. [Paper][TCAV] |

For the verified 128³ grid, factor-eight compression per spatial axis reduces spatial positions from `128³ = 2,097,152` to `16³ = 4,096`, a **512-fold** reduction. With four latent channels, each latent has 16,384 values, **128-fold** fewer values than the one-channel input. A same-width first convolution scales approximately as `DHW × k³ × C_in × C_out`. Spatial attention has quadratic token cost, so the design reserves attention for `8³` and `4³`. The savings apply to the iterative denoiser's grid. Full-resolution autoencoder activations still drive Stage 1 memory, and 50 guided steps imply 100 U-Net passes per candidate with sequential CFG. These are counts derived from the input contract and proposed model. [Input geometry][D-prep-report] · [Rombach §3][Rombach] · [MONAI U-Net source][MONAI-unet]

Choose latent diffusion as the starting architecture for regional conditioning and computational tractability. A 3D GAN can sample in one pass and should remain a timing and diversity comparator if resources permit. A pixel-space 3D DDPM avoids compression but has a much larger denoising grid. Compare against those alternatives on the same subject split and preprocessing, reporting peak memory, GPU time, target morphology, off-target measures and feature-space diversity. Prefer the compact model only if the Stage 1 reconstruction preserves the smallest target anatomy. Some useful baselines may fit, and no measured performance ranking exists for the three architectures on these scans. [Rombach §3][Rombach] · [Pinaya Methods][Pinaya] · [Khader Methods][Khader] · [Sun evaluation protocol][Sun]

## 2. End-to-end 3D latent diffusion architecture

The following network, hyperparameters and optimization policy are proposed design values. They are reproducible configurations for testing, with citations to the implemented MONAI interfaces and the verified input geometry. Use `monai==1.5.2`, `einops==0.8.1` and a pinned compatible PyTorch build. The locally verified code paths and runtime limits are described in Appendix B. [`AutoencoderKL`][MONAI-ae] · [`DiffusionModelUNet`][MONAI-unet] · [`DDPMScheduler`][MONAI-ddpm] · [Input geometry][M-base]

### 2.1 Stage 1 · reconstruction and latent regularization

Feed finite float32 `[B,1,128,128,128]` in the stored z-score intensity domain. Use the exact sidecar brain mask to weight fidelity, while retaining all input values and background. The decoder has a linear output. Keep the disk-space affine and qform/sform with each input. Apply no sigmoid, tanh, secondary per-volume min–max, or accidental masking in the generator. [Preprocessing `preprocess_mri_scan`][P-pipe] · [Base data config][M-base] · [MONAI AutoencoderKL][MONAI-ae]

| Step | Width / residual blocks | Spatial grid |
| --- | --- | --- |
| Input and 3³ encoder stem | 1 → 16 | 128³ |
| Encoder level 0 then stride-2 downsample | 16 / 1 | 128³ → 64³ |
| Encoder level 1 then stride-2 downsample | 32 / 1 | 64³ → 32³ |
| Encoder level 2 then stride-2 downsample | 64 / 2 | 32³ → 16³ |
| Encoder level 3 and output projection | 96 / 2 → 4 | 16³ |
| Independent 1³ posterior projections | `μ` and `σ`, each width 4 | 16³ |
| Decoder input projection and deep blocks | 4 → 96 / 2 | 16³ |
| Decoder upsample plus residual levels | 64 / 2 → 32 / 1 → 16 / 1 | 32³ → 64³ → 128³ |
| GroupNorm/activation/output convolution | 16 → 1 | 128³ |

Use eight GroupNorm groups, the release's default residual operations, no level attention or optional nonlocal AE block, and `use_checkpoint=True` during training. Let `z = μ + σε` with `ε∼N(0,I)`. Train `L_AE = weighted_L1 + 0.1 × edge_L1 + 1e-5 × KL_mean`. Weighted L1 uses `w=1+2M_brain` and divides by `sum(w)`. Edge L1 is the average L1 mismatch of first voxel differences on the three axes. `KL_mean=0.5 mean(μ² + σ² − log(σ²) − 1)` over all batch, channel and grid entries. Compute reductions in float32. Compare reconstructions of `decode(μ)` against input before Stage 2; the design uses posterior means for deterministic caches and source edits, while Stage 1 trains with samples. [MONAI `Encoder`, `Decoder`, `AutoencoderKL.encode/sampling/decode`][MONAI-ae] · [Rombach §3.1][Rombach]

After freezing Stage 1, compute a single float64-accumulated mean `m` and standard deviation `s` of posterior means across accepted train latents. Save them with the AE checkpoint and partition hash. Cache `z_scaled=(μ−m)/max(s,1e−6)` losslessly, with an explicit `1/max(s,1e−6)` scale. Fit once and apply the same transform in validation and inference. Decode `z_scaled/scale+m`. Reject a nonfinite or degenerate cached distribution. This generator-space scaling leaves the NIfTI voxel normalization contract intact. [MONAI latent diffusion tutorial, latent scale][MONAI-tutorial] · [Stage 1 architecture above](#21-stage-1--reconstruction-and-latent-regularization)

### 2.2 Stage 2 · conditional latent denoising

Use the same MONAI `DiffusionModelUNet` for training and sampling. Input and output are `[B,4,16,16,16]`. The 3³ stem projects to 64. Down blocks use two residual blocks each at widths `64,96,128` with grids `16³,8³,4³`. The middle at `4³` has residual → transformer → residual. Up blocks concatenate the matching skip tensors, use three residual blocks per level as implemented by MONAI, and return `4³→8³→16³` with widths `128,96,64`. Use one transformer block per attention-enabled level, three 32-wide heads at 96 channels, four at 128 channels, GroupNorm groups 8, and the standard final norm/activation/3³ projection to four predicted noise channels. Disable flash attention by default for 16 GB portability and upcast attention logits. [MONAI `DiffusionModelUNet.__init__` and `forward`][MONAI-unet]

Combine the timestep embedding with the MONAI diagnosis embedding. Use `num_class_embeds=K+1` with `K=3` by default. Morphology cross-attention gets a single 64-wide token obtained by a bias-free linear map from six continuous measurements plus six availability flags. Standardize each measurement using Train-CN medians and robust scales `max(IQR/1.349,1e−6)`, clip input covariates to ±5, and zero-fill unavailable values with the corresponding flag false. Training can start class-only with all-zero morphology inputs. Enable targeted conditioning only after an automated, separately validated measurement table exists. Mask the diagnosis and measurement context together in 10% of samples for learned unconditional CFG, and mask morphology on 15% of the remaining samples for explicit class-only control. [MONAI `forward(class_labels,context)`][MONAI-unet] · [Ho and Salimans][CFG] · [Sun Methods][Sun]

Train with the MONAI scaled-linear beta series, `1,000` train timesteps, `β_start=0.0015` and `β_end=0.0195`. Draw `t` uniformly. Set `z_t=√ᾱ_t z_scaled+√(1−ᾱ_t)ε` and regress `ε` using mean squared error. Use `ε_cfg=ε_uncond+w(ε_cond−ε_uncond)`, starting at `w=2.0` and validating `w∈{1,2,3.5}`. Use linear output with latent clipping disabled. [MONAI tutorial][MONAI-tutorial] · [MONAI scheduler][MONAI-ddpm] · [CFG][CFG]

**Full handoff trace** · `NIfTI (128,128,128), saved z-score` → `float32 [B,1,128³]` → `AE posterior μ,σ [B,4,16³]` → `scaled z [B,4,16³]` → `U-Net [B,4,16³] with CN/MCI/AD, optional subject anatomy, CFG` → `inverse latent scaling` → `AE decoder [B,1,128³]` → `voxel-space exterior lock` → `paired QC and anatomy gates` → `classifier-grid .nii.gz and applicable mask` → `checkpoint-specific layer 4 / pooled feature` → `CAV fitting`. [Producer pipeline][P-pipe] · [MONAI AE][MONAI-ae] · [MONAI U-Net][MONAI-unet] · [Classifier exporter and model][M-export]

## 3. Automated concept and neutral probe generation

### 3.1 [DESIGN SOLUTION / MISSING ARTIFACT] Atlas support and morphometry

The companion mask is a whole-brain mask. The audited code and project artifacts establish a reference template and brain masks. A released six-region atlas-mask bundle, a visit-linked normative morphometry table with the full concept schema, and a checkpointed DG-TCAV generator were not located in the audited engineering paths. Appendix B includes a standalone module to create the six 128³ supports using the published CerebrA labels. The atlas matches the documented symmetric 2009c template family and specifically includes entorhinal and inferior lateral ventricular labels. Use nearest-neighbor resampling after the qform/sform conversion in §0.2, version the atlas digest, verify bilateral presence and landmark overlays, and review THD especially carefully. The atlas paper reports low agreement with its original label for the inferior lateral ventricle and discusses registration bias. [Producer whole-brain mask][P-pipe] · [Phase 2 §2.1][M-prep] · [CerebrA Table 1 and Usage Notes][Atlas-paper] · [Official CerebrA atlas download][Atlas-data]

Use the following six label unions. For GCT, the parcel selection is the operational proposal for this project and covers posterior cingulate, precuneus, inferior parietal, rostral middle frontal, superior parietal and middle temporal parcels. VE steers the compartments labeled lateral ventricle, THD uses separate inferior lateral ventricle labels. In the method, large ROIs may need masks registered to subject anatomy to avoid boundaries moving outside the support. Retain each independent core ROI and use separate editable support and measurement ROIs. [CerebrA Table 1][Atlas-paper] · [Complete master formal concepts][D-tex]

| HA | ECT | VE | THD | GCT | AA |
| --- | --- | --- | --- | --- | --- |
| 48, 99 | 36, 87 | 41, 92 | 5, 56 | 47, 98, 31, 82, 10, 61, 1, 52, 9, 60, 28, 79 | 19, 70 |

**Subject morphometry.** Atlas labels establish where editing is allowed. Measures must vary with the actual generated tissue. Use an independent automated pipeline, such as standalone SynthSeg v2 for subcortical volumes and QC with temporal-horn labels, plus DL+DiReCT for subject cortical thickness. Validate both on real baseline/reconstruction pairs with the same affine-normalized, skull-stripped 128³ domain. Freeze that exact version and verify measurement repeatability before applying fixed thresholds. A volume is `label_voxel_count × |det(affine[:3,:3])|` on the measurement grid. Use an independently measured source ICV for same-subject edits or match a valid native-space ICV, with an explicit affine-scale reconciliation if numerator and denominator live on different grids. The skull-stripped mask volume does not supply ICV. For missing TIV, missing cortex/ventricular labels or invalid landmarks, record `measurement_unavailable` and withhold clinical probe acceptance. [SynthSeg v2 documentation and labels][SynthSeg] · [DL+DiReCT implementation][DiReCT] · [CerebrA Usage Notes][Atlas-paper]

For volume ratios, aggregate left and right labels and divide by a consistent subject ICV. For cortical measures, calculate the arithmetic mean of valid hemisphere thickness values for ECT and the preregistered parcel-area-weighted mean for GCT. Store native-versus-grid measurement provenance, labels, ICV source, registration scale and segmenter QC. At the approximately 1.2–1.5 mm output spacing, thin cortical regions span few voxels. Compare measurements on real scans with validated native-space measurements and quantify bias and repeatability by concept before claiming mm-valued thinning. If the 128³ measurement cannot support the project threshold, retain the candidate for an image-edit ablation and exclude it from the biomarker-positive bank. [Observed v4 spacing][D-prep-report] · [Proposal §5.5][D-proposal] · [DL+DiReCT implementation][DiReCT]

### 3.2 Candidate generation and control design

1. **Prepare an anchor bank.** Start from distinct eligible Train-CN subjects with validated PBM measurements. Use a matched baseline or screened visit; use no test identity or test-derived normalization. Build a stable subject order with seed 42, one selected visit per subject per generation pass, and balance ages and acquisition phases across positive and neutral banks where those verified covariates are available. Track selection and repeated subjects. The Train-CN data contain 429 baseline subjects before preprocessing, so 200 independent anchors is possible in principle and requires a qualified anchor count. [A manifest and master split][A-manifest] · [Master split][A-split] · [Proposal §5.5–5.7][D-proposal]
2. **Pick an edit dose.** Set the target anatomy token to the concept's Train-CN boundary or a dose further in the specified direction (start at the positive threshold, then try the 10th/90th percentile for atrophy/dilation where applicable). Retain the anchor's other measured anatomy values in the non-target token slots. Start with a CN diagnosis context to separate the anatomical target from diagnosis. Validate a disease-class context ablation on development subjects only. For the class-only initial pilot, mask and test local candidates but label them exploratory until the measured concept criteria pass. [Proposal §5.7][D-proposal] · [Sun Methods][Sun]
3. **Build per-anchor edit support.** Convert the approved atlas core to the exact scan grid and compare with independent subject segmentation. Derive a 2.5 mm pad in physical spacing around the core using `distance_transform_edt(~core,sampling=spacing)`, keep it inside the reviewed intracranial domain including allowable CSF, and version its bounds. The soft weights equal 1 inside the core and taper over the pad by `clip(1−distance/2.5,0,1)`. At `16³`, max-pool the binary support using kernel and stride 8. The resulting mask governs latent denoising only. Image-space composition governs the hard exterior. [Atlas coordinate solution above](#31-design-solution--missing-artifact-atlas-support-and-morphometry) · [Blended Latent Diffusion][Blended]
4. **Run 50 deterministic reverse steps.** Use one seed per attempt, same noise for the anchor's exterior at every step, and `t_max=589` as a proposed starting point. Round `linspace(t_max,0,50)` and require distinct timesteps. Warm-start by forward-noising the anchor posterior mean at the chosen first timestep. At each descending level, predict CFG noise inside the edit process, compute deterministic DDIM `ẑ0=(z_t−√(1−ᾱ_t)ε_cfg)/√ᾱ_t` and `z_next=√ᾱ_next ẑ0+√(1−ᾱ_next)ε_cfg`, and replace the exterior latent with the corresponding analytically noised original. Set `ᾱ_next=1` after step 0. This is noise-initialized editing. Deterministic inverse-DDIM for trajectory analysis is a separate experiment specified in Proposal §5.9, with reconstruction and inversion error measured before interpretation. [DDIM §4][DDIM] · [Blended Latent Diffusion][Blended] · [Proposal §5.9][D-proposal]
5. **Decode and composite.** Invert latent scaling, decode, and compute `x_out = where(support, x_anchor + w_voxel × (decoder−x_anchor), x_anchor)` in float32. Thus exterior voxels are exactly the anchor in model input space. Score pre-composite and post-composite candidates and test for boundary artifacts in the inner taper band. Use an updated, validated companion brain mask when a region edit affects the mask boundary, scoped to the support. Apply the corresponding classifier adapter after serialization and verify the intended feature survives its normalization. [DDIM][DDIM] · [Blended][Blended] · [CBAM loader][C-data]
6. **Build neutrals and matched shams.** PBM undergoes the same encoder/decoder/noise schedule, diagnosis pathway, support-size distribution and voxel composition while targeting retained anchor morphometry. It must pass PBM thresholds and differ from a verbatim training scan. In addition to a minimum 200 pooled PBM, generate concept-specific sham banks whenever the mask, schedule or anchor distributions differ. Keep positives and neutrals from each shared anchor on the same side of every CAV fitting/validation split. A separate random-CAV null permutes bank assignments within anchor/acquisition groups. The older proposal's random noise controls can be reported as an ablation; use matched MRI-domain controls for the primary test. [Proposal §5.7–5.8][D-proposal] · [Kim et al., §3][TCAV]
7. **Accept and publish atomically within the project.** Generate in fixed batches, initially up to 10 attempts per target output. Only accepted unique NIfTIs count. Store candidate rejections without saving them to the probe directories. Candidate IDs derive from run, concept, anchor, variant and seed. Write `probes/<concept>/<probe_id>.nii.gz`, the applicable companion mask, metadata, and accepted manifest transactionally. Record concept/dose, source identity via project-private mapping, source split, diagnoses, pipeline/checkpoint/template/atlas/split/scan hashes, code and environment versions, metrics and acceptance reason. Resume from the journal without counting duplicate hashes. Stop at 200 accepted positive volumes per concept and 200 PBM, or publish the measured shortfall when the attempt budget is exhausted. A successful class-only smoke test is a software milestone. A release of concept-positive probes requires the gates in §5. [User brief][Brief] · [Proposal §5.7][D-proposal]

## 4. Training protocol and GPU calibration

### 4.1 Proposed starting schedules

An epoch means one pass over eligible scans, with a patient-balanced sampler drawing one visit per subject per round and rotating visits between rounds. Weight sampling toward the observed diagnosis distribution initially. Use seeds `42,43,44` on the short comparison before investing in multiple long runs. Use actual accepted counts when calculating optimizer steps. The optimizer values are proposed baselines grounded in the MONAI tutorial design and mixed precision documentation. The audited classifier hardware runs do not prove diffusion fits. [MONAI training tutorial][MONAI-tutorial] · [NVIDIA mixed precision guide][NVIDIA-AMP] · [Phase 2 §2.1][M-prep]

| Setting | Stage 1 AutoencoderKL | Stage 2 latent U-Net plus anatomy projection |
| --- | --- | --- |
| Optimizer | AdamW, `lr=1e−4`, `β=(0.9,0.999)`, `eps=1e−8`, `weight_decay=1e−4` | AdamW, `lr=1e−4`, same β/ε/decay |
| Schedule | 500 optimizer updates linear warm-up from 10% of peak, cosine to `1e−6`, cap 100 epochs | 1,000 optimizer updates linear warm-up from 10%, cosine to `1e−6`, cap 200 epochs |
| Objective | `L_AE` from §2.1, no discriminator on initial run | Epsilon MSE from §2.2, EMA `0.999` after 500 completed optimizer updates, used for sampling |
| Gradient handling | Divide accumulated gradients by actual microsteps, unscale before clipping to norm 1.0, `zero_grad(set_to_none=True)` | Same, include U-Net and morphology module, save both EMA states |
| Validation / stopping | Every 2 epochs, patience 10 checks. Select for region-sensitive reconstruction and see §5, using val weighted L1 to break ties. | Every 5 epochs, patience 10 checks after first successful morphology pilot. Record fixed-timestep seeded epsilon loss and evaluate an unchanged visual/morphology validation panel. Freeze selection criteria in advance. |
| Resume provenance | Scheduler and AMP scaler, optimizer, per-epoch generator and data-order RNG, model/checkpoint/config/manifest/split hashes | Same plus frozen AE, latent calibration, cache keys, scheduler beta parameters, morphology encoder and EMA |

| Hardware profile | Stage 1 initial microbatch / accumulation | Stage 2 initial microbatch / accumulation | Precision and preflight |
| --- | --- | --- | --- |
| T4 16 GB | `1 × 4`, full 128³, AE checkpointing on | `2 × 4`, cached 16³ | FP16 with GradScaler, math attention, measure both update and validation peaks |
| P100 16 GB | `1 × 4`, AE checkpointing on | `1 × 8`, cached 16³ | FP16 with GradScaler only with a working sm60 build and supported kernels. FP32 64³ patch fallback if the CUDA op or full grid fails. |
| A100 40 GB | `2 × 2` after microbatch-1 preflight | `4 × 2` after microbatch-1 preflight | BF16 autocast when the runtime reports support; float32 objective reductions |

Treat each row as an admission target. Verify the actual accelerator and architecture, torch/CUDA build, allocator and free memory before fitting. Run one full training optimizer update, including backward, optimizer state creation and validation decoding, followed by 20 updates and a seeded sample on the intended profile. Record `torch.cuda.max_memory_allocated()`, `max_memory_reserved()`, host RAM, throughput and numerical overflow count. Admit a profile only if the maximum reserved memory remains below 85% of physically available device memory with finite losses and no operator fallback that invalidates throughput. Warm up the slow parts before timing. For FP16 nonfinite losses, first inspect reconstruction/norms and use FP32 KL and schedule math; reduce sample throughput if needed. [PyTorch CUDA semantics][Torch-CUDA] · [NVIDIA AMP][NVIDIA-AMP] · [A100 datatype support][NVIDIA-A100]

**Fallback order.** Reduce microbatch to 1 and increase accumulation to retain the effective batch. Retain AE checkpointing, enforce one forward at a time for CFG, stage cached latents on the VM disk, and keep metrics/segmentation in a separate process. If full-volume Stage 1 still exceeds preflight, train on `64³` aligned patches on origins divisible by eight with 50% foreground-centered sampling. Verify whole-volume inference and the patched AE against the full-volume region-sensitive validation gate. For tiled encoding/decoding, use `64³` tiles at stride 32, ignore or downweight patch margins through overlap weights, normalize blend weights, and verify seam error against a full reference on a larger device. Only validated AE checkpoints feed Stage 2. An `f=4` architecture is a separate ablation if `f=8` erases entorhinal or temporal-horn morphology. Its larger 32³ latent needs a new attention/memory preflight and cannot load the existing U-Net checkpoint by assumption. [MONAI AE source][MONAI-ae] · [MONAI U-Net source][MONAI-unet] · [CerebrA Table 1][Atlas-paper]

Training entry points should run in this dependency order · `audit_inputs → train_ae → validate_ae → cache_latents → train_ldm → pilot_probes → calibrate_qc → generate_probes → export_probe_features → frozen_tcav`. The blueprint contains reference implementations for target-grid masks, models/losses/masked sampling and numerical gates. The workflow still requires dataset wiring, defined project run directories, an independent morphometry validation baseline and an actual GPU profile. Keep ADNI data and participant-linked probe artifacts in authorized storage and check the project's applicable agreement before moving them to any hosted runtime or distributing them. [ADNI agreement terms][ADNI] · [User brief][Brief]

## 5. Automated quality gates before TCAV

The proposal gives `PSNR ≥28.5 dB`, `SSIM ≥0.88`, `axial-slice FVD ≤120` and `<1.5%` off-target volume change. These are project targets. Implement the metric definitions below, pin scales and extractors, and label any conditions that still require validation rather than aggregating them into a single “fidelity” score. The anatomical label and distribution gates can block release with all voxel metrics passing. [Proposal §5.7 and §5.11][D-proposal] · [SSIM source][SSIM] · [3D counterfactual benchmark][Sun] · [FVD reference][FVD]

| Gate | Exact implementation and acceptance |
| --- | --- |
| Integrity and geometry | Require finite decoded scalar array, approved grid, qform/sform, dtype and mm units, paired sidecar, canonical test exclusion, different content hash from source and existing outputs, and a valid signed-off regional measurement. Re-read the saved NIfTI and compare its classifier input against the intended array. [Pipeline output][P-pipe] · [`load_volume`][M-data] |
| Paired PSNR | Compute `10log10(1/MSE)` with `MSE = mean_R((x−y)²)`. For metrics only, map both scans using the same Train-fit 0.5th/99.5th voxel quantiles on equal-weighted Train brain samples, then clamp to `[0,1]`. `R` is the anchor's approved intracranial measurement region for self reconstruction/sham; for concept images it is `R_brain ∖ support`. Require `≥28.5 dB`, track support and taper error separately. Evaluate pre-composite exterior too, and report reconstruction and post-composite numbers separately. [Proposal §5.7][D-proposal] · [Paired metrics in Appendix B](#b3-paired-and-set-quality-metrics) |
| Volumetric SSIM | Compute local `((2μ_xμ_y+C1)(2σ_xy+C2))/((μ_x²+μ_y²+C1)(σ_x²+σ_y²+C2))` with 3D Gaussian `11³`, `σ=1.5 voxels` on the fixed grid, `C1=.01²,C2=.03²`. Mean over valid `R` centers whose windows lie inside `R`. Require `≥.88` as the proposal starting threshold. No valid centers yields failure. [Wang et al., §III][SSIM] · [Proposal §5.7][D-proposal] |
| Off-target voxel and structure invariance | Require bitwise-equivalent float32 exterior on reload. Independently compute `Σ_{c∈GM,WM,CSF}|V'_c−V_c|/Σ_cV_c` outside the support and within the frozen intracranial domain. Require `<.015` and validate segmentation QC. Check independent named off-target measures and support taper seams separately. For each non-target biomarker require shift within the frozen development repeatability envelope (`95th percentile` absolute paired no-op/reconstruction drift). Explicitly evaluate VE–THD nesting rather than forcing mathematically overlapping totals to be invariant. [Proposal §5.7][D-proposal] · [Sun minimality][Sun] |
| Target morphology | Require directional change beyond the matching repeatability envelope and threshold crossing or a documented dose within tolerance if already positive at baseline. Enforce the same measurement units and feature extractor/segmenter at calibration and release. For PBM require all six measurements in the agreed normal range, and verify the sham is neither an exact copy nor a hidden positive. Check that intended change survives the selected classifier's loader. [Proposal §5.5][D-proposal] · [Formal concept definitions][D-tex] · [Sun effectiveness/reversibility][Sun] |
| Fréchet distance in 3D MRI feature space | Use frozen 3D MRI features from a versioned pretrained encoder with no audit-subject overlap established, or a train-only external evaluator whose checkpoints and feature layers are frozen before calibration. This encoder must be independent of the classifier under explanation. Use the Gaussian feature-set distance `‖m_r−m_g‖²+tr(C_r+C_g−2(C_r^½ C_g C_r^½)^½)`, with a common Train-fitted PCA below the evaluation sample size. This is a cohort-level check. Calibrate the numerical threshold to the 95th percentile of validation real–real distances using age/phase-matched, subject-disjoint comparison sets of equal size, and freeze the numeric result per concept before evaluating probes. Initial plan `n=64` per set, `d≤32` features, 200 seeded resamples. If too few eligible validation subjects exist, use the same preregistered smaller `n≥32` for both comparisons and report the shortfall; lower counts leave the gate uncalibrated. [Heusel et al., §4][FID] · [Sun realism][Sun] |
| Axial slice FVD | Secondary metric for the proposal. Use the same orientation, ordered axial slice stack and fixed intensity transform for real and synthetic MRI. Fix grayscale-to-RGB replication, tensor layout `[N,T,H,W,3]`, `T=128`, resize and I3D hub model version to the FVD reference. Log sample count and exact extractor. Report `≤120` only with that implementation, with a sampled real–real reference as context. Anatomical slice continuity has no time axis; FVD is a stack diagnostic, separate from true longitudinal FVD. [Proposal §5.7][D-proposal] · [Unterthiner et al. code and preprocessing][FVD] |
| Evidence and artifact review | Require concept-level automated QA plus a blinded neuroradiology sample review before claiming clinical plausibility. Check duplicate/near-neighbor training anatomy, subgroup coverage, classifier geometry outcomes and artifact leakage in CAV banks. Report all acceptance rates and distinct subjects. [Proposal §5.10][D-proposal] · [Sun evaluation][Sun] |

**QC orchestration.** Route each finite candidate through geometry, paired/exterior metrics, independent morphology, regional safety and uniqueness checks. Keep provisional accepted files outside final release paths until the whole group passes matched feature-distribution and visual review. If fewer than 200 per group pass, return the actual numbers, failure distribution and the frozen config. No threshold changes after the final lock. Because exact exterior composition makes many post-composite global metrics optimistic, report true Stage 1 reconstruction, raw decoded masked proposal, and composed results side by side. [Proposal §5.7–5.11][D-proposal] · [Sun minimality and realism][Sun]

**TCAV estimation.** Fit a positive-versus-PBM linear SVM with a frozen CAV feature scaler trained on probe training anchors only. If `w` is fit in standardized coordinates, the corresponding raw-feature normal is `w/feature_std` before L2 normalization. Check held-out-anchor concept separability and matched sham behavior. For model score `s_k`, compute `D_{c,k}(x)=∇_h s_k(x)·v_c` and TCAV `mean(1[D>0])` over qualifying real patient subjects. With simple linear classifier heads and pooled features the derivative of a raw logit is constant across all subjects. Report this degeneracy and use a preregistered live nonlinear class score (e.g. grouped log probability) or a defined earlier feature representation for subject-varying interpretation, recording that as the selected TCAV variant. Build a subject-blocked permutation null by refitting CAVs with bank labels permuted within matched blocks, use `(1 + #null≥observed)/(1 + n_perm)` for the predeclared one-sided statistic or an explicitly centered two-sided version, and report bootstrap intervals clustered by evaluation subject. Use 1,000 permutations initially and Holm adjustment across the preregistered concept × class × layer family. Reuse of common anchors/patients invalidates treating all CAV runs as independent Welch-test observations. The proposal's Welch/Bonferroni protocol can be reproduced as a sensitivity analysis if its assumptions are satisfied. [Kim et al., §3][TCAV] · [Proposal Algorithm 1][D-proposal] · [Classifier fused linear head][M-model]

**Release criteria.** Pipeline and split audits pass. Full-resolution autoencoder reconstruction passes on development subjects with particular scrutiny of ECT and THD. A GPU profile completes the stated preflight. Both treatment and sham morphometry pass independent measurement. The seven probe quotas are met, distribution gates calibrated and passed, and TCAV consumes the exact frozen checkpoint contract. Claims remain at the level of synthetic probe generation and classifier sensitivity until an independent evaluation establishes the relevant clinical validity. [Proposal §§5.5–5.11][D-proposal] · [Sun benchmark][Sun]

## Appendix A. Source inventory


## Appendix B. Implementation references

The atlas code passed a six-region synthetic label round-trip using a real preprocessed scan's verified header. PyTorch 2.7.1 and MONAI 1.5.2 propagated all full-resolution model shapes on the meta device. Reduced-volume CPU tests ran both losses and backward passes, and completed 50 masked steps with unchanged exterior voxels. A fixture with absent Test files verified that the split index never resolves them. Paired metric and Fréchet tests verified identical exteriors and a known feature mean shift. These checks establish API and shape feasibility. They do not measure full-volume runtime/memory, atlas placement across the cohort, morphology accuracy or generation yield.

### B.1 Atlas masks on the pinned preprocessing grid

Save this block as `build_atlas_masks.py`. Install `nibabel`, `numpy` and `scipy` in the trusted work environment. Obtain the labeled symmetric 2009c CerebrA NIfTI from its DOI dataset, pin its checksum, then run `python build_atlas_masks.py ATLAS.nii.gz template/mni152.nii.gz VERIFIED_SCAN.nii.gz NEW_OUTPUT_DIR --atlas-sha256 VERIFIED_HASH`. The outputs are atlas supports, with the named label IDs and a provenance JSON. The release still requires correct overlays on real scans and independent per-subject morphometry. [CerebrA release][Atlas-data] · [CerebrA Table 1][Atlas-paper] · [Pinned preprocessing resize code][P-ops]

```python
"""Generate exact-grid edit masks from the published CerebrA Table 1 labels."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

TEMPLATE_SHA256 = "dbbd4542823c1e2b12265c9b7e3cf959ee02ff7829bbf05cdcfbecc0013a00fa"
REGIONS = {
    "HA": (48, 99),
    "ECT": (36, 87),
    "VE": (41, 92),
    "THD": (5, 56),
    "GCT": (47, 98, 31, 82, 10, 61, 1, 52, 9, 60, 28, 79),
    "AA": (19, 70),
}


def digest(path):
    with Path(path).open("rb") as handle:
        h = sha256()
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verified_grid(template, scan):
    qt, qtc = template.get_qform(coded=True)
    st, stc = template.get_sform(coded=True)
    qs, qsc = scan.get_qform(coded=True)
    ss, ssc = scan.get_sform(coded=True)
    if not all((qtc, stc, qsc, ssc)) or scan.shape != (128,) * 3:
        raise ValueError("Missing qform/sform or unexpected scan dimensions")
    expected = qt.copy()
    expected[:3, :3] = qt[:3, :3] @ np.diag(np.asarray(template.shape) / 128)
    if scan.header.get_xyzt_units()[0] != "mm" or not (
        np.allclose(qs, expected, atol=1e-5, rtol=0)
        and np.allclose(ss, expected, atol=1e-5, rtol=0)
    ):
        raise ValueError("Scan grid disagrees with the pinned preprocessing contract")
    return st @ np.linalg.solve(qt, ss)


def build_masks(atlas_path, template_path, scan_path, output, atlas_sha256):
    if digest(template_path) != TEMPLATE_SHA256:
        raise ValueError("Wrong DG-TCAV template version")
    if digest(atlas_path) != atlas_sha256:
        raise ValueError("Wrong atlas digest")
    atlas, template, scan = map(nib.load, (atlas_path, template_path, scan_path))
    if atlas.ndim != 3 or nib.aff2axcodes(atlas.affine) != ("R", "A", "S"):
        raise ValueError("Supply the verified canonical RAS CerebrA label volume")
    if not np.isfinite(atlas.affine).all() or abs(np.linalg.det(atlas.affine[:3, :3])) < 1e-8:
        raise ValueError("Invalid atlas affine")
    data = atlas.get_fdata()
    if not np.isfinite(data).all() or not np.array_equal(data, np.rint(data)):
        raise ValueError("Atlas must contain discrete integer labels")
    mni_affine = verified_grid(template, scan)
    on_grid = resample_from_to(atlas, (scan.shape, mni_affine), order=0).get_fdata()
    checked = {label for ids in REGIONS.values() for label in ids}
    missing = checked - set(np.unique(on_grid).astype(int))
    if missing:
        raise ValueError(f"Atlas labels absent on the target grid: {sorted(missing)}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for region, ids in REGIONS.items():
        mask = np.isin(on_grid, ids).astype(np.uint8)
        header = scan.header.copy()
        header.set_data_dtype(np.uint8)
        image = nib.Nifti1Image(mask, scan.affine, header)
        image.set_qform(scan.get_qform(), code=int(scan.header["qform_code"]))
        image.set_sform(scan.get_sform(), code=int(scan.header["sform_code"]))
        nib.save(image, output / f"{region}_support.nii.gz")
        counts[region] = int(mask.sum())
    metadata = {
        "atlas_sha256": atlas_sha256,
        "atlas_space": "MNI-ICBM152 nonlinear symmetric 2009c",
        "template_sha256": TEMPLATE_SHA256,
        "scan_grid_to_mni_world": mni_affine.tolist(),
        "scan_grid_internal_affine": scan.affine.tolist(),
        "labels": REGIONS,
        "voxels": counts,
    }
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("atlas")
    p.add_argument("template")
    p.add_argument("verified_scan")
    p.add_argument("output_directory")
    p.add_argument("--atlas-sha256", required=True)
    args = p.parse_args()
    build_masks(args.atlas, args.template, args.verified_scan, args.output_directory, args.atlas_sha256)
```

### B.2 MONAI model, losses and masked DDIM core

Save as `ldm_reference.py`. The tested imports use PyTorch 2.7.1, MONAI 1.5.2 and einops 0.8.1. Place `ae`, `unet`, `anatomy`, inputs and labels on the chosen device. Use float32 input/persisted latents, enable autocast/scaler in the training entry points according to §4, pass `scheduler.alphas_cumprod`, and use `.eval()` for sampling. Generate `support` and `alpha_voxel` as in §3.2. Set the seed and actual Train-fitted latent offset and scale. [MONAI AE][MONAI-ae] · [MONAI U-Net][MONAI-unet] · [MONAI DDPM][MONAI-ddpm] · [DDIM][DDIM]

```python
"""Reference MONAI model, training loss and 50-step masked sampling core."""
import torch
from torch import nn
from torch.nn import functional as F
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler


def build_models(num_classes=3, use_checkpoint=True):
    ae = AutoencoderKL(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 96), num_res_blocks=(1, 1, 2, 2),
        latent_channels=4, attention_levels=(False,) * 4, norm_num_groups=8,
        with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
        use_checkpoint=use_checkpoint, use_convtranspose=False,
    )
    unet = DiffusionModelUNet(
        spatial_dims=3, in_channels=4, out_channels=4,
        channels=(64, 96, 128), num_res_blocks=(2, 2, 2),
        attention_levels=(False, True, True), num_head_channels=(32, 32, 32),
        norm_num_groups=8, with_conditioning=True, cross_attention_dim=64,
        transformer_num_layers=1, num_class_embeds=num_classes + 1,
        upcast_attention=True, use_flash_attention=False,
    )
    anatomy = nn.Linear(12, 64, bias=False)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, schedule="scaled_linear_beta",
        beta_start=0.0015, beta_end=0.0195, prediction_type="epsilon",
        clip_sample=False,
    )
    return ae, unet, anatomy, scheduler


def autoencoder_loss(ae, x, brain):
    mu, sigma = ae.encode(x)
    recon = ae.decode(ae.sampling(mu, sigma)).float()
    x, brain, mu, sigma = x.float(), brain.float(), mu.float(), sigma.float()
    vox = (recon - x).abs()
    weights = 1 + 2 * brain
    fidelity = (vox * weights).sum() / weights.sum()
    edge = sum((recon.diff(dim=d) - x.diff(dim=d)).abs().mean() for d in (2, 3, 4)) / 3
    var = sigma.square().clamp_min(1e-12)
    kl = 0.5 * (mu.square() + var - var.log() - 1).mean()
    return fidelity + 0.1 * edge + 1e-5 * kl


def diffusion_loss(unet, anatomy, alphas, scaled_z, labels, morph, num_classes=3):
    batch = scaled_z.shape[0]
    device = scaled_z.device
    steps = torch.randint(len(alphas), (batch,), device=device)
    alpha = alphas.to(device=device, dtype=torch.float32)[steps].view(batch, 1, 1, 1, 1)
    eps = torch.randn_like(scaled_z)
    noisy = alpha.sqrt() * scaled_z + (1 - alpha).sqrt() * eps
    drop_label = torch.rand(batch, device=device) < 0.10
    drop_measurements = drop_label | (torch.rand(batch, device=device) < 0.15)
    labels = labels.masked_fill(drop_label, num_classes)
    morph = morph.masked_fill(drop_measurements[:, None], 0)
    pred = unet(noisy, steps, context=anatomy(morph).unsqueeze(1), class_labels=labels)
    return F.mse_loss(pred.float(), eps.float())


@torch.inference_mode()
def edit(ae, unet, anatomy, alphas, anchor, support, alpha_voxel, label,
         morph, latent_offset, latent_scale, seed, t_max=589, guidance=2.0,
         num_classes=3):
    """Use canonical intensity on the model path and retain the input outside support."""
    if not 49 <= t_max < len(alphas):
        raise ValueError("Need 50 distinct timesteps in the training range")
    if anchor.shape != support.shape or anchor.shape != alpha_voxel.shape:
        raise ValueError("Image and mask shapes must agree")
    if support.dtype != torch.bool or not bool(support.any()):
        raise ValueError("An edit requires nonempty Boolean support")
    if bool((~torch.isfinite(alpha_voxel) | (alpha_voxel < 0) | (alpha_voxel > 1)).any()):
        raise ValueError("Blend weights must be finite and in [0, 1]")
    z = (ae.encode(anchor)[0].float() - latent_offset) * latent_scale
    mask = F.max_pool3d(support.float(), kernel_size=8, stride=8) > 0
    if mask.shape[2:] != z.shape[2:]:
        raise ValueError("Support grid and latent shape disagree")
    device = z.device
    abar = alphas.to(device=device, dtype=torch.float32)
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(z.shape, device=device, dtype=z.dtype, generator=g)
    steps = torch.linspace(t_max, 0, 50).round().to(torch.int64).tolist()
    context = anatomy(morph).unsqueeze(1)
    null_context = torch.zeros_like(context)
    null_label = torch.full_like(label, num_classes)

    def anchor_at(a):
        return a.sqrt() * z + (1 - a).sqrt() * noise

    sample = anchor_at(abar[steps[0]])
    for j, now in enumerate(steps):
        current_alpha = abar[now]
        next_alpha = abar[steps[j + 1]] if j + 1 < len(steps) else abar.new_tensor(1)
        timestep = torch.full_like(label, now)
        eps_uncond = unet(sample, timestep, context=null_context, class_labels=null_label)
        eps_cond = unet(sample, timestep, context=context, class_labels=label)
        eps = eps_uncond.float() + guidance * (eps_cond.float() - eps_uncond.float())
        clean = (sample - (1 - current_alpha).sqrt() * eps) / current_alpha.sqrt()
        update = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * eps
        sample = torch.where(mask, update, anchor_at(next_alpha))

    proposal = ae.decode(sample / latent_scale + latent_offset).float()
    if not bool(torch.isfinite(proposal).all()):
        raise ValueError("Decoder produced a nonfinite candidate")
    return torch.where(support, anchor + alpha_voxel * (proposal - anchor), anchor)
```

### B.3 Paired and set quality metrics

Save as `probe_qc.py`. These functions accept volume, tissue and feature arrays from the specified upstream methods. The caller applies the fixed §5 gates, independently checks target morphometry, and performs set-level bootstrapped calibration using matched groups. [Proposal §5.7][D-proposal] · [Wang et al., §III][SSIM] · [Heusel et al., §4][FID]

```python
"""Paired voxel QC, non-target tissue checks, and a set-level distance."""
import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter


def paired_metrics(reference, candidate, region, train_low, train_high):
    """Return paired PSNR/SSIM on a fixed train-defined intensity scale."""
    reference, candidate = (np.asarray(a, dtype=np.float64) for a in (reference, candidate))
    region = np.asarray(region, dtype=bool)
    if reference.ndim != 3 or reference.shape != candidate.shape or region.shape != reference.shape:
        raise ValueError("Paired QC requires matching scalar 3-D grids")
    if not all(np.isfinite(a).all() for a in (reference, candidate, train_low, train_high)):
        raise ValueError("Nonfinite input to image QC")
    if not region.any() or train_high <= train_low:
        raise ValueError("Empty QC region or invalid training intensity scale")
    span = train_high - train_low
    x = np.clip((reference - train_low) / span, 0, 1)
    y = np.clip((candidate - train_low) / span, 0, 1)
    mse = np.mean((x[region] - y[region]) ** 2)

    def local(a):
        return gaussian_filter(a, sigma=1.5, truncate=5 / 1.5, mode="reflect")

    mx, my = local(x), local(y)
    vx = np.maximum(local(x * x) - mx * mx, 0)
    vy = np.maximum(local(y * y) - my * my, 0)
    xy = local(x * y) - mx * my
    ssim = ((2 * mx * my + 0.01 ** 2) * (2 * xy + 0.03 ** 2)
            / ((mx * mx + my * my + 0.01 ** 2) * (vx + vy + 0.03 ** 2)))
    interior = minimum_filter(region, size=11, mode="constant", cval=0)
    if not interior.any():
        raise ValueError("Need at least one 11-cube interior center")
    psnr = float("inf") if mse == 0 else float(-10 * np.log10(mse))
    return psnr, float(ssim[interior].mean())


def preservation(reference, candidate, support, measurement_domain, tissue_a, tissue_b):
    if not (reference.shape == candidate.shape == support.shape
            == measurement_domain.shape == tissue_a.shape == tissue_b.shape):
        raise ValueError("Voxel and segment grids differ")
    support = np.asarray(support, dtype=bool)
    exterior = ~support
    region = exterior & np.asarray(measurement_domain, dtype=bool)
    if not exterior.any() or not region.any():
        raise ValueError("No non-target brain domain")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Nonfinite voxel")
    # Provided arrays use a frozen coarse segmentation label map on this grid.
    if not np.isin(tissue_a[region], [0, 1, 2, 3]).all() or not np.isin(tissue_b[region], [0, 1, 2, 3]).all():
        raise ValueError("Use 0=other, 1=GM, 2=WM, 3=CSF")
    counts_a = np.array([(tissue_a[region] == label).sum() for label in (1, 2, 3)])
    counts_b = np.array([(tissue_b[region] == label).sum() for label in (1, 2, 3)])
    drift = abs(counts_b - counts_a).sum() / max(counts_a.sum(), 1)
    if not counts_a.sum():
        raise ValueError("No measurable reference tissue outside the support")
    a = np.asarray(reference[exterior], dtype=np.float32).view(np.uint32)
    b = np.asarray(candidate[exterior], dtype=np.float32).view(np.uint32)
    exact = np.array_equal(a, b)
    return exact, float(drift)


def frechet_distance(real_features, generated_features):
    """Squared Gaussian Fréchet distance for matched-size feature sets."""
    real, generated = (np.asarray(a, dtype=np.float64) for a in (real_features, generated_features))
    if real.ndim != 2 or generated.shape != real.shape or len(real) < 2:
        raise ValueError("Provide equally sized 2D feature sets")
    if not (np.isfinite(real).all() and np.isfinite(generated).all()):
        raise ValueError("Nonfinite feature set")
    cov_a = np.atleast_2d(np.cov(real, rowvar=False, ddof=1))
    cov_b = np.atleast_2d(np.cov(generated, rowvar=False, ddof=1))
    val, vec = np.linalg.eigh(cov_a)
    root_a = np.einsum("ik,jk,k->ij", vec, vec, np.sqrt(np.maximum(val, 0)))
    middle = np.einsum("ij,jk,kl->il", root_a, cov_b, root_a)
    trace_root = np.sqrt(np.maximum(np.linalg.eigvalsh(0.5 * (middle + middle.T)), 0)).sum()
    delta = real.mean(0) - generated.mean(0)
    value = np.sum(delta ** 2) + np.trace(cov_a) + np.trace(cov_b) - 2 * trace_root
    if not np.isfinite(value):
        raise ValueError("Nonfinite Fréchet distance")
    return max(0.0, float(value))
```

### B.4 Longitudinal split enforcement

Save as `index_diffusion.py` and call `generate_index` using the two versioned Drive inputs and the new visit-keyed QC CSV. The QC CSV columns are `subject_id,image_id,pipeline_version,template_sha256,status,volume_path,mask_path`; one final row per scan. Store paths relative to `volume_root`. The caller validates the actual header and voxel geometry before loading each admitted Train/Val scan and compares canonical decoded-voxel hashes as well as file digests if gzipped aliases are possible. [Master subject split][A-split] · [Longitudinal manifest][B-manifest] · [Existing baseline index][M-data]

```python
"""Build Train/Val-only indices without resolving or opening any test scan."""
from csv import DictReader
from hashlib import sha256
from pathlib import Path
import json


LABELS = {"CN": 0, "EMCI": 1, "LMCI": 1, "MCI": 1, "AD": 2}
VERSION = "masked_rigid_affine_centroid_seed42_v4"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(DictReader(handle))


def scan_key(row):
    image_id = row["image_id"].strip().removeprefix("I")
    if not image_id.isascii() or not image_id.isdigit():
        raise ValueError("Invalid image ID")
    return row["subject_id"].strip(), str(int(image_id))


def file_digest(path):
    with path.open("rb") as handle:
        h = sha256()
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def generate_index(master_split, longitudinal_manifest, qc_index, volume_root,
                   template_sha256, label_map=LABELS):
    splits = json.loads(Path(master_split).read_text())
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Invalid master split keys")
    owner = {}
    for partition, ids in splits.items():
        if not isinstance(ids, list) or not ids:
            raise ValueError("Expected a non-empty JSON subject list")
        for subject in ids:
            if (not isinstance(subject, str) or not subject
                    or subject != subject.strip() or subject in owner):
                raise ValueError("Invalid or overlapping master subject IDs")
            owner[subject] = partition

    manifest = read_csv(longitudinal_manifest)
    rows = {scan_key(row): row for row in manifest}
    if len(rows) != len(manifest) or len({key[1] for key in rows}) != len(rows):
        raise ValueError("Duplicate source image ID")
    if {key[0] for key in rows} != set(owner):
        raise ValueError("Master split and longitudinal manifest disagree")
    visits = {(key[0], row["visit"]) for key, row in rows.items()}
    if len(visits) != len(rows):
        raise ValueError("More than one scan in a subject visit")
    qc = {}
    for row in read_csv(qc_index):
        key = scan_key(row)
        if key not in rows or key in qc:
            raise ValueError("QC index requires a unique final row per original scan")
        qc[key] = row

    root = Path(volume_root).resolve(strict=True)
    seen_paths = set()
    image_hashes = {}
    result = {"train": [], "val": []}
    excluded = []

    def get_file(relative):
        p = Path(relative)
        if not relative or p.is_absolute() or ".." in p.parts:
            raise ValueError("Files must be relative to volume_root")
        p = (root / p).resolve(strict=True)
        if not p.is_relative_to(root) or not p.is_file() or p in seen_paths:
            raise ValueError("Invalid or reused volume path")
        seen_paths.add(p)
        return p

    for key, row in sorted(rows.items()):
        partition = owner[key[0]]
        if partition == "test":
            continue
        report = qc.get(key)
        if not report or (report["status"], report["pipeline_version"],
                          report["template_sha256"]) != ("SUCCESS", VERSION, template_sha256):
            excluded.append((key, "missing_or_ineligible_qc"))
            continue
        volume = get_file(report["volume_path"])
        mask = get_file(report["mask_path"])
        digest = file_digest(volume)
        if digest in image_hashes:
            raise ValueError("Repeated MRI bytes within or across development splits")
        image_hashes[digest] = key
        result[partition].append({
            "subject_id": key[0], "image_id": key[1], "split": partition,
            "label": label_map[row["visit_diagnosis"]],
            "source_label": row["visit_diagnosis"],
            "volume_path": str(volume), "mask_path": str(mask),
            "volume_sha256": digest, "mask_sha256": file_digest(mask),
        })
    expected_labels = set(label_map.values())
    for partition, data in result.items():
        if {row["label"] for row in data} != expected_labels:
            raise ValueError(f"Missing configured classes in {partition}")
        if any(row["subject_id"] in splits["test"] for row in data):
            raise ValueError("Test subject in the development index")
    return {
        "partitions": result,
        "excluded_development_scans": excluded,
        "master_split_sha256": file_digest(Path(master_split)),
        "manifest_sha256": file_digest(Path(longitudinal_manifest)),
        "qc_index_sha256": file_digest(Path(qc_index)),
    }
```

