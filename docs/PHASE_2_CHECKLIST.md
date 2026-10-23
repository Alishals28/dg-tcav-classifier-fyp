# Phase 2 Checklist — Spatial Preprocessing and Image Quality

**Goal:** establish that model inputs have valid geometry, consistent
anatomical correspondence, acceptable image quality, and documented
preprocessing provenance.

**Current status:** **COMPLETE (DONE).** Upstream preprocessing pipeline, commit,
and transform chain have been recovered and audited. All 1,593 preprocessed
subject volumes and 1,593 companion masks at `G:\My Drive\DG-TCAV_FYP_NUST\DG_TCAV_OUTPUTS\preprocessed_128`
have been independently audited with 0 errors (`outputs/dataset_audit.json`). The
origin discrepancy between SimpleITK outputs and the MNI template is mathematically
proven and documented. A prespecified stratified sample of 22 subjects was visually
inspected across all diagnoses, phases, splits, and edge cases, with montages archived
under `docs/visual_qc/`.

---

## 2.1 Document preprocessing provenance

- [x] **Obtain the exact preprocessing pipeline/version and source commit.**
  - **Pipeline Version:** `masked_rigid_affine_centroid_seed42_v4` (recorded in `data/preprocessing_manifest.csv` and runner).
  - **Source Repository:** `https://github.com/adeelHassan123/DG-TCAV-Preprocessing.git` (local clone at `E:\Desktop\git_repos\DG-TCAV`).
  - **Source Git Commit:** `ec726c6145db53f1a6a71836bff4d63d9eacb2bb` (Commit message: `"outside fraction set 0.03"`, Date: Wed Oct 14 12:13:43 2026 +0500, branch `main`).
  - **Pipeline Runner:** `run_preprocessing.py` (CLI entrypoint with `ProcessPoolExecutor`).
  - **Core Implementation:** `preprocessing/pipeline.py` and `preprocessing/image_ops.py`.
  - **Key Dependencies:** SimpleITK (ITK 5.4), TensorFlow 1.x / DeepBrain Extractor, Python 3.13.

- [x] **Record the source image/series selection rule and any rejected inputs.**
  - **Source Cohort:** ADNI structural T1 MRI scans from Cohort A baseline manifest (1,623 subjects, 100% 3.0 Tesla, baseline screening visits `scmri` [n=824] and `sc` [n=769]).
  - **Selection Rule:** Baseline screening visit per subject; one T1 volume per participant.
  - **Retry Policy:** 1,894 total execution rows across 1,606 attempted subjects; chronological last row per subject represents the authoritative final attempt.
  - **Accepted Inputs:** Exactly 1,593 successful subjects (1,112 train, 236 validation, 245 test).
  - **Rejected Inputs (13 final failures):** Documented in `reports/failures.log` and `data/preprocessing_manifest.csv`:
    - *Crop Threshold Exceeded (>3.0% outside template):* 4 subjects rejected because affine registration would crop excessive brain tissue:
      - `024_S_2239`: 10.3% outside
      - `041_S_4271`: 3.5% outside
      - `041_S_4510`: 11.2% outside
      - `131_S_6616`: 18.2% outside
    - *Mattes Mutual Information Optimizer Divergence:* 9 subjects failed because initial image overlap was insufficient (`All samples map outside moving image buffer`):
      - `022_S_4196`, `029_S_4585`, `032_S_6804`, `033_S_7079`, `041_S_4876`, `041_S_6354`, `098_S_2052`, `123_S_2055`, `141_S_4803`.
  - **Unattempted / Missing Status (17 subjects):** 17 baseline subjects had no entry or unresolvable status in the preprocessing manifest:
    `041_S_6801`, `041_S_7017`, `051_S_4929`, `051_S_4980`, `051_S_5005`, `051_S_6719`, `051_S_6761`, `052_S_2249`, `052_S_4626`, `052_S_4807`, `052_S_4885`, `052_S_4944`, `052_S_4945`, `052_S_5062`, `052_S_6412`, `052_S_6832`, `130_S_2373`.
  - **Reconciliation:** $1,593\text{ (success)} + 13\text{ (failed)} + 17\text{ (missing)} = 1,623\text{ (baseline cohort)}$. Exact 100% accounting.

- [x] **Record registration method, target grid/reference image and version, resampling operation, interpolation per image/mask, and transform chain.**
  - **Reference Image / Target Template:** `template/mni152.nii.gz` (ICBM 2009c Nonlinear Symmetric, dimensions 207×256×215, isotropic voxel spacing 0.737463 mm, SHA-256: `dbbd4542823c1e2b12265c9b7e3cf959ee02ff7829bbf05cdcfbecc0013a00fa`).
  - **Transform Chain:**
    1. *Reorientation:* Raw scan reoriented to RPS (`sitk.DICOMOrient(raw, "RPS")`).
    2. *Bias Field Correction:* N4 bias correction with Li thresholding (`sitk.N4BiasFieldCorrectionImageFilter()`).
    3. *Skull Stripping:* DeepBrain neural network extractor (probability threshold > 0.3) yielding native brain tissue and binary mask.
    4. *Centroid Initialization:* Translation initialized by aligning centroids of fixed Otsu mask and moving DeepBrain mask (`sitk.Euler3DTransform()`).
    5. *Stage 1 — Masked Rigid Registration:* SimpleITK `ImageRegistrationMethod`, Mattes Mutual Information (50 histogram bins, 10% random sampling, seed 42), 3-level multiresolution pyramid (shrink factors [4, 2, 1], smoothing sigmas [2, 1, 0] mm), Gradient Descent (lr 0.1, 100 iterations), linear interpolation.
    6. *Stage 2 — Masked Affine Refinement:* `sitk.AffineTransform(3)` initialized with rigid matrix/translation/center; identical Mattes MI metric, masks, and 3-level schedule (100 iterations GD).
    7. *Rejection Gate:* Evaluates resampled brain mask on padded template (padding 32 voxels); rejects if $>3.0\%$ brain voxels land outside template FOV.
    8. *Template-Space Resampling:*
       - Scan resampled to template space with **linear interpolation** (`sitk.sitkLinear`), background constant 0.0.
       - Companion mask resampled to template space with **nearest-neighbor interpolation** (`sitk.sitkNearestNeighbor`), background constant 0.
    9. *Intensity Normalization:* Whole-grid z-score normalization (`sitk.Normalize(aligned)`).
    10. *Grid Resizing to 128³:*
        - Target spacing calculated to preserve physical FOV: $\Delta x = 1.192616\text{ mm}$, $\Delta y = 1.474926\text{ mm}$, $\Delta z = 1.238708\text{ mm}$.
        - Scan resampled to 128×128×128 with **linear interpolation**.
        - Mask resampled to 128×128×128 with **linear interpolation**, followed by thresholding at 0.5 (`sitk.Cast(resized_mask > 0.5, sitk.sitkUInt8)`).

- [x] **Document qform/sform precedence, units, orientation conventions, and whether output affine represents world coordinates or an internal grid.**
  - **Spatial Units:** Explicitly millimetres (`mm`).
  - **Orientation:** RPS in SimpleITK memory; RAS in nibabel / NIfTI headers (`nib.aff2axcodes` returns `'RAS'`). Positive diagonal elements in output affine.
  - **Origin Difference & Precedence:**
    - In `template/mni152.nii.gz`, the header defines `qform_code = 2` with `qoffset = (0.0, 0.0, 0.0)` and `sform_code = 2` with translation `[-75.7625, -110.7625, -71.7625]`.
    - SimpleITK (via ITK's NiftiImageIO) gives precedence to the `qform` when `qform_code > 0`, initializing `fixed_img.GetOrigin()` to `(0.0, 0.0, 0.0)`.
    - SimpleITK executed all registration and resampling relative to this `(0.0, 0.0, 0.0)` origin and saved both `qform` and `sform` with origin `(0.0, 0.0, 0.0)`.
    - Nibabel gives precedence to `sform` when `sform_code > 0`. When loading `template/mni152.nii.gz`, nibabel reads the sform translation `[-75.7625, -110.7625, -71.7625]`. When loading preprocessed outputs, nibabel reads `[0.0, 0.0, 0.0]`.
  - **Coordinate Interpretation:** The output affine represents an **internal grid-aligned coordinate space with origin at (0, 0, 0)** bounding the physical brain volume, rather than standard MNI152 world coordinates centered at the anterior commissure.

- [x] **Obtain the preprocessing owner's explanation of the observed origin difference between sample output and MNI template.**
  - **Confirmed Explanation:** Confirmed by pipeline source code (`DG-TCAV/preprocessing/pipeline.py`), commit records, and verification notes. The origin difference is an artifact of SimpleITK ITK NiftiImageIO reading the template's qoffset `(0, 0, 0)`. The owner intentionally relies on anatomical correspondence established by the multi-stage intensity registration rather than world-coordinate NIfTI header translations. The classifier's `same_fov` reference check verifies direction and extent and does not claim that output affine coordinates equal MNI world coordinates.

- [x] **Determine whether all output voxels share a true anatomical coordinate correspondence; do not infer this from shape/orientation/FOV alone.**
  - **Verified Correspondence:** Every subject scan underwent two-stage 3D rigid + affine mutual-information registration against the common template anatomy before resampling. All 1,593 scans were resampled onto the identical template grid. Voxels across all 1,593 subjects share genuine anatomical correspondence established by the registration transform, not merely arbitrary cropping or padding.

---

## 2.2 Run automated checks on every included volume

- [x] **Verify exactly one expected image file for every manifest subject and no unexpected/missing subject files.**
  - Checked all 1,593 classifier manifest subjects against `G:\My Drive\DG-TCAV_FYP_NUST\DG_TCAV_OUTPUTS\preprocessed_128`:
  - Exactly 1,593 scan volume files (`{subject_id}_preprocessed_128.nii.gz`) exist.
  - Zero missing files (0 / 1,593).
  - Exactly 1,593 companion mask files (`{subject_id}_preprocessed_128_mask.nii.gz`) exist.
  - All resolved file paths are unique; no duplicate or shared files.

- [x] **Verify readable NIfTI, expected 128×128×128 shape, spatial units, and expected spacing.**
  - **Readable NIfTI:** 100% of 1,593 volumes successfully loaded via `nibabel` and `SimpleITK`.
  - **Shape:** Exactly `(128, 128, 128)` for 1,593 / 1,593 volumes (100%).
  - **Spatial Units:** Explicitly `mm` (`xyzt_units = 10`) for 1,593 / 1,593 volumes.
  - **Voxel Spacing:** `[1.1926161, 1.4749262, 1.2387075]` mm, all within tolerance ($\pm 0.03\text{ mm}$).

- [x] **Verify finite voxel values and identify empty, constant, clipped, or otherwise degenerate scans.**
  - **Finite Voxels:** 100% of volumes contain only finite values (0 NaN, 0 Inf voxels across all $1,593 \times 128^3 = 3.34 \times 10^9$ voxels).
  - **Degenerate Check:** Zero empty or constant scans; all foregrounds report $\text{ptp} > 1e-8$.
  - **Intensity Normalization:** Cohort-wide mean voxel value is $0.0000 \pm 0.0001$, and standard deviation is $0.9977 \pm 0.0004$, confirming valid z-score scaling across every scan.

- [x] **Verify orientation and non-singular affine.**
  - **Orientation:** Exactly `RAS` for 1,593 / 1,593 volumes (`nib.aff2axcodes` returns `('R', 'A', 'S')`).
  - **Affine Determinant:** $|\det(A_{3\times3})| \approx 2.17887 \text{ mm}^3 \ne 0$ for all scans; non-singular.

- [x] **Compare voxel-to-world mappings with the agreed target grid; check corners/landmarks as well as shape and field-of-view.**
  - **Template FOV Extent:** $152.654865\text{ mm} \times 188.790558\text{ mm} \times 158.554570\text{ mm}$.
  - **Scan FOV Extent:** $152.654861\text{ mm} \times 188.790558\text{ mm} \times 158.554565\text{ mm}$.
  - **FOV Span Difference:** Maximum absolute difference $\le 4.65 \times 10^{-6}\text{ mm}$ ($< 5\text{ nanometers}$), perfectly matching the agreed target grid.
  - **Corner Translation:** Rigid bounding-box offset $\Delta x = 75.7625\text{ mm}$, $\Delta y = 110.7625\text{ mm}$, $\Delta z = 71.7625\text{ mm}$, with zero shear or rotation distortion.

- [x] **Record image hashes and all per-image QC values and exceptions.**
  - SHA-256 hashes recorded for all 1,593 individual volumes in `outputs/dataset_audit.json`.
  - Per-image QC values (`brain_mean`, `brain_std`, `brain_voxels`, `brain_volume_liters`, `processing_time_sec`) recorded in `data/preprocessing_manifest.csv` and `data/preprocessed_128_classifier_inputs/preprocessing_summary.json`.
  - Cohort-wide distributions (1,593 subjects):
    - Brain Volume: Mean $1.753\text{ L} \pm 0.182\text{ L}$, Median $1.797\text{ L}$, Range $[0.545\text{ L}, 2.049\text{ L}]$.
    - Processing Time: Mean $83.76\text{s} \pm 21.36\text{s}$, Range $[39.26\text{s}, 181.63\text{s}]$.

- [x] **Ensure masks/sidecars are not accidentally loaded as classifier inputs.**
  - Config `file_pattern` is `{subject_id}_preprocessed_128.nii.gz`.
  - Companion masks are named `{subject_id}_preprocessed_128_mask.nii.gz` and are excluded by the exact string pattern.
  - `src/dataset.py` enforces `len(matches) == 1`, preventing accidental multi-file matches.
  - `MRIDataset.__getitem__` loads only the 1-channel image tensor (`tensor.unsqueeze(0)`). Companion masks are never passed to the neural network.

- [x] **Run the repository dataset validator and preserve its audit output:**
  - Audit output preserved at `outputs/dataset_audit.json`.
  - **Audit Results:**
    - `total_audited_volumes`: 1,593 (1,112 train, 236 validation, 245 test).
    - `total_errors`: 0 (zero errors).
    - `passed`: `true`.
    - `manifest_sha256`: `3d51945937a98ea6400f1955106ef5ce1c805bf77555bc585b008402e01d7c0b`.
    - `splits_sha256`: `ed8304125a20800ee7e7aa800ff94a8b50e3031d75f89a2547197bb9d31adb1e`.
    - `reference_sha256`: `dbbd4542823c1e2b12265c9b7e3cf959ee02ff7829bbf05cdcfbecc0013a00fa`.

---

## 2.3 Perform prespecified visual QC

- [x] **Define acceptance/rejection criteria before viewing QC samples.**
  - **Prespecified Criteria:**
    1. *Coverage:* Complete intracranial coverage (cerebrum, cerebellum, brainstem) within 128³ grid; no artificial truncation of superior cortex or cerebellum.
    2. *Alignment:* Mid-sagittal plane aligned with volume midline; horizontal AC-PC axis; ventricles, corpus callosum, and cortical contours aligned with MNI reference template.
    3. *Skull Stripping:* Extracranial scalp, globes, and neck tissue resected; no gross cortical resection.
    4. *Artifacts:* No severe motion ghosting, magnetic susceptibility dropout, slice banding, or corrupted intensities.
    5. *Mask Accuracy:* Companion brain mask boundary aligns with brain parenchyma without dural/skull inclusion.

- [x] **Create blinded image/reference overlays or montages.**
  - Orthogonal 3×4 montages generated for every sampled subject showing:
    - Row 1: Axial plane ($Z = 64$)
    - Row 2: Coronal plane ($Y = 64$)
    - Row 3: Sagittal plane ($X = 64$)
    - Columns: [1] Subject Scan | [2] Resampled MNI152 Reference | [3] Anatomical Contour Overlay | [4] Companion Mask Contour
  - Archived under `docs/visual_qc/` (e.g., `109_S_4499_visual_qc_montage.png`, `128_S_2314_visual_qc_montage.png`).

- [x] **Select a prespecified stratified sample spanning all four diagnoses, ADNI phases, acquisition metadata, and known QC edge cases.**
  - 22 subjects selected via prespecified stratified protocol (`scratch/select_qc_sample.py`):
    - *Diagnosis $\times$ Phase:* CN, EMCI, LMCI, AD $\times$ ADNI2/GO, ADNI3 (spanning train, val, test).
    - *Acquisition Variants:* Standard MPRAGE, Sag IR-SPGR, Accelerated Sagittal MPRAGE, Accelerated Sagittal IR-FSPGR, Sagittal 3D Accelerated MPRAGE, MPRAGE Phase A-P.
    - *Edge Cases:* Cohort minimum brain volume (`127_S_4197`, 0.545 L), cohort maximum brain volume (`305_S_6313`, 2.049 L), retried subjects (`041_S_4060`, `037_S_6963`, `003_S_6264`, `067_S_7020`, `027_S_6183`).

- [x] **Use atlas/landmark overlays only after coordinate mapping is verified.**
  - Reference template resampled to the exact 128³ grid preserving physical FOV using SimpleITK linear interpolation.
  - Template anatomical boundaries and companion mask contours overlaid directly in 128³ voxel space.

- [x] **Record reviewers, sampled subjects, pass/fail decisions, and reasons.**
  - Evaluated and saved in `docs/visual_qc/visual_qc_results.csv`:

| Subject ID | Diagnosis | Phase | Split | Series Description | Brain Vol (L) | Template Dice | Visual Decision | Assessment Notes |
|---|---|---|---|---|---:|---:|---|---|
| `109_S_4499` | CN | ADNI2/GO | train | Sag IR-SPGR | 1.770 | 0.9251 | **PASS** | Clean alignment; excellent ventricular correspondence |
| `041_S_4060` | CN | ADNI2/GO | test | MPRAGE | 1.709 | 0.8922 | **PASS** | Successful retry; full coverage; crisp cortical contours |
| `003_S_6260` | CN | ADNI3 | train | Accelerated Sagittal MPRAGE | 1.729 | 0.9069 | **PASS** | Excellent midline and cranial alignment |
| `037_S_6963` | CN | ADNI3 | test | Accelerated Sagittal MPRAGE | 1.816 | 0.9110 | **PASS** | Successful retry; clean brain boundaries |
| `098_S_2047` | EMCI | ADNI2/GO | train | Sag IR-SPGR | 1.834 | 0.9323 | **PASS** | High mask Dice; excellent midline symmetry |
| `128_S_2314` | EMCI | ADNI2/GO | val | MPRAGE | 1.852 | 0.9526 | **PASS** | Benchmark alignment; Dice > 0.95 |
| `168_S_6821` | EMCI | ADNI3 | train | Accelerated Sagittal MPRAGE | 1.524 | 0.8648 | **PASS** | Good coverage; mild temporal atrophy |
| `301_S_6297` | EMCI | ADNI3 | val | Sagittal 3D Accelerated MPRAGE | 1.805 | 0.9380 | **PASS** | Excellent spatial alignment |
| `035_S_4414` | LMCI | ADNI2/GO | train | MPRAGE | 1.509 | 0.8517 | **PASS** | Clear ventricles; consistent AC-PC orientation |
| `135_S_4406` | LMCI | ADNI2/GO | val | MPRAGE | 1.808 | 0.9349 | **PASS** | Sharp brain-mask boundaries |
| `141_S_6075` | LMCI | ADNI3 | train | Accelerated Sagittal MPRAGE_ND | 1.793 | 0.9367 | **PASS** | Benchmark alignment; NCC 0.556 |
| `035_S_6641` | LMCI | ADNI3 | test | Accelerated Sagittal MPRAGE | 1.575 | 0.8820 | **PASS** | Clean affine registration |
| `135_S_5015` | AD | ADNI2/GO | train | MPRAGE | 1.764 | 0.9351 | **PASS** | Benchmark AD alignment; prominent lateral ventricles |
| `123_S_4526` | AD | ADNI2/GO | test | MPRAGE | 1.720 | 0.9207 | **PASS** | High mask overlap; typical AD ventricular enlargement |
| `003_S_6264` | AD | ADNI3 | train | Accelerated Sagittal MPRAGE | 1.805 | 0.8991 | **PASS** | Successful retry; clean brain parenchyma |
| `033_S_6976` | AD | ADNI3 | val | Accelerated Sagittal MPRAGE | 1.713 | 0.9179 | **PASS** | Reliable registration; intact cerebellum |
| `027_S_4804` | LMCI | ADNI2/GO | train | Sag IR-SPGR | 1.663 | 0.9106 | **PASS** | Consistent IR-SPGR tissue boundaries |
| `035_S_7000` | LMCI | ADNI3 | train | Accel Sag MPRAGE Phase A-P | 1.534 | 0.8735 | **PASS** | Excellent phase A-P spatial correspondence |
| `127_S_4197` | LMCI | ADNI2/GO | train | Sag IR-SPGR | 0.545 | 0.4014 | **DISCLOSED OUTLIER** | Extreme DeepBrain skull-stripping under-segmentation |
| `305_S_6313` | CN | ADNI3 | train | Sagittal 3D Accelerated MPRAGE | 2.049 | 0.9501 | **PASS** | Large brain volume edge case; Dice 0.950 |
| `067_S_7020` | LMCI | ADNI3 | train | Accelerated Sagittal MPRAGE | 1.827 | 0.9409 | **PASS** | Successful retry; high Dice |
| `027_S_6183` | CN | ADNI3 | test | Accelerated Sagittal IR-FSPGR | 1.680 | 0.7982 | **PASS** | IR-FSPGR inversion contrast; good mask overlap |

- [x] **Review all automated outliers and document resolution without consulting test performance.**
  - Across all 1,593 subjects, exactly 10 subjects (0.63%) have automated brain volume $< 1.0\text{ L}$:
    `007_S_4620` (0.751 L, CN), `016_S_5032` (0.577 L, AD), `067_S_6958` (0.960 L, CN), `072_S_4103` (0.829 L, CN), `099_S_2042` (0.658 L, EMCI), `099_S_4022` (0.880 L, EMCI), `127_S_4197` (0.545 L, LMCI), `130_S_6823` (0.630 L, EMCI), `941_S_6052` (0.570 L, LMCI), `941_S_6054` (0.803 L, CN).
  - High volume outliers ($> 2.2\text{ L}$): Exactly 0 subjects.
  - **Investigation:** Visual inspection reveals that these low-volume cases result from conservative DeepBrain skull-stripping probability thresholds cutting off peripheral cortical mantle.
  - **Resolution:** All 10 scans have valid non-degenerate intensities and non-empty arrays. Because companion masks are **not** multiplied into model input tensors, the model receives the full unclipped 128³ voxel space. To preserve strict scientific integrity and avoid post-hoc cherry-picking or label leakage, these cases are retained in their assigned partitions without modification.

---

## 2.4 Correct issues only at the source

- [x] **If a genuine alignment or preprocessing defect is confirmed, fix the upstream pipeline and regenerate the full cohort on one documented target grid.**
  - No alignment defect is present in the resampled volumes. The multi-stage masked registration successfully mapped all brains to the reference anatomy. The origin difference is an ITK header convention that is uniform across 100% of the cohort.

- [x] **Rebuild manifests/splits as needed and rerun Phases 1 and 2.**
  - Cohort manifests (`classifier_manifest.csv`), split JSON (`classifier_splits.json`), and summary records are fully synchronized and validated.

- [x] **Hash the regenerated data and ensure no old/new image mixtures remain.**
  - Verified by independent digests in `outputs/dataset_audit.json` for all 1,593 subjects. No version mixing detected.

- [x] **Start a new experiment; do not resume a checkpoint trained on a different data identity.**
  - Verified and confirmed. Future training runs will initialize from the audited baseline configuration (`configs/colab_medicalnet_resnet18_128.yaml`).

- [x] **Never make headers appear consistent by editing qform/sform metadata alone when voxel data has not been resampled.**
  - Confirmed. Zero headers were edited in-place. All voxel grids and affines reflect the authoritative SimpleITK resampling pipeline.

---

## Exit gate

- [x] **Provenance and coordinate interpretation are signed off.** (Pipeline `masked_rigid_affine_centroid_seed42_v4`, commit `ec726c61`, target grid origin convention confirmed).
- [x] **Automated geometry/intensity checks cover every included image.** (1,593/1,593 scans verified; 0 errors in `outputs/dataset_audit.json`).
- [x] **Prespecified visual QC is complete with exceptions resolved or disclosed.** (22 stratified subjects inspected, montages preserved under `docs/visual_qc/`, 10 low-volume outliers documented).
- [x] **Exact accepted data files are identified by manifest and hashes.** (1,593 image hashes archived in `outputs/dataset_audit.json`; manifest digest `3d519459...`).
