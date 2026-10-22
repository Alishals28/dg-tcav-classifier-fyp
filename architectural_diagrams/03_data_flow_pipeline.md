# Data Flow & Preprocessing Pipeline Diagram

This document illustrates the complete end-to-end data transformation pipeline: from raw ADNI NIfTI scan volumes on disk to normalized, augmented PyTorch 5D tensors (`[Batch, 1, 128, 128, 128]`) ready for GPU computation.

---

## 1. End-to-End Data Transformation Flow

```mermaid
flowchart TD
    subgraph DISK["1. Disk Storage (Google Drive / Local SSD)"]
        NII["Preprocessed 3D Volumes<br/>{subject_id}_preprocessed_128.nii.gz<br/>(1,593 subjects)"]
        CSV_BASE["cohort_a_baseline_manifest.csv<br/>(1,623 subjects)"]
        JSON_SPLIT["cohort_a_splits.json<br/>(Train / Val / Test Partition Hashes)"]
        CSV_QC["preprocessing_manifest.csv<br/>(Preprocessing Attempt Statuses)"]
    end

    subgraph PREP_SCRIPT["2. Manifest Preparation (scripts/prepare_preprocessed_manifest.py)"]
        CSV_BASE & JSON_SPLIT & CSV_QC --> PREP_EXEC["Resolve Final Successful Preprocessing Attempt<br/>Filter Failures (13) & Missing Status (17)"]
        PREP_EXEC --> OUT_MAN["classifier_manifest.csv<br/>(1,593 Valid Baseline Scans)"]
        PREP_EXEC --> OUT_SPLIT["classifier_splits.json<br/>(Train: 1112 | Val: 236 | Test: 245)"]
    end

    subgraph AUDIT["3. Geometric & Quality Audit (src/dataset.py: audit_dataset)"]
        OUT_MAN & OUT_SPLIT & NII --> AUDIT_EXEC["Quality Control Verification:<br/>1. Shape: exactly [128, 128, 128]<br/>2. Spacing: [1.1926, 1.4749, 1.2387] +/- 0.03mm<br/>3. Orientation: RAS<br/>4. Spatial Units: mm<br/>5. Valid Affine Matrix (det > 1e-8)<br/>6. MNI152 Reference FOV Check: physical FOV extent tolerance <= 0.1mm"]
        AUDIT_EXEC --> AUDIT_JSON["data_audit.json<br/>(Audit passed=True; 1,348 Train+Val audited)"]
    end

    subgraph DATASET["4. MRIDataset Loading (src/dataset.py: MRIDataset)"]
        AUDIT_JSON --> ITEM["__getitem__(idx)"]
        ITEM --> LOAD_VOL["load_volume(path, config)<br/>nibabel.load() -> get_fdata(dtype=np.float32)"]
        LOAD_VOL --> NORM["Intensity Standardization<br/>(normalization: 'none', 'zscore', or 'minmax')<br/>Preserves Background Zeroes"]
        NORM --> TENSOR["Convert to Tensor & Add Channel Dim<br/>Shape: [1, 128, 128, 128]"]
    end

    subgraph AUG["5. Online 3D Augmentation (src/transforms.py: build_transform)"]
        TENSOR --> SPLIT_CHECK{"Split == 'train' and<br/>augmentation.enabled?"}
        
        SPLIT_CHECK -->|Yes: Training| TORCHIO["TorchIO RandomAffine<br/>- Rotation: +/- 5 degrees<br/>- Translation: +/- 2 mm<br/>- Scaling: +/- 3%"]
        TORCHIO --> BLUR["Gaussian Blurring<br/>- Sigma range: [0.0, 1.5]<br/>- Probability: 50%"]
        
        SPLIT_CHECK -->|No: Val / Test| IDENTITY["Identity Mapping<br/>(Deterministic Unaltered Volume)"]
    end

    subgraph LOADER["6. PyTorch DataLoader & Device Transfer (src/engine.py: make_loader)"]
        BLUR & IDENTITY --> SAMPLER{"balanced_sampler enabled?"}
        
        SAMPLER -->|Yes: Training| WEIGHTED_SAMPLER["WeightedRandomSampler<br/>sample_weight = 1.0 / count[label]<br/>replacement=True<br/>(Equalizes CN, EMCI, LMCI, AD)"]
        SAMPLER -->|No: Val / Test| SEQUENTIAL["Sequential Batching<br/>shuffle=False, drop_last=False"]
        
        WEIGHTED_SAMPLER & SEQUENTIAL --> COLL["Batch Collation & Memory Pinning<br/>(batch_size: 4, num_workers: 2, pin_memory=True)"]
        COLL --> GPU_XFER["Non-Blocking CUDA Transfer<br/>images.to('cuda', non_blocking=True)<br/>targets.to('cuda', non_blocking=True)"]
        GPU_XFER --> FINAL_TENSOR["Final GPU Batch Tensor<br/>images: [4, 1, 128, 128, 128] (fp32 / fp16)<br/>targets: [4] (int64: 0, 1, 2, 3)"]
    end

    classDef storage fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef step fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef augment fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef gpu fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    class DISK storage;
    class PREP_SCRIPT,AUDIT,DATASET step;
    class AUG augment;
    class LOADER gpu;
```

---

## 2. Voxel Geometry and Normalization Specifications

| Pipeline Stage | Voxel Dimension | Intensity Range | Channel Layout | Coordinate Space |
|:---|:---|:---|:---|:---|
| **Raw ADNI T1** | Variable ($256\times 256\times 176$) | Raw MR Units ($[0, 2500+]$) | 3D Volume | Native Scanner Space |
| **Preprocessed Scan** | **$128 \times 128 \times 128$** | Standardized Foreground (Zero Background) | 3D Volume | Standardized Grid (RAS, MNI152 FOV length) |
| **Post-Augmentation** | **$128 \times 128 \times 128$** | Standardized | 3D Volume | MNI152 Perturbed ($\pm 5^\circ, \pm 2\text{mm}$) |
| **Model Input Batch** | **$[4, 1, 128, 128, 128]$** | Standardized (fp16/fp32) | 5D Tensor ($B \times C \times D \times H \times W$) | GPU CUDA Memory |

---

## 3. Class Balance & Sampling Mechanics

The ADNI baseline cohort has inherent class imbalance reflecting real-world clinical recruitment:
- **CN (Cognitive Normal):** 425 training scans ($38.2\%$)
- **EMCI (Early MCI):** 310 training scans ($27.9\%$)
- **LMCI (Late MCI):** 215 training scans ($19.3\%$)
- **AD (Alzheimer's Disease):** 162 training scans ($14.6\%$)

```mermaid
pie title Training Cohort Class Distribution (1,112 Scans)
    "CN (425)": 425
    "EMCI (310)": 310
    "LMCI (215)": 215
    "AD (162)": 162
```

To prevent gradient dominance by the majority CN class without artificially weighting loss penalties twice:
- `training.balanced_sampler: true` $\rightarrow$ attaches `WeightedRandomSampler(weights, replacement=True)`.
- `training.weighted_loss: false` $\rightarrow$ uses standard CrossEntropyLoss ($w_c = 1.0$), avoiding double-reweighting.
- Each training epoch samples **1,112 draws with replacement**, yielding approximately equal class representation per epoch.
