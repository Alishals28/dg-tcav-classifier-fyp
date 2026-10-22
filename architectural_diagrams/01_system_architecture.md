# System Architecture Diagram

This document illustrates the end-to-end system architecture of the **DG-TCAV 3D MRI Classifier Subsystem**, its integration within the broader DG-TCAV (Diffusion-Generated Testing with Concept Activation Vectors) research framework, and its hardware deployment topology across local workstations, Google Colab GPU runners, and Google Drive storage.

---

## 1. End-to-End System Context Diagram

The classifier serves as the foundational **evaluation backbone** for the DG-TCAV project. It classifies 3D baseline brain MRI scans into four clinical stages: **Cognitive Normal (CN)**, **Early Mild Cognitive Impairment (EMCI)**, **Late Mild Cognitive Impairment (LMCI)**, and **Alzheimer's Disease (AD)**. It also exposes a spatial feature bottleneck at Layer 4 for extracting activations used to construct Concept Activation Vectors (CAVs).

```mermaid
flowchart TD
    subgraph S1["1. Data Acquisition & Cohort Extraction (External Upstream)"]
        ADNI["ADNI Database (ADNI 1, 2, 3, GO)"] --> RAW["Raw T1-weighted Structural MRI (MPRAGE / IR-SPGR)"]
        RAW --> BASE["Cohort A Baseline Filter<br/>(Exactly 1 baseline scan per subject, Zero patient leakage)"]
        BASE --> MAN["cohort_a_baseline_manifest.csv<br/>(1,623 candidate subjects)"]
    end

    subgraph S2["2. Standardized Neuroimaging Preprocessing (External Upstream Pipeline)"]
        MAN --> PREP["External 3D Preprocessing Pipeline<br/>(N4 Bias Field Correction + DeepBrain Skull Stripping)"]
        PREP --> REG["Affine Registration to MNI152 Bounding Box<br/>(Uniform 128³ Spacing & RAS Orientation)"]
        REG --> OUT128["Prepared 128x128x128 Standardized Scans<br/>(*_preprocessed_128.nii.gz, 1,593 subjects)"]
    end

    subgraph S3["3. DG-TCAV Classifier Subsystem (This Repository: dg-tcav-classifier-fyp)"]
        OUT128 --> AUDIT["Quality Control & Spatial Audit (src/dataset.py)<br/>Verifies 128³ shape, spacing, & template FOV extent"]
        AUDIT --> SPLIT["Stratified Subject-Level Partitions<br/>(Train: 1112 | Val: 236 | Test: 245)"]
        SPLIT --> LOADER["MRIDataset & Data Engine<br/>(Balanced Sampling + Online 3D Augmentation)"]
        
        MEDNET["MedicalNet Pretrained Weights<br/>(Med3D 23-dataset resnet_18_23dataset.pth)"] --> MODEL
        LOADER --> MODEL["3D ResNet-18 Classifier<br/>(Full 64-channel width, AdaptiveAvgPool3d)"]
        
        MODEL --> HEAD["Linear Classification Head<br/>(512 -> 4 Classes: CN, EMCI, LMCI, AD)"]
        MODEL --> METH["get_layer4_activations(x) Method<br/>[Batch, 512, 16, 16, 16] Live Autograd Tensor"]
    end

    subgraph S4["4. Downstream DG-TCAV Explanations (External Downstream FYP Modules)"]
        METH --> EXPORT["src/export_activations.py: export_features()<br/>Saves detached .npy arrays for linear CAV probes"]
        METH -.->|"Live autograd graph (forward_from_activations)"| GRAD["Directional Derivatives (TCAV)"]
        
        DIFF["Diffusion Model (DG Pipeline)<br/>(Generates synthetic concept MRI patches)"] --> CONCEPTS["AD Clinical Concepts<br/>(Hippocampal Atrophy, Ventricle Enlargement, etc.)"]
        
        CONCEPTS --> CAV["CAV Generator (Linear SVM / Probe)<br/>(Learns vector v_C orthogonal to concept boundary)"]
        EXPORT --> CAV
        CAV & GRAD --> SCORE["TCAV Score Calculation<br/>(Directional derivative S_C,k across disease cohorts)"]
    end

    classDef primary fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef model fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef downstream fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    class ADNI,RAW,BASE,MAN,PREP,REG,OUT128 primary;
    class AUDIT,SPLIT,LOADER,MEDNET,MODEL,HEAD,METH model;
    class EXPORT,GRAD,DIFF,CONCEPTS,CAV,SCORE downstream;
```

---

## 2. Infrastructure & Deployment Topology

The diagram below visualizes how code, weights, dataset volumes, and training artifacts flow between GitHub, the Google Colab GPU compute environment, and persistent Google Drive storage.

```mermaid
graph TB
    subgraph GITHUB["GitHub Repository (Alishals28/dg-tcav-classifier-fyp)"]
        CODE["Source Code (src/, scripts/, configs/)"]
        TEMPL["MNI152 Reference Template (template/mni152.nii.gz)"]
    end

    subgraph HF["Hugging Face Model Hub"]
        WEIGHTS["Tencent MedicalNet ResNet-18 Weights<br/>(resnet_18_23dataset.pth, 126 MB)"]
    end

    subgraph COLAB["Google Colab Runtime (T4 / V100 GPU)"]
        subgraph VRAM["GPU VRAM (15 GB)"]
            GPU_MODEL["ResNet-18 Model Parameters (33.0M / 32,988,228 params)"]
            GPU_BATCH["Batched 3D Volumes [4, 1, 128, 128, 128] (AMP fp16)"]
            GPU_GRAD["Scaled Gradients & AdamW Momentum"]
        end
        
        subgraph VM_DISK["Fast Local VM SSD (/content)"]
            SSD_SCANS["Cached 3D Volumes (/content/preprocessed_128)<br/>1,593 cohort volumes (~11 GB)"]
            SSD_MANIFESTS["Active Manifests (data/preprocessed_128_classifier_inputs)"]
            SSD_ENV["Virtual Environment (PyTorch 2.x, CUDA GPU Runtime, TorchIO)"]
        end
    end

    subgraph GDRIVE["Google Drive Persistent Storage (/content/drive/MyDrive)"]
        DRIVE_RAW["Preprocessed Master Archive (preprocessed_128/*.nii.gz)"]
        DRIVE_CKPT["Pretrained Backbone Weights (resnet_18_23dataset.pth)"]
        
        subgraph DRIVE_OUT["Experiment Runs (/classifier_outputs)"]
            OUT_BEST["best.pt (Highest Validation Macro-F1 Checkpoint)"]
            OUT_LAST["last.pt (Resume Checkpoint with RNG & Optimizer state)"]
            OUT_HIST["history.csv (rewritten each epoch)<br/>learning_curves.png (plotted at completion)"]
            OUT_AUDIT["data_audit.json & environment.json"]
            OUT_ACTS["features/ (Exported .npy feature arrays & labels)"]
        end
    end

    %% Interactions
    CODE -.->|"!git clone / !git pull"| SSD_ENV
    TEMPL -.-> SSD_MANIFESTS
    WEIGHTS -.->|"!wget -c directly to Drive"| DRIVE_CKPT
    DRIVE_CKPT -->|"Loaded at epoch 0"| GPU_MODEL
    
    DRIVE_RAW -->|"Multi-threaded copy (ThreadPoolExecutor)"| SSD_SCANS
    SSD_SCANS -->|"Multi-worker prefetch (num_workers=2)"| GPU_BATCH
    
    GPU_MODEL --> GPU_GRAD
    GPU_GRAD --> GPU_MODEL
    
    GPU_MODEL -->|"save_checkpoint(run_dir / 'best.pt')"| OUT_BEST
    GPU_MODEL -->|"save_checkpoint(run_dir / 'last.pt')"| OUT_LAST
    COLAB -->|"Update history each epoch"| OUT_HIST
```

---

## 3. Strict Patient Isolation and Cohort Partition Architecture

To guarantee **zero patient leakage** in clinical validation, all partitioning is performed strictly at the **Subject ID** level before any volumes are read. Exactly one baseline scan per subject is admitted into the scientific study.

```mermaid
flowchart LR
    A["Raw ADNI Cohort Baseline<br/>1,623 Unique Subjects<br/>(cohort_a_baseline_manifest.csv)"] --> B{"Preprocessing & QC Audit<br/>(13 failed, 17 missing status)"}
    
    B --> C["Preprocessed Valid Cohort<br/>1,593 Standardized 3D Volumes<br/>(classifier_manifest.csv)"]
    
    C --> D{"Subject-Level Stratified Hash Split"}
    
    D -->|70% Stratified| E["Train Split<br/>(1,112 Subjects)<br/>CN: 425 | EMCI: 310<br/>LMCI: 215 | AD: 162"]
    D -->|15% Stratified| F["Validation Split<br/>(236 Subjects)<br/>CN: 90 | EMCI: 66<br/>LMCI: 46 | AD: 34"]
    D -->|15% Stratified| G["Held-Out Test Split<br/>(245 Subjects)<br/>CN: 93 | EMCI: 69<br/>LMCI: 47 | AD: 36"]

    E -->|"Used for Parameter Updates"| TR_LOOP["Training Engine<br/>(src/train.py)"]
    F -->|"Used for Checkpoint Selection (best.pt)"| VAL_LOOP["Validation Loop<br/>(Selection Metric: Macro-F1)"]
    G -->|"STRICTLY ISOLATED until final frozen evaluation"| TEST_LOOP["Test Evaluator<br/>(src/evaluate.py --confirm-final-test)"]

    classDef split fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    class E,F,G split;
```
