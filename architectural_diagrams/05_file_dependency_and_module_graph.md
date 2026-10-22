# Module Dependencies & File Connection Graph

This document details the modular organization of the repository, illustrating how all Python source modules (`src/`), runnable CLI scripts (`scripts/`), configuration files (`configs/`), and test suites (`tests/`) depend on and communicate with each other.

---

## 1. Complete File-by-File Architecture & Dependency Graph

```mermaid
graph TD
    subgraph CONFIG_LAYER["Configuration Layer (configs/)"]
        BASE_CFG["base.yaml<br/>(Core Schema & Defaults)"]
        COLAB_MEDNET["colab_medicalnet_resnet18_128.yaml<br/>(Active Colab ResNet-18 Profile)"]
        BASE_CFG --> COLAB_MEDNET
    end

    subgraph UTILS_LAYER["Foundational Utilities & Constants"]
        CONST["src/constants.py<br/>- LABEL_TO_INDEX (CN:0..AD:3)<br/>- SPLIT_NAMES ('train','val','test')"]
        REPRO["src/reproducibility.py<br/>- set_seed()<br/>- collect_environment()<br/>- Warning filters"]
        METRICS["src/metrics.py<br/>- classification_metrics()<br/>- pairwise_metrics()<br/>- bootstrap_intervals()"]
        PLOTS["src/plots.py<br/>- plot_history()<br/>- plot_evaluation()"]
    end

    subgraph DATA_LAYER["Data Ingestion & Transformation Layer"]
        CONFIG_PY["src/config.py<br/>- load_config()<br/>- require_preprocessing()"]
        DATASET_PY["src/dataset.py<br/>- build_index()<br/>- load_volume()<br/>- audit_dataset()<br/>- MRIDataset"]
        TRANSFORMS_PY["src/transforms.py<br/>- build_transform()<br/>- TorchIO Affine & Blur"]
        
        CONST --> DATASET_PY
        CONFIG_PY --> DATASET_PY
        TRANSFORMS_PY --> DATASET_PY
    end

    subgraph MODEL_LAYER["Model & Weight Injection Layer"]
        MODEL_PY["src/model.py<br/>- ResNet18 (Production)<br/>- BasicBlock (Shortcut A)<br/>- TCAV Extraction Methods"]
        MEDNET_PY["src/medicalnet.py<br/>- load_medicalnet()<br/>- Weights compatibility report"]
        
        DATASET_PY -->|sha256_file| MEDNET_PY
        MODEL_PY --> MEDNET_PY
    end

    subgraph ENGINE_LAYER["Execution & Training Engine"]
        ENGINE_PY["src/engine.py<br/>- choose_device()<br/>- class_weights()<br/>- make_loader()<br/>- run_epoch() [with tqdm]"]
        
        DATASET_PY --> ENGINE_PY
        TRANSFORMS_PY --> ENGINE_PY
        METRICS --> ENGINE_PY
    end

    subgraph APP_LAYER["Application Entrypoints (src/ & scripts/)"]
        TRAIN_PY["src/train.py<br/>- train() main loop<br/>- checkpointing (best/last)"]
        EVAL_PY["src/evaluate.py<br/>- evaluate() on held-out test"]
        EXPORT_PY["src/export_activations.py<br/>- export_features() (.npy for CAV probing)"]
        
        TINY_SCRIPT["scripts/tiny_overfit.py<br/>- Sanity overfit check"]
        AUDIT_SCRIPT["scripts/validate_dataset.py<br/>- Standalone data audit"]
        PREP_SCRIPT["scripts/prepare_preprocessed_manifest.py<br/>- Manifest & Split generator"]
        INSPECT_SCRIPT["scripts/inspect_volume.py<br/>- Voxel inspection & slicing"]
    end

    %% Wiring into Application Layer
    COLAB_MEDNET -.-> CONFIG_PY
    CONFIG_PY --> TRAIN_PY
    DATASET_PY --> TRAIN_PY
    MODEL_PY --> TRAIN_PY
    MEDNET_PY --> TRAIN_PY
    ENGINE_PY --> TRAIN_PY
    REPRO --> TRAIN_PY
    PLOTS --> TRAIN_PY
    CONST --> TRAIN_PY

    TRAIN_PY --> EVAL_PY
    MODEL_PY --> EVAL_PY
    ENGINE_PY --> EVAL_PY
    PLOTS --> EVAL_PY

    MODEL_PY --> EXPORT_PY
    DATASET_PY --> EXPORT_PY
    ENGINE_PY --> EXPORT_PY

    CONFIG_PY --> TINY_SCRIPT
    ENGINE_PY --> TINY_SCRIPT
    MODEL_PY --> TINY_SCRIPT
    MEDNET_PY --> TINY_SCRIPT

    DATASET_PY --> AUDIT_SCRIPT
    CONFIG_PY --> AUDIT_SCRIPT

    classDef cfg fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef utils fill:#eceff1,stroke:#607d8b,stroke-width:2px;
    classDef data fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef model fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef engine fill:#ede7f6,stroke:#512da8,stroke-width:2px;
    classDef app fill:#fbe9e7,stroke:#d84315,stroke-width:2px;

    class CONFIG_LAYER cfg;
    class UTILS_LAYER utils;
    class DATA_LAYER data;
    class MODEL_LAYER model;
    class ENGINE_LAYER engine;
    class APP_LAYER app;
```

---

## 2. Source Module Responsibility Matrix

| File Path | Primary Responsibilities | Key Exported Functions / Classes |
|:---|:---|:---|
| [`src/constants.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/constants.py) | Immutable project constants, label mappings, and split identifiers | `LABEL_TO_INDEX`, `SPLIT_NAMES`, `REQUIRED_MANIFEST_COLUMNS` |
| [`src/config.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/config.py) | Recursive YAML inheritance (`extends`), validation rules, and schema checks | `load_config()`, `require_preprocessing()` |
| [`src/reproducibility.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/reproducibility.py) | Seed synchronization across Python, NumPy, CUDA; warning filters; hardware fingerprinting | `set_seed()`, `collect_environment()`, `save_environment()` |
| [`src/dataset.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/dataset.py) | NIfTI file verification, spatial affine auditing, intensity standardization, and PyTorch dataset | `build_index()`, `load_volume()`, `audit_dataset()`, `MRIDataset` |
| [`src/transforms.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/transforms.py) | Online 3D spatial transformations (TorchIO RandomAffine) and Gaussian blurring | `build_transform()` |
| [`src/model.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/model.py) | MedicalNet 3D ResNet-18, Shortcut A BasicBlock, and TCAV feature extractor interfaces | `ResNet18`, `BasicBlock`, `build_model()` |
| [`src/medicalnet.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/medicalnet.py) | MedicalNet Med3D weight loading, key mapping (`module.` stripping), and SHA-256 validation | `load_medicalnet()` |
| [`src/metrics.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/metrics.py) | Macro-F1, balanced accuracy, confusion matrices, bootstrap 95% confidence intervals | `classification_metrics()`, `pairwise_metrics()`, `bootstrap_intervals()` |
| [`src/engine.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/engine.py) | PyTorch DataLoader construction, class balancing samplers, AMP autocast, and training loop | `choose_device()`, `class_weights()`, `make_loader()`, `run_epoch()` |
| [`src/plots.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/plots.py) | Renders learning curve plots (`learning_curves.png`), ROC curves, and confusion matrix figures | `plot_history()`, `plot_evaluation()` |
| [`src/train.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/train.py) | Main driver for full training, early stopping, history logging (`history.csv`), and checkpointing (`best.pt`, `last.pt`) | `train()`, `save_checkpoint()`, `write_history_csv()`, `main()` |
| [`src/evaluate.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/evaluate.py) | Standalone evaluation of frozen checkpoints on held-out test data | `evaluate()`, `main()` |
| [`src/export_activations.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/export_activations.py) | Exports Layer 4 pooled features (`pooled_features.npy`) and optional spatial arrays (`.npy`) without gradients for offline CAV fitting | `export_features()`, `main()` |
| [`scripts/tiny_overfit.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/scripts/tiny_overfit.py) | Fast memorization sanity test on an 8-subject balanced subset | `main()` |
| [`scripts/validate_dataset.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/scripts/validate_dataset.py) | CLI utility to verify dataset geometry and affine alignment before training | `main()` |
| [`scripts/prepare_preprocessed_manifest.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/scripts/prepare_preprocessed_manifest.py) | Compiles raw cohort metadata and preprocessing statuses into final classifier manifests | `main()` |
