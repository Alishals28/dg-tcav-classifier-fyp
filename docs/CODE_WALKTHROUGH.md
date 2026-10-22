# Classifier design notes

Follow the code in this order: `dataset.py`, `model.py`, `engine.py`,
`train.py`, then `evaluate.py`.

The main call path is:

```text
load_config → require_preprocessing → build_index → audit_dataset
            → make_loader → build_model → load_medicalnet
            → run_epoch(train) → run_epoch(validation) → save checkpoints
```

Evaluation rebuilds the requested split, verifies the saved checkpoint and input
identity, runs the same inference loop, and then writes metrics and figures.

## Where each decision is implemented

| File and function | Responsibility |
|---|---|
| `config.py: load_config` | Combines the base YAML with an experiment override and rejects invalid settings |
| `config.py: require_preprocessing` | Prevents a real-data run until the preprocessing grid and QC are confirmed |
| `dataset.py: build_index` | Joins manifest rows to the fixed split and resolves exactly one scan per subject |
| `dataset.py: load_volume` | Checks the NIfTI grid and applies the configured per-image normalization |
| `dataset.py: audit_dataset` | Reads every selected scan and records errors and file hashes |
| `model.py: ResNet18` | MedicalNet 3D ResNet-18 backbone with 4-class head and TCAV feature interface |
| `medicalnet.py: load_medicalnet` | Loads compatible backbone tensors and writes a detailed loading report |
| `engine.py: make_loader` | Creates seeded data loaders and enables augmentation only for training |
| `engine.py: run_epoch` | Performs either parameter updates or inference and returns subject-aligned predictions |
| `train.py: train` | Coordinates auditing, optimization, validation selection, early stopping and resume |
| `evaluate.py: evaluate` | Checks checkpoint/data identity and writes held-out metrics and plots |
| `export_activations.py` | Saves features in the same row order as their subject metadata |

These boundaries make the experimental choices visible in configuration while
keeping the training and evaluation calculations in shared functions.

## Data and labels

The CSV provides scan identity and diagnosis. The split JSON assigns each subject
to training, validation or test. `build_index` checks the join before resolving
image paths. This keeps all experimental runs on the same partition.

Each scan becomes a float tensor of shape [batch, 1, 128, 128, 128] for the
current real-data profile. Spacing, orientation and affine are checked against
the configured reference. The single channel contains the already-normalized
preprocessing output. Class IDs are category indices, not numeric severity
targets.

The real-data profile does not normalize again. Geometry checks cannot establish
that preprocessing succeeded anatomically; those checks rely on preprocessing QC.

## Network and pretrained weights

A residual block applies two convolutions and adds its input through a shortcut.
This provides a shorter gradient path through the network. When dimensions change,
shortcut A subsamples the input and pads channels with zeros. It has no learned
parameters, and the implementation keeps its gradients attached.

The default backbone follows MedicalNet ResNet-18: two blocks per stage, shortcut
A, and dilation 2 and 4 in stages 3 and 4. Global average pooling produces 512
features, followed by a linear four-class head.

