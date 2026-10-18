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
| `model.py: ResNet18`, `ClassifierCNN` | Provides selectable four-class backbones and a shared feature/activation interface |
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

`configs/classifier_cnn_128.yaml` selects the alternative four-block dilated
`ClassifierCNN`:
four dilated Conv3d blocks with configurable InstanceNorm3d or BatchNorm3d, then a
flattened spatial representation, a 1024-dimensional `fc6` feature layer, and a
two-layer classifier head. Its flatten dimension is inferred on the meta device
from the configured input shape, avoiding an expensive real 3D dummy forward.
For the 128³ profile, Block 4 has shape [batch, 512, 3, 3, 3]. This architecture
does not use the MedicalNet checkpoint loader.

The supported official checkpoints are `resnet_18.pth` and
`resnet_18_23dataset.pth`, obtained from
[MedicalNet](https://github.com/Tencent/MedicalNet). They require width 64 and
shortcut A. The loader checks every backbone tensor and permits missing legacy
BatchNorm batch counters. Original head tensors are ignored; other missing or
incompatible tensors stop loading. The report records keys and a checkpoint hash.
Only trusted checkpoints should be supplied; loading uses `weights_only=True`.

## Loss and optimization

CrossEntropyLoss takes logits and applies log-softmax internally. Softmax is used
afterward for prediction probabilities.

The default profile uses class weights N_train / (4 × n_class_train), calculated
from training labels only. `ClassifierCNN` instead enables an inverse-frequency
`WeightedRandomSampler` and disables weighted loss, avoiding simultaneous
reweighting by both mechanisms. Weighted cross-entropy, when selected, divides by
the sum of target-class weights; `run_epoch` accumulates that denominator so epoch
loss is independent of how batches are grouped.

| Setting | Initial value |
|---|---|
| Default ResNet optimizer | AdamW; learning rate 1e-4, weight decay 1e-4 |
| ClassifierCNN optimizer | Adam; learning rate 1e-3, betas (0.9, 0.999) |
| Schedule | Cosine decay to 1e-6 over the configured maximum (50 default; 100 for ClassifierCNN) |
| Model selection | Highest validation macro-F1 |
| Early stopping | 10 epochs without improvement |
| Fine-tuning | All layers, including the selected normalization layers |
| Augmentation | Off for the initial comparison |

There is no warm-up or layer-freezing schedule. Small-batch BatchNorm behavior
needs checking during the real-data trial. Training uses one device.

`augmented.yaml` enables TorchIO Gaussian blur, small affine transforms in physical
coordinates, and mild Gaussian noise on training data only. The dedicated
ClassifierCNN profile disables added noise and instead enables balanced sampling.
No profile adds cropping, flipping, axis swaps, or elastic deformation.

`random_baseline.yaml` and `preprocessed_128.yaml` keep architecture, loss and
augmentation the same, changing only initialization. This isolates the effect of
pretraining. Changing both initialization and class weighting would confound
that comparison. This ResNet comparison does not describe the ClassifierCNN
profile, which has no pretrained checkpoint. The current CNN Colab workflow is
`colab_classifier_cnn_128.yaml`; it reads copied scans from the Colab VM disk and
writes outputs to Drive. The older `colab_preprocessed_128.yaml` and
`colab_local_128.yaml` remain for ResNet comparison runs.

## Training and checkpoints

Each epoch updates parameters on training images, then evaluates unaugmented
validation images with gradients disabled. Macro-F1 gives each class equal weight
in checkpoint selection; accuracy and per-class recall are also recorded.

`best.pt` stores the selected model. `last.pt` stores the latest model plus
optimizer, scheduler, AMP scaler, random states, data-loader state, history and
patience counter. Resume restores these at an epoch boundary and rejects changes
to configuration or data.

Python, NumPy, PyTorch and loader workers are seeded. Unsupported deterministic
CUDA operations produce warnings. Controlled CPU resume was tested; identical
results across hardware or software versions are not guaranteed.

Training checks test membership as metadata but never opens test images.
Use validation results to choose the configuration, then freeze it before the
final test evaluation. Image QC is separate from using predictions for tuning.

## Metrics and uncertainty

Evaluation reports accuracy, balanced accuracy, macro precision/recall/F1,
macro one-vs-rest AUC, per-class precision/recall/specificity/F1/AUC, and multiclass
Brier score. Predictions retain subject and image IDs for later analysis.

Macro-F1 includes all four labels with zero division set to zero. AUC is null when
positive or negative examples are absent. Balanced accuracy is null if any class
is missing; otherwise it equals macro recall. Per-phase results therefore need
to be read alongside their class supports.

The 95% intervals use class-stratified subject resampling for a fixed fitted model.
They condition on the observed class counts and do not capture variability from
retraining with different seeds.

AD-versus-CN and EMCI-versus-LMCI summaries report both the four-class decisions
on those subjects and forced decisions between the selected pair. Neither is a
separately trained binary classifier.

## Feature export and TCAV

Both live models expose spatial layer-4 activations and methods that continue to
the logits with gradients attached. ResNet exports its 512-dimensional pooled
features; ClassifierCNN exports the 1024-dimensional `fc6` features. Exported
NumPy arrays do not retain gradients. Subject rows keep arrays aligned with their
metadata.

For ResNet, there is an important limitation at the pooled bottleneck: for a
pooled vector z and a linear head, a class logit is wᵀz + b, so its gradient
with respect to z is w for every subject. The same issue occurs immediately
before global pooling. ClassifierCNN has a nonlinear classifier head after `fc6`,
but that alone does not establish a scientifically justified TCAV bottleneck.
A sign-based logit TCAV score still requires validation.

The TCAV stage still needs to choose a suitable nonlinear bottleneck or another
justified design, define concept/control examples, fit CAVs and test significance.
Feature export alone does not resolve those research decisions.
