# Classifier design notes

Follow the code in this order: `dataset.py`, `model.py`, `engine.py`,
`train.py`, then `evaluate.py`.

## Data and labels

The CSV provides scan identity and diagnosis. The split JSON assigns each subject
to training, validation or test. `build_index` checks the join before resolving
image paths. This keeps all experimental runs on the same partition.

Each scan becomes a float tensor of shape [batch, 1, 91, 109, 91]. The single
channel contains MRI intensity, and NIfTI spatial-axis order is retained
consistently. Class IDs are category indices, not numeric severity targets.

Normalization uses each image's nonzero foreground. It does not estimate
statistics across subjects. Geometry checks cannot establish that preprocessing
succeeded anatomically; those checks rely on the preprocessing team's QC.

## Network and pretrained weights

A residual block applies two convolutions and adds its input through a shortcut.
This provides a shorter gradient path through the network. When dimensions change,
shortcut A subsamples the input and pads channels with zeros. It has no learned
parameters, and the implementation keeps its gradients attached.

The backbone follows MedicalNet ResNet-18: two blocks per stage, shortcut A, and
dilation 2 and 4 in stages 3 and 4. Global average pooling replaces the segmentation
decoder, producing 512 features. A linear layer maps these to four logits.

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

Class weights are N_train / (4 × n_class_train), calculated from training labels
only. There is no additional weighted sampler. Weighted cross-entropy divides by
the sum of target-class weights; `run_epoch` accumulates that denominator so
epoch loss is independent of how batches are grouped.

| Setting | Initial value |
|---|---|
| Optimizer | AdamW; learning rate 1e-4, weight decay 1e-4 |
| Schedule | Cosine decay to 1e-6 over at most 50 epochs |
| Model selection | Highest validation macro-F1 |
| Early stopping | 10 epochs without improvement |
| Fine-tuning | All layers, including BatchNorm |
| Augmentation | Off for the initial comparison |

There is no warm-up or layer-freezing schedule. Small-batch BatchNorm behavior
needs checking during the real-data trial. Training uses one device.

`augmented.yaml` enables small TorchIO affine transforms in physical coordinates
and mild Gaussian noise, on training data only. It does not add flipping, cropping,
axis swaps, or elastic deformation.

`random_baseline.yaml` and `kaggle.yaml` keep architecture, loss and augmentation
the same, changing only initialization. This isolates the effect of pretraining.
Changing both initialization and class weighting would confound that comparison.

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

The live model exposes pooled features, spatial layer-4 activations and methods
that continue to the logits with gradients attached. Exported NumPy arrays do not
retain gradients. Subject rows keep the arrays aligned with their metadata.

There is an important limitation at this bottleneck: for a pooled vector z and
linear head, a class logit is wᵀz + b, so its gradient with respect to z is w for
every subject. The same issue occurs immediately before global pooling. A
sign-based logit TCAV score there can therefore be degenerate.

The TCAV stage still needs to choose a suitable nonlinear bottleneck or another
justified design, define concept/control examples, fit CAVs and test significance.
Feature export alone does not resolve those research decisions.
