# Explaining the classifier

Read in this order: config → dataset → model → engine → train → evaluate.

## Input and split handling

One 3D MRI per subject becomes [batch, 1, 91, 109, 91]. The single channel contains
MRI intensity. Spatial-axis order is preserved consistently from NIfTI; the code
does not secretly resize/register images. Class IDs are categorical, not numerical
severity targets. CSV supplies labels/scan identity; JSON supplies partition
membership. The join fails on ambiguity, missing files, duplicates, and overlap.

## Residual blocks and MedicalNet

A residual block learns a change to its input, then adds the input back. This
short path assists gradient flow. Shortcut A subsamples/pads channels without
learnable weights, preserving autograd. Later stages use MedicalNet's dilation.
Loading requires the full compatible backbone, not just a few matching layers.
The original segmentation head is replaced by a new four-class linear head.

## Logits, probabilities, and weighted loss

Four logits are unscaled scores. CrossEntropyLoss internally applies log-softmax;
do not softmax before loss. Softmax is used afterward for exported probabilities.
Class weights use training counts only and give smaller classes greater weight.
Weighted CE divides by the sum of target weights, so the epoch loss must accumulate
that denominator instead of averaging batch means. No weighted sampler doubles
the rebalancing. Macro-F1 gives every class equal importance in selection.

## One epoch

1. Train mode: augment training images, predict, compute loss, backpropagate, AdamW update.
2. Eval mode: unaugmented validation prediction with gradients disabled.
3. Select best.pt only if validation macro-F1 improves.
4. Step cosine schedule; save last.pt, history, and random states.
5. Stop at the cap or after the configured number of unimproved epochs.

best.pt is for model selection; last.pt is for resume. Resume restores optimizer,
scheduler, AMP scaler, RNG/loader state, patience counter, and history. Changed
configs/data are rejected. CPU resume is tested; cross-hardware identity is not promised.

## Preprocessing and test boundaries

The preprocessing teammate owns extraction/registration and visual QC. Headers
cannot prove either operation succeeded. The classifier checks the geometry and
requires an explicit handoff. Training opens no test images; test membership is
checked as metadata. Image integrity QC differs from using test predictions for
tuning. CLI confirmation reminds the researcher of this boundary but cannot
replace experimental discipline.

## What testing establishes

Artificial datasets check labels, gradients, checkpoints, metrics, and outputs.
Memorizing synthetic images is not evidence of disease-classification performance.
Actual MedicalNet weights, real-data QC/overfit, GPU behavior, and held-out results
still require verification. Compare random/pretrained with the same loss and
augmentation to isolate initialization; changing two factors confounds the result.

## TCAV limitations

Live features retain gradients; NumPy exports do not. A linear head gives constant
logit gradients at the final pooled vector, so it is not automatically a meaningful
standard sign-based TCAV bottleneck. The TCAV stage still chooses the bottleneck,
concept/control sets, CAV training scheme, and significance tests.
