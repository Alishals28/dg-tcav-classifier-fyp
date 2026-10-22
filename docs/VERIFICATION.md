# Verification record

Historical checks on the classifier code ran on synthetic data in September 2026.
They do not establish Colab GPU behavior. No test suite was run as part of the
128-cube profile changes.

## Historical synthetic checks

The prior record reported 41 CPU tests passing, synthetic two-epoch CLI training,
resume equivalence on CPU, a synthetic tiny-overfit check, feature export and
metric/report checks. Those results covered an earlier code/configuration state;
rerun the suite after the current changes before relying on them.

The MedicalNet architecture comparison used upstream source commit
20f76aaab5cac8056eaf50b79ed97c09dbfbd3bd. For real-data model evaluation:
- A 30-epoch from-scratch run on real ADNI data is recorded at outputs/dg-tcav-classifier-full/ (pretrained: false), achieving validation accuracy ~0.4286, validation macro-F1 ~0.3328, and macro OVR AUC ~0.6186. (The 2-epoch sanity trial in outputs/dg-tcav-classifier-trial/ verifies pipeline mechanics, recording validation macro-F1 0.088).
- The production MedicalNet-pretrained run (pretrained: true, 3D ResNet-18) on Colab T4 was executed on the full 1,112-train / 236-validation cohort (configs/colab_medicalnet_resnet18_128.yaml), completing 43 epochs with early stopping (peak validation Macro-F1 of 0.3993 at Epoch 28).
- Held-out test evaluation on the frozen est.pt checkpoint across all 245 unseen test subjects achieved 41.22% accuracy and 0.3744 macro-F1 (	est_evaluation/).

## Current 128-cube input observations

The seven local volume files and companion masks report 128³ dimensions, spacing
approximately 1.192616×1.474926×1.238708 mm, millimetre spatial units, and RAS
orientation. The `template/mni152.nii.gz` template reports 207×256×215 dimensions,
0.737463 mm isotropic spacing, and RAS orientation. The template and output have
matching physical field-of-view lengths to within the stored header precision.

The output sform origin is (0, 0, 0), while the MNI template origin is approximately
(-75.763, -110.763, -71.763) mm. The output files therefore preserve the template
extent and voxel orientation but do not retain its world-coordinate origin. The
user has confirmed anatomical alignment by preprocessing QC. The classifier's
`same_fov` reference check validates direction and extent; it intentionally does
not claim that output affine coordinates equal MNI world coordinates.

For the seven volume arrays, nonzero voxels have mean near zero and standard
deviation approximately 0.998. This supports using `normalization: none` to avoid
normalizing them a second time. The classifier's companion-mask data are not used
for this statistic or as model input.

## Current handoff status and remaining checks

The baseline cohort manifest and subject-level split cover 1,623 subjects. The
completed preprocessing report has 1,894 attempt rows across 169 subjects with
retries. With the agreed rule that the last row per subject is authoritative, 1,593
are successful (1,112 train, 236 validation, 245 test), 13 have a final failure,
and 17 baseline subjects had unresolvable or missing status. The generated
classifier manifest (`classifier_manifest.csv`), split file (`classifier_splits.json`),
and summary (`preprocessing_summary.json`) are fully reconciled with this 1,593-subject
cohort. Training class counts are CN 425, EMCI 310, LMCI 215, AD 162.

Before training, put the source manifests, report, template, checkpoint, and scan
files in the documented Drive layout. Run the preparation command in Colab to
recreate classifier inputs, audit every successful image, inspect visual QC
samples, and confirm the MedicalNet loading report. Then run tiny-overfit and a
short Colab GPU trial; assess memory, epoch time, BatchNorm and I/O. No real-data
training or Colab GPU trial has yet been established.

Rerun the software test suite after this migration before relying on the code.
When additional scans finish preprocessing, regenerate classifier inputs and
start a new training run. Do not resume a checkpoint trained on a different scan
set. Synthetic outputs provide no estimate of diagnostic performance.
