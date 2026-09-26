# Test record

The initial checks below ran on 25 September 2026. They used artificial images;
the real ADNI dataset and distributed MedicalNet checkpoint have not been tested.

## Results

| Check | Result |
|---|---|
| Installation | Editable install succeeded; pip check found no broken requirements |
| Automated tests | 41 passed on CPU, none skipped |
| Static checks | Ruff, Python compilation and Git whitespace checks passed |
| Notebook | nbformat schema validation passed |
| Synthetic CLI training | Two epochs; checkpoints, history, predictions and plots written |
| Resume | Interrupted and uninterrupted two-epoch CPU runs produced identical model tensors; num_workers=0 |
| Tiny overfit | Eight synthetic training images reached 100% training accuracy in eval mode after 11 epochs |
| Full-width input | Forward/backward passed at [1,1,91,109,91]; finite stem gradients |
| Model output | Four logits, 512 pooled features; 32,988,228 parameters |
| Architecture comparison | Layer-4 output matched the upstream MedicalNet implementation with identical weights and a controlled 16³ input |
| Preprocessing requirement | Incomplete real-training configuration was rejected |

Integration tests also cover evaluation reports, figures, feature export,
checkpoint/config mismatches, and refusal to overwrite evaluation results.
Data tests cover partition overlap/coverage, invalid labels/IDs, missing or
ambiguous files, geometry, invalid intensities, normalization and augmentation.
Metric tests use known answers and check weighted loss across batch sizes.

The architecture comparison used MedicalNet source commit
`20f76aaab5cac8056eaf50b79ed97c09dbfbd3bd`. It confirms backbone behavior, not
successful loading of the actual pretrained file.

## Environment

Python 3.12.14; PyTorch 2.14.0+cpu; NumPy 2.3.5; pandas 2.2.3;
NiBabel 5.4.2; scikit-learn 1.8.0; TorchIO 0.23.1; PyYAML 6.0.3;
Matplotlib 3.10.8; pytest 8.4.2. No CUDA device was available.
Each training run records its own package versions and Git commit.

## Documentation revision — 26 September 2026

Ruff lint/format checks, Python compilation, notebook schema validation and Git
whitespace checks passed. All 13 edited Python files have identical syntax trees
after excluding docstrings, confirming that their executable logic is unchanged.
The notebook now clones the default branch containing the merged implementation.

The full test suite was not rerun for this revision because the current runtime
lacks the training dependencies, including PyTorch. The 41-test result above is
from the initial implementation checks.

## Remaining checks

1. Receive the processed Cohort A scans, exact reference grid and preprocessing QC.
2. Audit the actual manifest, original splits and scan/image-ID correspondence.
3. Load the trusted MedicalNet checkpoint and inspect its loading report.
4. Run the overfit check on a small set of model-ready training scans.
5. Run a short Kaggle GPU trial to check AMP, memory, BatchNorm, I/O and epoch time.
6. Train the planned experiments, select on validation and evaluate the frozen model.

The Kaggle notebook has not been executed inside Kaggle. GPU determinism and
cross-device/version resume remain unverified. Synthetic results provide no
estimate of disease-classification accuracy.
