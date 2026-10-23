# Phase 9 Checklist — Explanations and Clinical Interpretation

**Goal:** analyze model behavior responsibly after predictive performance and
data validity are established; do not present explanation visuals as clinical
validation or causal proof.

**Current status:** **NOT STARTED.** Begin only after stable,
leakage-aware evaluation. Attribution outputs are supplementary diagnostics.

## 9.1 Establish explanation validity

- [ ] Confirm input, activation, and attribution tensors use the verified
  spatial grid and orientation.
- [ ] Generate Grad-CAM or other attribution maps for correctly and incorrectly
  classified subjects, covering all classes and relevant subgroups.
- [ ] Check map stability under small, plausible input perturbations.
- [ ] Perform model-randomization/sanity checks to verify maps depend on learned
  parameters and labels.
- [ ] Compare maps with blinded reviewers and document the protocol.
- [ ] Do not claim an anatomical biomarker is learned merely because a map
  visually overlaps an expected region.

## 9.2 Test shortcut dependence

- [ ] Evaluate sensitivity to scanner/site/phase and acquisition metadata.
- [ ] Inspect skull-stripping boundaries, background, registration artifacts,
  padding, and other non-anatomical cues.
- [ ] Where feasible, use controlled perturbation/ablation tests and compare
  development performance.
- [ ] Examine demographic proxies only under an appropriate, preplanned
  analysis and governance.
- [ ] Report limitations and avoid causal interpretations.

## 9.3 Perform TCAV only as a defined analysis

- [ ] Define concepts, concept examples, controls, and hypotheses before
  scoring.
- [ ] Verify concept examples are representative and do not encode labels or
  acquisition shortcuts inadvertently.
- [ ] Implement the actual TCAV scoring procedure using the required live
  model gradients; saved `.npy` feature exports alone are insufficient.
- [ ] Include negative/random concepts and appropriate statistical testing.
- [ ] Assess concept sensitivity across subjects and seeds.
- [ ] Report uncertainty, failures, and sensitivity to concept construction.

## Exit gate

- [ ] Explanation checks and controls are documented and reproducible.
- [ ] Conclusions are stated as model-behavior evidence, not clinical
  validation, causal neuroanatomy, or proof of disease mechanism.
