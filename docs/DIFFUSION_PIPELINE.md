# 3D latent diffusion and TCAV probe pipeline

Implements `RESEARCH_AND_DESIGN_BLUEPRINT.md` (sections 0-5, Appendix B). The blueprint sits at the
repository root, not under `docs/`.

| Step | Command | Module |
| --- | --- | --- |
| audit_inputs | `python -m src.diffusion.index_diffusion qc-index ...` then `audit --config configs/diffusion_ldm_128.yaml` | `index_diffusion.py` |
| atlas masks (once) | `python -m src.diffusion.build_atlas_masks ATLAS template/mni152.nii.gz SCAN OUT --atlas-sha256 H` | `build_atlas_masks.py` |
| train_ae | `python -m src.diffusion.train_stage1_ae --config ... --profile t4 [--preflight]` | `train_stage1_ae.py` |
| validate_ae | `... train_stage1_ae --validate-only` | `train_stage1_ae.py` |
| cache_latents | `python -m src.diffusion.cache_latents --config ...` | `cache_latents.py` |
| train_ldm | `python -m src.diffusion.train_stage2_ldm --config ... [--preflight]` | `train_stage2_ldm.py` |
| generate_probes | `python -m src.diffusion.generate_tcav_probes run --config ...` | `generate_tcav_probes.py`, `probe_qc.py` |
| export_probe_features | `... generate_tcav_probes export --classifier-config CFG`, then `python -m src.export_activations` | `generate_tcav_probes.py` |

## Contracts

* Input: `128^3` float32 NIfTI in the stored z-score domain (no min-max, no clipping) plus the
  `_mask.nii.gz` companion. The latent grid is `[4, 16, 16, 16]`.
* Classes: `CN=0`, `MCI=1` (EMCI/LMCI/MCI merged), `AD=2`; the CFG null index is `3`.
* Splits: the master subject split decides everything. Only `train` and `val` are indexed. Test
  scans are skipped before any path is resolved, opened or hashed, and the index, latent cache and probe
  anchors are re-checked against the 247-subject denylist.
* Today the index uses whichever scans have a `SUCCESS` QC row and files on disk (1,593 baseline scans).
  Extend the QC CSV with `--layout visit` rows as follow-up visits are preprocessed; nothing else changes.

## Honest limits

* Without a validated morphometry table (`data.morphometry_csv`, SynthSeg / DL+DiReCT measurements) the
  generator is class-only. Concept edits use an AD class context inside the atlas support, and acceptance uses a
  directional CSF-fraction proxy. Those probes are labeled `acceptance_level=exploratory` and carry
  `measurement_status=measurement_unavailable`. Pass an independent `measure_fn` to get `clinical`.
* `coarse_tissue_map` is a frozen intensity fallback for software QC, not a segmenter. The Frechet
  threshold and axial-slice FVD need the pinned feature extractors; only the Frechet distance and its
  real-real calibration are implemented here.
* GPU admission (`--preflight`) has only been exercised on CPU. Run it on each target profile first.
* Tests use tiny 32^3 models. Full-resolution shapes are verified on the meta device, not by training.
