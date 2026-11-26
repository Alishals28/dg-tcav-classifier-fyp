"""Generate exact-grid edit masks from the published CerebrA Table 1 labels.

Appendix B.1 of the blueprint. Run once per grid:

    python -m src.diffusion.build_atlas_masks ATLAS.nii.gz template/mni152.nii.gz \
        VERIFIED_SCAN.nii.gz NEW_OUTPUT_DIR --atlas-sha256 PINNED_HASH
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from .common import CONCEPTS, TEMPLATE_SHA256, sha256_file

REGIONS = {
    "HA": (48, 99),
    "ECT": (36, 87),
    "VE": (41, 92),
    "THD": (5, 56),
    "GCT": (47, 98, 31, 82, 10, 61, 1, 52, 9, 60, 28, 79),
    "AA": (19, 70),
}
assert tuple(REGIONS) == CONCEPTS


def verified_grid(template, scan, expected_shape=(128, 128, 128)):
    """Check the scan grid against the template and return A_mni = S_t @ inv(Q_t) @ A_disk."""
    qt, qtc = template.get_qform(coded=True)
    st, stc = template.get_sform(coded=True)
    qs, qsc = scan.get_qform(coded=True)
    ss, ssc = scan.get_sform(coded=True)
    if not all((qtc, stc, qsc, ssc)) or tuple(scan.shape) != tuple(expected_shape):
        raise ValueError("Missing qform/sform or unexpected scan dimensions")
    expected = qt.copy()
    expected[:3, :3] = qt[:3, :3] @ np.diag(np.asarray(template.shape) / np.asarray(expected_shape))
    if scan.header.get_xyzt_units()[0] != "mm" or not (
        np.allclose(qs, expected, atol=1e-5, rtol=0)
        and np.allclose(ss, expected, atol=1e-5, rtol=0)
    ):
        raise ValueError("Scan grid disagrees with the pinned preprocessing contract")
    return st @ np.linalg.solve(qt, ss)


def build_masks(atlas_path, template_path, scan_path, output, atlas_sha256,
                expected_shape=(128, 128, 128), regions=REGIONS):
    if sha256_file(template_path) != TEMPLATE_SHA256:
        raise ValueError("Wrong DG-TCAV template version")
    if atlas_sha256 is None or sha256_file(atlas_path) != atlas_sha256:
        raise ValueError("Wrong atlas digest (pin the CerebrA checksum with --atlas-sha256)")
    atlas, template, scan = map(nib.load, (atlas_path, template_path, scan_path))
    if atlas.ndim != 3 or nib.aff2axcodes(atlas.affine) != ("R", "A", "S"):
        raise ValueError("Supply the verified canonical RAS CerebrA label volume")
    if not np.isfinite(atlas.affine).all() or abs(np.linalg.det(atlas.affine[:3, :3])) < 1e-8:
        raise ValueError("Invalid atlas affine")
    data = atlas.get_fdata()
    if not np.isfinite(data).all() or not np.array_equal(data, np.rint(data)):
        raise ValueError("Atlas must contain discrete integer labels")
    mni_affine = verified_grid(template, scan, expected_shape)
    on_grid = resample_from_to(atlas, (scan.shape, mni_affine), order=0).get_fdata()
    checked = {label for ids in regions.values() for label in ids}
    missing = checked - set(np.unique(on_grid).astype(int))
    if missing:
        raise ValueError(f"Atlas labels absent on the target grid: {sorted(missing)}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for region, ids in regions.items():
        mask = np.isin(on_grid, ids).astype(np.uint8)
        header = scan.header.copy()
        header.set_data_dtype(np.uint8)
        image = nib.Nifti1Image(mask, scan.affine, header)
        image.set_qform(scan.get_qform(), code=int(scan.header["qform_code"]))
        image.set_sform(scan.get_sform(), code=int(scan.header["sform_code"]))
        nib.save(image, output / f"{region}_support.nii.gz")
        counts[region] = int(mask.sum())
    metadata = {
        "atlas_sha256": atlas_sha256,
        "atlas_space": "MNI-ICBM152 nonlinear symmetric 2009c",
        "template_sha256": TEMPLATE_SHA256,
        "scan_grid_to_mni_world": mni_affine.tolist(),
        "scan_grid_internal_affine": scan.affine.tolist(),
        "labels": {k: list(v) for k, v in regions.items()},
        "voxels": counts,
        "mask_sha256": {r: sha256_file(output / f"{r}_support.nii.gz") for r in regions},
    }
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_support_masks(directory, expected_shape=(128, 128, 128)):
    """Load the six atlas supports as boolean arrays keyed by concept."""
    directory = Path(directory)
    masks = {}
    for region in CONCEPTS:
        path = directory / f"{region}_support.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing atlas support: {path}")
        array = np.asarray(nib.load(str(path)).dataobj) > 0
        if array.shape != tuple(expected_shape) or not array.any():
            raise ValueError(f"Atlas support {region} is empty or has the wrong shape")
        masks[region] = array
    return masks


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("atlas")
    p.add_argument("template")
    p.add_argument("verified_scan")
    p.add_argument("output_directory")
    p.add_argument("--atlas-sha256", required=True)
    args = p.parse_args()
    meta = build_masks(args.atlas, args.template, args.verified_scan,
                       args.output_directory, args.atlas_sha256)
    print(json.dumps(meta["voxels"], indent=2))


if __name__ == "__main__":
    main()
