"""Print NIfTI geometry; optionally save a three-plane montage for manual QC."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--output", help="Optional PNG montage")
    args = parser.parse_args()
    image = nib.load(args.path)
    data = image.get_fdata(dtype=np.float32)
    print(
        {
            "shape": image.shape,
            "spacing": image.header.get_zooms(),
            "orientation": nib.aff2axcodes(image.affine),
            "units": image.header.get_xyzt_units(),
            "finite": bool(np.isfinite(data).all()),
            "nonzero_fraction": float((data != 0).mean()),
        }
    )
    print("Affine:\n", image.affine)
    print("These properties do not prove skull stripping or anatomical registration.")
    if args.output:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = nib.as_closest_canonical(image).get_fdata(dtype=np.float32)
        if data.ndim != 3:
            raise ValueError("Expected a 3D image")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, ax in enumerate(axes):
            ax.imshow(
                np.rot90(np.take(data, data.shape[axis] // 2, axis=axis)), cmap="gray"
            )
            ax.set_title(("Sagittal", "Coronal", "Axial")[axis])
            ax.axis("off")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.output, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
