"""Generate a tiny artificial cohort solely for software tests, never clinical results."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def make_synthetic(output, shape=(16, 16, 16)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    volumes = output / "volumes"
    volumes.mkdir()
    rng = np.random.default_rng(7)
    rows, splits = [], {"train": [], "val": [], "test": []}
    for split, count in (("train", 8), ("val", 4), ("test", 4)):
        for i in range(count):
            subject = f"synthetic_{len(rows):03d}"
            label = i % 4
            array = np.zeros(shape, dtype=np.float32)
            array[2:-2, 2:-2, 2:-2] = rng.uniform(
                0.1, 0.3, size=tuple(s - 4 for s in shape)
            )
            # An artificial spatial cue makes this useful for an engineering overfit test.
            start = 2 + label * 2
            array[start : start + 2, 4:12, 4:12] += 1
            image = nib.Nifti1Image(array, np.diag([2.0, 2.0, 2.0, 1.0]))
            image.header.set_xyzt_units("mm")
            nib.save(image, volumes / f"{subject}.nii.gz")
            rows.append(
                {
                    "subject_id": subject,
                    "image_id": str(len(rows) + 1),
                    "diagnosis": ("CN", "EMCI", "LMCI", "AD")[label],
                    "phase": "SYNTHETIC",
                }
            )
            splits[split].append(subject)
    pd.DataFrame(rows).to_csv(output / "manifest.csv", index=False)
    (output / "splits.json").write_text(json.dumps(splits, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/synthetic")
    args = parser.parse_args()
    print(make_synthetic(args.output))
