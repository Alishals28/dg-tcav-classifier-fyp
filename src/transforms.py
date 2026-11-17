"""Training-only 3D data augmentation:
- `conservative` (default): RandomBlur + RandomAffine + RandomNoise
- `yang2025` (authentic Yang et al., PLOS ONE 2025, Page 5 6-step TorchIO pipeline,
  matched to foreground Z-score normalization so train and val/test share identical intensity distributions):
  1. Intensity rescaling (dynamic range jitter on brain foreground)
  2. Random affine transformation (translation, rotation, scaling with zero background padding)
  3. Random flipping (`RandomFlip`)
  4. Random Gaussian noise or Gaussian blurring (`OneOf({RandomNoise, RandomBlur})`)
  5. Random 3D masking (`RandomMasking3D` cutout regions)
  6. Foreground Z-score normalization (`mean=0, std=1`, background preserved at `0.0`)
"""

import numpy as np
import torch


class ForegroundIntensityRescale3D:
    """Yang et al. (2025, Page 5): intensity rescaling, which adjusts the range of voxel values,
    while preserving 0.0 background outside the skull-stripped brain."""

    def __init__(self, scale_range=(0.90, 1.10), shift_range=(-0.10, 0.10), p: float = 0.8):
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.p = float(p)

    def __call__(self, subject_or_image):
        if np.random.rand() > self.p:
            return subject_or_image
        tensor = subject_or_image.data.clone()
        fg = tensor != 0
        if fg.any():
            alpha = float(np.random.uniform(*self.scale_range))
            beta = float(np.random.uniform(*self.shift_range))
            tensor[fg] = tensor[fg] * alpha + beta
        subject_or_image.set_data(tensor)
        return subject_or_image


class RandomMasking3D:
    """Yang et al. (2025, Page 5): random masking, which generates masked regions in the image."""

    def __init__(self, p: float = 0.5, min_fraction: float = 0.08, max_fraction: float = 0.18):
        self.p = float(p)
        self.min_fraction = float(min_fraction)
        self.max_fraction = float(max_fraction)

    def __call__(self, subject_or_image):
        if np.random.rand() > self.p:
            return subject_or_image
        tensor = subject_or_image.data.clone()
        _, d, h, w = tensor.shape
        fd = np.random.uniform(self.min_fraction, self.max_fraction)
        fh = np.random.uniform(self.min_fraction, self.max_fraction)
        fw = np.random.uniform(self.min_fraction, self.max_fraction)
        md, mh, mw = max(int(d * fd), 4), max(int(h * fh), 4), max(int(w * fw), 4)
        d0 = np.random.randint(0, max(d - md, 1))
        h0 = np.random.randint(0, max(h - mh, 1))
        w0 = np.random.randint(0, max(w - mw, 1))
        tensor[:, d0 : d0 + md, h0 : h0 + mh, w0 : w0 + mw] = 0.0
        subject_or_image.set_data(tensor)
        return subject_or_image


class ForegroundZNormalization3D:
    """Yang et al. (2025, Page 5): Z-normalization to mean=0 and std=1 on brain voxels,
    strictly matching `load_volume(..., normalization="zscore")` on validation/test scans."""

    def __call__(self, subject_or_image):
        tensor = subject_or_image.data.clone()
        fg = tensor.abs() > 1e-6
        tensor[~fg] = 0.0
        if fg.sum() > 2:
            vals = tensor[fg]
            std = float(vals.std().item())
            tensor[fg] = (vals - vals.mean()) / max(std, 1e-8)
        subject_or_image.set_data(tensor)
        return subject_or_image


def build_transform(config):
    if not config.get("enabled", False):
        return None
    import torchio as tio

    mode = config.get("mode", "conservative").lower()
    if mode == "yang2025":
        degrees = config.get("degrees", 10)
        translation = config.get("translation_mm", 5)
        scales = config.get("scales", 0.05)
        noise_std = config.get("noise_std", 0.05)
        blur_sigma = tuple(config.get("blur_sigma", (0.5, 1.5)))
        mask_prob = float(config.get("mask_prob", 0.5))
        flip_axes = tuple(config.get("flip_axes", (0,)))
        return tio.Compose(
            [
                ForegroundIntensityRescale3D(p=0.8),
                tio.RandomAffine(
                    scales=scales,
                    degrees=degrees,
                    translation=translation,
                    default_pad_value=0,
                    image_interpolation="linear",
                    p=0.8,
                ),
                tio.RandomFlip(axes=flip_axes, flip_probability=0.5),
                tio.OneOf(
                    {
                        tio.RandomNoise(mean=0.0, std=(0.0, max(float(noise_std), 0.01))): 0.5,
                        tio.RandomBlur(std=blur_sigma): 0.5,
                    },
                    p=0.5,
                ),
                RandomMasking3D(p=mask_prob),
                ForegroundZNormalization3D(),
            ]
        )

    return tio.Compose(
        [
            tio.RandomBlur(std=config.get("blur_sigma", (0.0, 1.5)), p=0.5),
            tio.RandomAffine(
                scales=config["scales"],
                degrees=config["degrees"],
                translation=config["translation_mm"],
                default_pad_value=0,
                image_interpolation="linear",
                p=0.5,
            ),
            *(
                [tio.RandomNoise(mean=0, std=(0, config["noise_std"]), p=0.25)]
                if config.get("noise_std", 0) > 0
                else []
            ),
        ]
    )
