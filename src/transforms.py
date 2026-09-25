"""Conservative training-only augmentation in physical image coordinates."""


def build_transform(config):
    if not config.get("enabled", False):
        return None
    import torchio as tio

    return tio.Compose(
        [
            tio.RandomAffine(
                scales=config["scales"],
                degrees=config["degrees"],
                translation=config["translation_mm"],
                default_pad_value=0,
                image_interpolation="linear",
                p=0.5,
            ),
            tio.RandomNoise(mean=0, std=(0, config["noise_std"]), p=0.25),
        ]
    )
