"""Shared label and cohort definitions."""

LABEL_TO_INDEX = {"CN": 0, "EMCI": 1, "LMCI": 2, "AD": 3}
SPLIT_NAMES = ("train", "val", "test")
REQUIRED_MANIFEST_COLUMNS = {
    "subject_id",
    "image_id",
    "diagnosis",
    "phase",
}
