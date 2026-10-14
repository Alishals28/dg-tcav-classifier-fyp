"""Shared label and cohort definitions."""

LABEL_TO_INDEX = {"CN": 0, "EMCI": 1, "LMCI": 2, "AD": 3}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}
SPLIT_NAMES = ("train", "val", "test")
REQUIRED_MANIFEST_COLUMNS = {
    "subject_id",
    "image_id",
    "diagnosis",
    "phase",
}
