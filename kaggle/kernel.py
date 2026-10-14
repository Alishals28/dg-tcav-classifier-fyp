"""Kaggle launcher: clone the repository and train on the attached scan dataset.

STAGE "trial" runs the dataset audit, the tiny overfit check and a 2-epoch trial.
STAGE "full" runs the planned training. scripts/push_kaggle_kernel.sh sets STAGE
when pushing. Everything under /kaggle/working/outputs is saved as kernel output.
"""

import subprocess
import sys
from pathlib import Path

STAGE = "trial"
REPO = "https://github.com/Alishals28/dg-tcav-classifier-fyp.git"
REF = "main"
REPO_DIR = Path("/tmp/dg-tcav-classifier-fyp")
OUTPUT_DIR = Path("/kaggle/working/outputs")


def run(command):
    print(f"$ {command}", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=REPO_DIR)


def find_scans():
    """Return the scan folder and file pattern; Kaggle may store .nii.gz uploads as .nii."""
    scans = sorted(Path("/kaggle/input").rglob("*_preprocessed_128.nii*"))
    if not scans:
        sys.exit("No *_preprocessed_128.nii[.gz] under /kaggle/input; attach the scan dataset")
    folders = {path.parent for path in scans}
    suffixes = {path.name.split("_preprocessed_128", 1)[1] for path in scans}
    if len(folders) != 1 or len(suffixes) != 1:
        sys.exit(f"Expected one folder and one extension: {sorted(folders)} {suffixes}")
    print(f"Found {len(scans)} scans in {next(iter(folders))}", flush=True)
    return folders.pop(), "{subject_id}_preprocessed_128" + suffixes.pop()


subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", REF, REPO, str(REPO_DIR)], check=True
)
run("git rev-parse HEAD")
run("pip install -q 'torchio>=0.20,<1' 'nibabel>=5.2,<6' && pip install -q -e . --no-deps")
run("nvidia-smi || true")
run("python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\"")

config = REPO_DIR / "configs" / "kaggle_run.yaml"
volume_dir, file_pattern = find_scans()
config.write_text(
    "extends: kaggle_128.yaml\n\ndata:\n"
    f"  volume_dir: {volume_dir}\n  file_pattern: \"{file_pattern}\"\n"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if STAGE == "trial":
    run(f"python -m scripts.validate_dataset --config {config} --output {OUTPUT_DIR}/dataset_audit.json")
    run(f"python -m scripts.tiny_overfit --config {config} --output {OUTPUT_DIR}/tiny_overfit")
    run(f"python -m src.train --config {config} --stop-after-epoch 2")
elif STAGE == "full":
    run(f"python -m src.train --config {config}")
else:
    sys.exit(f"Unknown STAGE: {STAGE}")
