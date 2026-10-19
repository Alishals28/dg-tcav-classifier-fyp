"""Experiment runner: launch seeds on Kaggle, collect results, write metrics, figures and LOG.md.

Usage (from the repository root):
    python experiments/run.py launch 001_resnet18_b8 --seeds 1 2 3
    python experiments/run.py wait 001_resnet18_b8
    python experiments/run.py collect 001_resnet18_b8 --baseline 000_name
    python experiments/run.py summarize

Each experiment lives in experiments/NNN_<slug>/ with a config.yaml that extends a base
config. The config and code must be committed and pushed: the Kaggle kernel clones that
exact commit. Raw Kaggle outputs (subject IDs, checkpoints) go to outputs/exp_runs/, which
is gitignored; only aggregate metrics and figures are written to the experiment folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR: Path = REPO_ROOT / "experiments"
RUNS_DIR: Path = REPO_ROOT / "outputs" / "exp_runs"
LOG_PATH: Path = EXPERIMENTS_DIR / "LOG.md"
REPO_URL: str = "https://github.com/Alishals28/dg-tcav-classifier-fyp.git"
KAGGLE_OWNER: str = "khizerinam08"
SCAN_DATASET: str = "khizerinam08/dg-tcav-preprocessed-128"
WEIGHTS_DATASET: str = "khizerinam08/medicalnet-resnet18-23dataset"
KAGGLE_CLI: str = str(Path.home() / ".venvs" / "kaggle" / "bin" / "kaggle")
CLASS_NAMES: list[str] = ["CN", "EMCI", "LMCI", "AD"]
EXPECTED_SPLITS_SHA256: str = "2dcf59474671623e97f9a27e03534e590a003a3285d23f486a4c5006fc9be1a0"
EXPECTED_MANIFEST_SHA256: str = "44d9c943a40ee75198b7fbdc777215b5242876dabb029b207ec496f39012376c"
EXPECTED_CLASS_COUNTS: dict[str, dict[str, int]] = {
    "train": {"CN": 215, "EMCI": 131, "LMCI": 112, "AD": 77},
    "val": {"CN": 49, "EMCI": 24, "LMCI": 20, "AD": 19},
    "test": {"CN": 47, "EMCI": 31, "LMCI": 24, "AD": 19},
}
TABLE_START: str = "<!-- table:start -->"
TABLE_END: str = "<!-- table:end -->"

BLUE: str = "#2a78d6"
GREY: str = "#b9b8b0"
INK: str = "#0b0b0b"
INK2: str = "#52514e"
GRID: str = "#e6e5e0"
REF: str = "#8a8984"
SEED_COLOURS: list[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100"]


# ---------------------------------------------------------------- shell helpers


def run_command(command: list[str], env: dict[str, str] | None = None, check: bool = True) -> str:
    result = subprocess.run(command, capture_output=True, text=True, env=env, cwd=REPO_ROOT)
    if check and result.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(command)}\n{result.stdout}\n{result.stderr}")
    return result.stdout + result.stderr


def kaggle_env() -> dict[str, str]:
    token = os.environ.get("KAGGLE_KHIZER_API_KEY")
    if not token:
        raise SystemExit("KAGGLE_KHIZER_API_KEY is not set")
    env = dict(os.environ)
    env.pop("KAGGLE_API_KEY", None)
    env["KAGGLE_CONFIG_DIR"] = tempfile.mkdtemp()
    env["KAGGLE_API_TOKEN"] = token
    return env


def kaggle(arguments: list[str], check: bool = True) -> str:
    return run_command([KAGGLE_CLI, *arguments], env=kaggle_env(), check=check)


def experiment_dir(name: str) -> Path:
    path = EXPERIMENTS_DIR / name
    if not (path / "config.yaml").exists():
        raise SystemExit(f"Missing {path / 'config.yaml'}")
    return path


def kernel_slug(name: str) -> str:
    return "dgtcav-exp-" + re.sub(r"[^a-z0-9-]", "-", name.lower())[:40].strip("-")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- split check


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_check(config_path: Path) -> dict[str, Any]:
    """Verify the experiment uses the fixed manifest and splits; abort if not."""
    sys.path.insert(0, str(REPO_ROOT))
    import pandas as pd

    from src.config import load_config

    config = load_config(config_path)
    manifest_path = REPO_ROOT / config["data"]["manifest_path"]
    splits_path = REPO_ROOT / config["data"]["splits_path"]
    splits = read_json(splits_path)
    manifest = pd.read_csv(manifest_path)
    diagnosis = dict(zip(manifest.subject_id, manifest.diagnosis))
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        counts[split] = {name: 0 for name in CLASS_NAMES}
        for subject in splits[split]:
            counts[split][diagnosis[subject]] += 1
    check = {
        "manifest_path": config["data"]["manifest_path"],
        "splits_path": config["data"]["splits_path"],
        "manifest_sha256": sha256_of(manifest_path),
        "splits_sha256": sha256_of(splits_path),
        "class_counts": counts,
    }
    check["passed"] = (
        check["manifest_sha256"] == EXPECTED_MANIFEST_SHA256
        and check["splits_sha256"] == EXPECTED_SPLITS_SHA256
        and counts == EXPECTED_CLASS_COUNTS
    )
    return check


# ---------------------------------------------------------------- kernel source


def kernel_source(name: str, commit: str, seeds: list[int]) -> str:
    """Python run on Kaggle: clone the commit, then run each seed on a free GPU."""
    return f'''"""Experiment {name}: seeds {seeds} at commit {commit}."""
import os, subprocess, sys, time
from pathlib import Path

REPO = "{REPO_URL}"
COMMIT = "{commit}"
EXPERIMENT = "{name}"
SEEDS = {seeds}
REPO_DIR = Path("/tmp/repo")
WORK = Path("/kaggle/working")

subprocess.run(["git", "clone", "--quiet", REPO, str(REPO_DIR)], check=True)
subprocess.run(["git", "checkout", "--quiet", COMMIT], check=True, cwd=REPO_DIR)
subprocess.run("pip install -q 'torchio>=0.20,<1' 'nibabel>=5.2,<6' && pip install -q -e . --no-deps", shell=True, check=True, cwd=REPO_DIR)
gpus = [line for line in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if line.startswith("GPU")]
print("GPUs:", gpus, flush=True)
if not gpus:
    sys.exit("No GPU available")

scans = sorted(Path("/kaggle/input").rglob("*_preprocessed_128.nii*"))
folders = {{p.parent for p in scans}}
suffixes = {{p.name.split("_preprocessed_128", 1)[1] for p in scans}}
if len(scans) != 768 or len(folders) != 1 or len(suffixes) != 1:
    sys.exit(f"Unexpected scan layout: {{len(scans)}} files, {{folders}}, {{suffixes}}")
volume_dir = folders.pop()
file_pattern = "{{subject_id}}_preprocessed_128" + suffixes.pop()
weights = sorted(Path("/kaggle/input").rglob("resnet_18_23dataset.pth"))

experiment_dir = REPO_DIR / "experiments" / EXPERIMENT
jobs = []
for seed in SEEDS:
    config = experiment_dir / f"kaggle_seed{{seed}}.yaml"
    text = (
        "extends: config.yaml\\n"
        f"experiment:\\n  seed: {{seed}}\\n  output_dir: {{WORK / ('seed' + str(seed))}}\\n"
        f"data:\\n  volume_dir: {{volume_dir}}\\n  file_pattern: \\"{{file_pattern}}\\"\\n"
    )
    if weights:
        text += f"model:\\n  pretrained_checkpoint: {{weights[0]}}\\n"
    config.write_text(text)
    jobs.append((seed, config))

env = dict(os.environ, TQDM_DISABLE="1", PYTHONUNBUFFERED="1")
running = {{}}
failed = []
started = time.time()
while jobs or running:
    free = [g for g in range(len(gpus)) if g not in running]
    while jobs and free:
        seed, config = jobs.pop(0)
        gpu = free.pop(0)
        log = open(WORK / f"seed{{seed}}.log", "w")
        process = subprocess.Popen(
            [sys.executable, "-m", "src.train", "--config", str(config)],
            cwd=REPO_DIR, stdout=log, stderr=subprocess.STDOUT,
            env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
        )
        running[gpu] = (seed, process, log)
        print(f"[{{time.time() - started:7.0f}}s] seed {{seed}} started on GPU {{gpu}}", flush=True)
    time.sleep(20)
    for gpu, (seed, process, log) in list(running.items()):
        if process.poll() is not None:
            log.close()
            del running[gpu]
            print(f"[{{time.time() - started:7.0f}}s] seed {{seed}} finished with code {{process.returncode}}", flush=True)
            if process.returncode != 0:
                failed.append(seed)

for path in WORK.rglob("last.pt"):
    path.unlink()
for seed in SEEDS:
    print(f"===== seed {{seed}} log tail =====")
    print("".join(open(WORK / f"seed{{seed}}.log").readlines()[-15:]), flush=True)
print("FAILED SEEDS:", failed, flush=True)
'''


# ---------------------------------------------------------------- launch / wait / collect


def launch(name: str, seeds: list[int], machine: str) -> None:
    path = experiment_dir(name)
    status = run_command(["git", "status", "--porcelain", "--", str(path), "src", "configs"])
    if status.strip():
        raise SystemExit(f"Commit the experiment config and code first:\n{status}")
    commit = run_command(["git", "rev-parse", "HEAD"]).strip()
    branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    if branch == "main":
        raise SystemExit("Experiments never run from main")
    remote = run_command(["git", "ls-remote", "origin", f"refs/heads/{branch}"]).split()
    if not remote or remote[0] != commit:
        raise SystemExit(f"Push {branch} first: Kaggle clones commit {commit} from GitHub")
    check = split_check(path / "config.yaml")
    write_json(path / "splits_check.json", check)
    if not check["passed"]:
        raise SystemExit(f"Split check failed, aborting: {json.dumps(check, indent=2)}")
    slug = kernel_slug(name)
    build = Path(tempfile.mkdtemp())
    (build / "kernel.py").write_text(kernel_source(name, commit, seeds))
    metadata = {
        "id": f"{KAGGLE_OWNER}/{slug}",
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": machine,
        "dataset_sources": [SCAN_DATASET, WEIGHTS_DATASET],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    write_json(build / "kernel-metadata.json", metadata)
    output = kaggle(["kernels", "push", "-p", str(build)])
    print(output.strip())
    write_json(
        path / "launch.json",
        {
            "kernel": metadata["id"],
            "private": True,
            "commit": commit,
            "branch": branch,
            "seeds": seeds,
            "machine": machine,
            "launched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )


def kernel_status(name: str) -> str:
    launch_info = read_json(experiment_dir(name) / "launch.json")
    output = kaggle(["kernels", "status", launch_info["kernel"]], check=False)
    match = re.search(r'status "KernelWorkerStatus\.([A-Z_]+)"', output)
    return match.group(1) if match else output.strip()


def wait(name: str, poll_seconds: int) -> str:
    while True:
        status = kernel_status(name)
        if status in {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"}:
            print(f"{name}: {status}")
            return status
        time.sleep(poll_seconds)


def seed_run_dirs(raw: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for history in raw.rglob("history.csv"):
        match = re.search(r"seed(\d+)", history.parent.parent.name)
        if match:
            result[int(match.group(1))] = history.parent
    return dict(sorted(result.items()))


def collect(name: str, baseline: str | None) -> None:
    path = experiment_dir(name)
    launch_info = read_json(path / "launch.json")
    raw = RUNS_DIR / name
    raw.mkdir(parents=True, exist_ok=True)
    kaggle(
        ["kernels", "output", launch_info["kernel"], "-p", str(raw), "-o", "-q",
         "--file-pattern", r"^(?!.*\.pt$).*$"],
    )
    runs = seed_run_dirs(raw)
    per_seed: dict[str, Any] = {}
    histories: dict[int, list[dict[str, float]]] = {}
    for seed, run in runs.items():
        import pandas as pd

        history = pd.read_csv(run / "history.csv")
        histories[seed] = history.to_dict("records")
        metrics_path = run / "validation_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = read_json(metrics_path)
        audit = read_json(run / "data_audit.json")
        environment = read_json(run / "environment.json")
        best_row = history.loc[history.val_macro_f1.idxmax()]
        dead = (run / "dead_run.json").exists()
        per_seed[str(seed)] = {
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy": metrics["accuracy"],
            "macro_roc_auc_ovr": metrics["macro_roc_auc_ovr"],
            "per_class_f1": {c: metrics["per_class"][c]["f1"] for c in CLASS_NAMES},
            "per_class_recall": {c: metrics["per_class"][c]["recall"] for c in CLASS_NAMES},
            "confusion_matrix": metrics["confusion_matrix"],
            "best_epoch": int(best_row.epoch),
            "epochs_run": int(history.epoch.max()),
            "seconds_total": float(history.seconds.sum()),
            "seconds_per_epoch": float(history.seconds.mean()),
            "dead_run": read_json(run / "dead_run.json") if dead else None,
            "splits_sha256": audit["splits_sha256"],
            "manifest_sha256": audit["manifest_sha256"],
            "git_commit": environment.get("git_commit"),
        }
        if audit["splits_sha256"] != EXPECTED_SPLITS_SHA256 or audit["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
            raise SystemExit(f"seed {seed} ran on a different split; results rejected")
    if not per_seed:
        raise SystemExit(f"No finished seeds found in {raw}")
    aggregate = aggregate_metrics(per_seed)
    result = {
        "experiment": name,
        "kernel": launch_info["kernel"],
        "commit": launch_info["commit"],
        "seeds_requested": launch_info["seeds"],
        "seeds_finished": [int(s) for s in per_seed],
        "per_seed": per_seed,
        "aggregate": aggregate,
        "runtime_hours": sum(v["seconds_total"] for v in per_seed.values()) / 3600,
        "baseline": baseline,
    }
    if baseline:
        result["comparison"] = compare(result, read_json(experiment_dir(baseline) / "metrics.json"))
    write_json(path / "metrics.json", result)
    write_json(path / "histories.json", {str(k): v for k, v in histories.items()})
    plot_experiment(path, result, histories)
    print(json.dumps({"aggregate": aggregate, "comparison": result.get("comparison")}, indent=2))


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else float("nan"),
        "n": int(len(array)),
    }


def aggregate_metrics(per_seed: dict[str, Any]) -> dict[str, Any]:
    seeds = list(per_seed.values())
    aggregate: dict[str, Any] = {
        key: mean_std([s[key] for s in seeds])
        for key in ("macro_f1", "balanced_accuracy", "accuracy", "macro_roc_auc_ovr")
    }
    aggregate["per_class_f1"] = {
        c: mean_std([s["per_class_f1"][c] for s in seeds]) for c in CLASS_NAMES
    }
    aggregate["worst_class_f1"] = min(v["mean"] for v in aggregate["per_class_f1"].values())
    aggregate["confusion_matrix_sum"] = np.sum(
        [np.asarray(s["confusion_matrix"]) for s in seeds], axis=0
    ).tolist()
    return aggregate


def compare(result: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """Noise rule: a win needs the mean gain to exceed the larger of the two seed stds."""
    new = result["aggregate"]["macro_f1"]
    old = base["aggregate"]["macro_f1"]
    gain = new["mean"] - old["mean"]
    threshold = max(new["std"], old["std"])
    return {
        "baseline": base["experiment"],
        "baseline_mean": old["mean"],
        "baseline_std": old["std"],
        "gain": gain,
        "threshold_max_std": threshold,
        "beats_noise": bool(gain > threshold),
        "worse_beyond_noise": bool(-gain > threshold),
    }


# ---------------------------------------------------------------- figures


def plot_setup() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.dpi": 130,
        }
    )
    return plt


def plot_experiment(path: Path, result: dict[str, Any], histories: dict[int, list[dict[str, float]]]) -> None:
    plt = plot_setup()
    title = f"{result['experiment']}: val macro-F1 {result['aggregate']['macro_f1']['mean']:.3f} ± {result['aggregate']['macro_f1']['std']:.3f}"

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for index, (seed, rows) in enumerate(sorted(histories.items())):
        colour = SEED_COLOURS[index % len(SEED_COLOURS)]
        epochs = [r["epoch"] for r in rows]
        axes[0, 0].plot(epochs, [r["train_loss"] for r in rows], color=colour, lw=1.2, ls="--")
        axes[0, 0].plot(epochs, [r["val_loss"] for r in rows], color=colour, lw=1.6, label=f"seed {seed}")
        axes[0, 1].plot(epochs, [r["val_macro_f1"] for r in rows], color=colour, lw=1.6, label=f"seed {seed}")
        axes[1, 0].plot(epochs, [r["val_balanced_accuracy"] for r in rows], color=colour, lw=1.6)
        axes[1, 0].plot(epochs, [r["train_balanced_accuracy"] for r in rows], color=colour, lw=1.0, ls="--")
        axes[1, 1].plot(epochs, [r["learning_rate"] for r in rows], color=colour, lw=1.2)
    axes[0, 0].set(title="Loss (dashed = train, solid = val)", xlabel="Epoch", ylabel="Loss")
    axes[0, 1].set(title="Val macro-F1 per epoch", xlabel="Epoch", ylabel="Macro-F1")
    axes[1, 0].axhline(0.25, color=REF, ls=":", lw=1)
    axes[1, 0].set(title="Balanced accuracy (dashed = train, solid = val; dotted = chance)", xlabel="Epoch")
    axes[1, 1].set(title="Learning rate", xlabel="Epoch")
    axes[0, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.grid(color=GRID)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path / "curves.png")
    plt.close(fig)

    seeds = result["per_seed"]
    matrices = [(f"seed {s}", np.asarray(v["confusion_matrix"])) for s, v in seeds.items()]
    matrices.append(("sum of seeds", np.asarray(result["aggregate"]["confusion_matrix_sum"])))
    fig, axes = plt.subplots(1, len(matrices), figsize=(3.3 * len(matrices), 3.4))
    for ax, (label, matrix) in zip(np.atleast_1d(axes), matrices):
        share = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if share[i, j] > 0.5 else INK, fontsize=9)
        ax.set_xticks(range(4), CLASS_NAMES)
        ax.set_yticks(range(4), CLASS_NAMES)
        ax.set_title(label)
        ax.set_xlabel("Predicted")
    np.atleast_1d(axes)[0].set_ylabel("True")
    fig.suptitle(f"{result['experiment']}: confusion at best val epoch (shade = row share)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path / "confusion.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    per_class = result["aggregate"]["per_class_f1"]
    x = np.arange(4)
    ax.bar(x, [per_class[c]["mean"] for c in CLASS_NAMES], color=BLUE, width=0.55,
           yerr=[0 if math.isnan(per_class[c]["std"]) else per_class[c]["std"] for c in CLASS_NAMES],
           capsize=4, ecolor=INK2)
    for index, (seed, values) in enumerate(seeds.items()):
        ax.scatter(x + (index - 1) * 0.08, [values["per_class_f1"][c] for c in CLASS_NAMES],
                   color=INK, s=12, zorder=3)
    ax.axhline(0.30, color=REF, ls="--", lw=1)
    ax.text(3.45, 0.31, "target floor 0.30", color=INK2, ha="right")
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Val F1 (mean ± std, dots = seeds)")
    ax.grid(axis="y", color=GRID)
    ax.set_title(f"{result['experiment']}: per-class F1", loc="left", fontweight="bold")
    fig.savefig(path / "per_class_f1.png")
    plt.close(fig)

    plot_comparison(path, result)


def all_results() -> list[dict[str, Any]]:
    results = []
    for metrics_path in sorted(EXPERIMENTS_DIR.glob("[0-9][0-9][0-9]_*/metrics.json")):
        result = read_json(metrics_path)
        result["verdict"] = read_verdict(metrics_path.parent)
        results.append(result)
    return results


def read_verdict(path: Path) -> str:
    notes = path / "notes.md"
    if notes.exists():
        match = re.search(r"verdict:\s*\**\s*([a-z-]+)", notes.read_text(), re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return "pending"


def plot_comparison(path: Path, result: dict[str, Any]) -> None:
    plt = plot_setup()
    results = [r for r in all_results() if r["experiment"] != result["experiment"]]
    shown = [r for r in results if r["verdict"] == "accept" or r["experiment"] == result.get("baseline")]
    shown.append(dict(result, verdict="this"))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(shown) + 1.6))
    for row, item in enumerate(shown):
        macro = item["aggregate"]["macro_f1"]
        colour = BLUE if item["verdict"] == "this" else GREY
        ax.errorbar(macro["mean"], row, xerr=0 if math.isnan(macro["std"]) else macro["std"],
                    fmt="o", color=colour, ecolor=colour, capsize=4, ms=7)
        for value in item["per_seed"].values():
            ax.scatter(value["macro_f1"], row, color=INK, s=10, zorder=3)
        label = item["experiment"] + (" (baseline)" if item["experiment"] == result.get("baseline") else "")
        ax.text(0.005, row + 0.28, label, color=INK if colour == BLUE else INK2, fontsize=8.5)
    ax.axvline(0.50, color=REF, ls="--", lw=1)
    ax.text(0.505, len(shown) - 0.6, "target 0.50", color=INK2)
    ax.set_yticks([])
    ax.set_xlim(0, 0.7)
    ax.set_ylim(-0.6, len(shown))
    ax.set_xlabel("Val macro-F1 (mean ± std over seeds; dots = seeds)")
    ax.grid(axis="x", color=GRID)
    ax.set_title(f"{result['experiment']} vs baseline and accepted runs", loc="left", fontweight="bold")
    fig.savefig(path / "comparison.png")
    plt.close(fig)


# ---------------------------------------------------------------- summarize


def summarize() -> None:
    results = all_results()
    lines = [
        TABLE_START,
        "| id | change | val macro-F1 mean±std | bal-acc mean±std | worst-class F1 (class) | seeds | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        aggregate = item["aggregate"]
        worst_class = min(CLASS_NAMES, key=lambda c: aggregate["per_class_f1"][c]["mean"])
        change = read_change(EXPERIMENTS_DIR / item["experiment"])
        lines.append(
            f"| {item['experiment'][:3]} | {change} | "
            f"{aggregate['macro_f1']['mean']:.3f} ± {aggregate['macro_f1']['std']:.3f} | "
            f"{aggregate['balanced_accuracy']['mean']:.3f} ± {aggregate['balanced_accuracy']['std']:.3f} | "
            f"{aggregate['worst_class_f1']:.3f} ({worst_class}) | {len(item['per_seed'])} | {item['verdict']} |"
        )
    lines.append(TABLE_END)
    text = LOG_PATH.read_text() if LOG_PATH.exists() else f"# Experiment log\n\n{TABLE_START}\n{TABLE_END}\n"
    start, end = text.index(TABLE_START), text.index(TABLE_END) + len(TABLE_END)
    LOG_PATH.write_text(text[:start] + "\n".join(lines) + text[end:])
    plot_progress(results)


def read_change(path: Path) -> str:
    notes = path / "notes.md"
    if notes.exists():
        match = re.search(r"change:\s*(.+)", notes.read_text(), re.IGNORECASE)
        if match:
            return match.group(1).strip().replace("|", "/")
    return path.name[4:]


def plot_progress(results: list[dict[str, Any]]) -> None:
    plt = plot_setup()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    best = -1.0
    steps_x: list[int] = []
    steps_y: list[float] = []
    for index, item in enumerate(results):
        macro = item["aggregate"]["macro_f1"]
        accepted = item["verdict"] == "accept"
        colour = BLUE if accepted else GREY
        ax.errorbar(index, macro["mean"], yerr=0 if math.isnan(macro["std"]) else macro["std"],
                    fmt="o", color=colour, ecolor=colour, capsize=3, ms=6)
        if accepted and macro["mean"] > best:
            best = macro["mean"]
        steps_x.append(index)
        steps_y.append(best if best >= 0 else np.nan)
    ax.step(steps_x, steps_y, where="post", color=BLUE, lw=1.5, alpha=0.6)
    ax.axhline(0.50, color=REF, ls="--", lw=1)
    ax.text(len(results) - 0.5, 0.505, "target 0.50", color=INK2, ha="right")
    ax.set_xticks(range(len(results)), [r["experiment"][:3] for r in results])
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Val macro-F1 (mean ± std)")
    ax.set_ylim(0, 0.7)
    ax.grid(axis="y", color=GRID)
    ax.set_title("Best accepted val macro-F1 so far (line); blue = accepted, grey = rejected/inconclusive",
                 loc="left", fontweight="bold")
    fig.savefig(EXPERIMENTS_DIR / "progress.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    launch_parser = commands.add_parser("launch")
    launch_parser.add_argument("name")
    launch_parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    launch_parser.add_argument("--machine", default="NvidiaTeslaT4")
    wait_parser = commands.add_parser("wait")
    wait_parser.add_argument("name")
    wait_parser.add_argument("--poll", type=int, default=120)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("name")
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("name")
    collect_parser.add_argument("--baseline")
    commands.add_parser("summarize")
    check_parser = commands.add_parser("check")
    check_parser.add_argument("name")
    args = parser.parse_args()
    if args.command == "launch":
        launch(args.name, args.seeds, args.machine)
    elif args.command == "wait":
        wait(args.name, args.poll)
    elif args.command == "status":
        print(kernel_status(args.name))
    elif args.command == "collect":
        collect(args.name, args.baseline)
    elif args.command == "summarize":
        summarize()
    elif args.command == "check":
        check = split_check(experiment_dir(args.name) / "config.yaml")
        print(json.dumps(check, indent=2))
        raise SystemExit(0 if check["passed"] else 1)


if __name__ == "__main__":
    main()
