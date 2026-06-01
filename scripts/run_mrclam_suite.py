"""Run the complete real-data MR.CLAM experiment suite.

The suite has two parts:

1. Dataset-level runs for MR.CLAM datasets 1--9.
2. Robot-held-out transfer runs on MR.CLAM Dataset 1, one held-out robot at a
   time.

Each run prepares real trajectories, trains all detectors, evaluates them, and
then writes aggregate mean/std tables under ``results/tables``.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "mrclam"
GENERATED_CFG_DIR = ROOT / "configs" / "generated"
RESULTS_DIR = ROOT / "results"


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _load_base_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _write_config(base: dict, name: str, seed: int, out: Path) -> None:
    cfg = dict(base)
    cfg["name"] = name
    cfg["seed"] = int(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _download_dataset(n: int) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / f"MRCLAM{n}.zip"
    candidates = [RAW_DIR / f"MRCLAM_Dataset{n}", RAW_DIR / f"MRSLAM_Dataset{n}"]
    for data_dir in candidates:
        if data_dir.exists():
            return data_dir
    if not zip_path.exists():
        url = f"ftp://asrl3.utias.utoronto.ca/MRCLAM/MRCLAM{n}.zip"
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    for data_dir in candidates:
        if data_dir.exists():
            return data_dir
    matches = sorted(RAW_DIR.glob(f"*Dataset{n}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"expected extracted dataset directory for dataset {n}")


def _run_one(
    *,
    name: str,
    cfg_path: Path,
    dataset_dir: Path,
    heldout_robot: int | None = None,
    retrain: bool = False,
    figures: bool = False,
) -> None:
    run_dir = RESULTS_DIR / "runs" / name
    data_dir = ROOT / "data" / name
    if retrain:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        if data_dir.exists():
            shutil.rmtree(data_dir)
    prep_cmd = [
        sys.executable,
        "scripts/prepare_mrclam.py",
        "--config",
        str(cfg_path),
        "--dataset-dir",
        str(dataset_dir),
    ]
    if heldout_robot is not None:
        prep_cmd += ["--heldout-robot", str(heldout_robot)]
    _run(prep_cmd)
    _run([sys.executable, "scripts/train.py", "--config", str(cfg_path), "--method", "all"])
    _run([sys.executable, "scripts/evaluate.py", "--config", str(cfg_path)])
    if figures:
        _run([sys.executable, "scripts/make_figures.py", "--config", str(cfg_path)])


def _aggregate(pattern: str, out_prefix: str, run_label: str) -> None:
    frames = []
    for csv_path in sorted((RESULTS_DIR / "runs").glob(pattern)):
        run_name = csv_path.parent.name
        df = pd.read_csv(csv_path)
        df[run_label] = run_name
        frames.append(df)
    if not frames:
        return
    all_df = pd.concat(frames, ignore_index=True)
    out_dir = RESULTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / f"{out_prefix}_all_results.csv"
    all_df.to_csv(all_path, index=False)

    metric_cols = [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "mean_delay",
        "median_delay",
        "miss_rate",
        "mean_delay_with_censoring",
    ]
    summary = (
        all_df.groupby(["method", "slice"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns
    ]
    summary_path = out_dir / f"{out_prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)

    all_slice = summary[summary["slice"] == "all"].copy()
    compact_cols = [
        "method",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
    ]
    compact = all_slice[compact_cols].sort_values("f1_mean", ascending=False)
    compact.to_csv(out_dir / f"{out_prefix}_all_slice_summary.csv", index=False)
    print(f"wrote {all_path}, {summary_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="configs/mrclam.yaml")
    ap.add_argument("--datasets", nargs="+", type=int, default=list(range(1, 10)))
    ap.add_argument("--heldout-dataset", type=int, default=1)
    ap.add_argument("--skip-dataset-runs", action="store_true")
    ap.add_argument("--skip-heldout-runs", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    base = _load_base_config(ROOT / args.base_config)
    GENERATED_CFG_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_dataset_runs:
        for n in args.datasets:
            dataset_dir = _download_dataset(n)
            name = f"mrclam_d{n}"
            cfg_path = GENERATED_CFG_DIR / f"{name}.yaml"
            _write_config(base, name=name, seed=base.get("seed", 0) + n - 1, out=cfg_path)
            _run_one(
                name=name,
                cfg_path=cfg_path,
                dataset_dir=dataset_dir,
                retrain=args.retrain,
                figures=args.figures and n == args.datasets[0],
            )
        _aggregate("mrclam_d[1-9]/main_results.csv", "mrclam_dataset", "dataset_run")

    if not args.skip_heldout_runs:
        dataset_dir = _download_dataset(args.heldout_dataset)
        for robot in range(1, 6):
            name = f"mrclam_d{args.heldout_dataset}_holdout_r{robot}"
            cfg_path = GENERATED_CFG_DIR / f"{name}.yaml"
            _write_config(
                base,
                name=name,
                seed=base.get("seed", 0) + 100 + robot,
                out=cfg_path,
            )
            _run_one(
                name=name,
                cfg_path=cfg_path,
                dataset_dir=dataset_dir,
                heldout_robot=robot,
                retrain=args.retrain,
                figures=False,
            )
        _aggregate(
            f"mrclam_d{args.heldout_dataset}_holdout_r*/main_results.csv",
            "mrclam_robot_holdout",
            "heldout_run",
        )


if __name__ == "__main__":
    main()
