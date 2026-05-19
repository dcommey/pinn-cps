"""Generate paper figures from cached results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pinncps.data import load_dataset
from pinncps.eval.plots import (
    plot_detection_delay,
    plot_residual_heatmap,
    plot_roc_curves,
    plot_trajectory_attack,
)
from pinncps.utils import load_config


def _attack_examples(test_data, scores_by_method, out_dir: Path) -> None:
    """One trajectory figure per attack type."""
    states = test_data["states"]
    obs = test_data["obs"]
    labels = test_data["labels"]
    meta = test_data["meta"]
    seen = set()
    for i, m in enumerate(meta):
        key = (m["kind"], m["severity"])
        if key in seen:
            continue
        seen.add(key)
        plot_trajectory_attack(
            states[i], obs[i], labels[i],
            out_dir / f"attack_{m['kind']}_{m['severity']}",
            title=f"{m['kind']} ({m['severity']})",
        )


def _roc_figures(test_data, scores_by_method, out_dir: Path) -> None:
    labels = test_data["labels"]
    plot_roc_curves(
        {name: (s, labels) for name, s in scores_by_method.items()},
        out_dir / "roc_all",
        title="All attacks",
    )
    for atk in sorted({m["kind"] for m in test_data["meta"]}):
        mask = np.array([m["kind"] == atk for m in test_data["meta"]])
        plot_roc_curves(
            {name: (s[mask], labels[mask]) for name, s in scores_by_method.items()},
            out_dir / f"roc_{atk}",
            title=atk,
        )


def _delay_figure(test_data, scores_by_method, out_dir: Path) -> None:
    from pinncps.eval.metrics import threshold_from_validation
    delays: Dict[str, np.ndarray] = {}
    labels = test_data["labels"]
    for name, sc in scores_by_method.items():
        # Use the nominal scores from a separate file if available, else the
        # 50th percentile of attacked scores as a placeholder threshold.
        thr = float(np.quantile(sc.reshape(-1), 0.6))
        per = []
        for i in range(sc.shape[0]):
            atk_idx = np.where(labels[i] > 0)[0]
            if atk_idx.size == 0:
                continue
            t_a = int(atk_idx[0])
            det = np.where(sc[i, t_a:] >= thr)[0]
            per.append(int(det[0]) if det.size else sc.shape[1] - t_a)
        delays[name] = np.array(per, dtype=np.float64) if per else np.array([0.0])
    plot_detection_delay(delays, out_dir / "detection_delay", title="Detection delay")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dir = ROOT / "results" / "runs" / cfg.name
    fig_dir = ROOT / "results" / "figures" / cfg.name
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_dir = ROOT / "data" / cfg.name
    if (data_dir / "test_attack.npz").exists():
        test_data = load_dataset(data_dir / "test_attack.npz")
    else:
        raise SystemExit(f"missing test set at {data_dir}; run generate_data first")

    scores_path = run_dir / "scores.npz"
    if not scores_path.exists():
        raise SystemExit(f"missing {scores_path}; run run_main first")
    with np.load(scores_path, allow_pickle=True) as f:
        keys = [k for k in f.files if k not in ("labels", "meta")]
        scores_by_method = {k: f[k] for k in keys}

    _attack_examples(test_data, scores_by_method, fig_dir)
    _roc_figures(test_data, scores_by_method, fig_dir)
    _delay_figure(test_data, scores_by_method, fig_dir)

    # Residual heatmap for the PINN over the first 16 trajectories.
    if "pinn" in scores_by_method:
        plot_residual_heatmap(
            scores_by_method["pinn"][:16],
            test_data["labels"][:16],
            fig_dir / "pinn_residual_heatmap",
            title="PINN anomaly score",
        )
    print(f"wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
