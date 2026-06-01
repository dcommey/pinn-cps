"""Generate paper figures from cached results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pinncps.data import load_dataset
from pinncps.models import PINN
from pinncps.models.detector import _smooth
from pinncps.models.pinn import PINNLoss, PINNLossConfigT
from pinncps.eval.plots import (
    plot_component_timeseries,
    plot_roc_curves,
    plot_trajectory_attack,
)
from pinncps.utils import load_config


def _display_attack(kind: str) -> str:
    if kind == "gps_spoofing":
        return "GPS-like pose spoofing"
    return kind.replace("_", " ")


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
            title=f"{_display_attack(m['kind'])} ({m['severity']})",
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
            title=_display_attack(atk),
        )


def _load_prm_components(cfg, run_dir: Path, dataset):
    ckpt = torch.load(run_dir / "pinn.pt", map_location="cpu", weights_only=False)
    obs_dim = ckpt.get("obs_dim", cfg.model.obs_dim)
    physics = PINNLoss(
        dt=ckpt["robot"]["dt"],
        energy_idle=ckpt["robot"]["energy_idle"],
        energy_lin=ckpt["robot"]["energy_lin"],
        energy_ang=ckpt["robot"]["energy_ang"],
        cfg=PINNLossConfigT(**ckpt["physics_cfg"]),
        obs_dim=obs_dim,
    )
    model = PINN(
        hidden=cfg.model.hidden,
        n_layers=cfg.model.n_layers,
        dropout=cfg.model.dropout,
        obs_dim=obs_dim,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    obs = dataset["obs"]
    commands = dataset["commands"]
    n, tp1, d = obs.shape
    s = torch.from_numpy(obs[:, :-1].reshape(-1, d)).float()
    u = torch.from_numpy(commands.reshape(-1, commands.shape[-1])).float()
    sn = torch.from_numpy(obs[:, 1:].reshape(-1, d)).float()
    preds = []
    with torch.no_grad():
        for i in range(0, s.shape[0], 4096):
            preds.append(model(s[i:i + 4096], u[i:i + 4096]))
        sp = torch.cat(preds, dim=0)
        pred = torch.linalg.vector_norm(sp - sn, dim=-1).cpu().numpy()
        kin = torch.linalg.vector_norm(physics._kin_residual(s, sn), dim=-1).cpu().numpy()
    pred = pred.reshape(n, tp1 - 1)
    kin = kin.reshape(n, tp1 - 1)
    pred_full = np.empty((n, tp1), dtype=np.float64)
    kin_full = np.empty((n, tp1), dtype=np.float64)
    pred_full[:, 0] = pred[:, 0]
    pred_full[:, 1:] = pred
    kin_full[:, 0] = kin[:, 0]
    kin_full[:, 1:] = kin
    return pred_full, kin_full


def _prm_scores(cfg, run_dir: Path, data_dir: Path, test_data):
    val = load_dataset(data_dir / "val.npz")
    val_pred, val_kin = _load_prm_components(cfg, run_dir, val)
    test_pred, test_kin = _load_prm_components(cfg, run_dir, test_data)
    pred_scale = np.std(val_pred) + 1e-9
    kin_scale = np.std(val_kin) + 1e-9
    return {
        "Prediction channel": _smooth(test_pred / pred_scale, 5),
        "PRM": _smooth(test_kin / kin_scale, 5),
        "Sum diagnostic": _smooth(test_pred / pred_scale + test_kin / kin_scale, 5),
    }


def _curated_scores(cfg, run_dir: Path, data_dir: Path, test_data, cached_scores):
    scores = _prm_scores(cfg, run_dir, data_dir, test_data)
    labels = {
        "oc_svm": "OC-SVM",
        "iso_forest": "Isolation Forest",
        "kalman": "EKF residual",
        "mlp": "MLP",
        "lstm_ae": "LSTM-AE",
    }
    for key, label in labels.items():
        if key in cached_scores:
            scores[label] = cached_scores[key]
    return scores


def _component_figure(test_data, component_scores, out_dir: Path) -> None:
    meta = test_data["meta"]
    labels = test_data["labels"]
    candidates = [
        i for i, m in enumerate(meta)
        if m["kind"] == "gps_spoofing" and m["severity"] == "overt"
    ]
    if not candidates:
        return
    idx = max(candidates, key=lambda i: float(np.max(component_scores["PRM"][i])))
    plot_component_timeseries(
        component_scores["PRM"][idx],
        component_scores["Prediction channel"][idx],
        labels[idx],
        out_dir / "prm_component_timeseries",
        title=f"{_display_attack(meta[idx]['kind'])} ({meta[idx]['severity']})",
    )


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

    curated_scores = _curated_scores(cfg, run_dir, data_dir, test_data, scores_by_method)
    _attack_examples(test_data, scores_by_method, fig_dir)
    _roc_figures(test_data, curated_scores, fig_dir)
    _component_figure(test_data, curated_scores, fig_dir)
    print(f"wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
