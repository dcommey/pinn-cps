"""Evaluate score-component ablations and simple fusion baselines.

This script reuses trained MR.CLAM detectors and computes:

* prediction channel only
* kinematic channel only (the PRM operating score)
* sum of prediction and kinematic channels
* max(prediction, kinematic)
* Mahalanobis combination of the two channels
* max-fusion of PRM and OC-SVM after nominal standardisation

It writes per-run and aggregated summaries under results/tables.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pinncps.data import load_dataset
from pinncps.eval import compute_detection_metrics, detection_delay, threshold_from_validation
from pinncps.models import PINN
from pinncps.models.detector import _smooth
from pinncps.models.pinn import PINNLoss, PINNLossConfigT
from pinncps.utils import load_config


def _predict_components(model, physics, obs, commands, device="cpu", chunk=4096):
    n, tp1, d = obs.shape
    s = torch.from_numpy(obs[:, :-1].reshape(-1, d)).float().to(device)
    u = torch.from_numpy(commands.reshape(-1, commands.shape[-1])).float().to(device)
    sn = torch.from_numpy(obs[:, 1:].reshape(-1, d)).float().to(device)
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, s.shape[0], chunk):
            preds.append(model(s[i:i + chunk], u[i:i + chunk]))
        sp = torch.cat(preds, dim=0)
        pred = torch.linalg.vector_norm(sp - sn, dim=-1).cpu().numpy()
        kin = torch.linalg.vector_norm(physics._kin_residual(s, sn), dim=-1).cpu().numpy()
    pred = pred.reshape(n, tp1 - 1)
    kin = kin.reshape(n, tp1 - 1)
    out_pred = np.empty((n, tp1), dtype=np.float64)
    out_kin = np.empty((n, tp1), dtype=np.float64)
    out_pred[:, 0] = pred[:, 0]
    out_pred[:, 1:] = pred
    out_kin[:, 0] = kin[:, 0]
    out_kin[:, 1:] = kin
    return out_pred, out_kin


def _load_prm_predictor(cfg, run_dir: Path):
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
    return model, physics


def _standardize(x, ref):
    return (x - np.mean(ref)) / (np.std(ref) + 1e-9)


def _mahalanobis_components(pred, kin, val_pred, val_kin):
    val = np.stack([val_pred.reshape(-1), val_kin.reshape(-1)], axis=1)
    mu = val.mean(axis=0)
    cov = np.cov(val, rowvar=False) + 1e-6 * np.eye(2)
    inv = np.linalg.pinv(cov)
    x = np.stack([pred.reshape(-1), kin.reshape(-1)], axis=1) - mu
    d2 = np.einsum("ij,jk,ik->i", x, inv, x)
    return np.sqrt(np.maximum(d2, 0.0)).reshape(pred.shape)


def _eval_scores(run_name, variant, scores, nominal_scores, labels, meta):
    thr = threshold_from_validation(nominal_scores, quantile=0.99)
    rows = []
    attack_types = sorted({m["kind"] for m in meta})
    severities = sorted({m["severity"] for m in meta})
    for atk_filter, sev_filter, tag in [
        (None, None, "all"),
        *((atk, None, atk) for atk in attack_types),
        *((None, sev, sev) for sev in severities),
    ]:
        mask = np.ones(len(meta), dtype=bool)
        if atk_filter:
            mask &= np.array([m["kind"] == atk_filter for m in meta])
        if sev_filter:
            mask &= np.array([m["severity"] == sev_filter for m in meta])
        if not mask.any():
            continue
        m = compute_detection_metrics(scores[mask], labels[mask], threshold=thr)
        d = detection_delay(scores[mask], labels[mask], threshold=thr)
        rows.append({
            "run": run_name,
            "variant": variant,
            "slice": tag,
            "threshold": thr,
            **m,
            **d,
        })
    fpr = float(np.mean(nominal_scores.reshape(-1) >= thr))
    for row in rows:
        row["nominal_fpr"] = fpr
    return rows


def evaluate_run(config_path: Path):
    cfg = load_config(config_path)
    run_dir = ROOT / "results" / "runs" / cfg.name
    data_dir = ROOT / "data" / cfg.name
    val = load_dataset(data_dir / "val.npz")
    nominal = load_dataset(data_dir / "test_nominal.npz")
    test = load_dataset(data_dir / "test_attack.npz")

    model, physics = _load_prm_predictor(cfg, run_dir)
    val_pred_raw, val_kin_raw = _predict_components(model, physics, val["obs"], val["commands"])
    nom_pred_raw, nom_kin_raw = _predict_components(model, physics, nominal["obs"], nominal["commands"])
    test_pred_raw, test_kin_raw = _predict_components(model, physics, test["obs"], test["commands"])

    pred_scale = np.std(val_pred_raw) + 1e-9
    kin_scale = np.std(val_kin_raw) + 1e-9
    nom_pred = _smooth(nom_pred_raw / pred_scale, 5)
    test_pred = _smooth(test_pred_raw / pred_scale, 5)
    nom_kin = _smooth(nom_kin_raw / kin_scale, 5)
    test_kin = _smooth(test_kin_raw / kin_scale, 5)
    nom_sum = nom_pred + nom_kin
    test_sum = test_pred + test_kin
    nom_max = np.maximum(nom_pred, nom_kin)
    test_max = np.maximum(test_pred, test_kin)
    nom_maha = _smooth(_mahalanobis_components(nom_pred_raw, nom_kin_raw, val_pred_raw, val_kin_raw), 5)
    test_maha = _smooth(_mahalanobis_components(test_pred_raw, test_kin_raw, val_pred_raw, val_kin_raw), 5)

    rows = []
    labels = test["labels"]
    meta = test["meta"]
    for variant, scores, nominal_scores in [
        ("prm_pred_only", test_pred, nom_pred),
        ("prm_kin_only", test_kin, nom_kin),
        ("prm_sum", test_sum, nom_sum),
        ("prm_max", test_max, nom_max),
        ("prm_mahalanobis", test_maha, nom_maha),
    ]:
        rows.extend(_eval_scores(cfg.name, variant, scores, nominal_scores, labels, meta))

    oc_path = run_dir / "oc_svm.pkl"
    if oc_path.exists():
        with oc_path.open("rb") as f:
            oc = pickle.load(f)
        nom_oc = oc.score_batch(nominal["obs"], nominal["commands"])
        test_oc = oc.score_batch(test["obs"], test["commands"])
        nom_fused = np.maximum(_standardize(nom_sum, nom_sum), _standardize(nom_oc, nom_oc))
        test_fused = np.maximum(_standardize(test_sum, nom_sum), _standardize(test_oc, nom_oc))
        rows.extend(_eval_scores(cfg.name, "fusion_max_prm_ocsvm", test_fused, nom_fused, labels, meta))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--pattern", default="configs/generated/mrclam_d[1-9].yaml")
    ap.add_argument("--out-prefix", default="mrclam_score_variants")
    args = ap.parse_args()

    if args.configs:
        config_paths = [Path(p) for p in args.configs]
    else:
        config_paths = sorted(ROOT.glob(args.pattern))
    rows = []
    for path in config_paths:
        print(f"evaluating {path}")
        rows.extend(evaluate_run(path))
    df = pd.DataFrame(rows)
    out_dir = ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{args.out_prefix}_all_results.csv", index=False)

    metric_cols = [
        "precision", "recall", "f1", "roc_auc", "pr_auc",
        "mean_delay", "median_delay", "miss_rate",
        "mean_delay_with_censoring", "nominal_fpr",
    ]
    summary = (
        df.groupby(["variant", "slice"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in summary.columns
    ]
    summary.to_csv(out_dir / f"{args.out_prefix}_summary.csv", index=False)
    all_slice = summary[summary["slice"] == "all"].copy()
    all_slice.to_csv(out_dir / f"{args.out_prefix}_all_slice_summary.csv", index=False)
    print(all_slice[["variant", "f1_mean", "f1_std", "roc_auc_mean", "pr_auc_mean", "nominal_fpr_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
