"""Evaluate all trained detectors and produce results tables / figures."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pinncps.data import load_dataset
from pinncps.eval import compute_detection_metrics, detection_delay, threshold_from_validation
from pinncps.models import (
    GRUPredictor,
    LSTMAutoencoder,
    LSTMPredictor,
    MLPPredictor,
    NeuralPredictorDetector,
    PINN,
    ReconstructionDetector,
)
from pinncps.models.pinn import PINNLoss, PINNLossConfigT
from pinncps.utils import load_config


def _load_detectors(cfg, val, out_dir: Path) -> Dict[str, object]:
    dets: Dict[str, object] = {}

    # PINN
    f = out_dir / "pinn.pt"
    if f.exists():
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        physics = PINNLoss(
            dt=ckpt["robot"]["dt"],
            energy_idle=ckpt["robot"]["energy_idle"],
            energy_lin=ckpt["robot"]["energy_lin"],
            energy_ang=ckpt["robot"]["energy_ang"],
            cfg=PINNLossConfigT(**ckpt["physics_cfg"]),
            obs_dim=ckpt.get("obs_dim", cfg.model.obs_dim),
        )
        m = PINN(hidden=cfg.model.hidden, n_layers=cfg.model.n_layers,
                 dropout=cfg.model.dropout, obs_dim=ckpt.get("obs_dim", cfg.model.obs_dim))
        m.load_state_dict(ckpt["model"])
        det = NeuralPredictorDetector(m, kind="pinn", sequence=False, physics_loss=physics)
        det.fit(val)
        dets["pinn"] = det

    # MLP
    f = out_dir / "mlp.pt"
    if f.exists():
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        m = MLPPredictor(hidden=cfg.model.hidden, n_layers=cfg.model.n_layers,
                         dropout=cfg.model.dropout, obs_dim=ckpt.get("obs_dim", cfg.model.obs_dim))
        m.load_state_dict(ckpt["model"])
        det = NeuralPredictorDetector(m, kind="mlp", sequence=False)
        det.fit(val)
        dets["mlp"] = det

    # LSTM
    f = out_dir / "lstm.pt"
    if f.exists():
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        m = LSTMPredictor(hidden=cfg.model.lstm_hidden, n_layers=1,
                          dropout=cfg.model.dropout, obs_dim=ckpt.get("obs_dim", cfg.model.obs_dim))
        m.load_state_dict(ckpt["model"])
        det = NeuralPredictorDetector(m, kind="lstm", sequence=True)
        det.fit(val)
        dets["lstm"] = det

    # GRU
    f = out_dir / "gru.pt"
    if f.exists():
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        m = GRUPredictor(hidden=cfg.model.lstm_hidden, n_layers=1,
                         dropout=cfg.model.dropout, obs_dim=ckpt.get("obs_dim", cfg.model.obs_dim))
        m.load_state_dict(ckpt["model"])
        det = NeuralPredictorDetector(m, kind="gru", sequence=True)
        det.fit(val)
        dets["gru"] = det

    # LSTM-AE
    f = out_dir / "lstm_ae.pt"
    if f.exists():
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        m = LSTMAutoencoder(hidden=cfg.model.lstm_hidden, latent=cfg.model.ae_latent,
                             dropout=cfg.model.dropout, obs_dim=ckpt.get("obs_dim", cfg.model.obs_dim))
        m.load_state_dict(ckpt["model"])
        det = ReconstructionDetector(m, window=ckpt["window"])
        det.fit(val)
        dets["lstm_ae"] = det

    # Pickled classical detectors
    for name in ("iso_forest", "oc_svm", "kalman"):
        f = out_dir / f"{name}.pkl"
        if f.exists():
            with open(f, "rb") as fh:
                dets[name] = pickle.load(fh)
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = ROOT / "results" / "runs" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir = ROOT / "results" / "tables"; tbl_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data" / cfg.name

    val = load_dataset(data_dir / "val.npz")
    test = load_dataset(data_dir / "test_attack.npz")
    nominal_test = load_dataset(data_dir / "test_nominal.npz")

    dets = _load_detectors(cfg, val, out_dir)
    print(f"loaded detectors: {list(dets)}")

    rows = []
    score_cache: Dict[str, np.ndarray] = {}
    attack_types = sorted({m["kind"] for m in test["meta"]})
    severities = sorted({m["severity"] for m in test["meta"]})
    for name, det in dets.items():
        ns = det.score_batch(nominal_test["obs"], nominal_test["commands"])
        thr = threshold_from_validation(ns, quantile=0.99)
        sc = det.score_batch(test["obs"], test["commands"])
        score_cache[name] = sc
        for atk_filter, sev_filter, tag in [
            (None, None, "all"),
            *((atk, None, atk) for atk in attack_types),
            *((None, sev, sev) for sev in severities),
            *((atk, sev, f"{atk}/{sev}") for atk in attack_types for sev in severities),
        ]:
            mask = np.ones(len(test["meta"]), dtype=bool)
            if atk_filter:
                mask &= np.array([m["kind"] == atk_filter for m in test["meta"]])
            if sev_filter:
                mask &= np.array([m["severity"] == sev_filter for m in test["meta"]])
            if not mask.any():
                continue
            m_metrics = compute_detection_metrics(sc[mask], test["labels"][mask], threshold=thr)
            d = detection_delay(sc[mask], test["labels"][mask], threshold=thr)
            rows.append({"method": name, "slice": tag, "threshold": thr, **m_metrics, **d})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "main_results.csv", index=False)
    print("\nResults (slice='all'):")
    print(df[df["slice"] == "all"].to_string(index=False))

    # F1 pivot by attack type.
    pivot = (
        df[df["slice"].isin(attack_types + ["all"])]
        .pivot_table(index="method", columns="slice", values="f1", aggfunc="mean")
        .round(3)
    )
    pivot.to_csv(tbl_dir / f"{cfg.name}_main_f1.csv")
    # ROC-AUC pivot too.
    roc = (
        df[df["slice"].isin(attack_types + ["all"])]
        .pivot_table(index="method", columns="slice", values="roc_auc", aggfunc="mean")
        .round(3)
    )
    roc.to_csv(tbl_dir / f"{cfg.name}_main_roc.csv")

    # Severity slice F1.
    sev = (
        df[df["slice"].isin(severities + ["all"])]
        .pivot_table(index="method", columns="slice", values="f1", aggfunc="mean")
        .round(3)
    )
    sev.to_csv(tbl_dir / f"{cfg.name}_main_f1_by_severity.csv")

    np.savez_compressed(
        out_dir / "scores.npz",
        **{k: v for k, v in score_cache.items()},
        labels=test["labels"],
        meta=np.array(test["meta"], dtype=object),
    )
    print(f"\nwrote results to {out_dir} and {tbl_dir}")


if __name__ == "__main__":
    main()
