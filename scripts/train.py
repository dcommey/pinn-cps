"""Train a single detector and persist it to results/runs/<config>/<method>.pt

Usage:
    python scripts/train.py --config configs/medium.yaml --method pinn
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pinncps.data import load_dataset
from pinncps.models import (
    GRUPredictor,
    IsolationForestDetector,
    KalmanResidualDetector,
    LSTMAutoencoder,
    LSTMPredictor,
    MLPPredictor,
    NeuralPredictorDetector,
    OCSVMDetector,
    PINN,
    ReconstructionDetector,
)
from pinncps.models.pinn import PINNLoss, PINNLossConfigT
from pinncps.training import train_autoencoder, train_predictor
from pinncps.utils import load_config, seed_all


METHODS = ["pinn", "mlp", "lstm", "gru", "lstm_ae", "iso_forest", "oc_svm", "kalman"]


def _physics(cfg) -> PINNLoss:
    return PINNLoss(
        dt=cfg.sim.dt,
        energy_idle=cfg.sim.energy_idle,
        energy_lin=cfg.sim.energy_lin,
        energy_ang=cfg.sim.energy_ang,
        cfg=PINNLossConfigT(**cfg.pinn_loss.__dict__),
        obs_dim=cfg.model.obs_dim,
    )


def train_one(method: str, cfg, train, val, out_dir: Path) -> dict:
    device = cfg.train.device
    seed_all(cfg.seed)
    t0 = time.time()
    if method == "pinn":
        physics = _physics(cfg)
        model = PINN(hidden=cfg.model.hidden, n_layers=cfg.model.n_layers,
                     dropout=cfg.model.dropout, obs_dim=cfg.model.obs_dim)
        hist = train_predictor(
            model, train, val, sequence=False, physics_loss=physics,
            epochs=cfg.train.epochs, batch_size=cfg.train.batch_size, lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay, patience=cfg.train.patience,
            window=cfg.data.window_len, stride=cfg.data.stride, device=device,
        )
        torch.save({"model": model.state_dict(),
                    "physics_cfg": cfg.pinn_loss.__dict__,
                    "obs_dim": cfg.model.obs_dim,
                    "robot": dict(dt=cfg.sim.dt, energy_idle=cfg.sim.energy_idle,
                                  energy_lin=cfg.sim.energy_lin, energy_ang=cfg.sim.energy_ang)},
                   out_dir / "pinn.pt")
    elif method == "mlp":
        model = MLPPredictor(hidden=cfg.model.hidden, n_layers=cfg.model.n_layers,
                              dropout=cfg.model.dropout, obs_dim=cfg.model.obs_dim)
        hist = train_predictor(
            model, train, val, sequence=False, physics_loss=None,
            epochs=cfg.train.epochs, batch_size=cfg.train.batch_size, lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay, patience=cfg.train.patience,
            window=cfg.data.window_len, stride=cfg.data.stride, device=device,
        )
        torch.save({"model": model.state_dict(), "obs_dim": cfg.model.obs_dim}, out_dir / "mlp.pt")
    elif method == "lstm":
        model = LSTMPredictor(hidden=cfg.model.lstm_hidden, n_layers=1,
                              dropout=cfg.model.dropout, obs_dim=cfg.model.obs_dim)
        hist = train_predictor(
            model, train, val, sequence=True, physics_loss=None,
            epochs=cfg.train.epochs, batch_size=cfg.train.batch_size, lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay, patience=cfg.train.patience,
            window=cfg.data.window_len, stride=cfg.data.stride, device=device,
        )
        torch.save({"model": model.state_dict(), "obs_dim": cfg.model.obs_dim}, out_dir / "lstm.pt")
    elif method == "gru":
        model = GRUPredictor(hidden=cfg.model.lstm_hidden, n_layers=1,
                             dropout=cfg.model.dropout, obs_dim=cfg.model.obs_dim)
        hist = train_predictor(
            model, train, val, sequence=True, physics_loss=None,
            epochs=cfg.train.epochs, batch_size=cfg.train.batch_size, lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay, patience=cfg.train.patience,
            window=cfg.data.window_len, stride=cfg.data.stride, device=device,
        )
        torch.save({"model": model.state_dict(), "obs_dim": cfg.model.obs_dim}, out_dir / "gru.pt")
    elif method == "lstm_ae":
        model = LSTMAutoencoder(hidden=cfg.model.lstm_hidden, latent=cfg.model.ae_latent,
                                 dropout=cfg.model.dropout, obs_dim=cfg.model.obs_dim)
        hist = train_autoencoder(
            model, train, val,
            epochs=cfg.train.epochs, batch_size=cfg.train.batch_size, lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay, patience=cfg.train.patience,
            window=cfg.data.window_len, stride=cfg.data.stride, device=device,
        )
        torch.save({"model": model.state_dict(),
                    "window": cfg.data.window_len,
                    "obs_dim": cfg.model.obs_dim}, out_dir / "lstm_ae.pt")
    elif method == "iso_forest":
        det = IsolationForestDetector(dt=cfg.sim.dt)
        det.fit(train)
        with open(out_dir / "iso_forest.pkl", "wb") as f:
            pickle.dump(det, f)
        hist = {"wall_time": time.time() - t0, "best_val": 0.0}
    elif method == "oc_svm":
        det = OCSVMDetector(dt=cfg.sim.dt)
        det.fit(train)
        with open(out_dir / "oc_svm.pkl", "wb") as f:
            pickle.dump(det, f)
        hist = {"wall_time": time.time() - t0, "best_val": 0.0}
    elif method == "kalman":
        det = KalmanResidualDetector(dt=cfg.sim.dt)
        det.fit(train)
        with open(out_dir / "kalman.pkl", "wb") as f:
            pickle.dump(det, f)
        hist = {"wall_time": time.time() - t0, "best_val": 0.0}
    else:
        raise ValueError(f"unknown method {method!r}")

    hist["wall_time"] = time.time() - t0
    with open(out_dir / f"{method}.json", "w") as f:
        json.dump({k: v for k, v in hist.items() if not isinstance(v, (np.ndarray,))},
                  f, indent=2, default=str)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=METHODS + ["all"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(1)

    out_dir = ROOT / "results" / "runs" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = ROOT / "data" / cfg.name
    train = load_dataset(data_dir / "train.npz")
    val = load_dataset(data_dir / "val.npz")

    methods = METHODS if args.method == "all" else [args.method]
    for m in methods:
        hist = train_one(m, cfg, train, val, out_dir)
        print(f"{m}: wall={hist['wall_time']:.1f}s val={hist.get('best_val', 0):.4f}")


if __name__ == "__main__":
    main()
