"""Train PINN + all baselines and produce the main comparison table.

Outputs:
    results/runs/<config>/main_results.csv     long-format per-method, per-attack
    results/tables/<config>_main.csv           pivot table for the paper
    results/tables/<config>_main.tex           LaTeX version of the pivot
    results/runs/<config>/scores.npz           cached scores for figure scripts
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pinncps.data import generate_attack_dataset, generate_nominal_dataset, load_dataset
from pinncps.eval import (
    compute_detection_metrics,
    detection_delay,
    threshold_from_validation,
)
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


def _df_to_latex_simple(df: pd.DataFrame) -> str:
    """Tiny LaTeX writer so we don't need jinja2."""
    cols = list(df.columns)
    lines = [
        "\\begin{tabular}{l" + "r" * len(cols) + "}",
        "\\toprule",
        " & ".join([df.index.name or ""] + [str(c) for c in cols]) + " \\\\",
        "\\midrule",
    ]
    for idx, row in df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append(" & ".join([str(idx)] + vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def _maybe_load_or_generate(cfg, out_dir: Path):
    """Load datasets from disk if present, otherwise regenerate."""
    data_dir = ROOT / "data" / cfg.name
    if all((data_dir / n).exists() for n in ("train.npz", "val.npz", "test_attack.npz", "test_nominal.npz")):
        train = load_dataset(data_dir / "train.npz")
        val = load_dataset(data_dir / "val.npz")
        test = load_dataset(data_dir / "test_attack.npz")
        nominal_test = load_dataset(data_dir / "test_nominal.npz")
        print(f"loaded cached datasets from {data_dir}")
    else:
        rng = np.random.default_rng(cfg.seed)
        train = generate_nominal_dataset(cfg.data.n_train, cfg, rng)
        val = generate_nominal_dataset(cfg.data.n_val, cfg, rng)
        test = generate_attack_dataset(cfg.data.n_test_per_attack, cfg, rng)
        nominal_test = generate_nominal_dataset(max(40, cfg.data.n_val), cfg, rng)
        print(f"generated datasets in memory (no cache at {data_dir})")
    return train, val, test, nominal_test


def _build_pinn_loss(cfg) -> PINNLoss:
    return PINNLoss(
        dt=cfg.sim.dt,
        energy_idle=cfg.sim.energy_idle,
        energy_lin=cfg.sim.energy_lin,
        energy_ang=cfg.sim.energy_ang,
        cfg=PINNLossConfigT(**cfg.pinn_loss.__dict__),
        obs_dim=cfg.model.obs_dim,
    )


# ---------------------------------------------------------------------------

def train_all_detectors(cfg, train, val) -> Dict[str, object]:
    """Return a dict of name -> fitted detector."""
    device = cfg.train.device
    detectors: Dict[str, object] = {}
    histories: Dict[str, dict] = {}

    # ---- PINN ----
    t0 = time.time()
    pinn_model = PINN(hidden=cfg.model.hidden, n_layers=cfg.model.n_layers,
                       dropout=cfg.model.dropout, obs_dim=cfg.model.obs_dim)
    physics = _build_pinn_loss(cfg)
    hist = train_predictor(
        pinn_model, train, val,
        sequence=False,
        physics_loss=physics,
        epochs=cfg.train.epochs, batch_size=cfg.train.batch_size,
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        patience=cfg.train.patience,
        window=cfg.data.window_len, stride=cfg.data.stride,
        device=device,
    )
    histories["pinn"] = hist
    det = NeuralPredictorDetector(pinn_model, kind="pinn", sequence=False,
                                   physics_loss=physics, device=device)
    det.fit(val)
    detectors["pinn"] = det
    print(f"  pinn trained in {time.time()-t0:.1f}s val={hist['best_val']:.4f}")

    # ---- MLP (no physics loss) ----
    t0 = time.time()
    mlp = MLPPredictor(hidden=cfg.model.hidden, n_layers=cfg.model.n_layers,
                        dropout=cfg.model.dropout, obs_dim=cfg.model.obs_dim)
    hist = train_predictor(
        mlp, train, val,
        sequence=False, physics_loss=None,
        epochs=cfg.train.epochs, batch_size=cfg.train.batch_size,
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        patience=cfg.train.patience,
        window=cfg.data.window_len, stride=cfg.data.stride, device=device,
    )
    histories["mlp"] = hist
    det = NeuralPredictorDetector(mlp, kind="mlp", sequence=False, device=device)
    det.fit(val)
    detectors["mlp"] = det
    print(f"  mlp trained in {time.time()-t0:.1f}s val={hist['best_val']:.4f}")

    # ---- LSTM ----
    t0 = time.time()
    lstm = LSTMPredictor(hidden=cfg.model.lstm_hidden, n_layers=1,
                          dropout=cfg.model.dropout, obs_dim=cfg.model.obs_dim)
    hist = train_predictor(
        lstm, train, val,
        sequence=True, physics_loss=None,
        epochs=cfg.train.epochs, batch_size=cfg.train.batch_size,
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        patience=cfg.train.patience,
        window=cfg.data.window_len, stride=cfg.data.stride, device=device,
    )
    histories["lstm"] = hist
    det = NeuralPredictorDetector(lstm, kind="lstm", sequence=True, device=device)
    det.fit(val)
    detectors["lstm"] = det
    print(f"  lstm trained in {time.time()-t0:.1f}s val={hist['best_val']:.4f}")

    # ---- GRU ----
    t0 = time.time()
    gru = GRUPredictor(hidden=cfg.model.lstm_hidden, n_layers=1,
                        dropout=cfg.model.dropout, obs_dim=cfg.model.obs_dim)
    hist = train_predictor(
        gru, train, val,
        sequence=True, physics_loss=None,
        epochs=cfg.train.epochs, batch_size=cfg.train.batch_size,
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        patience=cfg.train.patience,
        window=cfg.data.window_len, stride=cfg.data.stride, device=device,
    )
    histories["gru"] = hist
    det = NeuralPredictorDetector(gru, kind="gru", sequence=True, device=device)
    det.fit(val)
    detectors["gru"] = det
    print(f"  gru trained in {time.time()-t0:.1f}s val={hist['best_val']:.4f}")

    # ---- LSTM autoencoder ----
    t0 = time.time()
    ae = LSTMAutoencoder(hidden=cfg.model.lstm_hidden, latent=cfg.model.ae_latent,
                          dropout=cfg.model.dropout, obs_dim=cfg.model.obs_dim)
    hist = train_autoencoder(
        ae, train, val,
        epochs=cfg.train.epochs, batch_size=cfg.train.batch_size,
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        patience=cfg.train.patience,
        window=cfg.data.window_len, stride=cfg.data.stride, device=device,
    )
    histories["lstm_ae"] = hist
    det = ReconstructionDetector(ae, window=cfg.data.window_len, device=device)
    det.fit(val)
    detectors["lstm_ae"] = det
    print(f"  lstm_ae trained in {time.time()-t0:.1f}s val={hist['best_val']:.4f}")

    # ---- Isolation Forest ----
    t0 = time.time()
    iso = IsolationForestDetector(dt=cfg.sim.dt)
    iso.fit(train)
    detectors["iso_forest"] = iso
    print(f"  iso_forest fit in {time.time()-t0:.1f}s")

    # ---- One-Class SVM ----
    t0 = time.time()
    oc = OCSVMDetector(dt=cfg.sim.dt)
    oc.fit(train)
    detectors["oc_svm"] = oc
    print(f"  oc_svm fit in {time.time()-t0:.1f}s")

    # ---- Kalman ----
    t0 = time.time()
    kal = KalmanResidualDetector(dt=cfg.sim.dt)
    kal.fit(train)
    detectors["kalman"] = kal
    print(f"  kalman fit in {time.time()-t0:.1f}s")

    return detectors, histories


# ---------------------------------------------------------------------------

def evaluate(detectors, test, nominal_test) -> pd.DataFrame:
    rows = []
    score_cache: Dict[str, np.ndarray] = {}
    test_labels = test["labels"]
    test_obs = test["obs"]
    test_cmds = test["commands"]
    nom_obs = nominal_test["obs"]
    nom_cmds = nominal_test["commands"]
    attack_types = sorted({m["kind"] for m in test["meta"]})

    for name, det in detectors.items():
        nominal_scores = det.score_batch(nom_obs, nom_cmds)
        thr = threshold_from_validation(nominal_scores, quantile=0.99)
        scores = det.score_batch(test_obs, test_cmds)
        score_cache[name] = scores

        # overall metrics
        m = compute_detection_metrics(scores, test_labels, threshold=thr)
        d = detection_delay(scores, test_labels, threshold=thr)
        row = {"method": name, "attack": "all", "threshold": thr, **m, **d}
        rows.append(row)
        # per-attack
        for atk in attack_types:
            mask = np.array([mm["kind"] == atk for mm in test["meta"]])
            if not mask.any():
                continue
            m = compute_detection_metrics(scores[mask], test_labels[mask], threshold=thr)
            d = detection_delay(scores[mask], test_labels[mask], threshold=thr)
            row = {"method": name, "attack": atk, "threshold": thr, **m, **d}
            rows.append(row)

    return pd.DataFrame(rows), score_cache


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg.seed)
    torch.set_num_threads(1)

    run_dir = ROOT / "results" / "runs" / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir = ROOT / "results" / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)

    train, val, test, nominal_test = _maybe_load_or_generate(cfg, run_dir)

    print("training detectors...")
    detectors, histories = train_all_detectors(cfg, train, val)
    print("evaluating...")
    df, score_cache = evaluate(detectors, test, nominal_test)

    df.to_csv(run_dir / "main_results.csv", index=False)
    print(df[df["attack"] == "all"].to_string(index=False))

    # Pivot table by attack / method.
    pivot = (
        df.pivot_table(index="method", columns="attack", values="f1", aggfunc="mean")
        .round(3)
    )
    pivot.to_csv(tbl_dir / f"{cfg.name}_main_f1.csv")
    try:
        latex = pivot.to_latex(escape=True)
    except Exception:
        # Fallback: emit a hand-rolled tabular block so we don't need jinja2.
        latex = _df_to_latex_simple(pivot)
    with open(tbl_dir / f"{cfg.name}_main_f1.tex", "w") as f:
        f.write(latex)

    np.savez_compressed(
        run_dir / "scores.npz",
        **{k: v for k, v in score_cache.items()},
        labels=test["labels"],
        meta=np.array(test["meta"], dtype=object),
    )
    with open(run_dir / "history.json", "w") as f:
        # Strip non-JSON-serializable entries.
        clean = {
            k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (np.ndarray,))}
            for k, v in histories.items()
        }
        json.dump(clean, f, indent=2)


if __name__ == "__main__":
    main()
