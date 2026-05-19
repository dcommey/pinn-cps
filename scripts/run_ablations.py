"""Ablations and sensitivity studies for the PINN detector.

Runs:
  - lambda_kin   = 0     (no_kinematic)
  - lambda_eng   = 0     (no_energy)
  - lambda_smooth= 0     (no_smooth)
  - noise sweep  (x{0.5, 1.0, 2.0})
  - sparse obs sweep (dropout 0, 0.1, 0.25)
  - leave-one-attack-out (LOAO) generalisation

Writes results/tables/ablations.csv and a LaTeX table.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pinncps.data import generate_attack_dataset, generate_nominal_dataset, load_dataset
from pinncps.eval import compute_detection_metrics, detection_delay, threshold_from_validation
from pinncps.models import NeuralPredictorDetector, PINN
from pinncps.models.pinn import PINNLoss, PINNLossConfigT
from pinncps.training import train_predictor
from pinncps.utils import load_config, seed_all


def _train_pinn(cfg, train, val):
    physics = PINNLoss(
        dt=cfg.sim.dt,
        energy_idle=cfg.sim.energy_idle,
        energy_lin=cfg.sim.energy_lin,
        energy_ang=cfg.sim.energy_ang,
        cfg=PINNLossConfigT(**cfg.pinn_loss.__dict__),
        obs_dim=cfg.model.obs_dim,
    )
    model = PINN(hidden=cfg.model.hidden, n_layers=cfg.model.n_layers,
                  dropout=cfg.model.dropout, obs_dim=cfg.model.obs_dim)
    hist = train_predictor(
        model, train, val,
        sequence=False, physics_loss=physics,
        epochs=cfg.train.epochs, batch_size=cfg.train.batch_size,
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, patience=cfg.train.patience,
        window=cfg.data.window_len, stride=cfg.data.stride, device=cfg.train.device,
    )
    det = NeuralPredictorDetector(model, kind="pinn", sequence=False,
                                  physics_loss=physics, device=cfg.train.device)
    det.fit(val)
    return det, hist


def _eval_one(det, test, nominal_test) -> Dict:
    nominal_scores = det.score_batch(nominal_test["obs"], nominal_test["commands"])
    thr = threshold_from_validation(nominal_scores, quantile=0.99)
    scores = det.score_batch(test["obs"], test["commands"])
    m = compute_detection_metrics(scores, test["labels"], threshold=thr)
    d = detection_delay(scores, test["labels"], threshold=thr)
    return {**m, **d, "threshold": thr}


def _generate_split(cfg, rng, *, force_regen: bool = False):
    """Use cached datasets if available, else generate fresh.

    Most ablations vary only the PINN loss weights so they share the same
    datasets as the main run.  Sensitivity sweeps that change ``cfg.sensor``
    pass ``force_regen=True`` so the perturbed sensors are reflected.
    """
    data_dir = ROOT / "data" / cfg.name
    if not force_regen and all(
        (data_dir / n).exists()
        for n in ("train.npz", "val.npz", "test_attack.npz", "test_nominal.npz")
    ):
        return (
            load_dataset(data_dir / "train.npz"),
            load_dataset(data_dir / "val.npz"),
            load_dataset(data_dir / "test_attack.npz"),
            load_dataset(data_dir / "test_nominal.npz"),
        )
    train = generate_nominal_dataset(cfg.data.n_train, cfg, rng)
    val = generate_nominal_dataset(cfg.data.n_val, cfg, rng)
    test = generate_attack_dataset(cfg.data.n_test_per_attack, cfg, rng)
    nominal_test = generate_nominal_dataset(max(40, cfg.data.n_val), cfg, rng)
    return train, val, test, nominal_test


def run_lambda_ablations(cfg_base) -> List[Dict]:
    rows = []
    variants = {
        "full_pinn": (1.0, 1.0, 0.5, 0.05),
        "no_kinematic": (1.0, 0.0, 0.5, 0.05),
        "no_energy": (1.0, 1.0, 0.0, 0.05),
        "no_smooth": (1.0, 1.0, 0.5, 0.0),
        "no_physics_all": (1.0, 0.0, 0.0, 0.0),
    }
    for name, (ld, lk, le, ls) in variants.items():
        cfg = copy.deepcopy(cfg_base)
        cfg.pinn_loss.lambda_data = ld
        cfg.pinn_loss.lambda_kin = lk
        cfg.pinn_loss.lambda_energy = le
        cfg.pinn_loss.lambda_smooth = ls
        seed_all(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        train, val, test, nom_test = _generate_split(cfg, rng)
        det, _ = _train_pinn(cfg, train, val)
        m = _eval_one(det, test, nom_test)
        rows.append({"study": "lambda", "variant": name, **m})
        print(f"  lambda/{name}: f1={m['f1']:.3f} roc={m['roc_auc']:.3f}")
    return rows


def run_noise_sweep(cfg_base) -> List[Dict]:
    rows = []
    base_pos = cfg_base.sensor.noise_pos
    base_vel = cfg_base.sensor.noise_vel
    base_th = cfg_base.sensor.noise_heading
    base_w = cfg_base.sensor.noise_omega
    for mult in (0.5, 1.0, 2.0):
        cfg = copy.deepcopy(cfg_base)
        cfg.sensor.noise_pos = base_pos * mult
        cfg.sensor.noise_vel = base_vel * mult
        cfg.sensor.noise_heading = base_th * mult
        cfg.sensor.noise_omega = base_w * mult
        seed_all(cfg.seed); rng = np.random.default_rng(cfg.seed)
        train, val, test, nom_test = _generate_split(cfg, rng, force_regen=True)
        det, _ = _train_pinn(cfg, train, val)
        m = _eval_one(det, test, nom_test)
        rows.append({"study": "noise", "variant": f"x{mult}", **m})
        print(f"  noise/x{mult}: f1={m['f1']:.3f} roc={m['roc_auc']:.3f}")
    return rows


def run_dropout_sweep(cfg_base) -> List[Dict]:
    rows = []
    for p in (0.0, 0.1, 0.25):
        cfg = copy.deepcopy(cfg_base)
        cfg.sensor.dropout_prob = p
        seed_all(cfg.seed); rng = np.random.default_rng(cfg.seed)
        train, val, test, nom_test = _generate_split(cfg, rng, force_regen=True)
        det, _ = _train_pinn(cfg, train, val)
        m = _eval_one(det, test, nom_test)
        rows.append({"study": "dropout", "variant": f"p={p}", **m})
        print(f"  dropout/{p}: f1={m['f1']:.3f} roc={m['roc_auc']:.3f}")
    return rows


def run_loao(cfg_base) -> List[Dict]:
    rows = []
    all_attacks = list(cfg_base.attack.types)
    for held_out in all_attacks:
        cfg = copy.deepcopy(cfg_base)
        seed_all(cfg.seed); rng = np.random.default_rng(cfg.seed)
        train, val, _, nom_test = _generate_split(cfg, rng)
        # Test only on the held-out attack so we measure generalisation.
        test = generate_attack_dataset(cfg.data.n_test_per_attack, cfg, rng,
                                       attack_types=[held_out])
        det, _ = _train_pinn(cfg, train, val)
        m = _eval_one(det, test, nom_test)
        rows.append({"study": "loao", "variant": held_out, **m})
        print(f"  loao/{held_out}: f1={m['f1']:.3f} roc={m['roc_auc']:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--studies",
        nargs="+",
        default=["lambda", "noise", "dropout", "loao"],
        choices=["lambda", "noise", "dropout", "loao"],
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(1)

    rows: List[Dict] = []
    if "lambda" in args.studies:
        print("[ablation: lambda]")
        rows += run_lambda_ablations(cfg)
    if "noise" in args.studies:
        print("[ablation: noise]")
        rows += run_noise_sweep(cfg)
    if "dropout" in args.studies:
        print("[ablation: dropout]")
        rows += run_dropout_sweep(cfg)
    if "loao" in args.studies:
        print("[ablation: loao]")
        rows += run_loao(cfg)

    out = ROOT / "results" / "tables" / f"{cfg.name}_ablations.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.exists():
        try:
            prev = pd.read_csv(out)
            # Keep only studies not in this run; new run overrides them.
            prev = prev[~prev["study"].isin(df["study"].unique())]
            df = pd.concat([prev, df], ignore_index=True)
        except Exception:
            pass
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    try:
        latex = df.to_latex(index=False, float_format="%.3f", escape=True)
    except Exception:
        from io import StringIO
        sio = StringIO()
        sio.write("\\begin{tabular}{l" + "r" * (len(df.columns) - 1) + "}\n")
        sio.write("\\toprule\n")
        sio.write(" & ".join(df.columns) + " \\\\\n\\midrule\n")
        for _, row in df.iterrows():
            sio.write(" & ".join(
                f"{v:.3f}" if isinstance(v, float) else str(v) for v in row
            ) + " \\\\\n")
        sio.write("\\bottomrule\n\\end{tabular}\n")
        latex = sio.getvalue()
    with open(out.with_suffix(".tex"), "w") as f:
        f.write(latex)


if __name__ == "__main__":
    main()
