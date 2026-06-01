"""Ablate one-class SVM feature sets on cached MR.CLAM splits.

The main manuscript uses a strong OC-SVM baseline with engineered residual
features. This script separates that result into raw-signal and residual-feature
variants and estimates feature-group sensitivity by permutation on the attacked
test split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pinncps.data import load_dataset
from pinncps.eval import compute_detection_metrics, detection_delay, threshold_from_validation
from pinncps.models.detector import _smooth
from pinncps.utils import load_config


GROUPS = {
    "raw_motion": slice(0, 2),
    "first_difference": slice(2, 7),
    "kinematic_residual": slice(12, 15),
    "command_gap": slice(15, 17),
}


def _base_components(obs: np.ndarray, commands: np.ndarray, dt: float):
    if obs.ndim == 2:
        obs = obs[None]
        commands = commands[None]
        squeeze = True
    else:
        squeeze = False
    o0 = obs[:, :-1]
    o1 = obs[:, 1:]
    dobs = o1 - o0
    x_dot_fd = dobs[..., 0] / dt
    y_dot_fd = dobs[..., 1] / dt
    th_dot_fd = dobs[..., 2] / dt
    v_dot_fd = dobs[..., 3] / dt
    w_dot_fd = dobs[..., 4] / dt
    th_mid = 0.5 * (o0[..., 2] + o1[..., 2])
    v_mid = 0.5 * (o0[..., 3] + o1[..., 3])
    w_mid = 0.5 * (o0[..., 4] + o1[..., 4])
    rkx = x_dot_fd - v_mid * np.cos(th_mid)
    rky = y_dot_fd - v_mid * np.sin(th_mid)
    rkth = th_dot_fd - w_mid
    cmd_v_err = commands[..., 0] - o0[..., 3]
    cmd_w_err = commands[..., 1] - o0[..., 4]
    return {
        "obs": o0[..., :5],
        "cmd": commands,
        "delta": dobs[..., :5],
        "fd": np.stack([x_dot_fd, y_dot_fd, th_dot_fd, v_dot_fd, w_dot_fd], axis=-1),
        "kin": np.stack([rkx, rky, rkth], axis=-1),
        "cmd_gap": np.stack([cmd_v_err, cmd_w_err], axis=-1),
        "squeeze": squeeze,
    }


def features(obs: np.ndarray, commands: np.ndarray, dt: float, mode: str) -> np.ndarray:
    c = _base_components(obs, commands, dt)
    if mode == "raw_cmd":
        parts = [c["obs"], c["cmd"]]
    elif mode == "raw_delta_cmd":
        parts = [c["obs"], c["delta"], c["cmd"]]
    elif mode == "full_residual":
        parts = [
            c["obs"][..., 3:5],
            c["delta"],
            c["fd"],
            c["kin"],
            c["cmd_gap"],
        ]
    else:
        raise ValueError(f"unknown mode {mode}")
    out = np.concatenate(parts, axis=-1)
    return out[0] if c["squeeze"] else out


class FeatureOCSVM:
    def __init__(self, dt: float, mode: str, max_train: int = 4000):
        self.dt = float(dt)
        self.mode = mode
        self.max_train = int(max_train)
        self.model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
        self.mean = None
        self.std = None

    def fit(self, nominal: dict):
        x = features(nominal["obs"], nominal["commands"], self.dt, self.mode)
        flat = x.reshape(-1, x.shape[-1])
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0) + 1e-6
        if flat.shape[0] > self.max_train:
            rng = np.random.default_rng(0)
            flat = flat[rng.choice(flat.shape[0], self.max_train, replace=False)]
        self.model.fit((flat - self.mean) / self.std)

    def score_features(self, feats: np.ndarray) -> np.ndarray:
        flat = (feats.reshape(-1, feats.shape[-1]) - self.mean) / self.std
        scores = -self.model.decision_function(flat)
        return scores.reshape(feats.shape[:-1])

    def score_batch(self, obs: np.ndarray, commands: np.ndarray) -> np.ndarray:
        feats = features(obs, commands, self.dt, self.mode)
        scores = self.score_features(feats)
        full = np.empty(obs.shape[:2], dtype=np.float64)
        full[:, 0] = scores[:, 0]
        full[:, 1:] = scores
        return _smooth(full, 5)


def _eval(run: str, variant: str, scores, nominal_scores, labels, meta):
    thr = threshold_from_validation(nominal_scores, quantile=0.99)
    rows = []
    for tag, mask in [
        ("all", np.ones(len(meta), dtype=bool)),
        ("overt", np.array([m["severity"] == "overt" for m in meta])),
        ("stealthy", np.array([m["severity"] == "stealthy" for m in meta])),
    ]:
        metrics = compute_detection_metrics(scores[mask], labels[mask], threshold=thr)
        delays = detection_delay(scores[mask], labels[mask], threshold=thr)
        rows.append({"run": run, "variant": variant, "slice": tag, **metrics, **delays})
    return rows


def _permute_group(model: FeatureOCSVM, test: dict, group: str, seed: int) -> np.ndarray:
    feats = features(test["obs"], test["commands"], model.dt, "full_residual")
    permuted = feats.copy()
    sl = GROUPS[group]
    flat = permuted.reshape(-1, permuted.shape[-1])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(flat.shape[0])
    flat[:, sl] = flat[idx, sl]
    scores = model.score_features(permuted)
    full = np.empty(test["obs"].shape[:2], dtype=np.float64)
    full[:, 0] = scores[:, 0]
    full[:, 1:] = scores
    return _smooth(full, 5)


def evaluate_config(path: Path):
    cfg = load_config(path)
    data_dir = ROOT / "data" / cfg.name
    train = load_dataset(data_dir / "train.npz")
    nominal = load_dataset(data_dir / "test_nominal.npz")
    test = load_dataset(data_dir / "test_attack.npz")
    labels = test["labels"]
    meta = test["meta"]

    rows = []
    models = {}
    for mode in ["raw_cmd", "raw_delta_cmd", "full_residual"]:
        model = FeatureOCSVM(cfg.sim.dt, mode)
        model.fit(train)
        models[mode] = model
        nom_scores = model.score_batch(nominal["obs"], nominal["commands"])
        scores = model.score_batch(test["obs"], test["commands"])
        rows.extend(_eval(cfg.name, mode, scores, nom_scores, labels, meta))

    full = models["full_residual"]
    nominal_scores = full.score_batch(nominal["obs"], nominal["commands"])
    for group in GROUPS:
        scores = _permute_group(full, test, group, seed=cfg.seed + len(group))
        rows.extend(_eval(cfg.name, f"permute_{group}", scores, nominal_scores, labels, meta))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="configs/generated/mrclam_d[1-9].yaml")
    ap.add_argument("--out-prefix", default="mrclam_ocsvm_feature_ablation")
    args = ap.parse_args()

    rows = []
    for path in sorted(ROOT.glob(args.pattern)):
        print(f"evaluating {path}")
        rows.extend(evaluate_config(path))
    df = pd.DataFrame(rows)
    out_dir = ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{args.out_prefix}_all_results.csv", index=False)

    metric_cols = [
        "precision", "recall", "f1", "roc_auc", "pr_auc",
        "mean_delay", "median_delay", "miss_rate", "mean_delay_with_censoring",
    ]
    summary = df.groupby(["variant", "slice"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns]
    summary.to_csv(out_dir / f"{args.out_prefix}_summary.csv", index=False)
    all_slice = summary[summary["slice"] == "all"].copy()
    all_slice.to_csv(out_dir / f"{args.out_prefix}_all_slice_summary.csv", index=False)
    print(all_slice[["variant", "f1_mean", "roc_auc_mean", "pr_auc_mean", "miss_rate_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
