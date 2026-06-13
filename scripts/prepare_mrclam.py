"""Prepare real UTIAS MR.CLAM trajectories for PINN-CPS experiments.

This script converts the public MR.CLAM text files into the repository's NPZ
format. The nominal trajectories are real Vicon ground-truth robot poses and
logged odometry velocity commands. FDI test labels are created by applying
controlled cyber-channel perturbations to those real trajectories; no simulated
robot trajectories are generated here.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pinncps.attacks import AttackSpec, apply_attack
from pinncps.utils import load_config


def _load_dat(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def _interp_angle(t_src: np.ndarray, th_src: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.interp(t, t_src, np.unwrap(th_src))


def _resample_robot(dataset_dir: Path, robot: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    gt = _load_dat(dataset_dir / f"Robot{robot}_Groundtruth.dat")
    odom = _load_dat(dataset_dir / f"Robot{robot}_Odometry.dat")
    t0 = max(float(gt[0, 0]), float(odom[0, 0]))
    t1 = min(float(gt[-1, 0]), float(odom[-1, 0]))
    t = np.arange(t0, t1, dt)
    x = np.interp(t, gt[:, 0], gt[:, 1])
    y = np.interp(t, gt[:, 0], gt[:, 2])
    th = _interp_angle(gt[:, 0], gt[:, 3], t)
    v_cmd = np.interp(t[:-1], odom[:, 0], odom[:, 1])
    w_cmd = np.interp(t[:-1], odom[:, 0], odom[:, 2])

    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    v = dx * np.cos(th) + dy * np.sin(th)
    w = np.gradient(th, dt)
    obs = np.stack([x, y, th, v, w], axis=-1).astype(np.float32)
    commands = np.stack([v_cmd, w_cmd], axis=-1).astype(np.float32)
    return obs, commands


def _episodes(obs: np.ndarray, commands: np.ndarray, horizon: int, stride: int):
    out = []
    max_start = min(obs.shape[0] - horizon - 1, commands.shape[0] - horizon)
    for start in range(0, max_start, stride):
        end = start + horizon
        out.append((obs[start:end + 1], commands[start:end]))
    return out


def _pack_nominal(items, metas):
    n = len(items)
    horizon = items[0][1].shape[0]
    obs = np.stack([x[0] for x in items]).astype(np.float32)
    commands = np.stack([x[1] for x in items]).astype(np.float32)
    labels = np.zeros((n, horizon + 1), dtype=np.int64)
    return {
        "states": obs.copy(),
        "obs": obs,
        "commands": commands,
        "labels": labels,
        "meta": metas,
    }


def _save(d: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=d["states"],
        obs=d["obs"],
        commands=d["commands"],
        labels=d["labels"],
        meta=np.array(d["meta"], dtype=object),
    )


def _dataset_label(dataset_dir: Path) -> str:
    match = re.search(r"Dataset(\d+)", dataset_dir.name)
    return f"UTIAS MR.CLAM {match.group(1)}" if match else "UTIAS MR.CLAM"


def _attack_pack(items, metas, cfg, rng):
    attacks = [a for a in cfg.attack.types if a != "command_injection"]
    if not attacks:
        raise ValueError("MR.CLAM preparation does not support command_injection attacks")
    severities = list(cfg.attack.severity_levels)
    obs_out = []
    clean_out = []
    cmd_out = []
    lab_out = []
    meta_out = []
    for atk in attacks:
        for sev in severities:
            for idx, (obs, commands) in enumerate(items):
                horizon = commands.shape[0]
                start = int(rng.uniform(
                    cfg.attack.start_frac_low * (horizon + 1),
                    cfg.attack.start_frac_high * (horizon + 1),
                ))
                spec = AttackSpec(kind=atk, severity=sev, start=start, end=horizon + 1)
                result = apply_attack(obs, obs, commands, spec, rng)
                obs_out.append(result.obs.astype(np.float32))
                clean_out.append(obs.astype(np.float32))
                cmd_out.append(result.commands.astype(np.float32))
                lab_out.append(result.labels.astype(np.int64))
                base_meta = dict(metas[idx % len(metas)])
                base_meta.update({"kind": atk, "severity": sev, "start": start, "end": horizon + 1})
                meta_out.append(base_meta)
    return {
        # "states" holds the clean (pre-attack) observation series so that
        # downstream figure code can contrast ground truth with the attacked
        # channel; "obs" holds the attacked series consumed by detectors.
        "states": np.stack(clean_out).astype(np.float32),
        "obs": np.stack(obs_out).astype(np.float32),
        "commands": np.stack(cmd_out).astype(np.float32),
        "labels": np.stack(lab_out).astype(np.int64),
        "meta": meta_out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mrclam.yaml")
    ap.add_argument("--dataset-dir", default="data/raw/mrclam/MRCLAM_Dataset1")
    ap.add_argument("--outdir", default=None)
    ap.add_argument(
        "--heldout-robot",
        type=int,
        default=None,
        help="If set, train/validation episodes use all other robots and nominal/attack test episodes use this robot.",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.seed)
    dataset_dir = Path(args.dataset_dir)
    outdir = Path(args.outdir) if args.outdir else ROOT / "data" / cfg.name
    source = _dataset_label(dataset_dir)

    all_items = []
    all_metas = []
    for robot in range(1, 6):
        obs, commands = _resample_robot(dataset_dir, robot, cfg.sim.dt)
        for j, item in enumerate(_episodes(obs, commands, cfg.sim.horizon, cfg.data.stride)):
            all_items.append(item)
            all_metas.append({"kind": "nominal", "source": source, "robot": robot, "episode": j})

    if args.heldout_robot is None:
        order = rng.permutation(len(all_items))
        all_items = [all_items[i] for i in order]
        all_metas = [all_metas[i] for i in order]
        n_train = min(cfg.data.n_train, len(all_items))
        n_val = min(cfg.data.n_val, len(all_items) - n_train)
        remaining = len(all_items) - n_train - n_val
        n_test = min(cfg.data.n_test_per_attack, remaining)
        n_attack = min(cfg.data.n_test_per_attack, max(0, remaining - n_test))
        train_items = all_items[:n_train]
        val_items = all_items[n_train:n_train + n_val]
        test_items = all_items[n_train + n_val:n_train + n_val + n_test]
        attack_items = all_items[n_train + n_val + n_test:n_train + n_val + n_test + n_attack]
        train_meta = all_metas[:n_train]
        val_meta = all_metas[n_train:n_train + n_val]
        test_meta = all_metas[n_train + n_val:n_train + n_val + n_test]
        attack_meta = all_metas[n_train + n_val + n_test:n_train + n_val + n_test + n_attack]
    else:
        heldout = int(args.heldout_robot)
        if heldout < 1 or heldout > 5:
            raise ValueError("--heldout-robot must be between 1 and 5")
        train_pool = [(x, m) for x, m in zip(all_items, all_metas) if m["robot"] != heldout]
        test_pool = [(x, m) for x, m in zip(all_items, all_metas) if m["robot"] == heldout]
        train_order = rng.permutation(len(train_pool))
        test_order = rng.permutation(len(test_pool))
        train_pool = [train_pool[i] for i in train_order]
        test_pool = [test_pool[i] for i in test_order]
        n_train = min(cfg.data.n_train, len(train_pool))
        n_val = min(cfg.data.n_val, len(train_pool) - n_train)
        n_test = min(cfg.data.n_test_per_attack, len(test_pool))
        n_attack = min(cfg.data.n_test_per_attack, max(0, len(test_pool) - n_test))
        train_items = [x for x, _ in train_pool[:n_train]]
        train_meta = [m for _, m in train_pool[:n_train]]
        val_items = [x for x, _ in train_pool[n_train:n_train + n_val]]
        val_meta = [m for _, m in train_pool[n_train:n_train + n_val]]
        test_items = [x for x, _ in test_pool[:n_test]]
        test_meta = [m for _, m in test_pool[:n_test]]
        attack_items = [x for x, _ in test_pool[n_test:n_test + n_attack]]
        attack_meta = [m for _, m in test_pool[n_test:n_test + n_attack]]

    _save(_pack_nominal(train_items, train_meta), outdir / "train.npz")
    _save(_pack_nominal(val_items, val_meta), outdir / "val.npz")
    _save(_pack_nominal(test_items, test_meta), outdir / "test_nominal.npz")
    if not attack_items:
        raise RuntimeError("not enough MR.CLAM episodes to create a disjoint attack substrate")
    _save(_attack_pack(attack_items, attack_meta, cfg, rng), outdir / "test_attack.npz")
    split = f", heldout robot {args.heldout_robot}" if args.heldout_robot is not None else ""
    print(f"wrote MR.CLAM real-data experiment to {outdir} ({source}{split}; {len(train_items)} train, {len(val_items)} val, {len(test_items)} nominal threshold episodes, {len(attack_items)} attack-substrate episodes)")


if __name__ == "__main__":
    main()
