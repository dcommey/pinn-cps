"""Dataset generation and PyTorch wrappers.

A "trajectory" is one episode: T+1 timesteps of observed state plus T command
vectors.  We store them stacked as:

    obs:      (N, T+1, OBS_DIM)
    states:   (N, T+1, STATE_DIM)
    commands: (N, T,   CTRL_DIM)
    labels:   (N, T+1) - 0/1 attack label per timestep
    meta:     list[dict] with attack type, severity, start, end, trajectory_kind

The windowed PyTorch dataset slides a length-L window across the timesteps; the
sample-level label is the max over the window so a window is positive iff any
attacked step is inside it.  For point-level metrics we also expose the raw
per-step labels.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is optional for non-training callers
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore

from ..sim import (
    RobotParams,
    OBS_DIM,
    STATE_DIM,
    apply_sensor_noise,
    generate_command_sequence,
    simulate_trajectory,
)
from ..attacks import AttackSpec, apply_attack


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def _random_initial_state(rng: np.random.Generator, battery_init: float) -> np.ndarray:
    x = rng.uniform(-0.5, 0.5)
    y = rng.uniform(-0.5, 0.5)
    th = rng.uniform(-np.pi, np.pi)
    return np.array([x, y, th, 0.0, 0.0, battery_init], dtype=np.float64)


def _make_clean_trajectory(
    rng: np.random.Generator,
    robot_params: RobotParams,
    sensor_cfg,
    horizon: int,
    trajectory_kind: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (states, obs, commands) for one nominal episode."""
    commands = generate_command_sequence(
        trajectory_kind, horizon, robot_params.dt, rng
    )
    s0 = _random_initial_state(rng, robot_params.battery_init)
    states = simulate_trajectory(s0, commands, robot_params)
    obs = apply_sensor_noise(states, sensor_cfg, rng)
    return states, obs, commands


def generate_nominal_dataset(
    n: int,
    cfg,
    rng: np.random.Generator,
):
    robot_params = RobotParams(
        dt=cfg.sim.dt,
        v_max=cfg.sim.v_max,
        omega_max=cfg.sim.omega_max,
        energy_idle=cfg.sim.energy_idle,
        energy_lin=cfg.sim.energy_lin,
        energy_ang=cfg.sim.energy_ang,
        battery_init=cfg.sim.battery_init,
    )
    horizon = cfg.sim.horizon
    types = cfg.sim.trajectory_types
    states = np.empty((n, horizon + 1, STATE_DIM), dtype=np.float32)
    obs = np.empty_like(states)
    commands = np.empty((n, horizon, 2), dtype=np.float32)
    labels = np.zeros((n, horizon + 1), dtype=np.int64)
    meta: List[dict] = []
    for i in range(n):
        kind = types[i % len(types)]
        s, o, c = _make_clean_trajectory(rng, robot_params, cfg.sensor, horizon, kind)
        states[i] = s.astype(np.float32)
        obs[i] = o.astype(np.float32)
        commands[i] = c.astype(np.float32)
        meta.append({"kind": "nominal", "trajectory": kind})
    return {
        "states": states,
        "obs": obs,
        "commands": commands,
        "labels": labels,
        "meta": meta,
    }


def generate_attack_dataset(
    n_per_attack: int,
    cfg,
    rng: np.random.Generator,
    attack_types: Optional[Sequence[str]] = None,
):
    robot_params = RobotParams(
        dt=cfg.sim.dt,
        v_max=cfg.sim.v_max,
        omega_max=cfg.sim.omega_max,
        energy_idle=cfg.sim.energy_idle,
        energy_lin=cfg.sim.energy_lin,
        energy_ang=cfg.sim.energy_ang,
        battery_init=cfg.sim.battery_init,
    )
    horizon = cfg.sim.horizon
    types = cfg.sim.trajectory_types
    attack_types = list(attack_types or cfg.attack.types)
    severities = list(cfg.attack.severity_levels)
    total = len(attack_types) * n_per_attack
    all_states = np.empty((total, horizon + 1, STATE_DIM), dtype=np.float32)
    all_obs = np.empty_like(all_states)
    all_cmds = np.empty((total, horizon, 2), dtype=np.float32)
    all_labels = np.zeros((total, horizon + 1), dtype=np.int64)
    meta: List[dict] = []

    idx = 0
    for atk in attack_types:
        for j in range(n_per_attack):
            kind = types[(idx + j) % len(types)]
            states, obs, commands = _make_clean_trajectory(
                rng, robot_params, cfg.sensor, horizon, kind
            )
            sev = severities[j % len(severities)]
            start = int(
                rng.uniform(
                    cfg.attack.start_frac_low * (horizon + 1),
                    cfg.attack.start_frac_high * (horizon + 1),
                )
            )
            end = horizon + 1
            params = {}
            if atk == "command_injection":
                params = dict(
                    robot_params=robot_params,
                    sensor_cfg=cfg.sensor,
                    s0=states[0].astype(np.float64),
                )
            spec = AttackSpec(kind=atk, severity=sev, start=start, end=end, params=params)
            result = apply_attack(obs, states, commands, spec, rng)
            all_states[idx] = states.astype(np.float32)
            all_obs[idx] = result.obs.astype(np.float32)
            all_cmds[idx] = result.commands.astype(np.float32)
            all_labels[idx] = result.labels.astype(np.int64)
            meta.append(
                {
                    "kind": atk,
                    "trajectory": kind,
                    "severity": sev,
                    "start": int(start),
                    "end": int(end),
                }
            )
            idx += 1
    return {
        "states": all_states,
        "obs": all_obs,
        "commands": all_cmds,
        "labels": all_labels,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_dataset(d: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=d["states"],
        obs=d["obs"],
        commands=d["commands"],
        labels=d["labels"],
        meta=np.array(d["meta"], dtype=object),
    )


def load_dataset(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as f:
        return {
            "states": f["states"],
            "obs": f["obs"],
            "commands": f["commands"],
            "labels": f["labels"],
            "meta": list(f["meta"]),
        }


# ---------------------------------------------------------------------------
# Windowed dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):  # type: ignore[misc]
    """Sliding-window dataset for predictor training.

    Each sample is a window starting at step ``t``:

        x_obs       : (L, OBS_DIM)
        x_cmd       : (L, CTRL_DIM)        commands aligned with obs[0..L-1]
        target_obs  : (L, OBS_DIM)         obs at steps 1..L (one-step shifted)
        label_win   : int                  1 iff any step in window is attacked

    For one-step predictors we predict ``target_obs[-1]`` from
    ``(x_obs[:-1], x_cmd[:-1])``; for sequence models we use the full window.
    """

    def __init__(self, data: dict, window: int, stride: int = 1):
        self.window = window
        self.stride = stride
        self.obs = data["obs"]
        self.commands = data["commands"]
        self.labels = data["labels"]
        N, Tp1, _ = self.obs.shape
        T = Tp1 - 1
        self.T = T
        # Build index: list of (traj_idx, t0) where window is [t0, t0+window)
        idxs = []
        for i in range(N):
            for t0 in range(0, T - window + 1, stride):
                idxs.append((i, t0))
        self._idxs = idxs

    def __len__(self) -> int:
        return len(self._idxs)

    def __getitem__(self, k):
        i, t0 = self._idxs[k]
        L = self.window
        x_obs = self.obs[i, t0 : t0 + L]
        # Commands are length T, aligned with steps 0..T-1; pair them with obs[0..L-1].
        x_cmd = self.commands[i, t0 : t0 + L]
        target = self.obs[i, t0 + 1 : t0 + L + 1]
        label_win = int(self.labels[i, t0 : t0 + L + 1].max())
        if torch is None:
            return x_obs, x_cmd, target, label_win
        return (
            torch.from_numpy(x_obs).float(),
            torch.from_numpy(x_cmd).float(),
            torch.from_numpy(target).float(),
            torch.tensor(label_win, dtype=torch.long),
        )
