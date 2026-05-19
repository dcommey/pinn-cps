"""Trajectory command generators.

Each generator returns a `(T, 2)` array of (v_cmd, omega_cmd) commands.  We
include four regimes so the dataset is not dominated by one motion pattern:

    - figure8     - smooth, periodic, exercises both linear and angular axes
    - waypoint    - random waypoint chase with a simple pure-pursuit controller
    - lawnmower   - coverage pattern with sharp 180-degree turns
    - square      - piecewise-constant commands tracing a square loop

These are deterministic given an RNG so dataset generation is reproducible.
"""
from __future__ import annotations

import numpy as np

TRAJECTORY_TYPES = ("figure8", "waypoint", "lawnmower", "square")


def _figure8(T: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    period = rng.uniform(8.0, 14.0)
    v_amp = rng.uniform(0.4, 0.9)
    w_amp = rng.uniform(0.6, 1.2)
    t = np.arange(T) * dt
    v_cmd = v_amp * (0.7 + 0.3 * np.sin(2 * np.pi * t / period))
    omega_cmd = w_amp * np.sin(4 * np.pi * t / period)
    return np.stack([v_cmd, omega_cmd], axis=1)


def _square(T: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    seg = max(int(2.5 / dt), 4)
    v_amp = rng.uniform(0.4, 0.8)
    w_amp = rng.uniform(0.8, 1.3)
    cmds = np.zeros((T, 2))
    phase = 0
    for start in range(0, T, seg):
        end = min(start + seg, T)
        if phase % 2 == 0:
            cmds[start:end, 0] = v_amp
            cmds[start:end, 1] = 0.0
        else:
            cmds[start:end, 0] = 0.05
            cmds[start:end, 1] = w_amp
        phase += 1
    return cmds


def _lawnmower(T: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    seg = max(int(3.0 / dt), 4)
    v_amp = rng.uniform(0.4, 0.7)
    w_amp = rng.uniform(1.0, 1.4) * (1 if rng.random() < 0.5 else -1)
    cmds = np.zeros((T, 2))
    phase = 0
    for start in range(0, T, seg):
        end = min(start + seg, T)
        if phase % 2 == 0:
            cmds[start:end, 0] = v_amp
            cmds[start:end, 1] = 0.0
        else:
            cmds[start:end, 0] = 0.05
            cmds[start:end, 1] = w_amp
            w_amp = -w_amp
        phase += 1
    return cmds


def _waypoint(T: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Generate commands by chasing random waypoints with a simple controller.

    We pre-simulate a kinematic point and emit (v_cmd, omega_cmd) targeting the
    next waypoint.  The actual robot simulator still handles the lag.
    """
    n_pts = rng.integers(3, 7)
    pts = rng.uniform(-3.0, 3.0, size=(n_pts, 2))
    x, y, theta = 0.0, 0.0, 0.0
    idx = 0
    cmds = np.zeros((T, 2))
    for t in range(T):
        dx = pts[idx, 0] - x
        dy = pts[idx, 1] - y
        dist = float(np.hypot(dx, dy))
        if dist < 0.3 and idx < n_pts - 1:
            idx += 1
            continue
        desired_theta = np.arctan2(dy, dx)
        err = np.arctan2(np.sin(desired_theta - theta), np.cos(desired_theta - theta))
        v_cmd = float(np.clip(0.7 * np.cos(err) * min(dist, 1.0), 0.0, 0.9))
        omega_cmd = float(np.clip(1.2 * err, -1.4, 1.4))
        cmds[t] = (v_cmd, omega_cmd)
        # Lightweight forward sim (without lag) so the controller progresses.
        v = v_cmd
        omega = omega_cmd
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt
    return cmds


_GENERATORS = {
    "figure8": _figure8,
    "waypoint": _waypoint,
    "lawnmower": _lawnmower,
    "square": _square,
}


def generate_command_sequence(
    kind: str,
    T: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if kind not in _GENERATORS:
        raise ValueError(f"unknown trajectory type {kind!r}")
    return _GENERATORS[kind](T, dt, rng)
