"""2D unicycle robot dynamics with battery model.

State vector (size 6):
    s = [x, y, theta, v, omega, battery]

Observation vector exposes the same states (sensors are co-located with state
in this minimal model; the FDI attacks operate on the observation channel and,
for command injection, on the control input ``u``).

Control vector (size 2):
    u = [v_cmd, omega_cmd]

Continuous-time dynamics::

    x_dot      = v cos(theta)
    y_dot      = v sin(theta)
    theta_dot  = omega
    v_dot      = (v_cmd - v) / tau_v
    omega_dot  = (omega_cmd - omega) / tau_omega
    batt_dot   = - (e_idle + e_lin |v| + e_ang |omega|)

The first-order lags on v and omega make the closed-loop physically realistic
without requiring full torque dynamics: the controller cannot teleport speed.

These equations are intentionally analytically simple so that the PINN
residual loss can be expressed without auto-differentiating through a black-box
integrator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATE_DIM = 6  # x, y, theta, v, omega, battery
OBS_DIM = 6
CTRL_DIM = 2  # v_cmd, omega_cmd


@dataclass
class RobotParams:
    """Physical and actuator parameters for the unicycle robot."""

    dt: float = 0.1
    tau_v: float = 0.4  # velocity tracking time constant (s)
    tau_omega: float = 0.3  # yaw-rate tracking time constant (s)
    v_max: float = 1.0
    omega_max: float = 1.5
    energy_idle: float = 0.05
    energy_lin: float = 0.10
    energy_ang: float = 0.05
    battery_init: float = 100.0


def dynamics(s: np.ndarray, u: np.ndarray, p: RobotParams) -> np.ndarray:
    """Continuous-time state derivative.

    Both `s` and `u` may be batched on the leading axis.
    """
    x, y, theta, v, omega, batt = (
        s[..., 0], s[..., 1], s[..., 2], s[..., 3], s[..., 4], s[..., 5],
    )
    v_cmd, omega_cmd = u[..., 0], u[..., 1]
    # Saturate commands to the actuator envelope.
    v_cmd = np.clip(v_cmd, -p.v_max, p.v_max)
    omega_cmd = np.clip(omega_cmd, -p.omega_max, p.omega_max)
    dx = v * np.cos(theta)
    dy = v * np.sin(theta)
    dtheta = omega
    dv = (v_cmd - v) / p.tau_v
    domega = (omega_cmd - omega) / p.tau_omega
    dbatt = -(p.energy_idle + p.energy_lin * np.abs(v) + p.energy_ang * np.abs(omega))
    return np.stack([dx, dy, dtheta, dv, domega, dbatt], axis=-1)


def rk4_step(s: np.ndarray, u: np.ndarray, p: RobotParams) -> np.ndarray:
    """One classical Runge-Kutta 4 integration step."""
    dt = p.dt
    k1 = dynamics(s, u, p)
    k2 = dynamics(s + 0.5 * dt * k1, u, p)
    k3 = dynamics(s + 0.5 * dt * k2, u, p)
    k4 = dynamics(s + dt * k3, u, p)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_trajectory(
    s0: np.ndarray,
    commands: np.ndarray,
    p: RobotParams,
) -> np.ndarray:
    """Roll out the closed-loop system given a sequence of commands.

    Parameters
    ----------
    s0 : (STATE_DIM,) initial state
    commands : (T, CTRL_DIM) control sequence
    p : robot parameters

    Returns
    -------
    states : (T+1, STATE_DIM) trajectory including the initial state.
    """
    T = commands.shape[0]
    states = np.empty((T + 1, STATE_DIM), dtype=np.float64)
    states[0] = s0
    s = s0
    for t in range(T):
        s = rk4_step(s, commands[t], p)
        # Battery floored at 0.
        s[5] = max(s[5], 0.0)
        states[t + 1] = s
    return states
