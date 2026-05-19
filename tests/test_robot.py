"""Tests for the unicycle simulator."""
import numpy as np

from pinncps.sim import RobotParams, simulate_trajectory, generate_command_sequence


def test_battery_monotone_nonincreasing():
    p = RobotParams()
    rng = np.random.default_rng(0)
    cmds = generate_command_sequence("figure8", 100, p.dt, rng)
    s0 = np.array([0, 0, 0, 0, 0, p.battery_init], dtype=np.float64)
    states = simulate_trajectory(s0, cmds, p)
    assert np.all(np.diff(states[:, 5]) <= 0)
    assert states[-1, 5] < p.battery_init


def test_zero_command_zero_motion():
    p = RobotParams()
    s0 = np.array([1.0, 2.0, 0.4, 0.0, 0.0, 50.0], dtype=np.float64)
    cmds = np.zeros((20, 2))
    states = simulate_trajectory(s0, cmds, p)
    # Position should not change because v starts at 0 and stays at 0.
    np.testing.assert_allclose(states[:, 0], s0[0], atol=1e-6)
    np.testing.assert_allclose(states[:, 1], s0[1], atol=1e-6)


def test_kinematic_residual_small_on_clean_states():
    """Nominal RK4 trajectory should have small kinematic residual (trapezoidal)."""
    p = RobotParams()
    rng = np.random.default_rng(0)
    cmds = generate_command_sequence("waypoint", 200, p.dt, rng)
    s0 = np.array([0, 0, 0, 0, 0, p.battery_init], dtype=np.float64)
    states = simulate_trajectory(s0, cmds, p)
    dt = p.dt
    x = states[:, 0]; y = states[:, 1]; th = states[:, 2]
    v = states[:, 3]; w = states[:, 4]
    rx = (x[1:] - x[:-1]) / dt - 0.5 * (v[:-1] * np.cos(th[:-1]) + v[1:] * np.cos(th[1:]))
    ry = (y[1:] - y[:-1]) / dt - 0.5 * (v[:-1] * np.sin(th[:-1]) + v[1:] * np.sin(th[1:]))
    rth = (th[1:] - th[:-1]) / dt - 0.5 * (w[:-1] + w[1:])
    # Trapezoidal residual is exact for affine derivatives; for the unicycle
    # nonlinearity it is O(dt^2) at steady state but larger during velocity
    # transients (the first ~tau seconds).  The relevant bound for the PINN
    # residual loss is the steady-state regime.
    assert np.median(np.abs(rx)) < 5e-3
    assert np.median(np.abs(ry)) < 5e-3
    assert np.median(np.abs(rth)) < 5e-3
    assert np.max(np.abs(rx)) < 5e-2
    assert np.max(np.abs(ry)) < 5e-2
    assert np.max(np.abs(rth)) < 5e-2
