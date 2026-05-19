"""Smoke tests for the attack registry."""
import numpy as np

from pinncps.attacks import AttackSpec, apply_attack
from pinncps.sim import RobotParams, generate_command_sequence, simulate_trajectory, apply_sensor_noise
from pinncps.utils.config import SensorConfig


def _make_episode(kind: str, T: int = 80, seed: int = 0):
    p = RobotParams()
    rng = np.random.default_rng(seed)
    cmds = generate_command_sequence(kind, T, p.dt, rng)
    s0 = np.array([0, 0, 0, 0, 0, p.battery_init], dtype=np.float64)
    states = simulate_trajectory(s0, cmds, p)
    obs = apply_sensor_noise(states, SensorConfig(), rng)
    return p, rng, states, obs, cmds


def test_gps_spoofing_drifts_position():
    p, rng, states, obs, cmds = _make_episode("figure8")
    spec = AttackSpec(kind="gps_spoofing", severity="overt", start=20, end=70)
    res = apply_attack(obs, states, cmds, spec, rng)
    assert res.labels[10:20].sum() == 0  # before
    assert res.labels[20:70].sum() == 50  # during
    assert np.max(np.abs(res.obs[20:70, 0] - obs[20:70, 0])) > 0.1


def test_replay_substitutes_history():
    p, rng, states, obs, cmds = _make_episode("waypoint")
    spec = AttackSpec(kind="replay", severity="overt", start=40, end=70)
    res = apply_attack(obs, states, cmds, spec, rng)
    # Replayed window should equal the source window
    assert res.labels[40:70].any()
    assert np.allclose(res.obs[40:70, :2], obs[40 - 30:40 - 30 + (70 - 40), :2], atol=1e-6)


def test_command_injection_changes_trajectory():
    p, rng, states, obs, cmds = _make_episode("square")
    from pinncps.utils.config import SensorConfig as SC
    spec = AttackSpec(
        kind="command_injection",
        severity="overt",
        start=20,
        end=70,
        params=dict(robot_params=p, sensor_cfg=SC(), s0=states[0]),
    )
    res = apply_attack(obs, states, cmds, spec, rng)
    # The resimulated trajectory should diverge from the original after the
    # attack start.
    diff = np.linalg.norm(res.obs[70:, :2] - obs[70:, :2], axis=-1)
    assert diff.mean() > 0.05
