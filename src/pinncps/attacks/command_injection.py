"""False command injection.

The attacker modifies the control packet that reaches the actuator.  The
observations themselves are not touched: the resulting trajectory is the
physically-correct response to the corrupted command, but it disagrees with the
commands recorded in the supervisory log (which we expose to the detector).

This is the most interesting case: the observations look noisy-but-consistent,
yet the command-conditioned twin should flag a violation.
"""
from __future__ import annotations

import numpy as np

from .base import Attack, AttackResult, AttackSpec
from ..sim.robot import simulate_trajectory, RobotParams
from ..sim.sensors import apply_sensor_noise


class CommandInjectionAttack(Attack):
    kind = "command_injection"

    @classmethod
    def apply(cls, obs, states, commands, spec: AttackSpec, rng):
        """Note: this attack must be applied *before* the sensor model because it
        actually changes the robot's trajectory.  We accept the original clean
        observations and re-simulate.

        Extra params expected via ``spec.params``:
            robot_params: RobotParams used for the simulation
            sensor_cfg:   sensor noise config
            s0:           initial state used to (re)simulate
        """
        robot_params: RobotParams = spec.params["robot_params"]
        sensor_cfg = spec.params["sensor_cfg"]
        s0 = spec.params["s0"]

        T = commands.shape[0]
        end = T if spec.end < 0 else spec.end
        if spec.severity == "stealthy":
            v_scale = spec.params.get("v_scale", 1.0 + rng.choice([-1, 1]) * 0.15)
            w_scale = spec.params.get("w_scale", 1.0 + rng.choice([-1, 1]) * 0.15)
        else:
            v_scale = spec.params.get("v_scale", 1.0 + rng.choice([-1, 1]) * 0.4)
            w_scale = spec.params.get("w_scale", 1.0 + rng.choice([-1, 1]) * 0.5)

        actual_cmds = commands.copy()
        actual_cmds[spec.start:end, 0] *= v_scale
        actual_cmds[spec.start:end, 1] *= w_scale

        # Re-simulate physical states under the corrupted commands.
        new_states = simulate_trajectory(s0, actual_cmds, robot_params)
        # Sensors observe the new physical reality.
        new_obs = apply_sensor_noise(new_states, sensor_cfg, rng)

        # Detector sees the *logged* (intended) commands, not the corrupted ones,
        # so commands returned here are the original logged commands.
        labels = np.zeros(T + 1, dtype=np.int64)  # obs is length T+1
        labels[spec.start:end] = 1
        return AttackResult(obs=new_obs, commands=commands.copy(), labels=labels, spec=spec)
