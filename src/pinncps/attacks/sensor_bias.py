"""Sensor bias: constant additive offset on velocity or heading channel."""
from __future__ import annotations

import numpy as np

from .base import Attack, AttackResult, AttackSpec


class SensorBiasAttack(Attack):
    kind = "sensor_bias"

    @classmethod
    def apply(cls, obs, states, commands, spec: AttackSpec, rng):
        T = obs.shape[0]
        end = T if spec.end < 0 else spec.end
        labels = np.zeros(T, dtype=np.int64)
        obs = obs.copy()

        channel = spec.params.get("channel", None)
        if channel is None:
            channel = rng.choice(["v", "theta"])
        if spec.severity == "stealthy":
            mag = spec.params.get("magnitude", 0.05 if channel == "v" else 0.05)
        else:
            mag = spec.params.get("magnitude", 0.20 if channel == "v" else 0.25)
        sign = rng.choice([-1.0, 1.0])
        if channel == "v":
            obs[spec.start:end, 3] += sign * mag
        elif channel == "theta":
            obs[spec.start:end, 2] += sign * mag
        elif channel == "omega":
            obs[spec.start:end, 4] += sign * mag
        else:
            raise ValueError(f"unknown bias channel {channel!r}")
        labels[spec.start:end] = 1
        return AttackResult(obs=obs, commands=commands.copy(), labels=labels, spec=spec)
