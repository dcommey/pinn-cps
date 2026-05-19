"""GPS spoofing: gradual drift injected on the (x, y) channels.

The stealthy variant uses a slow ramp whose magnitude per step is below the
sensor noise standard deviation, which is exactly the regime where statistical
detectors struggle but a physics-residual detector still sees inconsistency
with the velocity reading.
"""
from __future__ import annotations

import numpy as np

from .base import Attack, AttackResult, AttackSpec


class GPSSpoofingAttack(Attack):
    kind = "gps_spoofing"

    @classmethod
    def apply(cls, obs, states, commands, spec: AttackSpec, rng):
        T = obs.shape[0]
        end = T if spec.end < 0 else spec.end
        labels = np.zeros(T, dtype=np.int64)
        obs = obs.copy()

        if spec.severity == "stealthy":
            drift_per_step = spec.params.get("drift", 0.012)
        else:
            drift_per_step = spec.params.get("drift", 0.06)
        # Random direction.
        ang = rng.uniform(0, 2 * np.pi)
        dx_step = drift_per_step * np.cos(ang)
        dy_step = drift_per_step * np.sin(ang)
        for t in range(spec.start, end):
            k = t - spec.start + 1
            obs[t, 0] += dx_step * k
            obs[t, 1] += dy_step * k
            labels[t] = 1
        return AttackResult(obs=obs, commands=commands.copy(), labels=labels, spec=spec)
