"""Packet delay / drop attack on the sensor channel.

For each compromised step we either drop the packet (hold previous value) or
delay it by a few steps.  The label is 1 on every compromised step.
"""
from __future__ import annotations

import numpy as np

from .base import Attack, AttackResult, AttackSpec


class PacketAttack(Attack):
    kind = "packet"

    @classmethod
    def apply(cls, obs, states, commands, spec: AttackSpec, rng):
        T = obs.shape[0]
        end = T if spec.end < 0 else spec.end
        labels = np.zeros(T, dtype=np.int64)
        obs = obs.copy()

        if spec.severity == "stealthy":
            drop_prob = spec.params.get("drop_prob", 0.15)
            max_delay = spec.params.get("max_delay", 1)
        else:
            drop_prob = spec.params.get("drop_prob", 0.4)
            max_delay = spec.params.get("max_delay", 3)
        for t in range(max(spec.start, 1), end):
            if rng.uniform() < drop_prob:
                # Hold the previous reading.
                obs[t] = obs[t - 1]
                labels[t] = 1
            elif max_delay > 0 and rng.uniform() < 0.5 * drop_prob:
                d = int(rng.integers(1, max_delay + 1))
                src = max(0, t - d)
                obs[t] = obs[src]
                labels[t] = 1
        return AttackResult(obs=obs, commands=commands.copy(), labels=labels, spec=spec)
