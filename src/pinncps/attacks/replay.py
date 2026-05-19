"""Replay attack: substitute current observations with an older window."""
from __future__ import annotations

import numpy as np

from .base import Attack, AttackResult, AttackSpec


class ReplayAttack(Attack):
    kind = "replay"

    @classmethod
    def apply(cls, obs, states, commands, spec: AttackSpec, rng):
        T = obs.shape[0]
        end = T if spec.end < 0 else spec.end
        labels = np.zeros(T, dtype=np.int64)
        obs = obs.copy()

        if spec.severity == "stealthy":
            offset = spec.params.get("offset", 10)  # short replay, harder to spot
        else:
            offset = spec.params.get("offset", 30)
        src_start = max(0, spec.start - offset)
        length = min(end - spec.start, spec.start - src_start)
        if length <= 0:
            return AttackResult(obs=obs, commands=commands.copy(), labels=labels, spec=spec)
        obs[spec.start:spec.start + length] = obs[src_start:src_start + length]
        labels[spec.start:spec.start + length] = 1
        return AttackResult(obs=obs, commands=commands.copy(), labels=labels, spec=spec)
