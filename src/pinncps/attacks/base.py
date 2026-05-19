"""Base types for the attack registry.

An attack receives a clean observation series and the corresponding ground-truth
states / commands, and returns:

* attacked observations
* possibly modified commands (only command_injection does this)
* a binary per-timestep ground-truth label (1 where the attacked window is
  active, 0 otherwise)
* metadata describing the attack

This separation makes attacks composable and makes ground truth unambiguous
for evaluation (the labels are derived from the injection mask, not from a
detector's notion of "anomalous").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

ATTACK_TYPES = (
    "gps_spoofing",
    "sensor_bias",
    "replay",
    "packet",
    "command_injection",
)


@dataclass
class AttackSpec:
    """Per-trajectory attack specification."""

    kind: str
    severity: str = "stealthy"  # "stealthy" or "overt"
    start: int = 0  # inclusive start step (in observation index)
    end: int = -1  # exclusive end step, -1 means run to end
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackResult:
    obs: np.ndarray
    commands: np.ndarray
    labels: np.ndarray  # (T,) {0,1}
    spec: AttackSpec


class Attack:
    """Abstract attack. Subclasses implement :meth:`apply`."""

    kind: str = "base"

    @classmethod
    def apply(
        cls,
        obs: np.ndarray,
        states: np.ndarray,
        commands: np.ndarray,
        spec: AttackSpec,
        rng: np.random.Generator,
    ) -> AttackResult:  # pragma: no cover - abstract
        raise NotImplementedError
