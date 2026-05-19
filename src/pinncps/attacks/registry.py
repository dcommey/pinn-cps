"""Attack registry + convenience apply function."""
from __future__ import annotations

from typing import Dict, Type

import numpy as np

from .base import Attack, AttackResult, AttackSpec
from .gps_spoofing import GPSSpoofingAttack
from .sensor_bias import SensorBiasAttack
from .replay import ReplayAttack
from .packet import PacketAttack
from .command_injection import CommandInjectionAttack

_REGISTRY: Dict[str, Type[Attack]] = {
    GPSSpoofingAttack.kind: GPSSpoofingAttack,
    SensorBiasAttack.kind: SensorBiasAttack,
    ReplayAttack.kind: ReplayAttack,
    PacketAttack.kind: PacketAttack,
    CommandInjectionAttack.kind: CommandInjectionAttack,
}


def build_attack(kind: str) -> Type[Attack]:
    if kind not in _REGISTRY:
        raise KeyError(f"unknown attack {kind!r}; have {list(_REGISTRY)}")
    return _REGISTRY[kind]


def apply_attack(
    obs: np.ndarray,
    states: np.ndarray,
    commands: np.ndarray,
    spec: AttackSpec,
    rng: np.random.Generator,
) -> AttackResult:
    return build_attack(spec.kind).apply(obs, states, commands, spec, rng)
