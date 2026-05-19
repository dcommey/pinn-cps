"""YAML config loading with dataclass schema."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SimConfig:
    dt: float = 0.1
    horizon: int = 200
    v_max: float = 1.0
    omega_max: float = 1.5
    energy_idle: float = 0.05
    energy_lin: float = 0.10
    energy_ang: float = 0.05
    battery_init: float = 100.0
    trajectory_types: List[str] = field(
        default_factory=lambda: ["figure8", "waypoint", "lawnmower", "square"]
    )


@dataclass
class SensorConfig:
    noise_pos: float = 0.02
    noise_vel: float = 0.02
    noise_heading: float = 0.01
    noise_omega: float = 0.01
    noise_battery: float = 0.05
    dropout_prob: float = 0.0


@dataclass
class AttackConfig:
    # Fraction of test trajectories per attack type.
    types: List[str] = field(
        default_factory=lambda: [
            "gps_spoofing",
            "sensor_bias",
            "replay",
            "packet",
            "command_injection",
        ]
    )
    # Each attack has overt and stealthy parameter sets, see attacks/registry.
    severity_levels: List[str] = field(default_factory=lambda: ["stealthy", "overt"])
    start_frac_low: float = 0.3
    start_frac_high: float = 0.7


@dataclass
class DataConfig:
    n_train: int = 400
    n_val: int = 80
    n_test_per_attack: int = 60
    window_len: int = 32
    stride: int = 4


@dataclass
class ModelConfig:
    hidden: int = 64
    n_layers: int = 2
    lstm_hidden: int = 64
    ae_latent: int = 16
    dropout: float = 0.0
    obs_dim: int = 6


@dataclass
class PINNLossConfig:
    lambda_data: float = 1.0
    lambda_kin: float = 1.0
    lambda_energy: float = 0.5
    lambda_smooth: float = 0.05


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 6
    device: str = "cpu"


@dataclass
class ExperimentConfig:
    seed: int = 0
    name: str = "default"
    sim: SimConfig = field(default_factory=SimConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pinn_loss: PINNLossConfig = field(default_factory=PINNLossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _from_dict(cls, d: Dict[str, Any]):
    if d is None:
        return cls()
    # only fill known fields
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in d.items() if k in valid})


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    cfg = ExperimentConfig(
        seed=raw.get("seed", 0),
        name=raw.get("name", path.stem),
        sim=_from_dict(SimConfig, raw.get("sim")),
        sensor=_from_dict(SensorConfig, raw.get("sensor")),
        attack=_from_dict(AttackConfig, raw.get("attack")),
        data=_from_dict(DataConfig, raw.get("data")),
        model=_from_dict(ModelConfig, raw.get("model")),
        pinn_loss=_from_dict(PINNLossConfig, raw.get("pinn_loss")),
        train=_from_dict(TrainConfig, raw.get("train")),
    )
    return cfg
