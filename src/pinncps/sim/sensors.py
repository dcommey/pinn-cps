"""Sensor noise model applied to ground-truth states.

We model each observation channel with additive Gaussian noise.  Per-channel
standard deviations come from the experiment SensorConfig.  Optional Bernoulli
``dropout`` simulates lost packets independently of the structured packet attack.
"""
from __future__ import annotations

import numpy as np


def apply_sensor_noise(
    states: np.ndarray,
    sensor_cfg,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add independent Gaussian noise per channel.

    Returns an observation array of the same shape as `states`.
    """
    std = np.array(
        [
            sensor_cfg.noise_pos,
            sensor_cfg.noise_pos,
            sensor_cfg.noise_heading,
            sensor_cfg.noise_vel,
            sensor_cfg.noise_omega,
            sensor_cfg.noise_battery,
        ],
        dtype=np.float64,
    )
    noise = rng.normal(0.0, std, size=states.shape)
    obs = states + noise
    if sensor_cfg.dropout_prob > 0.0:
        mask = rng.uniform(size=states.shape[:-1]) < sensor_cfg.dropout_prob
        # On dropout, hold previous observation.
        for t in range(1, obs.shape[0]):
            if mask[t]:
                obs[t] = obs[t - 1]
    return obs
