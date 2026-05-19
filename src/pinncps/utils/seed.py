"""Seeding utilities for reproducibility."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms can be slow; opt-in by env var.
        if os.environ.get("PINNCPS_DETERMINISTIC") == "1":
            torch.use_deterministic_algorithms(True)
    except Exception:
        pass
