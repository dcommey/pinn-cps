"""Generate train / val / test datasets given a config."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pinncps.data import (
    generate_attack_dataset,
    generate_nominal_dataset,
    save_dataset,
)
from pinncps.utils import load_config, seed_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    out_dir = Path(args.out or ROOT / "data" / cfg.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = generate_nominal_dataset(cfg.data.n_train, cfg, rng)
    val = generate_nominal_dataset(cfg.data.n_val, cfg, rng)
    test = generate_attack_dataset(cfg.data.n_test_per_attack, cfg, rng)
    nominal_test = generate_nominal_dataset(
        max(40, cfg.data.n_val),
        cfg,
        rng,
    )
    save_dataset(train, out_dir / "train.npz")
    save_dataset(val, out_dir / "val.npz")
    save_dataset(test, out_dir / "test_attack.npz")
    save_dataset(nominal_test, out_dir / "test_nominal.npz")
    print(f"wrote datasets to {out_dir}")
    print(f"  train  {train['states'].shape}")
    print(f"  val    {val['states'].shape}")
    print(f"  test+  {test['states'].shape}")
    print(f"  test0  {nominal_test['states'].shape}")


if __name__ == "__main__":
    main()
