# Physics-Residual Monitoring for MR.CLAM Robot Telemetry

This repository contains the code for a small research pipeline on false-data
injection detection in mobile-robot telemetry. The main experiment uses the
public UTIAS MR.CLAM datasets: real Vicon pose traces and logged odometry
commands are converted into monitoring episodes, then controlled cyber-channel
perturbations are applied to the observation stream.

The current detector is intentionally simple. It uses a trapezoidal kinematic
residual as the primary alarm score, with a learned one-step predictor kept as
an auxiliary diagnostic channel. That turned out to be the honest story in the
experiments: the residual signal matters more than the neural architecture.

## What is included

- MR.CLAM preprocessing for all nine public sub-datasets.
- Attack construction for GPS-like spoofing, sensor bias, replay, and packet
  hold/delay perturbations.
- Baselines: MLP, LSTM, GRU, LSTM autoencoder, Isolation Forest, one-class SVM,
  and an EKF residual detector.
- Evaluation code for F1, ROC-AUC, PR-AUC, delay, miss rate, score ablations,
  and robot-held-out transfer.
- Unit tests for the core attack, metric, robot, and residual logic.

Generated datasets, trained checkpoints, result tables, figures, and manuscript
build artifacts are intentionally not tracked. They are reproducible from the
scripts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the tests:

```bash
pytest tests
```

## Data

Download the UTIAS MR.CLAM archives from the original project site and extract
them under:

```text
data/raw/mrclam/
```

The suite accepts the usual extracted directory names, including
`MRCLAM_Dataset1` and the Dataset 4 `MRSLAM_Dataset4` spelling.

## Reproducing the Main Experiment

Run the full real-data suite:

```bash
python scripts/run_mrclam_suite.py --retrain
```

Then run the score ablations:

```bash
python scripts/evaluate_score_variants.py
python scripts/evaluate_score_variants.py \
  --pattern 'configs/generated/mrclam_d1_holdout_r[1-5].yaml' \
  --out-prefix mrclam_holdout_score_variants
```

To regenerate the figures used for a single MR.CLAM run:

```bash
python scripts/make_figures.py --config configs/generated/mrclam_d1.yaml
```

Outputs are written under `results/`. That directory is ignored by git so the
repository stays lightweight.

## Notes

The repository does not redistribute MR.CLAM data. It also does not include
trained checkpoints or generated result artifacts. This keeps the public code
focused on the implementation and makes the experimental outputs traceable to a
fresh run.
