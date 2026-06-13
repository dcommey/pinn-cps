# PINN-CPS

Code for physics-residual monitoring of false-data injection attacks in
mobile-robot telemetry.

The main experiments use UTIAS MR.CLAM robot traces. The pipeline converts real
Vicon poses and logged odometry commands into monitoring episodes, injects
cyber-channel perturbations, and evaluates residual-based detectors against
classical and neural baselines.

## Contents

- MR.CLAM preprocessing and experiment configs.
- Attacks: pose spoofing, sensor bias, replay, and packet hold/delay.
- Detectors: physics-residual monitor, neural predictors, OC-SVM, Isolation
  Forest, and EKF residual baseline.
- Evaluation scripts for F1, ROC-AUC, PR-AUC, detection delay, ablations, and
  robot-held-out transfer.
- Unit tests for attacks, metrics, robot simulation, and residual logic.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data Layout

Place the extracted MR.CLAM folders under:

```text
data/raw/mrclam/
```

Expected names include `MRCLAM_Dataset1` through `MRCLAM_Dataset9`; Dataset 4
may also appear as `MRSLAM_Dataset4`.

## Run

```bash
python scripts/run_mrclam_suite.py --retrain
python scripts/evaluate_score_variants.py
python scripts/evaluate_ocsvm_feature_ablation.py
```

Robot-held-out score variants:

```bash
python scripts/evaluate_score_variants.py \
  --pattern 'configs/generated/mrclam_d1_holdout_r[1-5].yaml' \
  --out-prefix mrclam_robot_holdout_score_variants
```

Regenerate figures for one run:

```bash
python scripts/make_figures.py --config configs/generated/mrclam_d1.yaml
```

Run tests:

```bash
pytest tests
```

Outputs are written under `results/`.
