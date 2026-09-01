# CS2 Match Prediction

An end-to-end probabilistic machine-learning system for predicting professional
Counter-Strike 2 matches from strictly historical pre-match information. It covers the
full pipeline — data auditing, chronological feature engineering, model training and
evaluation, Monte-Carlo tournament simulation, and a deployed FastAPI + Next.js
application — with every final model frozen before its evaluation.

## Overview

Two prediction modes serve two different tasks:

- **Pre-veto series prediction** — predicts one BO1/BO3/BO5 series before the map veto,
  using a Random Forest (V2) over 17 engineered historical features. Also drives the
  tournament simulator.
- **Known-map prediction** — predicts one selected map with an XGBoost model (V3) over
  120 engineered features; per-map probabilities are composed into an exact series
  probability by dynamic programming.

## Methodology

All features are built strictly chronologically: every feature for a match is computed
only from information with `datetime < prediction_time` — no current or future match
information ever enters a feature. Feature families include ELO-style team strength,
recent form, activity, and roster and map statistics. Data is split into chronological
train / validation / test partitions; hyperparameters are tuned with train-only
expanding-window cross-validation, and training uses mirrored (team-order-swapped)
augmentation to enforce prediction symmetry.

## Models

| Task | Model | Role |
|---|---|---|
| Series | Logistic Regression V2 | Linear baseline |
| Series | Random Forest V2 | Final pre-veto model |
| Series | XGBoost V2 | Boosting comparison |
| Known-map | XGBoost V3 | Final known-map model |

Earlier V1 models exist only as development history; the deployed application uses
Random Forest V2 and XGBoost V3.

## Results

**Series validation (chronological held-out period)** — Random Forest V2 achieved the
strongest probability-quality metrics among the tuned series models:

| Metric | RF V2 |
|---|---|
| Log Loss | 0.6514 |
| Brier | 0.2298 |
| Accuracy | 0.6068 |
| ROC-AUC | 0.6566 |

**Known-map sealed test (opened once, after freezing)** — XGBoost V3:

| Metric | XGB V3 |
|---|---|
| Log Loss | 0.6521 |
| Brier | 0.2301 |
| Accuracy | 0.6132 |
| ROC-AUC | 0.6489 |

Its performance was statistically indistinguishable from the strong overall-ELO
baseline under paired bootstrap analysis.

**External event evaluation (IEM Cologne Major 2026)** — the frozen RF V2, scored on
106 official matches only after the pre-event freeze:

| Metric | Frozen RF V2 |
|---|---|
| Log Loss | 0.6316 |
| Brier | 0.2208 |
| Accuracy | 0.6415 |
| ROC-AUC | 0.6968 |

This is an external-event evaluation on one real tournament — encouraging, but not
proof of universal generalization.

## Tournament Simulation

The full 32-team Major (Swiss stages + playoffs) was simulated 50,000 times before the
event from a frozen pre-event probability matrix — 5.3 million simulated matches in
total. The frozen forecast ranked the actual champion, Team Falcons, 4th of 32 with an
8.93% championship probability.

## Project Architecture

```
Raw match data
    ↓
Cleaning + chronological feature engineering
    ↓
├── Series prediction → Random Forest V2
└── Known-map prediction → XGBoost V3
    ↓
Evaluation (validation · sealed test · external event)
    ↓
Tournament simulation / API / web application
```

## Repository Structure

| Directory | Contents |
|---|---|
| `data/` | Raw data, engineered features, splits, frozen evaluation and deployment artifacts. |
| `data_preparation/` | Raw-data audit, cleaning, canonical datasets, team-identity policy. |
| `feature_engineering/` | Chronological feature engines, per-model preprocessing, state snapshots. |
| `training/` | Model training and tuning code (series and map models). |
| `models/` | Frozen final model artifacts (RF V2, XGB V3). |
| `evaluation/` | Feature ablations, sealed internal test, bootstrap uncertainty, Cologne evaluation. |
| `tournament/` | Swiss + playoff rules engine and the Monte-Carlo simulation pipeline. |
| `application/` | Inference core, model-grounded explanations, FastAPI, tournament service. |
| `config/` | Feature schemas, frozen evaluation protocols, application registries. |
| `validation/` | Per-phase validation gates that re-verify artifacts, hashes and invariants. |
| `reports/` | Phase reports, figures, tables, and the presentation. |
| `tests/` | Test suite (features, models, evaluation, tournament, application). |
| `web/` | Next.js PWA frontend (calls the API only — no model logic). |

## Installation

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running

All Python commands run from the repository root.

```
# API (FastAPI, http://127.0.0.1:8000, health: /api/v1/health/ready)
python -m application.api.run_application_api

# Web frontend (proxies /api/* to the backend)
cd web && npm install && npm run dev

# Tests
python -m pytest

# Validation gates (example)
python -m validation.validate_phase9d
```

A fresh clone needs the large rebuildable state snapshots under `data/interim/`
(~220 MB, regenerated via `feature_engineering/state/`, intentionally not committed)
before the API's readiness check passes.

## Reproducibility

All randomized steps use a fixed seed (42). Splits are chronological and versioned;
the final models, their preprocessing specifications, and the evaluation protocols are
frozen as committed artifacts, and all evaluation results are saved to disk. The
application verifies model/state/config hashes at startup and refuses to serve if
anything has drifted; the Cologne pre-event simulation and its inputs are recorded in
immutable, hash-carrying receipts.

## Limitations

- Single public dataset source; static historical snapshot (data ends 28 Jun 2026).
- BO5 series are nearly absent from the training data.
- No fitted probability calibration layer.
- Concept drift and roster changes are only partially modelled.
- The known-map XGBoost did not significantly outperform the overall-ELO baseline.

## Tech Stack

Python, NumPy, pandas, scikit-learn, XGBoost, FastAPI, Next.js.

## Data Source

The raw data is a public dataset of professional CS2 matches (Jan 2023 – Jun 2026,
three tournament tiers), stored unmodified under `data/raw/` and audited in
`reports/phases/data_audit.md`.
