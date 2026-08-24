# CS2 Match Prediction

An end-to-end probabilistic machine-learning system for professional Counter-Strike 2:
pre-match series prediction, known-map prediction, Monte-Carlo simulation of a full Major
tournament, model-grounded explanations, a FastAPI serving layer, and a Next.js PWA
frontend. Every model is developed under strict chronological (leakage-safe) evaluation,
frozen before its tests, and validated against a real external event (IEM Cologne Major
2026).

## Project Overview

```
CS2 Match Prediction
│
├── Data
│   └── audited professional CS2 matches
│
├── Feature Engineering
│   ├── team strength / ELO
│   ├── recent form
│   ├── map features
│   └── player / roster features
│
├── Models
│   ├── Logistic Regression
│   ├── Random Forest
│   └── XGBoost
│
├── Evaluation
│   ├── chronological validation
│   ├── sealed map test
│   └── Cologne external evaluation
│
├── Tournament Simulation
│   └── 50,000 Monte-Carlo Majors
│
└── Application
    ├── inference
    ├── explanations
    ├── FastAPI
    └── Next.js PWA
```

Two prediction modes serve two different tasks:

```
Maps unknown (pre-veto)   →  Random Forest V2                       →  P(series win)
Maps known  (post-veto)   →  XGBoost V3 per map → DP composition    →  P(series win)

Tournament: RF V2 probabilities → Bernoulli Monte Carlo → Major outcome distributions
```

This tree is the *logical* architecture shown to reviewers. Physical file paths are kept
stable on purpose — see [Reproducibility & Provenance](#reproducibility--provenance).

## Architecture

```
Raw Match Data
      ↓
Audit & Cleaning
      ↓
Chronological Historical State
      ↓
Feature Engineering
      ↓
LR / RF / XGBoost
      ↓
Evaluation
      ↓
Frozen Models
      ↓
Tournament Simulation
      ↓
FastAPI
      ↓
Next.js PWA
```

## Repository Structure

| Directory | Contents |
|---|---|
| `config/` | Feature schemas, frozen evaluation protocols, tournament definition, application registries. |
| `data/` | Raw, intermediate, engineered, modeling, evaluation and deployment artifacts (see lifecycle below). |
| `models/` | Frozen trained model artifacts and their metadata. |
| `scripts/` | All research, feature, training, evaluation, tournament and application logic. Physically flat **intentionally**, for provenance stability. |
| `tests/` | Phase-named test suite mirroring the project's audit trail. |
| `reports/` | Phase reports, figures, tables and presentation material. |
| `web/` | Next.js PWA frontend (calls the API only — contains no model logic). |
| `src/` | Intentionally empty — validation gates assert it stays empty ("no repo restructuring"). |
| `reference/` | Course reference material. |

Data lifecycle inside `data/`:
`raw/` (immutable inputs) → `interim/` (cleaned/canonical) → `features/` (engineered
datasets and state snapshots) → `modeling/` (splits, CV folds, preprocessing specs,
selected configs) → `evaluation/` (frozen test and Cologne artifacts) → `deployment/`
(post-Cologne application state and receipts). `tournaments/` holds frozen tournament
source snapshots. `processed/` is optional local legacy/reserved space, not an active
repository component.

## Where to Start

Recommended reading order for reviewers:

1. `reports/data_audit.md` — raw-data audit (including the broken winner label it caught)
2. `reports/phase3_feature_engineering.md` — leakage-safe feature engine
3. `reports/model_comparison_tuned_v1features.md` — tuned LR vs RF vs XGB comparison
4. `reports/phase7_internal_test_results.md` — known-map model, sealed internal test
5. `reports/phase8d_cologne_pre_event_simulation.md` — frozen pre-event Major simulation
6. `reports/phase8e_cologne_simulation_vs_reality.md` — simulation vs the real Major
7. `reports/phase9b_application_inference_core.md` — application inference core
8. `reports/phase9d_application_api.md` — FastAPI layer
9. `reports/phase9e_application_tournament_service.md` — Major simulation service
10. `reports/phase10a_pwa_foundation_predict.md` — PWA frontend

## Models

| Task | Model | Purpose |
|---|---|---|
| Pre-veto series prediction | **Random Forest V2** | Predict a match before the map veto (also drives the tournament simulator) |
| Known-map prediction | **XGBoost V3** (+ dynamic-programming series composition) | Predict each selected map, compose exact P(series win) |
| Development baseline | Logistic Regression (from scratch) | Simple, interpretable linear baseline |
| Development comparison | Random Forest / XGBoost | Compared under the same chronological methodology |

RF V2 and XGB V3 serve **different application tasks** — they are not competitors on a
single final task.

## Data & Evaluation

- Professional CS2 matches, 2023–2026, three tournament tiers.
- Strictly chronological processing: every feature is computed from information available
  before the match; no future information enters historical features.
- Chronological train / validation / test split; hyperparameters tuned with train-only
  expanding-window CV.
- The known-map model was scored once on a **sealed test** partition, opened only after
  the model was frozen.
- **Frozen Cologne external evaluation**: model and state frozen before IEM Cologne Major
  2026; 50,000 pre-event tournament simulations permanently saved; the 106 official Major
  matches were compared against those frozen predictions only afterwards.

## Main Results

Three separate evaluations of two different models on different data — they are **not**
directly comparable to each other:

| Evaluation | Model / setting | Metrics |
|---|---|---|
| Pre-veto **validation** (held-out chronological period) | RF V2, series-level | AUC ≈ 0.657 · Log Loss ≈ 0.651 · Brier ≈ 0.230 |
| Known-map **sealed test** (internal, opened once) | XGB V3, map-level | Accuracy ≈ 0.613 · AUC ≈ 0.649 · Log Loss ≈ 0.652 · Brier ≈ 0.230 |
| Cologne **external event** (106 official matches) | RF V2, frozen pre-event | Accuracy ≈ 0.642 · AUC ≈ 0.697 · Log Loss ≈ 0.632 · Brier ≈ 0.221 |

The Cologne result is a demonstration on one real external event — encouraging, but not
proof of universal generalization. The actual champion (Team Falcons) carried a pre-event
championship probability of 8.9% — rank 4 of 32 in the frozen forecast.

## Script Guide

`scripts/` is physically flat (see provenance note below). Logical grouping:

### Data Audit & Cleaning
- `audit_data.py` — full raw-data audit; caught the broken `team1_win` label
- `build_canonical_datasets.py` — conservative cleaning into canonical series/map tables
- `orientation_analysis.py` — analysis of the team-ordering bias in the raw data

### Team Identity
- `team_identity_analysis.py` — shows raw team IDs are per-match, not persistent
- `build_team_identity_policy.py` / `unresolved_team_review.py` — canonical name policy and manual-review trail

### Feature Engineering
- `feature_engine.py` — core chronological series feature engine (ELO, form, activity); shared by training and live inference
- `map_feature_engine.py`, `modern_map_feature_engine.py`, `team_form_engine.py`, `player_roster_feature_engine.py` — map / form / roster state engines
- `rich_map_feature_composer.py`, `rich_modern_map_feature_composer.py` — compose the full known-map feature vector (historical and future/synthetic paths)
- `build_series_features_v1.py` … `build_series_features_v4_roster.py`, `build_map_features_v1.py` … `build_map_features_v3_modern_map.py` — feature dataset builders
- `build_series_split_v1.py`, `build_map_split_v1.py`, `build_map_cv_folds_v1.py` — chronological splits and CV folds
- `build_pre_cologne_*.py` / `build_deployment_*.py` — frozen pre-Cologne and post-Cologne state snapshots
- `preprocessing_*.py` — per-model input preprocessing (fit on train only, saved as artifacts)

### Training
- `train_logistic_regression_v{1,2}.py`, `train_random_forest_v{1,2}.py`, `train_xgboost_v{1,2}.py` — series-model training
- `*_tuning_*.py`, `random_forest_cv_folds_v2.py` — train-only chronological hyperparameter search
- `train_map_models_v1.py`, `finalize_map_xgboost_v3.py` — known-map models; the final frozen XGB V3
- `models/` (subpackage) — from-scratch LR implementation and RF/XGB model wrappers/tuning

### Evaluation
- `evaluate_series_feature_sets_v{2,3,4}.py`, `evaluate_map_feature_sets_v3.py` — paired feature-set ablations under frozen configs
- `freeze_phase7_protocol.py`, `evaluate_phase7_test_once.py`, `phase7_test_bootstrap.py`, `phase7_test_reports.py`, `phase7_test_visualizations.py` — sealed internal test (protocol frozen first, scored once, cluster-bootstrap uncertainty)
- `reconcile_cologne_actual_results.py`, `phase8e_*.py`, `build_phase8e_*.py` — Cologne simulation-vs-reality evaluation (results opened only after the pre-event freeze)

### Tournament Simulation
- `tournament_engine.py` — pure Swiss + playoff rules engine (Valve rulebook; ML-free)
- `pre_veto_series_predictor.py` — frozen RF V2 prediction adapter
- `generate_cologne_pre_event_probability_matrix.py` — all 2,976 matchup probabilities, precomputed
- `cologne_pre_event_simulation.py`, `run_phase8d_pipeline.py`, `phase8d_figures.py` — the 50,000-run frozen pre-event simulation

### Application
- `application_inference.py` — versioned inference core (both prediction modes, DP series composition)
- `application_explanations.py` — model-grounded factor attributions (TreeSHAP for XGB, exact tree-path decomposition for RF)
- `application_api.py` / `run_application_api.py` — FastAPI transport with startup contract verification
- `application_tournament_service.py` / `application_tournament_router.py` — Major simulation service (historical frozen views + interactive simulation)
- `build_application_*.py`, `build_explanation*.py` — application registries, fixtures and receipts

### Validation
- `validate_phase*.py` — 28 phase-specific validation gates that re-verify artifacts,
  hashes and invariants; together they act as the project's structural regression suite.

## Important Entry Points

| Goal | File |
|---|---|
| Run the API | `scripts/run_application_api.py` |
| Application API | `scripts/application_api.py` |
| Inference | `scripts/application_inference.py` |
| Explanations | `scripts/application_explanations.py` |
| Tournament engine | `scripts/tournament_engine.py` |
| Tournament service | `scripts/application_tournament_service.py` |
| Frontend | `web/` |
| Tests | `tests/` |

## Running the Project

Backend (Windows):

```
.venv\Scripts\activate
python scripts\run_application_api.py
```

FastAPI serves at `http://127.0.0.1:8000` (health: `/api/v1/health/ready`).

Frontend:

```
cd web
npm run dev
```

Next.js runs on the development port shown by npm and proxies `/api/*` to the backend.

Tests: `pytest` from the repository root.

## Reproducibility & Provenance

- **Chronological evaluation** — features, splits and tuning never see the future; the
  same feature code serves training and live inference.
- **Frozen models** — the selected models (RF V2, XGB V3) were frozen before their
  final evaluations; no post-test retuning anywhere.
- **Frozen Cologne artifacts** — the pre-event simulation and its inputs are recorded in
  immutable, hash-carrying receipts; real results were opened only after that freeze, and
  the historical artifacts are never regenerated.
- **Hash-verified application contracts** — the API re-verifies model/state/config hashes
  at startup and refuses to serve if anything drifted.

The flat `scripts/` directory is preserved deliberately: later project phases hash and
reference exact script paths as part of frozen scientific receipts and validation
contracts, so moving these files would invalidate historical provenance. The logical
responsibilities are indexed in this README while the physical paths remain stable to
preserve reproducibility. For the same reason `src/` is intentionally empty (validation
gates assert it stays that way), and the tracked zero-byte root `config.yaml` is a
vestigial placeholder kept for stability.

## Current Application Status

- `/predict` implemented — maps-unknown (pre-veto) and maps-known modes
- Model-grounded explanations implemented (shown as per-prediction factors)
- FastAPI application layer implemented (versioned, contract-verified)
- Major simulation backend implemented (exposed via the API)
- PWA foundation implemented; dedicated Major UI pages are staged as next-release shells

## Limitations / Future Work

- Deployment data snapshot ends 28 Jun 2026 (not live data)
- One external tournament evaluated; BO5 nearly absent from the data
- Richer roster/transfer modelling
- Time-decayed and opponent-adjusted rating features
- Second data source (e.g. GRID) for freshness and identity cross-validation
- Probability calibration and ensemble research
