# CS2 Match Prediction

An end-to-end probabilistic machine-learning system for professional Counter-Strike 2:
pre-match series prediction, known-map prediction, Monte-Carlo simulation of a full Major
tournament, model-grounded explanations, a FastAPI serving layer, and a Next.js PWA
frontend. Every model is developed under strict chronological (leakage-safe) evaluation,
frozen before its tests, and validated against a real external event (IEM Cologne Major
2026).

> **Two branches, one experiment.** The tag **`frozen-v1`** contains the original,
> path-stable scientific/application snapshot: every receipt, hash and validator verifies
> there exactly as it was frozen. The active **`reviewer-refactor`** branch (this tree)
> reorganizes the *source layout* for maintainability; path-sensitive application
> contracts were re-baselined only after exact behavioral parity verification. Historical
> Phase 8D/8E evaluation artifacts were **not** regenerated. See
> [Reproducibility & Provenance](#reproducibility--provenance).

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

## Architecture

```
Raw Match Data                 data/raw/
      ↓
Audit & Cleaning               data_preparation/
      ↓
Chronological Historical State feature_engineering/state/
      ↓
Feature Engineering            feature_engineering/
      ↓
LR / RF / XGBoost              training/
      ↓
Evaluation                     evaluation/
      ↓
Frozen Models                  models/
      ↓
Tournament Simulation          tournament/
      ↓
FastAPI                        application/
      ↓
Next.js PWA                    web/
```

## Repository Structure

The directory tree mirrors the pipeline:

| Directory | Contents |
|---|---|
| `data_preparation/` | Raw-data audit, conservative cleaning, canonical datasets, team-identity policy. |
| `feature_engineering/` | `series/` (ELO/form/activity engine), `form/`, `maps/`, `roster/` engines and dataset builders; `preprocessing/` (per-model input preprocessing); `state/` (pre-Cologne and post-Cologne state snapshots). |
| `training/` | `logistic_regression/`, `random_forest/`, `xgboost/` (series models) and `map_models/` (known-map RF/XGB tuning and the final XGB V3). Code only — no artifacts. |
| `models/` | Trained model artifacts: `series/` (LR, RF, XGB series models) and `map/` (map models incl. the frozen `map_xgboost_v3_final`). See [Model artifacts](#model-artifacts). |
| `evaluation/` | `validation/` (feature-set ablations), `internal_test/` (Phase 7 sealed test), `uncertainty/` (cluster bootstrap), `cologne_2026/` (Phase 8E simulation-vs-reality). |
| `tournament/` | `engine/` (Phase 8C Swiss + playoff engine, byte-identical to the frozen version), `simulation/` (pre-veto predictor adapter, Monte-Carlo runner), `cologne_2026/` (Phase 8D frozen pre-event simulation). |
| `application/` | `inference/`, `explanations/`, `api/` (FastAPI), `tournament/` (Major simulation service + router). |
| `validation/` | The 28 phase validation gates (`validate_phase*.py`), see [Validators](#validators). |
| `_common/` | Shared compatibility utilities retained under the original import name to preserve frozen executable contracts (`from _common import ROOT`). |
| `config/` | `features/` (feature schemas), `evaluation/` (frozen test/8D/8E protocols), `application/` (application registries), `tournaments/` (frozen Cologne definition — path unchanged). |
| `data/` | `raw/` → `interim/` → `features/` → `modeling/` → `evaluation/` → `deployment/` (+ `tournaments/`). Paths unchanged from `frozen-v1`. |
| `tests/` | Phase-named tests grouped as `features/`, `models/`, `evaluation/`, `tournament/`, `application/`. |
| `reports/` | `phases/` (all phase reports, filenames unchanged), `figures/`, `tables/`, `presentation/` (deck, notes, Q&A, sources, presentation-only figures and build scripts). |
| `web/` | Next.js PWA frontend (calls the API only — contains no model logic). |
| `src/` | Intentionally empty — validation gates assert it stays empty. |
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

1. `reports/phases/data_audit.md` — raw-data audit (including the broken winner label it caught)
2. `reports/phases/phase3_feature_engineering.md` — leakage-safe feature engine
3. `reports/phases/model_comparison_tuned_v1features.md` — tuned LR vs RF vs XGB comparison
4. `reports/phases/phase7_internal_test_results.md` — known-map model, sealed internal test
5. `reports/phases/phase8d_cologne_pre_event_simulation.md` — frozen pre-event Major simulation
6. `reports/phases/phase8e_cologne_simulation_vs_reality.md` — simulation vs the real Major
7. `reports/phases/phase9b_application_inference_core.md` — application inference core
8. `reports/phases/phase9d_application_api.md` — FastAPI layer
9. `reports/phases/phase9e_application_tournament_service.md` — Major simulation service
10. `reports/phases/phase10a_pwa_foundation_predict.md` — PWA frontend

Phase reports cite the original `scripts/…` paths of `frozen-v1`; the mapping to the
current layout is the [Script Guide](#script-guide) below.

## Models

| Task | Model | Purpose |
|---|---|---|
| Pre-veto series prediction | **Random Forest V2** | Predict a match before the map veto (also drives the tournament simulator) |
| Known-map prediction | **XGBoost V3** (+ dynamic-programming series composition) | Predict each selected map, compose exact P(series win) |
| Development baseline | Logistic Regression (from scratch) | Simple, interpretable linear baseline |
| Development comparison | Random Forest / XGBoost | Compared under the same chronological methodology |

RF V2 and XGB V3 serve **different application tasks** — they are not competitors on a
single final task.

### Model artifacts

Versioned (final runtime models required by the frozen application contracts):
`models/series/random_forest_v2.joblib` + `random_forest_v2.json`,
`models/map/map_xgboost_v3_final.json` + `map_xgboost_v3_final_metadata.json`.
Their preprocessing/config artifacts live in `data/modeling/` (versioned).

Local-only (gitignored) experimental artifacts: LR v1/v2, RF v1 (138 MB, obsolete
untuned baseline), XGB v1/v2, map RF v1, map XGB v1. They are referenced by phase reports
and historical validators but not by the deployed application.

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

### Data Audit & Cleaning — `data_preparation/`
- `audit_data.py` — full raw-data audit; caught the broken `team1_win` label
- `build_canonical_datasets.py` — conservative cleaning into canonical series/map tables
- `orientation_analysis.py` — analysis of the team-ordering bias in the raw data

### Team Identity — `data_preparation/`
- `team_identity_analysis.py` — shows raw team IDs are per-match, not persistent
- `build_team_identity_policy.py` / `unresolved_team_review.py` — canonical name policy and manual-review trail

### Feature Engineering — `feature_engineering/`
- `series/feature_engine.py` — core chronological series feature engine (ELO, form, activity); shared by training and live inference
- `maps/map_feature_engine.py`, `maps/modern_map_feature_engine.py`, `form/team_form_engine.py`, `roster/player_roster_feature_engine.py` — map / form / roster state engines
- `maps/rich_map_feature_composer.py`, `maps/rich_modern_map_feature_composer.py` — compose the full known-map feature vector (historical and future/synthetic paths)
- `series/build_series_features_v1.py` … `roster/build_series_features_v4_roster.py`, `maps/build_map_features_v1.py` … `maps/build_map_features_v3_modern_map.py` — feature dataset builders
- `series/build_series_split_v1.py`, `maps/build_map_split_v1.py`, `maps/build_map_cv_folds_v1.py` — chronological splits and CV folds
- `state/build_pre_cologne_*.py` / `state/build_deployment_*.py`, `state/run_phase9a_pipeline.py` — frozen pre-Cologne and post-Cologne state snapshots
- `preprocessing/preprocessing_*.py` — per-model input preprocessing (fit on train only, saved as artifacts)

### Training — `training/`
- `logistic_regression/` — from-scratch LR (`logistic_regression_scratch.py`), `train_logistic_regression_v{1,2}.py`, tuning
- `random_forest/` — `random_forest_v1.py` wrapper, `train_random_forest_v{1,2}.py`, `random_forest_tuning_v2.py`, `random_forest_cv_folds_v2.py`
- `xgboost/` — `xgboost_v1.py` wrapper, `train_xgboost_v{1,2}.py`, `xgboost_tuning_v2.py`
- `map_models/` — `train_map_models_v1.py`, `map_*_tuning_v1.py`, `map_xgboost_v3_final_tuning.py`, `finalize_map_xgboost_v3.py` (the frozen XGB V3), shared `map_modeling_common.py`

### Evaluation — `evaluation/`
- `validation/evaluate_series_feature_sets_v{2,3,4}.py`, `validation/evaluate_map_feature_sets_v3.py` — paired feature-set ablations under frozen configs
- `internal_test/freeze_phase7_protocol.py`, `internal_test/evaluate_phase7_test_once.py`, `internal_test/phase7_test_reports.py`, `internal_test/phase7_test_visualizations.py`, `uncertainty/phase7_test_bootstrap.py` — sealed internal test (protocol frozen first, scored once, cluster-bootstrap uncertainty)
- `cologne_2026/reconcile_cologne_actual_results.py`, `cologne_2026/phase8e_*.py`, `cologne_2026/build_phase8e_*.py` — Cologne simulation-vs-reality evaluation (results opened only after the pre-event freeze)

### Tournament Simulation — `tournament/`
- `engine/tournament_engine.py` — pure Swiss + playoff rules engine (Valve rulebook; ML-free; byte-identical to `frozen-v1`)
- `simulation/pre_veto_series_predictor.py` — frozen RF V2 prediction adapter
- `simulation/generate_cologne_pre_event_probability_matrix.py` — all 2,976 matchup probabilities, precomputed
- `simulation/cologne_pre_event_simulation.py`, `cologne_2026/run_phase8d_pipeline.py`, `cologne_2026/phase8d_figures.py` — the 50,000-run frozen pre-event simulation

### Application — `application/`
- `inference/application_inference.py` — versioned inference core (both prediction modes, DP series composition)
- `explanations/application_explanations.py` — model-grounded factor attributions (TreeSHAP for XGB, exact tree-path decomposition for RF)
- `api/application_api.py` / `api/run_application_api.py` — FastAPI transport with startup contract verification
- `tournament/application_tournament_service.py` / `tournament/application_tournament_router.py` — Major simulation service (historical frozen views + interactive simulation)
- `*/build_*.py` — application registries, fixtures and receipts

### Validation — `validation/`
- `validate_phase*.py` — 28 phase-specific validation gates that re-verify artifacts,
  hashes and invariants; see [Validators](#validators) for which ones are active on this
  branch and which are authoritative only at `frozen-v1`.

## Important Entry Points

| Goal | File |
|---|---|
| Run the API | `application/api/run_application_api.py` (`python -m application.api.run_application_api`) |
| Application API | `application/api/application_api.py` |
| Inference | `application/inference/application_inference.py` |
| Explanations | `application/explanations/application_explanations.py` |
| Tournament engine | `tournament/engine/tournament_engine.py` |
| Tournament service | `application/tournament/application_tournament_service.py` |
| Frontend | `web/` |
| Tests | `tests/` |

## Running the Project

All Python commands run from the repository root (the root is the package root; no
`PYTHONPATH` or `sys.path` tricks are required).

Backend (Windows):

```
.venv\Scripts\activate
python -m application.api.run_application_api
```

FastAPI serves at `http://127.0.0.1:8000` (health: `/api/v1/health/ready`).

Frontend:

```
cd web
npm run dev
```

Next.js runs on the development port shown by npm and proxies `/api/*` to the backend.

Tests: `python -m pytest` from the repository root. Validators: `python -m validation.validate_phase9d` (etc.).

A fresh clone needs the large historical/deployment state snapshots under `data/interim/`
(rebuildable via `feature_engineering/state/`; ~220 MB, intentionally not committed) before
the API's readiness check passes. `data/interim/team_identity_policy.csv` **is** versioned:
it is a small, deterministic runtime identity contract.

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

### `frozen-v1` vs `reviewer-refactor`

`frozen-v1` is the authoritative record of the original file paths, executable hashes,
Phase 9B–9E receipts and scientific/application provenance. On `reviewer-refactor`:

- Source files were moved into packages (`git mv`; history preserved). Only import lines,
  path constants and subprocess commands changed — no model mathematics, feature logic,
  targets, hyperparameters, evaluation or tournament rules.
- `tournament/engine/tournament_engine.py`, `feature_engineering/series/feature_engine.py`,
  `_common/__init__.py`, `preprocessing_common.py`, `preprocessing_common_map_v2.py` and
  `player_roster_feature_engine.py` are **byte-identical** to `frozen-v1`, so the runtime
  check of the engine hash against the pre-registration Phase 8D receipt still passes.
- The Phase 9B–9E application registries/receipts on this branch are
  **reviewer-refactor executable compatibility contracts**: they were re-generated by
  their own hash-only builders after import-line edits changed the SHA-256 of the
  pipeline modules they enumerate. Model, state and prediction artifacts were not touched
  (the regenerated OpenAPI snapshot is byte-identical). The originals remain at `frozen-v1`.
- Phase 7/8B/8D/8E receipts and protocol YAMLs are untouched historical records; the
  `scripts/…` paths they mention describe `frozen-v1`.

### Validators

| Class | Gates | Status on `reviewer-refactor` |
|---|---|---|
| Active reviewer-refactor validators | `validate_phase9a`, `9b`, `9c`, `9d`, `9e`, `8b`, `8c`, `8d`, `8e`, `7`, `2`, `3` | Expected PASS (locators updated; hashes of frozen data/models unchanged; 9x contracts re-baselined). |
| Historical frozen-layout validators | `validate_phase4a…4c1`, `5a…5c1`, `6a…6d` | Authoritative at `frozen-v1`. They embed SHA-256 values of scripts whose import lines changed, and several re-run feature builders (which would regenerate feature tables). Their expected path/hash differences on this branch are documented, not "fixed". |

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
