# Phase 8A — Application Prediction Contracts + Pre-Veto Series Engine Audit

**This is an audit-and-design deliverable. No model was fit, tuned, evaluated, or modified while producing this report. No Phase 1–7 artifact was touched. Cologne match results were not inspected.**

---

## A. Pre-Veto Series Model Inventory

Every artifact related to the pre-veto **series** prediction task (Team A vs Team B, no map information), inventoried read-only.

### Serialized, ready-to-use models

| model | feature version | feature count | hyperparameters | serialized model | preprocessing saved | selected via | pre-Phase-7? |
|---|---|---|---|---|---|---|---|
| `models/random_forest_v2.joblib` / `.json` | V1 (`config/series_features_v1.yaml`) | 17 raw / 19 transformed | `n_estimators=300, max_depth=8, min_samples_leaf=20, min_samples_split=10, max_features=sqrt, bootstrap=True, criterion=gini` (`data/modeling/random_forest_v2_selected_config.json`) | yes | yes (`data/modeling/random_forest_preprocessing_v2.json`) | Phase 4B.1 chronological CV (6,619-match TRAIN) | yes — Phase 4-era |
| `models/xgboost_v2.json` | V1 | 17 raw / 19 transformed | `learning_rate=0.02, max_depth=4, min_child_weight=20, subsample=0.6, colsample_bytree=0.9, gamma=2.0, reg_alpha=0.01, reg_lambda=1.0, n_estimators=98` (`data/modeling/xgboost_v2_selected_config.json`) | yes | yes (`data/modeling/xgboost_preprocessing_v2.json`) | Phase 4C.1 chronological CV | yes |
| `models/logistic_regression_scratch_v2.json/.npz` | V1 | 17 raw / 19 transformed | scratch-implemented LR, tuned (`data/modeling/logistic_regression_v2_selected_config.json`) | yes | yes (`data/modeling/logistic_preprocessing_v2.json`) | Phase 4A.1 chronological CV | yes |
| `models/random_forest_v1.joblib`, `models/xgboost_v1.json` | V1 | 17/19 | untuned baselines | yes | yes | Phase 4B/4C (not selected — superseded by V2 tuning) | yes |

All three V2 models were evaluated **exactly once** on the 1,419-match main **series** VALIDATION partition (`data/modeling/series_split_v1.csv`) — a wholly different partition from the known-map task's `map_split_v1.csv`. Their selection long predates Phase 5A (the first map-task phase), let alone Phase 7.

**Validation metrics (Phase 4B.1/4C.1, pre-Phase-7):**

| model | Log Loss | Brier | ROC-AUC | Accuracy | train-val ROC-AUC gap |
|---|---|---|---|---|---|
| **RF V2** | **0.6514** | **0.2298** | **0.6566** | 0.6068 | 0.0550 |
| **XGB V2** | 0.6542 | 0.2311 | 0.6504 | **0.6117** | **0.0073** |
| LR V2 (scratch) | 0.6581 | 0.2329 | 0.6412 | 0.6131 | n/a (LR, not a tree ensemble) |

### Feature-version ablations that exist but produced no serialized model

| dataset | phase | rows/cols | evaluated how | verdict (frozen-config CV) | model ever fit+saved on it? |
|---|---|---|---|---|---|
| `series_features_v2_map_pool.parquet` | 5B.1 | 9,456 × 47 | paired CV, **frozen V1-selected** RF V2/XGB V2 hyperparameters re-applied | **HELP** (RF), **HELP** (XGB) — `reports/phase5b1_series_map_pool_cv_results.md` | **no** |
| `series_features_v3_form.parquet` | 5B.3 | 9,456 × 59 | same, frozen configs | **HELP** / **HELP** — `reports/phase5b3_team_form_cv_results.md` | **no** |
| `series_features_v4_roster.parquet` | 5C.1 | 9,456 × 90 | same, frozen configs | **HELP CLEARLY** (full-data) — `reports/phase5c1_player_roster_cv_results.md` | **no** |

`models/` contains no RF/XGB/LR artifact trained on V2, V3, or V4 series features — every ablation reused the **exact frozen** V1-selected hyperparameters unchanged, never a retune. See section C for what this means for application readiness.

### Supporting infrastructure already in place

- **Inference primitive**: `feature_engine.build_features(store, team1, team2, prediction_time, best_of, tier=None)` (scripts/feature_engine.py:258) — pure, read-only, returns exactly the 17-feature dict the V1 preprocessing modules expect. No target/score parameter.
- **Preprocessing**: `scripts/preprocessing_random_forest_v1.py`, `preprocessing_xgboost_v1.py`, `preprocessing_logistic_v1.py` — all accept an arbitrary DataFrame (including a synthetic 1-row future frame), the same pattern already proven in `finalize_map_xgboost_v3.py`'s own future-inference-parity check.
- **State**: `data/features/series_team_state_v1_full.json` (full-development snapshot — **not** strictly pre-Cologne, see section C) and `data/features/pre_cologne_team_state_v1_full.json` (strictly pre-Cologne, see section C).
- **Tests**: `tests/test_feature_engine.py`, `test_preprocessing_{logistic,random_forest,xgboost}.py`, `test_random_forest_v1.py`, `test_xgboost_v1.py`, `test_random_forest_cv_v2.py`, `test_xgboost_tuning_v2.py`, `test_logistic_regression_tuning_v2.py` — the V1-feature RF/XGB/LR pipeline is well-covered.

---

## B. Recommended Existing Series Application Candidate

Applying the predefined hierarchy exactly (probability quality first — Log Loss/Brier — then ROC-AUC, then Accuracy, then stability, then inference readiness), using **only** the pre-Phase-7 validation numbers in section A:

1. **Log Loss / Brier**: RF V2 wins both (0.6514 vs 0.6542; 0.2298 vs 0.2311).
2. **ROC-AUC**: RF V2 wins (0.6566 vs 0.6504).
3. **Accuracy**: XGB V2 wins (0.6117 vs 0.6068) — but accuracy is criterion 3, and the simulator consumes probabilities, not thresholded labels.

RF V2 wins all three primary/ranking criteria. Under the frozen hierarchy this is decisive before criterion 4 (stability) is even consulted — a lower-priority criterion does not overturn a result already settled by higher-priority ones.

**Recommendation: Random Forest V2**, described precisely as a **frozen existing pre-veto application candidate selected from already-completed development evidence** — not a newly optimized, newly tuned, or newly trained model. No new CV, fitting, comparison, or evaluation was run to reach this conclusion.

**Caveat to document prominently**: XGB V2's train-validation ROC-AUC gap (0.0073) is roughly 1/7 of RF V2's (0.0550). This is real, pre-existing evidence of a much more stable train→validation transition for XGB V2, and means RF V2's validation-set advantage carries more single-split risk than XGB V2's numbers do. This does not change the recommendation under the frozen hierarchy (criterion 4 only applies after 1–3 are exhausted, and they were not tied), but the application should treat RF V2 as "the frozen candidate selected by the primary criteria, with a documented higher single-split-variance caveat" rather than an unambiguously superior model. No additional pre-Phase-7 artifact was found that materially changes this picture (the V2/V3/V4 ablations in section A were run against **both** RF V2 and XGB V2's frozen configs and did not alter either model's own V1 validation numbers).

---

## C. Whether Any Model Refit Would Be Required

**RF V2 (the section B recommendation) requires no refit.** Serialized model, preprocessing, and config all exist and are directly usable as-is (`models/random_forest_v2.joblib`, `data/modeling/random_forest_preprocessing_v2.json`, `data/modeling/random_forest_v2_selected_config.json`).

**On V4 series features specifically** (the richest, most favorably-evaluated feature set per section A):

- V4 richer series features (`series_features_v4_roster.parquet`, 90 columns) exist.
- Frozen-config ablations (5B.1→5B.3→5C.1) showed favorable development evidence at every step, culminating in "HELP CLEARLY" for the full V4 feature set.
- **No serialized V4 series model exists.**
- **No complete application-ready V4 preprocessing/model artifact exists** (only the ablation-harness preprocessing used transiently inside the Phase 5C.1 evaluation script, never saved as a deployable artifact).
- Creating one now would require new model fitting.
- Any new fitting would reopen model development, which Phase 8A is explicitly not permitted to do.
- **Therefore V4 is not used by the application under the current freeze.**

V4 may contain useful additional predictive information — the ablation evidence is real and pre-Phase-7 — but it is **not currently an application-ready frozen prediction system**. No V4 reconstruction or training occurred or is proposed here. If the user wants V4 available to the application, that is a distinct, explicit model-development decision for a future phase, not something Phase 8A silently performs or assumes.

---

## D. Frozen Known-Map Contract (audit)

Confirmed present and internally consistent:

| artifact | path |
|---|---|
| model | `models/map_xgboost_v3_final.json` |
| metadata | `models/map_xgboost_v3_final_metadata.json` |
| preprocessing | `data/modeling/map_xgboost_v3_final_preprocessing.json` |
| config | `data/modeling/map_xgboost_v3_final_config.json` |
| feature schema | `config/map_features_v3_modern_map.yaml` (120 raw / 131 transformed) |
| feature composer (historical) | `scripts/rich_modern_map_feature_composer.py` |
| future/synthetic composer | `scripts/rich_modern_map_feature_composer.build_future_modern_rich_map_features` |

`build_future_modern_rich_map_features` already produces a complete 120-raw/131-transformed synthetic prediction — proven in Phase 6D's own future-inference-parity check and reconfirmed by Phase 7's frozen artifacts — **without** touching TEST or Cologne. Threshold remains 0.5, `final_n_estimators=118`, no calibration. Nothing new is required for MODE B inference; this phase only confirms readiness.

**Phase 7 TEST status is unaffected by anything in this report.** Phase 7 scored the already-built chronological TEST feature rows (`map_features_v3_modern_map.parquet` restricted to `split=="test"`) with the frozen model — a fixed historical dataset, not a synthetic/future composer call. The state-snapshot finding in section E concerns **future synthetic inference, historical pre-event replay, Cologne simulation, and later application composition only**. `data/evaluation/map_test_predictions_v1.parquet` is not reopened, altered, or regenerated by this report or by anything it recommends.

---

## E. Audit Finding — Strict Pre-Cologne Series State

**Finding.** `data/features/series_team_state_v1_full.json` — used as the `series_state` parameter in every Phase 6A/6C/6D "pre-Cologne sufficiency" validator check (`validate_phase6a.py`, `validate_phase6c.py`, `validate_phase6d.py`) — has a **maximum team-history datetime of 2026-06-21T16:00:00**, which is *after* the Cologne cutoff (**2026-06-02T13:30:00**, from `map_stream_common.cologne_cutoff()`). It is the full-development snapshot ("development" = not Cologne itself, but legitimately including non-Cologne events running concurrently with or shortly after Cologne's start), **not** a strict pre-Cologne one.

The strictly correct snapshot already exists: `data/features/pre_cologne_team_state_v1_full.json`. **Validated in this audit** (read-only): its maximum team-history datetime is **2026-05-30T19:30:00**, strictly before the 2026-06-02T13:30:00 cutoff — confirmed to contain no state update at or after the first Cologne match.

**Scope of impact.** This never affected known-map feature or model correctness — the companion map/form/roster/modern-map state stores used in those same Phase 6A/6C/6D checks *were* correctly restricted to their own `pre_cologne_*.json` snapshots. Only the series-ELO component (`series_state.teams[team].elo`, feeding the map-specialization "vs overall strength" features) of those particular synthetic sufficiency smoke tests was less strict than documented. This is a **read-only audit finding**, not a defect requiring repair: per this phase's mandate, no Phase 6 validator or artifact is modified. It is recorded here so future phases use the correct file.

**Cologne pre-event inference contract (recorded for future phases, not implemented here):**

- Strict pre-Cologne states only: `pre_cologne_team_state_v1_full.json` (series), `pre_cologne_map_state_v1.json`, `pre_cologne_form_state_v1.json`, `pre_cologne_player_roster_state_v1.json`, `pre_cologne_modern_map_state_v1.json`.
- No Cologne match result may update any state used for the pre-event prediction.
- No full-development state snapshot (e.g. `series_team_state_v1_full.json`) may be substituted for the strict pre-Cologne one in this context.
- Prediction state remains frozen throughout the entire pre-event Monte Carlo simulation — states are not incrementally updated with simulated or real intra-tournament results.
- The prediction datetime/cutoff passed to every composer call must represent information available strictly before the first Cologne match.

---

## F. Unknown-Map Inference Contract (MODE A)

```python
def predict_series_unknown_maps(team_a: str, team_b: str, best_of: int,
                                  prediction_datetime, tier: str | None = None) -> dict:
    """
    Pre-veto series prediction. Uses ONLY the frozen RF V2 series model
    (section B) - no map selection is consulted or required.

    Composition (all pieces already exist and are already tested):
        1. feature_engine.build_features(series_state, team_a, team_b,
           prediction_datetime, best_of, tier=tier)          -> 17-feature dict
        2. preprocessing_random_forest_v1.transform(row_df, params,
           model_features)                                    -> (X, feature_names)
        3. RandomForestClassifier.predict_proba(X)             -> P(team_a wins series)

    `series_state` is a feature_engine.StateStore loaded from the
    appropriate snapshot for the calling context (see section E for the
    strict pre-Cologne rule; a "live" context would use the current
    full-development or later state instead - a distinct, later decision).

    Returns
    -------
    {
        "team_a": str, "team_b": str, "best_of": int,
        "probability_team_a": float,   # in [0, 1]
        "probability_team_b": float,   # == 1 - probability_team_a
        "model_id": "series_random_forest_v2",
        "prediction_mode": "pre_veto",
    }
    """
```

No implementation occurs in Phase 8A - this is the frozen contract Phase 8B implements against.

---

## G. Known-Map Inference Contract (MODE B)

```python
def predict_map(team_a: str, team_b: str, map_name: str, best_of: int,
                 prediction_datetime, tier: str | None = None) -> dict:
    """
    Known-map prediction for ONE selected map. Uses the frozen Phase 6D/7
    system exclusively - zero modification.

    Composition (all pieces already exist and are already tested):
        1. rich_modern_map_feature_composer.build_future_modern_rich_map_features(
               team_a, team_b, best_of, map_name, prediction_datetime,
               series_state, map_state, form_state, player_roster_state,
               modern_map_state, tier=tier)                    -> 120-feature dict
        2. preprocessing_xgboost_map_v3.transform(row_df, params, roles) -> (X, names)
        3. XGBClassifier(...).load_model("models/map_xgboost_v3_final.json")
           .predict_proba(X)                                   -> P(team_a wins the map)

    Threshold, calibration, and ensemble policy are exactly the frozen
    Phase 6D/7 contract: threshold 0.5 (not applied here - this returns the
    raw probability), no calibration, no ensemble.

    Returns
    -------
    {
        "team_a": str, "team_b": str, "map_name": str, "best_of": int,
        "probability_team_a": float,   # in [0, 1]
        "probability_team_b": float,   # == 1 - probability_team_a
        "model_id": "map_xgboost_v3_final",
        "prediction_mode": "known_map",
    }
    """
```

---

## H. Known-Map Series Composer Contract

For BO*n* (best of `n`, `n` odd, needing `ceil(n/2)` wins), given ordered per-map win probabilities `[p_1, ..., p_n]` for team A (later maps used only if the series is not yet decided), compute `P(team A wins the series)` via one general dynamic-program rather than three hard-coded formulas:

```python
def compose_series_probability(map_probabilities: list[float], best_of: int) -> float:
    """
    General DP: state = (maps_played, wins_a). Transition on map i's own
    probability p_i. A state is TERMINAL as soon as either side reaches
    ceil(best_of/2) wins - later maps are never reached from a terminal
    state, so the recursion naturally accounts for "later maps only play
    if required" without a separate case per format.

        wins_needed = (best_of + 1) // 2
        dp[0][0] = 1.0
        for i in range(len(map_probabilities)):
            for wins_a in range(0, i + 1):
                if wins_a >= wins_needed or (i - wins_a) >= wins_needed:
                    continue  # already terminal, do not propagate further
                p = map_probabilities[i]
                dp[i+1][wins_a+1] += dp[i][wins_a] * p
                dp[i+1][wins_a]   += dp[i][wins_a] * (1 - p)
        P(team A wins series) = sum(dp[i][wins_a] for all terminal states with wins_a == wins_needed)

    BO1 (wins_needed=1) collapses to dp trivially returning map_probabilities[0].
    BO3 (wins_needed=2) and BO5 (wins_needed=3) fall out of the SAME code path.

    len(map_probabilities) must be >= 2*wins_needed - 1 (the maximum possible
    maps for that format); probabilities beyond the series' natural length
    are never consulted structurally (dp stops propagating once terminal).
    """
```

No implementation occurs in Phase 8A.

---

## I. Explanation Contract

Non-causal wording only - grouped permutation/feature-family importance describes association under the frozen model, never a causal claim.

```python
def explain_prediction(...) -> dict:
    """
    Returns
    -------
    {
        "factors_for_team_a": [ {"category": str, "description": str, "strength": float}, ... ],
        "factors_for_team_b": [ {"category": str, "description": str, "strength": float}, ... ],
        "risk_factors": [ {"category": str, "description": str}, ... ],
        "confidence_summary": str,
    }

    Candidate high-level categories (from the frozen models' own feature
    families, e.g. map_feature_families.py / evaluate_map_feature_sets_v3.V3_FAMILIES):
        overall team strength, recent form, opponent strength, map pool,
        selected-map strength, player performance, current roster quality,
        roster stability, map experience.
    """
```

No SHAP or other attribution computation is implemented in Phase 8A - contract only.

---

## J. Major Simulator Contract

```python
def simulate_major(tournament: dict, n_simulations: int, random_seed: int,
                    prediction_engine: str = "known_map_preferred") -> dict:
    """
    One full stochastic tournament run per simulation, using predict_series_unknown_maps
    / predict_map + compose_series_probability for every matchup, sampling a
    winner from each series probability, advancing the bracket, and repeating
    n_simulations times with a single RNG seeded by random_seed.

    Returns aggregated, per-team probabilities: reach_stage_2, reach_stage_3,
    reach_playoffs, reach_semifinal, reach_final, win_tournament - plus the
    per-simulation matchup/probability/sampled-winner record and stage
    advancement trail needed to reconstruct any individual simulated bracket.

    No tournament logic (stage structure, seeding, Swiss pairing) is
    implemented in Phase 8A - contract only.
    """
```

---

## K. API / PWA Architecture

```
Python inference/simulation core
  ├── unknown-map series predictor      (predict_series_unknown_maps, section F)
  ├── known-map predictor               (predict_map, section G)
  ├── series probability composer       (compose_series_probability, section H)
  ├── explanation engine                (explain_prediction, section I)
  └── Major simulation engine           (simulate_major, section J)
        ↓
      API layer
        ↓
      Next.js PWA
        ↓
   interactive user interface
```

No feature engineering or XGBoost/RF logic is duplicated in TypeScript. All prediction/simulation logic stays Python-side; the frontend calls the API layer only.

**Frontend product/UX requirements** (recorded verbatim for future reference, not built): premium, modern, esports-oriented, data-driven, interactive, fast, professional - explicitly avoiding excessive neon, stereotypical gaming-dashboard UI, clutter, oversized glowing cards, excessive gradients, tiny analytical text. Visual direction: "premium esports analytics product," not "gaming dashboard template." Core future routes: `/` (overview/landing), `/predict` (match predictor: Team A/B selectors, BO1/3/5, maps-unknown vs maps-known toggle with ordered map selectors scaling 1/up-to-3/up-to-5, map-by-map + series probability output, visual probability bar, favored team, factors/risks), `/major` (interactive Major simulator), `/major/cologne-2026` (frozen historical simulation, section L below), `/major/cologne-2026/results` (simulation vs reality). Major UX: Stage 1 → Stage 2 → Stage 3 → Quarterfinal → Semifinal → Grand Final, each matchup card eventually showing team identity, current Swiss record, format, model probability, predicted/sampled winner, expandable explanation; three user modes later (model simulation, interactive Pick'Em, Monte Carlo probability view).

---

## L. Cologne 2026 Evaluation Lifecycle (clarification)

IEM Cologne Major 2026 is a **one-time external historical simulation/evaluation**, not a permanently-excluded dataset. The scientific lifecycle has five stages, and mixing them would corrupt the historical evaluation - each stage's boundary is a hard, one-directional gate:

**A. Before Cologne simulation.** Cologne remains completely unavailable to the prediction system. Use only: the frozen pre-event model (section B/D), strict pre-Cologne team/map/player/roster states (section E), and tournament information legitimately known before the event (bracket, seeding, format - never outcomes). Do NOT use Cologne winners, scores, map outcomes, player statistics, roster-performance updates, or any state update produced by a Cologne match. The objective: *"What would the system have predicted immediately before IEM Cologne Major 2026 began?"*

**B. Run and freeze the pre-event simulation.** Before inspecting actual Cologne outcomes, permanently save the simulation outputs. Future immutable artifacts should record, at minimum: simulation/model version, state version, prediction cutoff, model/config/state hashes, tournament-definition hash, RNG seed, Monte Carlo simulation count, individual matchup probabilities, Stage 1/2/3 advancement probabilities, playoff/semifinal/final/championship probabilities, and (if produced) the deterministic model-most-likely path. Once committed, these become the permanent historical pre-event predictions and must **never** later be regenerated with knowledge of Cologne outcomes.

**C. Only then open actual Cologne results.** After the pre-event artifacts are frozen: load the actual results, compare simulation against reality, evaluate realistic-tournament behavior (predicted vs actual advancement at every stage, probability assigned to the actual outcomes, correctly-predicted-team counts, bracket/path similarity, Monte Carlo probability of the actual champion, match-level quality where appropriate). This becomes a permanent **"Cologne 2026 — Simulation vs Reality"** historical evaluation record.

**D. After Cologne evaluation is complete.** Once (1) the pre-event simulation is frozen and (2) the simulation-vs-reality evaluation is saved, Cologne 2026 may become normal historical information for future application/deployment work - incorporated into historical datasets, team/player/roster/map state, future training datasets, and future deployment models. This does **not** retroactively invalidate the historical evaluation, because those predictions were generated and permanently saved *before* Cologne outcomes were ever used.

**E. Never rewrite history.** The historical simulation must never be regenerated using a model trained on Cologne, post-Cologne team/player/map state, or future tournament information. The `Cologne 2026 — Simulation vs Reality` PWA page must always read the original frozen pre-Cologne prediction artifacts. A future live/deployment model may be newer and may legitimately know Cologne - that model belongs to a **separate application context** and must never overwrite or be confused with the historical replay.

### Two system timelines - never mixed

```
HISTORICAL EVALUATION                    FUTURE DEPLOYMENT
strict pre-Cologne data/state            historical data before Cologne
        |                                + Cologne 2026
frozen pre-event prediction model        + later legitimate matches
        |                                        |
Cologne 2026 Monte Carlo simulation      updated states
        |                                        |
immutable prediction artifact            potential future deployment model/version
        |                                        |
open actual Cologne results              future match/tournament predictions
        |
simulation-vs-reality evaluation
        |
permanent historical replay
```

The PWA may eventually expose both: **(1) Historical Replay** - frozen original prediction, what the model believed before Cologne, comparison with reality; **(2) Live / Latest Predictor** - current dataset/state, may legitimately include Cologne and later events.

### Future versioning requirement

Later application artifacts should be explicitly versioned so historical evaluation can never be accidentally replaced. Illustrative naming (use actual repository conventions when implementation begins): research model `series_rf_v2_pre_cologne`; research state `pre_cologne_2026`; historical simulation `cologne_2026_pre_event_simulation_v1`; historical evaluation `cologne_2026_simulation_vs_reality_v1`; future deployment model `series_model_<future_version>`; future state `state_post_cologne_<version>`. The essential requirement is explicit separation between pre-event historical-evaluation artifacts and post-event deployment artifacts - not the specific names.

### Status during Phase 8A itself

**Cologne remains completely untouched in this phase and in every phase until the pre-event simulation and external evaluation artifacts described above are permanently frozen.** The permission to incorporate Cologne as ordinary historical data applies only *after* stages B and C are both complete and committed - it does not apply now, and nothing in this report reads, infers from, or references any actual Cologne match/map/player/roster outcome.

---

## Status

- **NO MODEL FITTING** occurred in Phase 8A.
- **NO V4 RECONSTRUCTION** occurred or is proposed for immediate action.
- **NO TEST REOPENING** — `data/evaluation/map_test_predictions_v1.parquet` was not read, altered, or regenerated.
- **NO COLOGNE RESULT INSPECTION** — no Cologne match/map/player/roster outcome was read.
- **NO APPLICATION CODE, SIMULATOR, TOURNAMENT ENGINE, API, OR PWA** was implemented — contracts only.
- **RF V2 recommendation** = frozen existing pre-veto application candidate selected from already-completed pre-Phase-7 development evidence, not a newly optimized model.
- **V4 series features** = documented as containing favorable pre-Phase-7 evidence but not currently application-ready; no fitting performed or scheduled by this report.
- **Pre-Cologne state finding** = documented as a read-only audit finding; no Phase 6 artifact modified.

Phase 8A complete. Awaiting review before Phase 8B.
