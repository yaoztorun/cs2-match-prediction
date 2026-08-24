# Phase 9C — Model-Grounded Explanation Core

Non-causal, deterministic, model-grounded explanations layered on the frozen Phase 9B
prediction contract. No model was fit, retrained, tuned, or calibrated. No API, frontend,
or LLM-generated explanations.

Module: `scripts/application_explanations.py`
Registries: `config/application_explanations_v1.yaml`,
`config/application_explanation_feature_groups_v1.yaml`
Receipt: `data/deployment/application_explanation_receipt_v1.json`

## A. Purpose / non-causal contract

Every explanation carries `explanation_type = "model_feature_attribution"` and `causal = false`.
Language is associational throughout ("the model favored Team A primarily due to...", never
"Team A will win because..."). Explanation code never changes a prediction: every explain
function reconstructs the model's own output from its attribution and asserts it matches the
authoritative Phase 9B prediction within a frozen tolerance before returning anything.

## B. Attribution-method audit

No explanation library is installed (`shap`/`eli5`/`lime`/`treeinterpreter` all absent) — per
instruction, none was auto-installed. Both models nonetheless get exact, deterministic local
attribution using only what's already present:

**XGB V3 — native TreeSHAP.** `xgboost`'s C++ core exposes exact Shapley-value contributions via
the low-level `Booster.predict(dmatrix, pred_contribs=True)` — no `shap` package required, always
built in. Output space: **log-odds (raw margin)** — `sigmoid(base_value + Σcontributions) ==
predict_proba` verified directly. **Tree-range audit (amendment #8):** the frozen model has *no*
`best_iteration`, `best_ntree_limit`, or early-stopping metadata anywhere (`booster.attributes()
== {}`, `booster.num_boosted_rounds() == 118 == metadata["final_n_estimators"]`) — TreeSHAP with
no `iteration_range` restriction therefore uses exactly the same 118 trees `predict_proba` does;
there is nothing to restrict. **Feature-order proof (amendment #9):** `DMatrix` is constructed
with explicit `feature_names=transformed_feature_names`, and `dm.feature_names` is asserted equal
to the frozen preprocessing contract's own list before every call — not implied by array length.

**RF V2 — Saabas-style tree path decomposition (Saabas 2014).** sklearn ships no native
contribution API and `treeinterpreter` isn't installed, so this is implemented directly from
sklearn's own exposed tree internals (`tree_.feature`, `tree_.threshold`, `tree_.children_left/
right`, `tree_.value`) — zero new dependencies. **This is explicitly NOT SHAP, NOT Shapley
values, NOT TreeSHAP, and NOT globally-consistent feature importance** — it is path-dependent,
local, tree-structure-dependent, and lacks SHAP's symmetry/consistency guarantees. "Exact" here
means exact reconstruction of *this specific forest's own prediction*, verified empirically (not
assumed): **max reconstruction error 8.88e-16 across 500 diverse real feature vectors**,
including many cold-start pairs, and independently re-verified on 200 fresh fixtures in both the
test suite and the validator. Output space: **probability** directly (RF leaves already store
class probabilities).

**A real precision bug was caught and fixed during this audit**, directly because of the "prove
empirically over a broad fixture set, not a handful of manual examples" requirement: an initial
implementation reconstructed correctly on hand-picked vectors but showed up to a **1.28e-3**
error on a 150-case random batch. Root cause: **sklearn's Cython tree-traversal code compares
splits using `float32`, even though `tree_.threshold` is exposed as `float64` in the Python API**
— a small number of near-threshold feature values landed on the opposite side of a split under
float64 comparison versus sklearn's actual internal float32 comparison. Fixed by casting the
input vector to `float32` before every split comparison in the Saabas walk, dropping the error to
machine epsilon (~1e-16). This is documented here because it is exactly the kind of bug broad
empirical validation is supposed to catch, and it did.

**Class semantics** were verified robustly, not assumed: `model.classes_ == [0, 1]` and all 300
individual tree estimators' `.classes_ == [0, 1]` (index 1 = class 1, consistently across the
whole forest) — checked directly, not inferred from array position.

## C. RF feature taxonomy

The forest splits on the **19 transformed** features, never the 17 raw ones directly (amendment
#1) — the low-level attribution surface is therefore the 19-dimensional transformed vector.
`bestOf` → `{bestOf_BO3, bestOf_BO5}` and `tier` → `{tier_tier2, tier_tier3}` (the other 15 raw
features map 1:1); this expansion is derived programmatically from the frozen preprocessing
contract's own `transformed_feature_names`, not hand-typed. Product groups derived directly from
the 17 raw features' actual semantics (no pre-existing RF family table exists):
`overall_strength` (ELO, all-time win rate), `recent_performance` (last-5/10 form, activity,
rest), `historical_experience` (match counts, data-sufficiency flags), `event_context`
(bestOf/tier). **`opponent_strength`, `map_pool`, `selected_map_strength`, `map_experience`,
`player_strength`, `roster_stability`, `roster_map_familiarity` are explicitly NOT created for RF**
— none of those concepts exist in its 17-feature input, and the feature-group config asserts
this exclusion is intentional (`factor_groups_not_applicable_to_this_model`), not an oversight.

## D. XGB feature taxonomy

Reuses the **existing, already-published Phase 6D family taxonomy**
(`reports/tables/map_xgboost_v3_final_feature_importance.csv` /
`..._group_importance.csv`) rather than reinventing one. **Exact set equality** verified
programmatically (not just the 131-count sum) between the family table's feature names and the
frozen `transformed_feature_names` — 131/131, zero duplicates, zero gaps in either direction.
The 16 lettered families (A–P) collapse to 9 product-facing groups, frozen explicitly with a
documented rationale per family (`config/application_explanation_feature_groups_v1.yaml`):

| Family | Product group | Rationale |
|---|---|---|
| A | `overall_strength` | original series V1 signal, reused unchanged |
| B, C, D | `map_pool` | pool depth, same-map matchup advantage, pool confidence |
| E | `opponent_strength` | strength of schedule / residual form |
| F, G | `recent_form` | time-decayed form + confidence flags |
| H | `player_strength` | individual player performance stats |
| I, J | `roster_stability` | lineup continuity + confidence flags |
| K | `selected_map_strength` | the team's own RAW historical performance on the specific map |
| L | `event_context` | categorical map/bestOf/tier |
| M, N | `map_experience` | **recency/opponent-adjusted** map performance (M) and map specialization relative to overall/pool strength (N) — grouped together because both are *comparative/contextual* views, deliberately kept separate from K's raw strength numbers |
| O, P | `roster_map_familiarity` | current-roster player performance and lineup continuity on the specific map |

The historical ablation AUC-decrease values from the Phase 6D tables are used only as taxonomy
context (family labels) — **never as local explanation contributions**, per instruction.

## E. Feature-to-factor mapping

`config/application_explanation_feature_groups_v1.yaml`, built by
`scripts/build_explanation_feature_groups.py`, which raises before writing anything if coverage
is incomplete. RF: 19/19 features mapped. XGB: 131/131 features mapped. No feature falls into an
undocumented `technical_other` bucket for either model — full explicit coverage was achievable
for both.

## F. Unknown-map explanations

`explain_series_unknown_maps(context_id, team_a, team_b, best_of, prediction_datetime=None,
tier=None)` reuses the exact same prepared feature vector Phase 9B's own
`predict_series_unknown_maps` builds (via the shared `_prepare_rf_prediction` /
`_predict_rf_from_prepared` helpers extracted from `application_inference.py` — amendment #5;
the vector is generated exactly once per request). Returns `{prediction, explanation}` with
`base_value`, 19 `feature_contributions` (raw value + transformed value + contribution + factor
group), `grouped_factors` (ranked, signed, with `supporting_features`), `team_a_factors` /
`team_b_factors` / `neutral_factors`, top-3 positive/negative factors, deterministic
`human_readable_summary` templates, `input_provenance` (cold-start default provenance, kept
separate from `model_contribution` — amendment #21), and a `reconstruction_check`.

Example (Team Vitality vs Team Falcons, BO3, deployment context): `probability_team_a = 0.5523`,
top positive factor `overall_strength` (+0.068), top negative factor `recent_performance`
(-0.037), reconstruction error 6e-16.

## G. Known-map explanations

`explain_map(...)` mirrors the same shape using the 131-feature TreeSHAP vector, plus
`state_support` metadata carried through **at the top level, separate from `explanation`**
(amendment #20) — a `player_map_state_available: false` never gets converted into a team_a/
team_b factor; it is a data-support fact, not a model contribution, unless a real feature
actually produced a signed contribution supporting that interpretation.

Example (same matchup, Mirage): `probability_team_a = 0.5710`, top positive `map_pool` (+0.208),
`overall_strength` (+0.146), `player_strength` (+0.079); top negative `opponent_strength`
(-0.085). `state_support`: all 6 flags true, `fallbacks_used: []` for this history-rich pair.

## H. Known-series map-level explanations

`explain_series_known_maps(...)` never sums per-map log-odds TreeSHAP contributions into a
series-level number (amendment #19 — mathematically invalid, since the series probability lives
in a different space produced by the DP, not by any single feature vector). Two explicitly
separate top-level keys: `map_level_explanations` (one full `explain_map` result per ordered map,
untouched TreeSHAP attribution) and `series_composition` (leverage + reach probability only, DP
probability space). A `note` field states this separation explicitly in every response.

## I. Series-composition leverage

`series_composition_leverage_i = P_series(p_i=1) − P_series(p_i=0)`, all other map probabilities
held fixed — computed via the existing, untouched `compose_series_probability`, never SHAP, never
model attribution, never a causal effect. Verified: **leverage is independent of the original
value of p_i** (changing `map_probabilities[0]` from 0.1 to 0.9 while holding the rest fixed
leaves `leverage[0]` unchanged, to 1e-12) — a direct consequence of the DP being multilinear in
each individual map probability. A mathematical identity was also discovered and verified: **for
the final map slot in any BO series, `leverage == probability_map_is_reached` exactly** (if the
last map decides the series only when reached, forcing its probability to 1 vs 0 swings the
series outcome by exactly the probability of reaching it).

## J. Reach probability

`probability_map_is_reached` computed via a DP variant sharing the exact same `(maps_played,
wins_a)` state space as `compose_series_probability` (not a separate ad hoc formula). Verified
against the analytical closed forms in the amendment: BO1 `P(map1)=1`; BO3 `P(map1)=P(map2)=1`,
`P(map3) = p1(1-p2) + (1-p1)p2` (matched exactly, 1e-12); BO5 `P(map1)=P(map2)=1`, with maps 3–5
verified numerically monotonic-non-increasing and bounded in `[0,1]` across 20 random fixtures,
plus the exact structural BO5 example in the amendment. Monotonicity (`P(reach i+1) <= P(reach
i)`) holds in every tested case.

## K. Cold-start / partial-support behavior

`input_provenance` (RF) distinguishes which raw values came from a real history entry vs. a
cold-start default (`ELO_INITIAL`, 0.5 win-rate defaults, zero match counts, NaN
`days_since_last_match` before median imputation) — explicitly labeled as **input defaults, not
model contributions in themselves**. `state_support`/`fallbacks_used` (XGB) are surfaced but never
converted into `team_a_factors`/`team_b_factors`. **THUNDERdOWNUNDER lifecycle**, tested directly:
historical context → `team_a_cold_start: true`, one provenance note; deployment context →
`team_a_cold_start: false`, 4 legitimate matches, zero provenance notes — the exact same
before/after pattern Phase 9A established, now visible through the explanation layer without any
team-specific special-casing.

## L. Determinism

Repeated identical calls return byte-identical `base_value`, `feature_contributions`,
`grouped_factors` (including `rank`), and `human_readable_summary` — verified directly, not
assumed, in both the test suite and the validator.

## M. Performance

Explanation adds attribution computation on top of the (already-measured, Phase 9B) prediction
latency. RF Saabas walks 300 trees × depth 8 (≤2,400 node visits) in pure Python/numpy — cheap.
XGB TreeSHAP is one native C++ call per map. Explanation latency was not optimized for; if it
proves materially slower than prediction alone in a future API, lazy/on-demand explanation
loading is a reasonable choice for that later phase, not addressed here.

## N. Validation

`pytest tests/test_phase9c_application_explanations.py` — 35 tests, including a 200-fixture RF
additivity re-verification (independent of the report's own 500-fixture check), 9-map XGB
reconstruction re-check, DP reach/leverage analytical and random-fixture invariants, the
THUNDERdOWNUNDER lifecycle, JSON safety, determinism, zero state mutation, and — critically — two
tests that **run the actual Phase 9B `pytest` suite and `validate_phase9b.py` as real
subprocesses** to prove the `application_inference.py` refactor (amendment #5) didn't drift
anything: full repository suite passes; `scripts/validate_phase9c.py` independently re-runs the
same real-command regression gate (never importing a test module — amendment #7) and reports all
checks passed.

## O. Limitations

- **RF attribution is Saabas, not SHAP** — stated plainly throughout, including in the registry
  and every explanation's `attribution_method` field, specifically so a future API/PWA never
  mislabels it.
- **RF and XGB contributions live in different output spaces** (probability vs. log-odds) and
  are never presented as directly comparable magnitudes; every explanation carries its
  `attribution_output_space` explicitly, and no contribution is ever labeled a "percentage impact"
  since neither space mathematically supports that interpretation.
- **The K/M/N XGB family split** (`selected_map_strength` vs. `map_experience`) is a documented
  judgment call, not a uniquely correct partition — the rationale is recorded in the registry for
  anyone who wants to reconsider it later.
- **No API, no PWA, no LLM explanation generation** — out of scope for this phase by design.

---

```
APPLICATION EXPLANATION CORE = IMPLEMENTED
EXPLANATIONS = MODEL-GROUNDED
EXPLANATIONS = NON-CAUSAL
PHASE 9B PREDICTIONS = UNCHANGED
RF V2 = UNCHANGED
XGB V3 = UNCHANGED
NO RETRAINING
NO API YET
NO PWA YET
```
