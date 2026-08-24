"""
Tests for Phase 4C.1's XGBoost chronological tuning.

The headline property under test is the THREE-STAGE temporal separation:

    inner_fit  <  inner_early_stop  <  outer_fold_validation

so that the block used for early stopping is never the block a candidate is
scored on (which would make CV scores optimistic).
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_engineering.preprocessing.preprocessing_common import (
    DIRECTIONAL_DIFF_FEATURES, SYMMETRIC_COUNT_FEATURES, BINARY_FEATURES,
)
from feature_engineering.preprocessing.preprocessing_xgboost_v1 import build_augmented_training_raw, fit_preprocessing, transform
from training.xgboost.xgboost_tuning_v2 import (
    split_inner_early_stop, build_random_candidates, all_candidates, select_winner,
    complexity_key, derive_final_n_estimators, ANCHOR_CONFIGS, SEARCH_KEYS,
    LOG_LOSS_EQUIVALENCE_EPSILON, N_RANDOM_CANDIDATES, INNER_EARLY_STOP_FRACTION,
)

ROOT = Path(__file__).resolve().parents[2]
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "random_forest_cv_folds_v2.csv"
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
FEATURES_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
EVAL_MANIFEST = ROOT / "data" / "interim" / "evaluation_manifest.csv"

# RF V2 fold manifest must be reused byte-identically
RF_CV_FOLDS_SHA256 = "152864c64ef558139af8b588d80e94102a13f52786275dc386357b52ac524247"

MODEL_FEATURES = DIRECTIONAL_DIFF_FEATURES + SYMMETRIC_COUNT_FEATURES + BINARY_FEATURES + ["bestOf", "tier"]


def _synthetic_fold_train(n=600, seed=0, with_missing=True):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01")
    data = {
        "match_id": np.arange(n),
        "datetime": [base + pd.Timedelta(hours=i // 3) for i in range(n)],
        "team1_series_win": rng.integers(0, 2, size=n).astype(float),
    }
    for c in DIRECTIONAL_DIFF_FEATURES:
        data[c] = rng.normal(scale=50, size=n)
    for c in SYMMETRIC_COUNT_FEATURES:
        data[c] = rng.integers(0, 100, size=n).astype(float)
    for c in BINARY_FEATURES:
        data[c] = rng.integers(0, 2, size=n).astype(float)
    data["bestOf"] = rng.choice([1, 3, 5], size=n)
    data["tier"] = rng.choice(["tier1", "tier2", "tier3"], size=n)
    df = pd.DataFrame(data)
    if with_missing:
        idx = rng.choice(n, size=max(1, n // 10), replace=False)
        df.loc[idx, "days_since_last_match_diff"] = np.nan
    return df


# ---------------- RF fold manifest reused exactly ----------------

def test_rf_cv_fold_manifest_reused_byte_identically():
    assert CV_FOLDS_PATH.exists()
    actual = hashlib.sha256(CV_FOLDS_PATH.read_bytes()).hexdigest()
    assert actual == RF_CV_FOLDS_SHA256, "Phase 4C.1 must reuse RF V2's fold manifest unchanged"


def test_cv_folds_contain_only_train_ids_and_no_holdout_ids():
    cv = pd.read_csv(CV_FOLDS_PATH)
    split = pd.read_csv(SPLIT_PATH)
    train_ids = set(split.loc[split.split == "train", "match_id"])
    val_ids = set(split.loc[split.split == "validation", "match_id"])
    test_ids = set(split.loc[split.split == "test", "match_id"])

    cv_ids = set(cv["match_id"])
    assert cv_ids <= train_ids
    assert cv_ids.isdisjoint(val_ids)
    assert cv_ids.isdisjoint(test_ids)

    em = pd.read_csv(EVAL_MANIFEST)
    cologne = set(em.loc[em.evaluation_group == "cologne_2026", "match_id"])
    post = set(em.loc[em.evaluation_group == "post_cologne", "match_id"])
    assert cv_ids.isdisjoint(cologne)
    assert cv_ids.isdisjoint(post)


def test_outer_fold_chronology_holds_on_real_folds():
    cv = pd.read_csv(CV_FOLDS_PATH, parse_dates=["datetime"])
    for fold in sorted(cv["fold"].unique()):
        tr = cv[(cv.fold == fold) & (cv.role == "train")]
        va = cv[(cv.fold == fold) & (cv.role == "validation")]
        assert tr["datetime"].max() < va["datetime"].min()


def test_no_timestamp_group_spans_outer_fold_train_and_validation():
    cv = pd.read_csv(CV_FOLDS_PATH, parse_dates=["datetime"])
    for fold in sorted(cv["fold"].unique()):
        sub = cv[cv.fold == fold]
        per_dt_role = sub.groupby("datetime")["role"].nunique()
        assert (per_dt_role == 1).all()


# ---------------- THREE-STAGE temporal separation ----------------

def test_inner_split_is_chronological_and_group_preserving():
    train = _synthetic_fold_train(n=600, seed=1)
    fit, es = split_inner_early_stop(train)

    assert len(fit) > 0 and len(es) > 0
    assert len(fit) + len(es) == len(train)
    assert fit["datetime"].max() < es["datetime"].min()
    # no exact-timestamp group appears on both sides
    assert set(fit["datetime"]).isdisjoint(set(es["datetime"]))


def test_inner_early_stop_fraction_is_approximately_as_configured():
    train = _synthetic_fold_train(n=1000, seed=2)
    fit, es = split_inner_early_stop(train)
    frac = len(es) / len(train)
    assert abs(frac - INNER_EARLY_STOP_FRACTION) < 0.05


def test_three_stage_separation_on_every_real_fold():
    """inner_fit < inner_early_stop < outer_fold_validation, on the real data."""
    cv = pd.read_csv(CV_FOLDS_PATH, parse_dates=["datetime"])
    feats = pd.read_parquet(FEATURES_PATH, engine="fastparquet")

    for fold in sorted(cv["fold"].unique()):
        train_ids = set(cv.loc[(cv.fold == fold) & (cv.role == "train"), "match_id"])
        val_ids = set(cv.loc[(cv.fold == fold) & (cv.role == "validation"), "match_id"])
        fold_train = feats[feats.match_id.isin(train_ids)].reset_index(drop=True)
        fold_val = feats[feats.match_id.isin(val_ids)].reset_index(drop=True)

        inner_fit, inner_es = split_inner_early_stop(fold_train)

        assert inner_fit["datetime"].max() < inner_es["datetime"].min(), f"fold {fold} stage 1<2"
        assert inner_es["datetime"].max() < fold_val["datetime"].min(), f"fold {fold} stage 2<3"
        # the three stages are disjoint in both ids and timestamp groups
        assert set(inner_fit.match_id).isdisjoint(set(inner_es.match_id))
        assert set(inner_es.match_id).isdisjoint(set(fold_val.match_id))
        assert set(inner_fit["datetime"]).isdisjoint(set(inner_es["datetime"]))
        assert set(inner_es["datetime"]).isdisjoint(set(fold_val["datetime"]))


def test_early_stop_block_is_never_the_scoring_block():
    """The inner early-stop block must be strictly inside fold-train, i.e. it can
    never overlap the outer fold-validation block that scores the candidate."""
    cv = pd.read_csv(CV_FOLDS_PATH, parse_dates=["datetime"])
    feats = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    for fold in sorted(cv["fold"].unique()):
        train_ids = set(cv.loc[(cv.fold == fold) & (cv.role == "train"), "match_id"])
        val_ids = set(cv.loc[(cv.fold == fold) & (cv.role == "validation"), "match_id"])
        fold_train = feats[feats.match_id.isin(train_ids)].reset_index(drop=True)
        _, inner_es = split_inner_early_stop(fold_train)
        assert set(inner_es.match_id) <= train_ids
        assert set(inner_es.match_id).isdisjoint(val_ids)


# ---------------- mirroring / preprocessing inside folds ----------------

def test_mirroring_applies_only_to_inner_fit():
    train = _synthetic_fold_train(n=500, seed=3)
    inner_fit, inner_es = split_inner_early_stop(train)
    augmented = build_augmented_training_raw(inner_fit)

    assert len(augmented) == 2 * len(inner_fit)
    assert augmented["team1_series_win"].mean() == pytest.approx(0.5, abs=1e-12)
    # inner early-stop untouched
    assert len(inner_es) == len(train) - len(inner_fit)


def test_fold_preprocessing_preserves_nan_and_applies_no_scaling():
    train = _synthetic_fold_train(n=500, seed=4, with_missing=True)
    inner_fit, inner_es = split_inner_early_stop(train)
    augmented = build_augmented_training_raw(inner_fit)
    params = fit_preprocessing(augmented, MODEL_FEATURES)

    X, names = transform(augmented, params)
    idx = names.index("days_since_last_match_diff")
    assert int(np.isnan(X[:, idx]).sum()) == int(augmented["days_since_last_match_diff"].isna().sum())
    assert params["scaling_applied"] is False
    assert params["imputation_applied"] is False
    assert params["missing_value_policy"] == "preserve_nan_native_xgboost"

    # unscaled: raw values pass through untouched
    e = names.index("elo_diff")
    np.testing.assert_array_equal(X[:, e], augmented["elo_diff"].to_numpy(dtype=float))

    # inner early-stop transformed with the inner-fit-derived artifact, still unmirrored
    X_es, _ = transform(inner_es, params)
    assert X_es.shape[0] == len(inner_es)
    assert X_es.shape[1] == 19


# ---------------- candidate generation ----------------

def test_candidate_generation_is_deterministic_and_duplicate_free():
    a = build_random_candidates()
    b = build_random_candidates()
    assert a == b
    assert len(a) == N_RANDOM_CANDIDATES

    cands = all_candidates()
    assert len(cands) == len(ANCHOR_CONFIGS) + N_RANDOM_CANDIDATES
    ids = [c["candidate_id"] for c in cands]
    assert len(ids) == len(set(ids))
    tuples = [tuple(c[k] for k in SEARCH_KEYS) for c in cands]
    assert len(tuples) == len(set(tuples))


def test_anchors_present_and_v1_reference_is_selectable():
    cands = all_candidates()
    ids = {c["candidate_id"] for c in cands}
    assert "xgb_v1_structure_reference" in ids, "V1 structural reference must be a candidate"
    for concept in ["anchor_shallow_depth2_conservative", "anchor_shallow_depth3_conservative",
                     "anchor_moderate_depth3_mcw5", "anchor_moderate_depth4_mcw3",
                     "anchor_strong_reg_gamma", "anchor_strong_reg_lambda_alpha",
                     "anchor_strong_subsampling", "anchor_low_lr_0p02", "anchor_faster_lr_0p10"]:
        assert concept in ids
    v1 = next(c for c in cands if c["candidate_id"] == "xgb_v1_structure_reference")
    assert v1["learning_rate"] == 0.05 and v1["max_depth"] == 4 and v1["min_child_weight"] == 1


# ---------------- selection rule ----------------

def _agg(rows):
    return pd.DataFrame(rows)


def test_selection_primary_lowest_log_loss():
    p = {"a": dict(max_depth=4, min_child_weight=1, gamma=0.0, reg_lambda=1.0, reg_alpha=0.0, candidate_id="a"),
         "b": dict(max_depth=4, min_child_weight=1, gamma=0.0, reg_lambda=1.0, reg_alpha=0.0, candidate_id="b")}
    agg = _agg([
        {"candidate_id": "a", "val_log_loss_mean": 0.650, "val_roc_auc_mean": 0.60, "val_log_loss_std": 0.01, "median_effective_n_estimators": 100},
        {"candidate_id": "b", "val_log_loss_mean": 0.680, "val_roc_auc_mean": 0.70, "val_log_loss_std": 0.01, "median_effective_n_estimators": 100},
    ])
    w, stage = select_winner(agg, p)
    assert w == "a" and "primary" in stage


def test_selection_epsilon_tie_resolved_by_roc_auc():
    p = {k: dict(max_depth=4, min_child_weight=1, gamma=0.0, reg_lambda=1.0, reg_alpha=0.0, candidate_id=k) for k in "ab"}
    agg = _agg([
        {"candidate_id": "a", "val_log_loss_mean": 0.650, "val_roc_auc_mean": 0.60, "val_log_loss_std": 0.01, "median_effective_n_estimators": 100},
        {"candidate_id": "b", "val_log_loss_mean": 0.650 + LOG_LOSS_EQUIVALENCE_EPSILON / 2, "val_roc_auc_mean": 0.66, "val_log_loss_std": 0.01, "median_effective_n_estimators": 100},
    ])
    w, stage = select_winner(agg, p)
    assert w == "b" and "secondary" in stage


def test_selection_tertiary_resolved_by_lower_log_loss_std():
    p = {k: dict(max_depth=4, min_child_weight=1, gamma=0.0, reg_lambda=1.0, reg_alpha=0.0, candidate_id=k) for k in "ab"}
    agg = _agg([
        {"candidate_id": "a", "val_log_loss_mean": 0.650, "val_roc_auc_mean": 0.64, "val_log_loss_std": 0.05, "median_effective_n_estimators": 100},
        {"candidate_id": "b", "val_log_loss_mean": 0.6505, "val_roc_auc_mean": 0.64, "val_log_loss_std": 0.01, "median_effective_n_estimators": 100},
    ])
    w, stage = select_winner(agg, p)
    assert w == "b" and "tertiary" in stage


def test_selection_complexity_tiebreak_prefers_shallower_then_more_regularized():
    p = {
        "deep": dict(max_depth=6, min_child_weight=1, gamma=0.0, reg_lambda=1.0, reg_alpha=0.0, candidate_id="deep"),
        "shallow": dict(max_depth=2, min_child_weight=20, gamma=2.0, reg_lambda=10.0, reg_alpha=1.0, candidate_id="shallow"),
    }
    agg = _agg([
        {"candidate_id": "deep", "val_log_loss_mean": 0.650, "val_roc_auc_mean": 0.64, "val_log_loss_std": 0.02, "median_effective_n_estimators": 100},
        {"candidate_id": "shallow", "val_log_loss_mean": 0.6505, "val_roc_auc_mean": 0.64, "val_log_loss_std": 0.02, "median_effective_n_estimators": 100},
    ])
    w, stage = select_winner(agg, p)
    assert w == "shallow" and "complexity" in stage


def test_complexity_key_excludes_subsample_and_colsample():
    """subsample/colsample_bytree must NOT participate in the complexity ranking."""
    a = dict(max_depth=4, min_child_weight=5, gamma=0.0, reg_lambda=1.0, reg_alpha=0.0,
             subsample=0.60, colsample_bytree=0.60, candidate_id="x")
    b = dict(max_depth=4, min_child_weight=5, gamma=0.0, reg_lambda=1.0, reg_alpha=0.0,
             subsample=1.00, colsample_bytree=1.00, candidate_id="x")
    assert complexity_key(a, 100) == complexity_key(b, 100)


def test_complexity_key_orders_shallower_as_simpler():
    shallow = dict(max_depth=2, min_child_weight=20, gamma=2.0, reg_lambda=10.0, reg_alpha=1.0, candidate_id="s")
    deep = dict(max_depth=6, min_child_weight=1, gamma=0.0, reg_lambda=0.5, reg_alpha=0.0, candidate_id="d")
    assert complexity_key(shallow, 100) < complexity_key(deep, 100)


# ---------------- final_n_estimators rule ----------------

def test_final_n_estimators_is_median_of_best_iteration_plus_one():
    assert derive_final_n_estimators([9, 19, 29, 39]) == int(round(np.median([10, 20, 30, 40])))  # 25
    assert derive_final_n_estimators([100, 100, 100, 100]) == 101
    # even count -> numpy median averages the middle two
    assert derive_final_n_estimators([10, 20, 30, 40]) == 26
