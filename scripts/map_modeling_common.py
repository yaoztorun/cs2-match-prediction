"""
[PROJECT ADDITION - Phase 6B]

Shared harness for the KNOWN-MAP models: dataset/manifest loading with the
leakage assertions re-run every time, the per-fold preprocessing CACHE (so 36
RF candidates x 4 folds do not rebuild the same matrices 144 times), metric
helpers, series-macro diagnostics, the chronological inner early-stop split,
the ELO reference baselines, and deterministic search-plan/checkpoint support.

TRAIN-ONLY BY CONSTRUCTION
--------------------------
`load_cv_manifest()` reads data/modeling/map_cv_folds_v1.csv, which Phase 6A
built exclusively from TRAIN match_ids. Tuning scripts import from here and
never open data/modeling/map_split_v1.csv - the main VALIDATION partition is
therefore structurally absent from hyperparameter selection, not merely
"avoided by convention". Only the final evaluation script loads the split
manifest, and only after every configuration is frozen.
"""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss, confusion_matrix,
)

from _common import ROOT
from build_series_split_v1 import pick_boundary_index
from feature_engine import elo_expected, ELO_INITIAL
from preprocessing_common_map_v2 import (
    load_map_v2_roles, build_augmented_training_raw, assert_augmented_symmetry,
)
import preprocessing_random_forest_map_v2 as prep_rf
import preprocessing_xgboost_map_v2 as prep_xgb

CONFIG_PATH = ROOT / "config" / "map_features_v2_rich.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "map_features_v2_rich.parquet"
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "map_cv_folds_v1.csv"
SPLIT_PATH = ROOT / "data" / "modeling" / "map_split_v1.csv"
MODELING_DIR = ROOT / "data" / "modeling"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
N_FOLDS = 4
LOG_LOSS_EQUIVALENCE_EPSILON = 0.002   # fixed BEFORE any search runs - never altered afterwards

EXPECTED_TOTAL_MAP_ROWS = 10318
EXPECTED_TRAIN_N = 7762
EXPECTED_VAL_N = 1129
EXPECTED_TEST_N = 1427

# Columns that must never reach a model matrix (brief section 4). Checked
# against the raw feature frame before any fit.
FORBIDDEN_PREDICTORS = [
    "score1_game", "score2_game", "kills", "deaths", "assists", "adr", "kast", "kddiff",
    "player_id", "player_name", "lineup", "team1_win", "team1_series_win", "map_id",
]


# ---------------------------------------------------------------------------
# loading + leakage assertions
# ---------------------------------------------------------------------------

def load_roles():
    return load_map_v2_roles(CONFIG_PATH)


def load_features():
    """The full known-map feature frame. Loaded ONCE per script."""
    df = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    assert len(df) == EXPECTED_TOTAL_MAP_ROWS, f"unexpected map row count: {len(df)}"
    return df


def assert_target_and_no_forbidden_columns(df, roles):
    """Brief section 4, re-asserted before every modeling run."""
    y = df[roles["target"]]
    assert y.notna().all(), "target has missing values"
    assert set(pd.unique(y)) <= {0, 1}, f"target is not binary: {sorted(set(pd.unique(y)))}"

    model_cols = set(roles["model_features"])
    bad = sorted(c for c in model_cols if c in FORBIDDEN_PREDICTORS)
    assert not bad, f"forbidden column present in the model feature list: {bad}"
    token_bad = sorted(
        c for c in model_cols
        if any(t in c.lower() for t in ("score1_game", "score2_game", "kddiff", "player1", "player2",
                                         "player3", "player4", "player5", "_kills", "_deaths", "lineup"))
    )
    assert not token_bad, f"forbidden token in the model feature list: {token_bad}"
    missing = sorted(model_cols - set(df.columns))
    assert not missing, f"config-declared features absent from the parquet: {missing}"


def load_cv_manifest(verify_against_split=True):
    """The frozen TRAIN-only map CV manifest, with brief section 3's checks
    re-run rather than assumed.

    `verify_against_split=True` reads data/modeling/map_split_v1.csv to prove the
    manifest is TRAIN-only. That read is itself leakage-safe (it inspects only
    which ids are TRAIN), but tuning scripts still pass False so that they never
    open the split manifest at all - the stronger structural guarantee. The
    validator performs the id-level cross-check independently."""
    cv = pd.read_csv(CV_FOLDS_PATH, parse_dates=["datetime"])
    assert sorted(cv["fold"].unique()) == list(range(1, N_FOLDS + 1))
    assert set(cv["role"]) == {"train", "validation"}

    for fold in range(1, N_FOLDS + 1):
        f = cv[cv["fold"] == fold]
        tr, va = f[f["role"] == "train"], f[f["role"] == "validation"]
        assert len(tr) and len(va), f"fold {fold} has an empty side"
        # one role per match_id within a fold; maps of a series stay together
        assert (f.groupby("match_id")["role"].nunique() == 1).all(), \
            f"fold {fold}: a match_id appears in both roles"
        # chronology, re-derived not assumed
        assert tr["datetime"].max() < va["datetime"].min(), f"fold {fold}: outer chronology violated"
        # exact-timestamp groups are never split across the boundary
        assert set(tr["datetime"]).isdisjoint(set(va["datetime"])), \
            f"fold {fold}: an exact timestamp group crosses the train/validation boundary"

    if verify_against_split:
        split = pd.read_csv(SPLIT_PATH)
        train_ids = set(split.loc[split["split"] == "train", "match_id"])
        other_ids = set(split.loc[split["split"] != "train", "match_id"])
        assert set(cv["match_id"]) <= train_ids, "CV manifest contains a non-TRAIN match_id"
        assert set(cv["match_id"]).isdisjoint(other_ids), "CV manifest overlaps validation/test"
    return cv


def fold_frames(cv, features_df, fold):
    """The ORIGINAL (unmirrored) raw rows of one outer fold."""
    ids_tr = set(cv.loc[(cv["fold"] == fold) & (cv["role"] == "train"), "match_id"])
    ids_va = set(cv.loc[(cv["fold"] == fold) & (cv["role"] == "validation"), "match_id"])
    tr = features_df[features_df["match_id"].isin(ids_tr)].sort_values(
        ["series_datetime", "match_id", "game_id"]).reset_index(drop=True)
    va = features_df[features_df["match_id"].isin(ids_va)].sort_values(
        ["series_datetime", "match_id", "game_id"]).reset_index(drop=True)
    return tr, va


# ---------------------------------------------------------------------------
# per-fold preprocessing cache
# ---------------------------------------------------------------------------

class FoldCache:
    """Builds each fold's RAW frames and BOTH models' transformed matrices
    exactly ONCE, then hands the same arrays to every candidate. Without this,
    36 RF candidates x 4 folds would re-fit and re-apply identical preprocessing
    144 times for no reason.

    Per fold it holds:
        raw_train / raw_val / raw_train_augmented
        rf: X_aug, X_train_orig, X_val, y_*, fitted preprocessing params
        xgb: the same, under XGB's NaN-preserving policy
    """

    def __init__(self, cv, features_df, roles, build_rf=True, build_xgb=True):
        self.roles = roles
        self.target = roles["target"]
        self.folds = {}
        for fold in range(1, N_FOLDS + 1):
            raw_tr, raw_va = fold_frames(cv, features_df, fold)
            aug_tr = build_augmented_training_raw(raw_tr, roles)
            assert len(aug_tr) == 2 * len(raw_tr)
            assert abs(float(aug_tr[self.target].mean()) - 0.5) < 1e-9
            assert_augmented_symmetry(aug_tr, roles)

            entry = {
                "raw_train": raw_tr, "raw_val": raw_va, "raw_train_augmented": aug_tr,
                "y_aug": aug_tr[self.target].to_numpy(dtype=float),
                "y_train_orig": raw_tr[self.target].to_numpy(dtype=float),
                "y_val": raw_va[self.target].to_numpy(dtype=float),
                "n_train_unique": len(raw_tr), "n_train_augmented": len(aug_tr), "n_val": len(raw_va),
            }
            if build_rf:
                p = prep_rf.fit_preprocessing(aug_tr, roles)
                entry["rf_params"] = p
                entry["rf_X_aug"], self.feature_names = prep_rf.transform(aug_tr, p, roles)
                entry["rf_X_train_orig"], _ = prep_rf.transform(raw_tr, p, roles)
                entry["rf_X_val"], _ = prep_rf.transform(raw_va, p, roles)
            if build_xgb:
                p = prep_xgb.fit_preprocessing(aug_tr, roles)
                entry["xgb_params"] = p
                entry["xgb_X_aug"], self.feature_names = prep_xgb.transform(aug_tr, p, roles)
                entry["xgb_X_train_orig"], _ = prep_xgb.transform(raw_tr, p, roles)
                entry["xgb_X_val"], _ = prep_xgb.transform(raw_va, p, roles)
            self.folds[fold] = entry

    def __getitem__(self, fold):
        return self.folds[fold]


# ---------------------------------------------------------------------------
# chronological inner early-stop split (XGB only)
# ---------------------------------------------------------------------------

INNER_EARLY_STOP_FRACTION = 0.15   # latest ~15% of outer-fold-train timestamp groups


def split_inner_early_stop(fold_train_df, fraction=INNER_EARLY_STOP_FRACTION):
    """Split one outer fold's TRAIN history chronologically into INNER FIT
    (earlier) and INNER EARLY STOP (later). Grouping is by `series_datetime`,
    which keeps every map of a series - and every exact-timestamp group -
    entirely on one side. The outer fold's validation block is untouched."""
    d = fold_train_df.sort_values(["series_datetime", "match_id", "game_id"]).reset_index(drop=True)
    group_sizes = d.groupby("series_datetime").size().sort_index()
    datetimes = group_sizes.index.tolist()
    cum, running = [], 0
    for c in group_sizes.values.tolist():
        running += c
        cum.append(running)

    b = pick_boundary_index(cum, (1.0 - fraction) * len(d))
    b = max(0, min(b, len(datetimes) - 2))     # both sides non-empty
    cutoff = datetimes[b]

    inner_fit = d[d["series_datetime"] <= cutoff].reset_index(drop=True)
    inner_es = d[d["series_datetime"] > cutoff].reset_index(drop=True)
    assert len(inner_fit) and len(inner_es), "inner split produced an empty side"
    assert inner_fit["series_datetime"].max() < inner_es["series_datetime"].min()
    assert set(inner_fit["match_id"]).isdisjoint(set(inner_es["match_id"])), \
        "a series' maps were split across the inner early-stop boundary"
    return inner_fit, inner_es


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_proba, y_pred=None, with_confusion=False):
    if y_pred is None:
        y_pred = (np.asarray(y_proba) >= 0.5).astype(int)
    y_true = np.asarray(y_true)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(set(y_true.tolist())) > 1 else float("nan"),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }
    if with_confusion:
        out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        out["majority_class_accuracy"] = float(max(y_true.mean(), 1 - y_true.mean()))
    return out


def series_macro_metrics(match_ids, y_true, y_proba):
    """Brief section 25. Multiple maps of one series are dependent observations,
    so alongside the PRIMARY map-level metrics this averages each series' own
    mean per-map log loss / Brier / correctness, then averages those per-series
    values equally across match_ids - so a BO5 does not outweigh a BO1.

    Deliberately NO per-series ROC-AUC: most series have too few maps (and often
    a single class) for an AUC to mean anything."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_proba, dtype=float), 1e-15, 1 - 1e-15)
    per_map_ll = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    per_map_brier = (p - y_true) ** 2
    per_map_correct = ((p >= 0.5).astype(float) == y_true).astype(float)

    d = pd.DataFrame({"match_id": np.asarray(match_ids), "ll": per_map_ll,
                       "brier": per_map_brier, "correct": per_map_correct})
    per_series = d.groupby("match_id")[["ll", "brier", "correct"]].mean()
    return {
        "n_series": int(len(per_series)),
        "series_macro_log_loss": float(per_series["ll"].mean()),
        "series_macro_brier": float(per_series["brier"].mean()),
        "series_macro_accuracy": float(per_series["correct"].mean()),
    }


# ---------------------------------------------------------------------------
# ELO reference baselines (brief section 11) - never tuned
# ---------------------------------------------------------------------------

def baseline_probabilities(df, kind):
    """kind in {"half", "overall_elo", "map_elo"}. The ELO baselines invert the
    project's unchanged expected-score formula (feature_engine.elo_expected) on
    the already-computed pre-series diffs - no refitting, no parameters."""
    if kind == "half":
        return np.full(len(df), 0.5)
    col = {"overall_elo": "elo_diff", "map_elo": "map_elo_diff"}[kind]
    diff = df[col].to_numpy(dtype=float)
    assert np.isfinite(diff).all(), f"{col} unexpectedly non-finite - baseline would be undefined"
    # elo_expected(r_a, r_b) depends only on (r_a - r_b), which is exactly the diff column
    return np.array([elo_expected(ELO_INITIAL + d, ELO_INITIAL) for d in diff])


# ---------------------------------------------------------------------------
# deterministic search-plan + checkpoint/resume
# ---------------------------------------------------------------------------

def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_versions():
    import sklearn
    import xgboost
    return {"numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}


def build_search_plan(model_name, candidates, extra):
    """A fully deterministic description of what this search WILL do. Contains
    NO wall-clock timestamp, so its hash is stable across reruns and can gate
    checkpoint resumption."""
    plan = {
        "model": model_name,
        "seed": RANDOM_STATE,
        "n_folds": N_FOLDS,
        "log_loss_equivalence_epsilon": LOG_LOSS_EQUIVALENCE_EPSILON,
        "candidates": candidates,
        "artifact_hashes": {
            "config/map_features_v2_rich.yaml": sha256_file(CONFIG_PATH),
            "data/features/map_features_v2_rich.parquet": sha256_file(FEATURES_PATH),
            "data/modeling/map_cv_folds_v1.csv": sha256_file(CV_FOLDS_PATH),
        },
        "package_versions": package_versions(),
        **extra,
    }
    plan["plan_hash"] = hashlib.sha256(
        json.dumps({k: v for k, v in plan.items() if k != "plan_hash"},
                    sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return plan


def load_checkpoint(progress_path, plan_hash):
    """Returns {(candidate_id, fold): record} for records written under the
    IDENTICAL plan hash. A hash mismatch discards everything rather than mixing
    results from two different searches."""
    done = {}
    if not progress_path.exists():
        return done
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("plan_hash") != plan_hash:
            continue
        done[(rec["candidate_id"], int(rec["fold"]))] = rec
    return done


def append_checkpoint(progress_path, record):
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def reset_checkpoint_if_stale(progress_path, plan_hash):
    """If the file exists but was written under a different plan, start clean -
    old and new results are never mixed."""
    if not progress_path.exists():
        return False
    hashes = set()
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            hashes.add(json.loads(line).get("plan_hash"))
    if hashes and hashes != {plan_hash}:
        progress_path.unlink()
        return True
    return False
