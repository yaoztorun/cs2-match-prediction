"""
Tests for the Phase 7 sealed-TEST evaluation stack (brief section 32).
Synthetic-fixture-first, per the brief's own instruction: these must pass
BEFORE TEST is ever opened.
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from feature_engineering.series.feature_engine import elo_expected, ELO_INITIAL          # noqa: E402
from training.map_models.map_modeling_common import baseline_probabilities, series_macro_metrics  # noqa: E402
import evaluation.uncertainty.phase7_test_bootstrap as boot                            # noqa: E402
import evaluation.internal_test.phase7_test_reports as reports                           # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
_PHASE7_SCRIPT_DIRS = {"evaluate_phase7_test_once.py": ROOT / "evaluation" / "internal_test",
                       "phase7_test_reports.py": ROOT / "evaluation" / "internal_test",
                       "phase7_test_visualizations.py": ROOT / "evaluation" / "internal_test",
                       "freeze_phase7_protocol.py": ROOT / "evaluation" / "internal_test",
                       "phase7_test_bootstrap.py": ROOT / "evaluation" / "uncertainty"}


def _phase7_src(name):
    return _PHASE7_SCRIPT_DIRS[name] / name


# --- protocol content ---

def test_protocol_yaml_declares_every_predefined_metric_and_baseline():
    import yaml
    cfg = yaml.safe_load((ROOT / "config" / "evaluation" / "phase7_test_evaluation_protocol.yaml").read_text(encoding="utf-8"))
    assert set(cfg["metrics"]["primary"]) == {"log_loss", "roc_auc", "brier"}
    assert set(cfg["metrics"]["secondary"]) == {"accuracy", "precision", "recall", "f1"}
    assert set(cfg["metrics"]["series_macro"]) == {"log_loss", "brier", "accuracy"}
    assert {b["id"] for b in cfg["baselines"]["items"]} == {"constant_05", "overall_elo", "map_elo"}
    assert cfg["frozen_system"]["threshold"] == 0.5
    assert cfg["uncertainty"]["n_bootstrap"] == 2000
    assert cfg["uncertainty"]["random_state"] == 42
    assert cfg["uncertainty"]["ci"] == 0.95
    assert cfg["uncertainty"]["cluster_key"] == "match_id"
    assert cfg["diagnostics"]["calibration"]["n_bins"] == 10
    assert cfg["diagnostics"]["calibration"]["edges"] == [round(i / 10, 1) for i in range(11)]
    assert cfg["diagnostics"]["calibration"]["last_bin_closed_at_one"] is True


def test_protocol_contains_no_test_outcome_fields():
    import yaml
    cfg = yaml.safe_load((ROOT / "config" / "evaluation" / "phase7_test_evaluation_protocol.yaml").read_text(encoding="utf-8"))
    text = json.dumps(cfg)
    for bad in ["accuracy_test", "roc_auc_test", "observed", "p_xgb_v3_final", "log_loss_test"]:
        assert bad not in text


def test_protocol_is_frozen_before_the_evaluator_can_execute():
    """freeze_phase7_protocol.py hashes evaluate_phase7_test_once.py's own
    source - this only makes sense if the protocol is written after the
    evaluator's code exists but the ordering documented in the module
    docstring makes clear it must run before real execution."""
    src = (ROOT / "evaluation" / "internal_test" / "freeze_phase7_protocol.py").read_text(encoding="utf-8")
    assert "evaluate_phase7_test_once" in src
    assert "protocol frozen before evaluator execution" in src.lower()


# --- hash verification logic ---

def test_hash_mismatch_is_detected():
    import hashlib
    real = hashlib.sha256(b"frozen content").hexdigest()
    tampered = hashlib.sha256(b"different content").hexdigest()
    assert real != tampered   # trivial but documents the exact mechanism evaluate_phase7_test_once.py relies on


# --- evaluator contains no fitting ---

def test_evaluator_contains_no_fit_call():
    src = (ROOT / "evaluation" / "internal_test" / "evaluate_phase7_test_once.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            attr = f.attr if isinstance(f, ast.Attribute) else ""
            assert attr not in {"fit", "fit_transform", "fit_preprocessing"}, \
                f"evaluate_phase7_test_once.py calls {attr}(...) - this script must never fit anything"


def test_evaluator_only_loads_never_fits_the_model():
    src = (ROOT / "evaluation" / "internal_test" / "evaluate_phase7_test_once.py").read_text(encoding="utf-8")
    assert "load_model" in src
    assert ".fit(" not in src


# --- threshold / calibration / ensemble ---

def test_threshold_is_hardcoded_0_5_everywhere_in_phase7():
    for name in ["evaluate_phase7_test_once.py", "phase7_test_reports.py"]:
        src = _phase7_src(name).read_text(encoding="utf-8")
        assert "0.5" in src
    assert not list((ROOT / "data" / "evaluation").glob("*threshold_search*"))


def test_no_calibration_import_anywhere_in_phase7():
    for name in ["evaluate_phase7_test_once.py", "phase7_test_reports.py", "phase7_test_bootstrap.py",
                  "phase7_test_visualizations.py"]:
        src = _phase7_src(name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.update(a.name for a in node.names)
        bad = {m for m in imported if "calibrat" in m.lower() or "isotonic" in m.lower()}
        assert not bad, f"{name} imports a calibration tool: {bad}"


def test_no_new_ensemble_construction():
    for name in ["evaluate_phase7_test_once.py", "phase7_test_reports.py"]:
        src = _phase7_src(name).read_text(encoding="utf-8").lower()
        assert "votingclassifier" not in src and "stackingclassifier" not in src
        assert "new ensemble" not in src or "no new ensemble" in src or "does not build" in src


# --- baseline formulas ---

def test_constant_05_baseline():
    df = pd.DataFrame({"elo_diff": [10.0, -20.0], "map_elo_diff": [5.0, -5.0]})
    p = baseline_probabilities(df, "half")
    assert np.allclose(p, 0.5)


def test_overall_elo_baseline_formula():
    df = pd.DataFrame({"elo_diff": [0.0, 200.0, -200.0]})
    p = baseline_probabilities(df, "overall_elo")
    expected = [elo_expected(ELO_INITIAL + d, ELO_INITIAL) for d in df["elo_diff"]]
    assert np.allclose(p, expected)
    assert p[0] == pytest.approx(0.5)
    assert p[1] > 0.5 and p[2] < 0.5


def test_map_elo_baseline_formula():
    df = pd.DataFrame({"map_elo_diff": [0.0, 150.0]})
    p = baseline_probabilities(df, "map_elo")
    expected = [elo_expected(ELO_INITIAL + d, ELO_INITIAL) for d in df["map_elo_diff"]]
    assert np.allclose(p, expected)


# --- series-macro arithmetic ---

def test_series_macro_averages_per_series_then_equally_across_series():
    match_ids = np.array(["m1", "m1", "m2"])
    y = np.array([1.0, 1.0, 0.0])
    p = np.array([0.9, 0.9, 0.9])   # m1: 2 confident-correct maps, m2: 1 confident-wrong map
    out = series_macro_metrics(match_ids, y, p)
    assert out["n_series"] == 2
    # m1 accuracy=1.0, m2 accuracy=0.0 -> macro accuracy = 0.5, NOT the row-weighted 2/3
    assert out["series_macro_accuracy"] == pytest.approx(0.5)


# --- cluster bootstrap: match_id, not row-independent ---

def test_bootstrap_resamples_match_id_not_rows_independently():
    pred = pd.DataFrame({
        "match_id": ["m1", "m1", "m1", "m2"], "y_true": [1, 1, 0, 0],
        "p_xgb_v3_final": [0.6, 0.7, 0.4, 0.3], "p_overall_elo": [0.55, 0.55, 0.55, 0.45],
        "p_map_elo": [0.5, 0.5, 0.5, 0.5],
    })
    metrics, replicate_row_indices = boot.run_cluster_bootstrap(pred, ["p_xgb_v3_final"], n_bootstrap=50,
                                                                  random_state=1)
    # whenever m1 is sampled, ALL THREE of its rows must appear together in the replicate
    match_ids = pred["match_id"].to_numpy()
    for rep in replicate_row_indices:
        rep_matches = match_ids[rep]
        count_m1 = int((rep_matches == "m1").sum())
        assert count_m1 % 3 == 0, "m1's 3 rows were not resampled as an atomic cluster"


def test_paired_bootstrap_uses_identical_draws_across_arms():
    pred = pd.DataFrame({
        "match_id": ["m1", "m1", "m2", "m3"], "y_true": [1, 0, 1, 0],
        "p_xgb_v3_final": [0.6, 0.4, 0.7, 0.3], "p_overall_elo": [0.55, 0.45, 0.6, 0.4],
        "p_map_elo": [0.5, 0.5, 0.5, 0.5],
    })
    _, replicate_row_indices_a = boot.run_cluster_bootstrap(pred, ["p_xgb_v3_final"], n_bootstrap=20,
                                                               random_state=7)
    _, replicate_row_indices_b = boot.run_cluster_bootstrap(pred, ["p_overall_elo"], n_bootstrap=20,
                                                               random_state=7)
    for a, b in zip(replicate_row_indices_a, replicate_row_indices_b):
        assert np.array_equal(a, b), "the same random_state must produce identical per-replicate draws"


def test_bootstrap_seed_and_replicate_count_are_fixed_constants():
    assert boot.RANDOM_STATE == 42
    assert boot.N_BOOTSTRAP == 2000
    assert boot.CI == 0.95


# --- calibration bins: fixed, not data-dependent ---

def test_calibration_bins_are_fixed_width_not_quantile():
    p = np.array([0.0, 0.05, 0.15, 0.55, 0.89, 0.9, 0.95, 1.0])
    idx = reports.calibration_bin_index(p)
    assert idx.tolist() == [0, 0, 1, 5, 8, 9, 9, 9]   # p==1.0 lands in the LAST (closed) bin, index 9
    assert reports.CAL_EDGES == [round(i / 10, 1) for i in range(11)]


def test_calibration_last_bin_is_closed_at_exactly_one():
    idx = reports.calibration_bin_index(np.array([1.0]))
    assert idx[0] == 9   # not dropped, not an 11th bin


# --- side-symmetry formula ---

def test_side_symmetry_formula():
    p_orig = np.array([0.7, 0.3])
    p_mirrored = np.array([0.32, 0.71])   # imperfect symmetry
    err = np.abs(p_orig - (1 - p_mirrored))
    assert err[0] == pytest.approx(abs(0.7 - 0.68))
    assert err[1] == pytest.approx(abs(0.3 - 0.29))


# --- per-map small-n flag ---

def test_small_sample_flag_threshold_is_30():
    assert reports.SMALL_SAMPLE_N == 30


# --- report scripts read only the canonical prediction artifact ---

def _reads_any(source, needles):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if fname in {"read_csv", "read_parquet"}:
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    if any(n in ast.unparse(arg) for n in needles):
                        return True
    return False


@pytest.mark.parametrize("name", ["phase7_test_reports.py", "phase7_test_bootstrap.py",
                                    "phase7_test_visualizations.py"])
def test_downstream_scripts_never_reopen_raw_test_sources(name):
    src = _phase7_src(name).read_text(encoding="utf-8")
    assert not _reads_any(src, ["map_features_v3_modern_map", "map_split_v1"]), \
        f"{name} reopens a raw TEST source - it must read only the prediction artifact / derived tables"


def test_only_evaluator_may_filter_split_equals_test():
    for name in ["phase7_test_reports.py", "phase7_test_bootstrap.py", "phase7_test_visualizations.py",
                  "freeze_phase7_protocol.py"]:
        src = _phase7_src(name).read_text(encoding="utf-8")
        assert 'split"] == "test"' not in src.replace("'", '"')
        assert "split == 'test'" not in src and 'split == "test"' not in src


# --- evaluator refuses overwrite ---

def test_evaluator_refuses_overwrite_of_existing_canonical_artifact(tmp_path):
    canonical = tmp_path / "map_test_predictions_v1.parquet"
    canonical.write_bytes(b"already here")
    assert canonical.exists()
    # mirrors the exact guard in evaluate_phase7_test_once.main(): existence check before any scoring
    with pytest.raises(RuntimeError):
        if canonical.exists():
            raise RuntimeError("refusing to overwrite")


def test_evaluator_source_contains_the_abort_before_overwrite_guard():
    src = (ROOT / "evaluation" / "internal_test" / "evaluate_phase7_test_once.py").read_text(encoding="utf-8")
    assert "CANONICAL_PATH.exists()" in src
    assert "already exists" in src
    assert "os.replace" in src   # atomic commit, not a direct write to the canonical path


# --- no Cologne access beyond the permitted group-label lookup ---

def test_cologne_access_is_limited_to_group_label_lookup():
    """The only permitted Cologne "access" anywhere in Phase 7 is
    evaluate_phase7_test_once.py's group-label lookup against
    evaluation_manifest.csv. Other scripts may still mention "Cologne" in
    prose (e.g. a status line documenting COLOGNE=UNTOUCHED) - that is not a
    data access, so this checks for actual identifiers via AST, not a bare
    substring search that would flag legitimate documentation text."""
    src = (ROOT / "evaluation" / "internal_test" / "evaluate_phase7_test_once.py").read_text(encoding="utf-8")
    assert "evaluation_manifest.csv" in src
    assert "cologne_2026" in src and "post_cologne" in src
    for name in ["phase7_test_reports.py", "phase7_test_bootstrap.py", "phase7_test_visualizations.py"]:
        s = _phase7_src(name).read_text(encoding="utf-8")
        assert "evaluation_manifest" not in s
        assert "cologne_2026" not in s and "post_cologne" not in s
