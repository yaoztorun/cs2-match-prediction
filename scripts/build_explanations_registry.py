"""Builds config/application_explanations_v1.yaml (Phase 9C)."""

import yaml

from _common import ROOT
import phase9a_common as p9a  # reuses sha256_file, nothing else

SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "config"
sha256_file = p9a.sha256_file


def build():
    import build_application_registries as bar

    registry = {
        "explanation_version": "application_explanations_v1",
        "prediction_contract": "phase9b",
        "causal": False,
        "explanation_type": "model_feature_attribution",
        "models": {
            "series_random_forest_v2": {
                "attribution_method": "saabas_path_decomposition",
                "attribution_output_space": "probability",
                "note": "Saabas-style tree path decomposition (Saabas 2014) - NOT SHAP/Shapley values. "
                        "Exact reconstruction of this forest's own predict_proba output, path/tree-structure "
                        "dependent, lacks SHAP's symmetry/consistency guarantees.",
            },
            "map_xgboost_v3_final": {
                "attribution_method": "xgboost_native_treeshap",
                "attribution_output_space": "log_odds",
                "note": "Native TreeSHAP via Booster.predict(pred_contribs=True) - exact Shapley values, "
                        "built into xgboost's C++ core, zero new dependencies.",
            },
        },
        "hashes": {
            "application_explanations_py": sha256_file(SCRIPTS / "application_explanations.py"),
            "feature_groups_config": sha256_file(CONFIG / "application_explanation_feature_groups_v1.yaml"),
            "application_inference_py": sha256_file(SCRIPTS / "application_inference.py"),
            "phase9b_context_registry": sha256_file(CONFIG / "application_inference_contexts_v1.yaml"),
            "rf_v2_model": sha256_file(bar.RF_PIPELINE["rf_model"]),
            "rf_v2_preprocessing": sha256_file(bar.RF_PIPELINE["rf_preprocessing"]),
            "xgb_v3_model": sha256_file(bar.XGB_PIPELINE["xgb_model"]),
            "xgb_v3_preprocessing": sha256_file(bar.XGB_PIPELINE["xgb_preprocessing"]),
        },
    }
    out_path = CONFIG / "application_explanations_v1.yaml"
    out_path.write_text(yaml.safe_dump(registry, sort_keys=False, default_flow_style=False), encoding="utf-8")
    print(f"wrote {out_path}")
    return registry


if __name__ == "__main__":
    build()
