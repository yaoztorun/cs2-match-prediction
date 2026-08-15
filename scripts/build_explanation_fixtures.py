"""
Computes explanation outputs for the PREDECLARED fixture manifest
(config/application_explanation_fixtures_v1.yaml, frozen before this script
ever ran - amendment #22). Writes one JSON file per fixture group under
data/deployment/phase9c_fixtures/. No actual match outcome labels - only
model-generated prediction/explanation output.
"""

import json

import yaml

from _common import ROOT
import application_explanations as ae

MANIFEST_PATH = ROOT / "config" / "application_explanation_fixtures_v1.yaml"
OUT_DIR = ROOT / "data" / "deployment" / "phase9c_fixtures"


def build():
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = {}

    um_out = []
    for fx in manifest["fixtures"]["unknown_map"]:
        result = ae.explain_series_unknown_maps(fx["context_id"], fx["team_a"], fx["team_b"], fx["best_of"])
        um_out.append({"fixture_id": fx["id"], "purpose": fx["purpose"], "result": result})
    p = OUT_DIR / "unknown_map_fixtures_v1.json"
    p.write_text(json.dumps(um_out, indent=2), encoding="utf-8")
    written["unknown_map"] = p

    km_out = []
    for fx in manifest["fixtures"]["known_map"]:
        result = ae.explain_map(fx["context_id"], fx["team_a"], fx["team_b"], fx["map_name"], fx["best_of"])
        km_out.append({"fixture_id": fx["id"], "purpose": fx["purpose"], "result": result})
    p = OUT_DIR / "known_map_fixtures_v1.json"
    p.write_text(json.dumps(km_out, indent=2), encoding="utf-8")
    written["known_map"] = p

    ks_out = []
    for fx in manifest["fixtures"]["known_series"]:
        result = ae.explain_series_known_maps(fx["context_id"], fx["team_a"], fx["team_b"], fx["best_of"],
                                               fx["ordered_maps"])
        ks_out.append({"fixture_id": fx["id"], "purpose": fx["purpose"], "result": result})
    p = OUT_DIR / "known_series_fixtures_v1.json"
    p.write_text(json.dumps(ks_out, indent=2), encoding="utf-8")
    written["known_series"] = p

    for name, path in written.items():
        print(f"wrote {path}")
    return written


if __name__ == "__main__":
    build()
