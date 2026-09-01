"""
Persists the full computed-metrics detail (everything phase8e_metrics.py
produces that isn't already its own named artifact) as one hashable JSON
artifact, consumed by the receipt as the "metric tables / figure source
tables" hash target.
"""

import json

from _common import ROOT
import evaluation.cologne_2026.phase8e_metrics as pm

OUT_PATH = ROOT / "data" / "evaluation" / "cologne_2026_phase8e_metrics_detail_v1.json"


def build():
    m = pm.compute_all_metrics()
    detail = {k: v for k, v in m.items() if not k.endswith("_df") and k != "pred_df"}
    detail["swiss_terminal_record_detail"] = m["swiss_record_df"].to_dict(orient="records")
    detail["playoff_seed_detail"] = m["playoff_seed_df"].to_dict(orient="records")
    OUT_PATH.write_text(json.dumps(detail, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return detail


if __name__ == "__main__":
    build()
