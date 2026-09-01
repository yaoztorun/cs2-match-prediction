"""
Phase 8C synthetic deterministic demo. Runs one complete Major tournament
through the pure mechanics engine using a wholly synthetic 32-team fixture
(S1_*/S2_*/S3_*) and HigherSeedWinsProvider - fully deterministic, no ML,
no real Cologne teams. The real frozen YAML is loaded only for RULES
(never `participants`). Writes a readable trace to
data/tournaments/phase8c_synthetic_trace.json, deterministically (sorted
keys, no timestamps/UUIDs), so re-running this script reproduces a
byte-identical file.
"""

import json

from _common import ROOT
import tournament.engine.tournament_engine as te

TRACE_PATH = ROOT / "data" / "tournaments" / "phase8c_synthetic_trace.json"


def main():
    rules = te.load_frozen_rules()
    stage1, stage2_direct, stage3_direct = te.build_synthetic_fixture()
    provider = te.HigherSeedWinsProvider()

    result = te.run_major_tournament(stage1, stage2_direct, stage3_direct, rules, provider)
    result_dict = result.to_dict()
    canonical_hash = te.trace_hash(result_dict)

    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"engine": "tournament_engine.py (Phase 8C)", "fixture": "synthetic_32_team",
               "outcome_provider": "HigherSeedWinsProvider", "canonical_trace_sha256": canonical_hash,
               "result": result_dict}
    TRACE_PATH.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"champion: {result.champion}")
    print(f"stage1 matches: {len(result.stage1.trace)}  advancers: "
          f"{[t.team_id for t in result.stage1.advancers]}")
    print(f"stage2 matches: {len(result.stage2.trace)}  advancers: "
          f"{[t.team_id for t in result.stage2.advancers]}")
    print(f"stage3 matches: {len(result.stage3.trace)}  advancers (playoff seeds 1-8): "
          f"{[t.team_id for t in result.stage3.advancers]}")
    print(f"playoff matches: {len(result.playoffs.trace)}  runner-up: {result.playoffs.runner_up}")
    print(f"canonical trace SHA-256: {canonical_hash}")
    print(f"trace written to: {TRACE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
