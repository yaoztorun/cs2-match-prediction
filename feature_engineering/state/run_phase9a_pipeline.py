"""
Phase 9A transactional orchestrator (amendment #10/#11/#12): preflight ->
build the manifest -> build all five deployment states TWICE, independently,
into two separate staging directories -> require byte-identical hashes
between them (a real determinism proof, not "overwrite and re-hash") ->
promote staging A to the final locations -> validate -> receipt LAST.

Every builder is called with a FRESH empty StateStore each time (amendment
#12) - none of them load pre_cologne_*.json or patch an existing full
state; state_type is rebuilt from scratch from the deployment-history
manifest on every invocation, in both staging runs.
"""

import hashlib
import json
import shutil

import pandas as pd

from _common import ROOT, raw_file_hashes
import feature_engineering.state.phase9a_common as p9a
import feature_engineering.state.build_deployment_history_manifest as bhm
import feature_engineering.state.build_deployment_series_state as bss
import feature_engineering.state.build_deployment_map_state as bms
import feature_engineering.state.build_deployment_form_state as bfs
import feature_engineering.state.build_deployment_roster_state as brs
import feature_engineering.state.build_deployment_modern_map_state as bmms

RECEIPT_PATH = p9a.DEPLOY / "deployment_state_receipt_v1.json"
STAGING_A = p9a.DEPLOY / ".phase9a_staging_a"
STAGING_B = p9a.DEPLOY / ".phase9a_staging_b"
AUDIT_PATH = p9a.DEPLOY / "deployment_state_consumption_audit_v1.csv"

BUILDERS = [
    ("series", bss.build, p9a.NEW_DEPLOYMENT_ARTIFACTS["series_state"]),
    ("map", bms.build, p9a.NEW_DEPLOYMENT_ARTIFACTS["map_state"]),
    ("form", bfs.build, p9a.NEW_DEPLOYMENT_ARTIFACTS["form_state"]),
    ("roster", brs.build, p9a.NEW_DEPLOYMENT_ARTIFACTS["roster_state"]),
    ("modern_map", bmms.build, p9a.NEW_DEPLOYMENT_ARTIFACTS["modern_map_state"]),
]


def preflight():
    if RECEIPT_PATH.exists():
        existing = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        if existing.get("committed") is True:
            raise RuntimeError(f"STOP: a valid Phase 9A receipt already exists at {RECEIPT_PATH} "
                                f"(committed=True) - ABORTING rather than overwriting.")
    partial = [name for name, path in p9a.NEW_DEPLOYMENT_ARTIFACTS.items() if path.exists()]
    if partial and not RECEIPT_PATH.exists():
        print(f"RECOVERY DIAGNOSTIC: partial canonical Phase 9A outputs exist without a valid receipt: "
              f"{partial}. This run will rebuild everything from scratch (fresh StateStores, staged, "
              f"validated, then promoted) and overwrite them - this is a clean rebuild, not a resume.")


def build_into(staging_dir):
    staging_dir.mkdir(parents=True, exist_ok=True)
    audits = []
    for name, fn, final_path in BUILDERS:
        out_path = staging_dir / final_path.name
        _store, audit_df, _meta = fn(output_path=out_path)
        audits.append(audit_df)
    return pd.concat(audits, ignore_index=True)


def _strip_nondeterministic_fields(obj):
    """The only legitimately non-semantic field any Phase 9A state/meta JSON contains is a
    wall-clock `generated_at` timestamp - stripped before hashing so the determinism check
    verifies actual STATE CONTENT, not build wall-clock time (amendment #11: canonicalize
    explicitly rather than weaken the test)."""
    if isinstance(obj, dict):
        return {k: _strip_nondeterministic_fields(v) for k, v in obj.items() if k != "generated_at"}
    if isinstance(obj, list):
        return [_strip_nondeterministic_fields(v) for v in obj]
    return obj


def canonical_hash(path):
    if path.suffix == ".json":
        obj = _strip_nondeterministic_fields(json.loads(path.read_text(encoding="utf-8")))
        canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_dir(d):
    return {p.name: canonical_hash(p) for p in sorted(d.glob("*")) if p.is_file()}


def main():
    preflight()
    hashes_before_raw = raw_file_hashes()
    historical_replay_before = p9a.hash_historical_replay_record()

    bhm.build()  # deterministic/idempotent, written directly at its final location

    print("\n=== building staging A ===")
    audit_a = build_into(STAGING_A)
    print("\n=== building staging B (independent second build, for the determinism proof) ===")
    audit_b = build_into(STAGING_B)

    hashes_a, hashes_b = hash_dir(STAGING_A), hash_dir(STAGING_B)
    if hashes_a != hashes_b:
        mismatches = {k: (hashes_a.get(k), hashes_b.get(k)) for k in set(hashes_a) | set(hashes_b)
                      if hashes_a.get(k) != hashes_b.get(k)}
        raise RuntimeError(f"STOP: non-deterministic rebuild - staging A and B differ: {mismatches}")
    print(f"\ndeterminism check PASSED: {len(hashes_a)} artifacts byte-identical across two independent builds")

    audit_a_sorted = audit_a.sort_values(["state_type", "match_id"]).reset_index(drop=True)
    audit_b_sorted = audit_b.sort_values(["state_type", "match_id"]).reset_index(drop=True)
    if not audit_a_sorted.equals(audit_b_sorted):
        raise RuntimeError("STOP: consumption audit differs between the two independent builds")

    # ---- promote staging A to final locations ----
    for name, fn, final_path in BUILDERS:
        for suffix in (".json", ".parquet"):
            src = (STAGING_A / final_path.name).with_suffix(suffix)
            if src.exists():
                dst = final_path.with_suffix(suffix)
                shutil.copy2(src, dst)
                print(f"promoted {src.name} -> {dst}")
    shutil.rmtree(STAGING_A)
    shutil.rmtree(STAGING_B)

    audit_a_sorted.to_csv(AUDIT_PATH, index=False)
    print(f"wrote {AUDIT_PATH} ({len(audit_a_sorted)} rows)")

    assert raw_file_hashes() == hashes_before_raw, "data/raw/ was modified during the Phase 9A pipeline run"

    print("\n=== running validation/validate_phase9a.py before committing the receipt ===")
    import subprocess
    import sys
    import os
    result = subprocess.run([sys.executable, "-m", "validation.validate_phase9a"],
                             cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                             env={"PYTHONIOENCODING": "utf-8", **os.environ})
    print(result.stdout[-3000:])
    if result.returncode != 0:
        raise RuntimeError(f"STOP: validation/validate_phase9a.py failed - refusing to write the receipt. "
                            f"stderr: {result.stderr[-2000:]}")

    import feature_engineering.state.build_deployment_state_receipt as receipt_mod
    receipt_mod.build(historical_replay_before_hashes=historical_replay_before)
    print("\nPhase 9A pipeline complete.")


if __name__ == "__main__":
    main()
