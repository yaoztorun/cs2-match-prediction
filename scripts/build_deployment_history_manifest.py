"""
Phase 9A: the authoritative deployment-history manifest. Answers exactly one
question per raw series row: "is this legitimate historical information for
a FUTURE deployment state?" It says nothing about whether any particular
state engine can actually consume a legitimate row (see
build_deployment_state_consumption_audit.py for that - amendment #3/#4).

Cologne inclusion is a POSITIVE 106-ID WHITELIST sourced from the frozen
Phase 8E canonical actual-results artifact (never "cologne_2026 minus
showmatch", never re-derived from tournament name or BO arithmetic).
Every cologne_2026-tagged row must resolve to exactly one of
{official Cologne match, frozen showmatch exclusion} - anything else is a
hard STOP, since it would mean the Phase 8E authority and this manifest have
silently drifted apart.

Writes: data/deployment/deployment_history_manifest_v1.parquet
"""

import pandas as pd

from _common import INTERIM, ROOT

EVAL = ROOT / "data" / "evaluation"
DEPLOY_DIR = ROOT / "data" / "deployment"
CANONICAL_COLOGNE_PATH = EVAL / "cologne_2026_actual_series_results_v1.parquet"
RECONCILIATION_PATH = EVAL / "cologne_2026_result_reconciliation_v1.csv"


def load_official_cologne_match_ids():
    canonical = pd.read_parquet(CANONICAL_COLOGNE_PATH, engine="fastparquet")
    ids = set(canonical["source_match_id"].astype(int))
    if len(ids) != 106:
        raise ValueError(f"STOP: expected 106 official Cologne match_ids from the frozen Phase 8E canonical "
                          f"actual-results artifact, got {len(ids)}")
    return ids


def load_showmatch_match_ids():
    recon = pd.read_csv(RECONCILIATION_PATH)
    excluded = recon[~recon["included_in_official_event"]]
    if len(excluded) != 1:
        raise ValueError(f"STOP: expected exactly 1 excluded row in the frozen Phase 8E reconciliation artifact, "
                          f"got {len(excluded)}")
    return set(excluded["source_match_id"].astype(int))


def build():
    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    sb = pd.read_parquet(INTERIM / "series_base.parquet", engine="fastparquet",
                          columns=["match_id", "datetime", "tournament", "tier"])
    rej = pd.read_csv(INTERIM / "rejected_series_rows.csv")[["match_id", "datetime", "tournament", "tier",
                                                               "reject_reason"]]

    if len(em) != 9923:
        raise ValueError(f"STOP: expected 9,923 evaluation_manifest rows, got {len(em)}")

    official_cologne_ids = load_official_cologne_match_ids()
    showmatch_ids = load_showmatch_match_ids()
    if not official_cologne_ids.isdisjoint(showmatch_ids):
        raise ValueError("STOP: official Cologne match_ids and the frozen showmatch exclusion overlap")

    retained_ids = set(sb["match_id"])
    rejected_ids = set(rej["match_id"])
    if not retained_ids.isdisjoint(rejected_ids):
        raise ValueError("STOP: a match_id is both retained (series_base) and rejected (rejected_series_rows)")

    cologne_group_ids = set(em.loc[em["evaluation_group"] == "cologne_2026", "match_id"])
    if not rejected_ids.isdisjoint(cologne_group_ids):
        raise ValueError("STOP: a cologne_2026-tagged match_id is also a Phase-2 reject - upstream assumption violated")
    post_cologne_group_ids = set(em.loc[em["evaluation_group"] == "post_cologne", "match_id"])
    if not rejected_ids.isdisjoint(post_cologne_group_ids):
        raise ValueError("STOP: a post_cologne-tagged match_id is also a Phase-2 reject - upstream assumption violated")

    sb_meta = sb.set_index("match_id")
    rej_meta = rej.set_index("match_id")

    rows = []
    n_unexplained_cologne = 0
    for match_id, group in zip(em["match_id"], em["evaluation_group"]):
        if match_id in rejected_ids:
            r = rej_meta.loc[match_id]
            rows.append({"match_id": int(match_id), "datetime": r["datetime"], "tournament": r["tournament"],
                         "tier": r["tier"], "history_status": "excluded_existing_reject",
                         "history_reason": f"phase2_reject: {r['reject_reason']}"})
            continue

        r = sb_meta.loc[match_id]
        base = {"match_id": int(match_id), "datetime": str(r["datetime"]), "tournament": r["tournament"],
                "tier": r["tier"]}

        if group == "cologne_2026":
            if match_id in official_cologne_ids:
                rows.append({**base, "history_status": "included",
                             "history_reason": "official Cologne series (Phase 8E canonical actual-results whitelist)"})
            elif match_id in showmatch_ids:
                rows.append({**base, "history_status": "excluded_showmatch",
                             "history_reason": "non_tournament_showmatch (frozen Phase 8E reconciliation artifact)"})
            else:
                n_unexplained_cologne += 1
                rows.append({**base, "history_status": "ERROR_unexplained_cologne_row",
                             "history_reason": "cologne_2026-tagged match_id is neither in the Phase 8E official "
                                                "106-match whitelist nor the frozen showmatch exclusion"})
        else:
            rows.append({**base, "history_status": "included", "history_reason": f"legitimate {group} series"})

    if n_unexplained_cologne:
        raise ValueError(f"STOP: {n_unexplained_cologne} cologne_2026-tagged match_id(s) could not be explained by "
                          "the Phase 8E official whitelist or the frozen showmatch exclusion - the manifest and "
                          "the Phase 8E authority have drifted apart.")

    manifest = pd.DataFrame(rows)
    if len(manifest) != 9923:
        raise ValueError(f"STOP: manifest has {len(manifest)} rows, expected 9,923")

    counts = manifest["history_status"].value_counts()
    n_included, n_showmatch, n_reject = counts.get("included", 0), counts.get("excluded_showmatch", 0), \
        counts.get("excluded_existing_reject", 0)
    print(f"history_status counts: included={n_included}, excluded_showmatch={n_showmatch}, "
          f"excluded_existing_reject={n_reject}")
    if n_included != 9800:
        raise ValueError(f"STOP: expected 9,800 included rows, got {n_included}")
    if n_showmatch != 1:
        raise ValueError(f"STOP: expected 1 excluded_showmatch row, got {n_showmatch}")
    if n_reject != 122:
        raise ValueError(f"STOP: expected 122 excluded_existing_reject rows, got {n_reject}")

    included_cologne = manifest[(manifest["match_id"].isin(cologne_group_ids)) & (manifest["history_status"] == "included")]
    if len(included_cologne) != 106:
        raise ValueError(f"STOP: expected 106 included official Cologne rows, got {len(included_cologne)}")

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEPLOY_DIR / "deployment_history_manifest_v1.parquet"
    manifest = manifest.sort_values(["datetime", "match_id"]).reset_index(drop=True)
    manifest.to_parquet(out_path, engine="fastparquet", index=False)
    print(f"wrote {out_path} ({len(manifest)} rows)")

    included = manifest[manifest["history_status"] == "included"]
    cutoff = included["datetime"].max()
    latest_row = included.loc[included["datetime"].idxmax()]
    print(f"deployment_cutoff = {cutoff}")
    print(f"latest included match_id = {latest_row['match_id']}, tournament = {latest_row['tournament']}")

    post_cologne = manifest[manifest["match_id"].isin(post_cologne_group_ids)]
    print(f"post_cologne raw rows = {len(post_cologne)}, included = "
          f"{(post_cologne['history_status'] == 'included').sum()}")
    print("post_cologne by tournament:")
    print(post_cologne.groupby("tournament").size())
    print("post_cologne by tier:")
    print(post_cologne.groupby("tier").size())

    return manifest


if __name__ == "__main__":
    build()
