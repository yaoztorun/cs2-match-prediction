"""
Phase 3 item 1: mechanically apply the already-approved Phase 2.5 proposed
decisions to produce a machine-readable team identity policy. No new manual
identity judgments are made here.

Reads:
    data/interim/team_aliases.csv              (792 rows, Phase 2)
    data/interim/unresolved_team_decisions.csv  (25 rows, Phase 2.5)

Writes:
    data/interim/team_identity_policy.csv
"""

import pandas as pd

from _common import INTERIM

ELIGIBLE_DECISIONS = {"", "KEEP_AS_SINGLE_TEAM"}


def main():
    aliases = pd.read_csv(INTERIM / "team_aliases.csv", dtype=str, keep_default_na=False)
    decisions = pd.read_csv(INTERIM / "unresolved_team_decisions.csv", dtype=str, keep_default_na=False)
    decision_map = dict(zip(decisions["team_name"], decisions["proposed_decision"]))

    rows = []
    for _, r in aliases.iterrows():
        name = r["original_team_name"]
        phase25_decision = decision_map.get(name, "")
        eligible = 1 if phase25_decision in ELIGIBLE_DECISIONS else 0
        notes = r["notes"]
        if phase25_decision:
            note_suffix = f"Phase 2.5 proposed decision: {phase25_decision} (see reports/unresolved_team_review.md)."
            notes = (notes + " " + note_suffix).strip() if notes else note_suffix
        rows.append({
            "team_name": name,
            "canonical_team_name": r["canonical_team_name"],
            "phase25_decision": phase25_decision,
            "identity_feature_eligible": eligible,
            "notes": notes,
        })

    policy = pd.DataFrame(rows)
    policy.to_csv(INTERIM / "team_identity_policy.csv", index=False, encoding="utf-8")

    n_eligible = int(policy["identity_feature_eligible"].sum())
    n_ineligible = len(policy) - n_eligible

    # canonical-level eligibility (conservative: eligible only if every original
    # name mapping to that canonical name is eligible) - printed here for visibility;
    # orchestration scripts derive this themselves from the same policy file.
    canon_elig = policy.groupby("canonical_team_name")["identity_feature_eligible"].min()
    n_canon_ineligible = int((canon_elig == 0).sum())

    print(f"team_identity_policy.csv: {len(policy)} rows | "
          f"identity_feature_eligible=1: {n_eligible}, =0: {n_ineligible}")
    print(f"distinct canonical names: {len(canon_elig)} | canonical-level ineligible: {n_canon_ineligible}")
    by_decision = policy[policy["phase25_decision"] != ""]["phase25_decision"].value_counts().to_dict()
    print(f"phase25_decision breakdown (non-blank only): {by_decision}")


if __name__ == "__main__":
    main()
