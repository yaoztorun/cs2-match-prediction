# Phase 8D.1 — Pre-Result Cutoff Timestamp Provenance Audit

Additive artifact only. Does not modify `reports/phase8d_cologne_pre_event_simulation.md`,
`config/tournaments/iem_cologne_major_2026_pre_event.yaml`, or any hash-checked Phase 8D/8B
artifact. No Cologne result data (winner, score, standing, qualifier, champion, map result,
player statistic) was read to produce this report — only `datetime` fields (scheduling
metadata) and pre-existing frozen Phase 8B/8D documentation.

Companion machine-readable artifact: `data/evaluation/cologne_2026_pre_event_cutoff_provenance_v1.json`

---

## 1. What `2026-06-02T13:30:00` actually is

It is **not** an independently chosen buffer time. `scripts/map_stream_common.cologne_cutoff()`
derives it as `min(series_datetime)` over every match tagged `evaluation_group == "cologne_2026"`
in `data/interim/evaluation_manifest.csv`, read from `data/interim/series_base.parquet`. Re-running
this function (read-only, datetime column only — the same "metadata boundary" read every
pre-Cologne state builder already performs, explicitly whitelisted in `validate_phase6d.py`)
reconfirms:

```
cologne_cutoff() -> Timestamp('2026-06-02 13:30:00'), tzinfo=None, 107 match_ids
```

The column dtype is `datetime64[us]` — **naive**, no UTC offset, no timezone attached anywhere
in the parquet, the manifest, or any downstream JSON/YAML artifact. This value is simply the raw
dataset's own recorded start time for the first `cologne_2026`-tagged match, carried through
unmodified.

## 2. Repository timezone basis: not documented

An exhaustive search (`grep` across every script and report for `timezone|UTC|CEST|CET|GMT|
tz_convert|tz_localize`) found **zero** timezone-handling code anywhere in the pipeline. Every
`datetime` column at every stage (`data_audit.md`, Phase 2 canonicalization, `feature_engine.py`,
`map_stream_common.py`, `phase8d_common.py`, the frozen state JSONs) is naive `datetime64[us]`/
ISO strings with no offset.

More importantly, the repository's own existing audit explicitly flags this as an **open,
unresolved question** — not something previously verified and merely undocumented:

> `scripts/audit_data.py` / `reports/data_audit.md` §14, Open Question 1: *"`datetime` semantics
> — assumed to be the scheduled/start time, but this is not verified against the data source. If
> it is instead a completion/scrape time, any chronological train/test split logic would need to
> change."*

So the repository does not establish a timezone basis, and does not even fully establish that
`datetime` is a start/schedule time rather than a scrape/completion time (though the latter is
the working assumption used everywhere, including here).

The **one** operational data point touching timezone at all is in
`reports/phase8b_cologne_tournament_definition.md` line 34: when querying the GitHub commits API
for the rulebook pin, the cutoff was passed as `until=2026-06-02T13:30:00Z` — i.e. treated as UTC
when a timezone-aware string was required by an external API. This was an ad hoc formatting
choice made to satisfy that API's input contract, not a verified or documented property of the
`kaggle_ektarr` source dataset. It is weak, not dispositive, evidence for a naive-UTC convention.

No citation, source URL, or collection-methodology note for `kaggle_ektarr` exists anywhere in
the repository (checked `SOURCE = "kaggle_ektarr"` occurrences and all `*.md`/`*.json` for
`kaggle.com`/citation text) — there is no external documentation to fall back on either.

## 3. Comparison against the public schedule instant

Per the task's given pre-event scheduling fact: first Cologne match at **12:30 CEST**
(UTC+2 in June) on 2026-06-02 → **2026-06-02T10:30:00Z**.

Two candidate repository conventions were tested against this fixed point:

| Hypothesis | repository_cutoff_utc | vs. first_match_utc (10:30:00Z) |
|---|---|---|
| **naive UTC** (repo value read literally as `Z`) | `2026-06-02T13:30:00Z` | **+10,800 s (3h00m LATER)** |
| **UTC+3 wall clock** (repo value is local time in a UTC+3 zone) | `2026-06-02T10:30:00Z` | **0 s (exact match)** |

UTC+3 is the unique offset that reconciles the two instants exactly (13:30 wall-clock vs. 12:30
CEST wall-clock is a flat 1-hour nominal difference; CEST is UTC+2, so the reconciling zone is
UTC+2+1 = UTC+3). Nothing in the repository contradicts UTC+3, but nothing in the repository
proves it either — see §2. Plausible real-world explanations exist (e.g. a scraper converting
HLTV epoch timestamps to naive local time on a machine set to a UTC+3 zone such as
Moscow/Istanbul/EEST), but these are speculation, not verified from repository or source
semantics.

**Verdict: provenance is genuinely unresolved from available repository evidence.** UTC+3 is the
best-supported hypothesis (unique exact reconciliation, zero contradicting evidence) against a
naive-UTC hypothesis supported only by one unrelated, ad hoc API-formatting choice. Neither is
proven. This is reported plainly rather than silently resolved in either direction.

## 4. State leakage — reconfirmed independently, unaffected by the ambiguity

Recomputed directly from `data/features/pre_cologne_team_state_v1_full.json` (not read from the
frozen report — recomputed fresh by scanning every embedded match `datetime`/`dt` field, 18,984
values):

```
max embedded history datetime = 2026-05-30T19:30:00
min embedded history datetime = 2023-01-10T09:30:00
```

This matches the frozen Phase 8A/8D claim (`2026-05-30T19:30:00`) exactly.

Both state max and the repository cutoff are drawn from the **same undocumented-but-internally-
consistent** `datetime` column, so their relative ordering (2 days 18 hours apart) holds
regardless of which absolute-UTC hypothesis is correct. Translating both endpoints under either
hypothesis still leaves a comfortable margin before the true first-match instant:

- naive-UTC hypothesis: state max `2026-05-30T19:30:00Z` → **63h before** true kickoff (`10:30Z` Jun 2)
- UTC+3 hypothesis: state max `2026-05-30T16:30:00Z` → **66h before** true kickoff

**No Cologne state update entered the simulation under either interpretation.**

## 5. Could probabilities differ due to time-derived features anyway?

One feature depends on the exact cutoff instant rather than just its date:
`feature_engine.py` computes `days_since_last_match = (as_of - hist[-1].dt).total_seconds() /
86400.0` (continuous, not day-rounded). A 3-hour cutoff ambiguity shifts this feature by a
constant `+0.125` days for every team's Cologne prediction, depending on which hypothesis is
correct. Given typical `days_since_last_match` magnitudes (multiple days), a 0.125-day constant
shift is a small perturbation; whether it moves any RF V2 leaf split cannot be asserted without
re-running inference, which this phase does not do (no model calls, per protocol). This is flagged
as a possible, bounded, unverified source of tiny probability drift — not a leakage issue and not
evidence either way on the timezone question itself.

## 6. Recommendation

Given:
- state leakage is proven absent under both hypotheses (§4),
- no repository or source evidence proves a *genuine* cutoff error (the "3h later" reading is one
  of two unproven hypotheses, not a confirmed fact — §3),
- the only feature sensitive to the ambiguity would shift by a small, bounded amount (§5),

this audit does **not** find grounds to declare a confirmed Case B (genuine pre-result error) and
therefore does **not** trigger V1→V2 regeneration. This is treated as an additive Case-A-style
provenance clarification, with the caveat — stated plainly rather than suppressed — that the
"same instant" conclusion rests on the UTC+3 hypothesis being correct, which is well-supported
but not proven from repository semantics alone. Phase 8D V1 remains authoritative and unmodified.

If the user has independent knowledge of the `kaggle_ektarr` source's documented timezone
convention (e.g. from the original Kaggle dataset page, which is outside this repository and was
not fetched), that would resolve §3 conclusively and should supersede this report's inference.

---

## Answers to the required checklist

```
CUTOFF RAW                      = 2026-06-02T13:30:00  (naive, datetime64[us], no tz anywhere in repo)
TIMEZONE BASIS                  = NOT DOCUMENTED IN REPOSITORY. Best-supported inferred basis:
                                   source-local wall clock, UTC+3 (unique exact reconciliation with
                                   public schedule; unproven — see report body).
                                   Weak internal counter-signal: Phase 8B treated it as naive-UTC once,
                                   ad hoc, for an external API call only.
CUTOFF UTC                      = 2026-06-02T10:30:00Z  (under the UTC+3 basis)
                                   2026-06-02T13:30:00Z  (under the naive-UTC alternate hypothesis)
FIRST MATCH LOCAL TIME          = 2026-06-02T12:30:00+02:00 (CEST, as given)
FIRST MATCH UTC                 = 2026-06-02T10:30:00Z
DIFFERENCE                      = 0 seconds under UTC+3 basis; +10,800 seconds (repo cutoff later)
                                   under the naive-UTC alternate hypothesis
EQUIVALENT INSTANT = TRUE/FALSE = INDETERMINATE FROM REPOSITORY EVIDENCE ALONE — best-supported
                                   answer is TRUE (via the UTC+3 basis), not proven to certainty.
STATE MAX TIMESTAMP             = 2026-05-30T19:30:00 (independently recomputed, matches frozen report)
STATE STRICTLY PRE-EVENT = TRUE/FALSE = TRUE (holds under both hypotheses, by 63-66 hours margin)
PHASE 8D REGENERATION REQUIRED = TRUE/FALSE = FALSE (no confirmed genuine error; provenance
                                   clarification only, per §6)
```
