# Phase 10A — Next.js PWA Foundation + Prediction Experience

## A. Frontend architecture

- **Location**: `web/` — a fully isolated, additive frontend directory. No Python/model/data/state
  file changed (verified via `git status`: the only new tracked path is `web/`; the Python tree is
  byte-identical). `web/.gitignore` is nested; no nested git repository was created
  (`create-next-app --disable-git`, verified `web/.git` absent).
- **Toolchain** (resolver-established, per amendment #1 — versions recorded from `npm ls` after
  scaffold, typecheck/build run immediately): `next@16.3.1`, `react@19.2.8` / `react-dom@19.2.8`,
  **`typescript@5.9.3`** (create-next-app's compatible resolution — TS 7.0.2 being npm's latest
  dist-tag was deliberately NOT forced), `tailwindcss@4.3.3`, `eslint@9.39.5`,
  `lucide-react@1.31.0`. Dev/test: `vitest@4.1.10`, `@testing-library/react@16.3.2`,
  `@testing-library/user-event`, `@testing-library/jest-dom`, `jsdom@29.1.1`,
  `@vitejs/plugin-react`. **No Redux, no chart library, no UI framework, no auth, no PWA package.**
  All pinned in `web/package-lock.json`; `npm install` is lockfile-reproducible.
- **Routes** (App Router): `/` and `/predict` fully implemented; `/major`,
  `/major/cologne-2026`, `/major/cologne-2026/results` are polished "next release" shells via a
  shared `ComingNextPhase` component — descriptive text only, no fake brackets, no fake
  percentages, no inert controls (amendment #30). All 9 routes prerender fully static.
- **Component architecture**: `components/shell/` (Nav, FreshnessBadge, ComingNextPhase,
  SWRegister) and `components/predict/` (TeamSelector, BestOfSelector, MapModeSelector,
  OrderedMapSelector, ProbabilityBar, FactorCards, MapBreakdown, SupportNotice,
  PredictionDetails, PredictionResult, PredictClient) — no monolithic page file; `predict/page.tsx`
  is 12 lines.
- Next 16.3's bundled `node_modules/next/dist/docs/` was consulted for current conventions
  (global `LayoutProps`/`PageProps` helpers, `app/manifest.ts`) before writing code.

## B. Design system

- **Dark-first single theme** as CSS custom properties in `app/globals.css`, mapped into Tailwind
  v4 `@theme` tokens: background layers (`--canvas`/`--panel`/`--panel-2`/`--panel-glass`), lines
  (`--line`/`--line-strong`), text (`--ink`/`--ink-2`/`--ink-3`), semantic (`--team-a` cool cyan,
  `--team-b` warm amber, `--accent`, `--warn`, `--danger`, `--neutral-mid` for the 50/50
  midpoint), radii scale. No per-component colors anywhere.
- **Typography**: one family system — Geist Sans for text, **Geist Mono with tabular numerals for
  every probability/stat** (`.t-stat`). Fixed scale: `.t-display/.t-heading/.t-body/.t-label/
  .t-meta/.t-stat`. No italics.
- **Product name**: "CS2 Match Intel" is a working name, defined ONCE in `lib/app-config.ts`
  (`APP_NAME`/`APP_SHORT_NAME`/`APP_INITIALS`) and consumed by the header wordmark, document
  metadata, manifest, and offline page (amendment #3). No permanent visual logo was created.
- **Navigation**: compact top bar on desktop (wordmark + Predict/Major + freshness chip; only two
  destinations, so no left rail), fixed bottom navigation on mobile with
  `env(safe-area-inset-bottom)` reservation and a `.pb-nav` main-content clearance so the last
  content stays scrollable above it (amendment #28). No redundant page headings anywhere.

## C. API client

- `web/lib/api/`: `types.ts` (typed contracts), `client.ts` (single fetch wrapper),
  `endpoints.ts` (`getMeta`, `getHealthReady`, `getContexts`, `searchTeams`, `getMaps`,
  `predictSeries`, `predictMap`), `tournament.types.ts` (Phase 10B type definitions only — no
  Major UI).
- **Fixture-verified types** (amendment #21): before any component was written, deterministic
  REAL responses were captured from the running FastAPI service into `web/test/fixtures/` —
  meta, health-ready, teams-search, maps, pre-veto BO3, known-maps BO1/BO3/BO5, a genuine
  cold-start case (Team Germany vs MOUZ — one of only two identity-eligible zero-history teams
  in the deployment snapshot), a partial-support case (THUNDERdOWNUNDER vs MOUZ known-maps,
  `fallbacks_used` non-empty), and a structured `unknown_team` error. Every TS field exists in a
  captured payload; nothing was invented. All fixtures are **summary**-detail (amendment #22 —
  no 131-feature `full` payloads committed).
- **Error handling**: the wrapper parses `{error:{code,message,detail},request_id}` into a typed
  `ApiError` (code/status/requestId/detail) and network failures into `ApiUnreachableError`; raw
  backend JSON never reaches components.
- **Base URL** (amendment on §9): two documented modes in `.env.local`/`.env.example` — default
  is same-origin `/api/*` proxied by a `next.config.ts` rewrite to `API_PROXY_TARGET`
  (dev default `http://localhost:8000`), which decouples the frontend origin from the backend
  CORS allowlist; alternatively `NEXT_PUBLIC_API_BASE_URL` switches to direct cross-origin calls.
  The proxy mode was added when the port-3000 CORS assumption met reality: an unrelated user
  process occupies port 3000 locally, and altering the backend CORS config would have changed a
  Phase 9D-receipted artifact — the proxy solves it with zero backend changes.
- The client never sends `prediction_datetime` (backend snapshot cutoff governs) and always
  requests `include_explanation=true, explanation_detail="summary"` on the product path.

## D. App shell

Sticky translucent top bar (wordmark, desktop nav, freshness chip) + mobile bottom nav +
`<main>` container. Backend health (`/health/ready`) is probed once per predict-page mount by
`useHealth` and gates ONLY API-dependent actions: home, navigation, Major shells, and methodology
content all render with the backend down (amendment #18). Unavailability renders a polished
"Prediction service unavailable" panel with a Retry button; no raw status, no stack traces.

## E. Prediction workflow

Input card (not a form): two `TeamSelector` comboboxes arranged TEAM A · vs · TEAM B, segmented
`BestOfSelector` (BO1/BO3/BO5 → typed `1|3|5`), two-card `MapModeSelector` ("Maps unknown —
Before veto / map selection" / "Maps known — Use the exact ordered maps"; model names appear only
in the details drawer), and `OrderedMapSelector` in known mode. `Predict match` stays disabled
until the request contract is satisfiable (both teams, distinct, and in known mode exactly
1/3/5 unique maps).

- **TeamSelector**: full ARIA combobox (role, `aria-expanded`, `aria-controls`,
  `aria-activedescendant`, `aria-autocomplete`, listbox/option roles, ArrowUp/Down + Enter +
  Escape, outside-click dismissal) — behavior, not just attributes (amendment #27). Server search
  with 250 ms debounce, minimum query length 1, empty query performs no request (amendment #11),
  and **dual stale-response protection**: AbortController cancellation plus a monotonic
  request-sequence guard, with a dedicated test proving an older slow response can never
  overwrite a newer query's results (amendment #10). Selected state renders a team identity
  panel (canonical name + "N matches in snapshot" / "No match history in this snapshot") — text
  identity, no invented logos. The team chosen on the other side is rendered disabled
  ("already selected") and unclickable.
- **Result integrity** (amendment #12): each response is stored with the canonical
  `predictionRequestSignature` of the inputs that produced it; the result renders only while
  signatures match. Changing any prediction-defining input collapses the result to "Inputs
  changed — run Predict to update the result." — verified live in a screenshot and by two tests
  (including signature-restoring round-trip BO3→BO5→BO3).
- **URL state** (amendments #13/#14/#15): `?a=&b=&bo=&mode=&maps=` synced via shallow
  `history.replaceState` (never triggers prediction). Auto-run happens at most once per mount and
  only for a complete valid URL (test asserts exactly one `/predict/series` call). Canonical team
  names verbatim with standard URL encoding; restore resolves them through `searchTeams` requiring
  an exact canonical match — an unavailable team produces "X isn't available in this model
  snapshot", never a silent substitution. Known-maps order is preserved exactly
  (`maps=Mirage,Inferno,Nuke` = slots 1/2/3; never sorted).
- **Loading**: "Analyzing matchup…" inside a fixed-min-height result region (no layout shift), no
  fake progress, no rotating messages.

## F. Unknown-map experience

Result-first hero: team names in team colors, large Geist Mono percentages, one horizontal split
probability bar (labels, 50/50 midpoint tick, single entry transition), "Favored: X" or "Even
matchup" when `prediction_is_tied`, caption "Pre-veto prediction · BOn". A stale-snapshot
freshness warning from the backend would surface verbatim if ever present (the frontend itself
never requests a future datetime).

## G. Known-map experience

Composed series hero (identical shape, caption "Known-map series prediction"), then **Map-by-map**:
each ordered map as a row with its own mini probability bar, favored side, and — as visually
separate series-mechanics metadata under a divider (amendment #9) — "Chance map is played: N%"
(`probability_map_is_reached`, shown for every map, e.g. 48% for Map 3 in the live BO3 capture)
and "Series leverage: 0.48" (`series_composition_leverage`, plain-language tooltip; never called
SHAP/feature importance/causal). Per-map TreeSHAP factors are expandable per map, headed "Why the
model leans this way on {map}" — map attributions are NEVER merged into a single series factor
ranking (amendment #8), and the page states this explicitly.

## H. Explanation UX

"Why the model leans this way": two columns (Supporting {Team A} / Supporting {Team B}) of factor
cards rendered ONLY from `grouped_factors` the API returned — the live pre-veto capture shows
exactly the four RF concepts (Overall strength, Recent performance, Historical experience, Series
format & tier) with no XGB-only concepts, and tests assert their absence. Display copy is built
from STRUCTURED fields (`factor_group` + `direction` + canonical names → "Overall strength favors
Team Vitality"); the backend's deterministic `human_readable_summary` prose is **never
string-mutated** and is preserved verbatim in the details drawer under "Model summary (verbatim)"
(amendment #6). Each card has a thin relative-magnitude bar normalized to the max absolute group
contribution WITHIN that one explanation only, titled "Relative model contribution (within this
prediction only)", with no `%` label anywhere (amendment #7 — tested). The non-causal disclosure
appears exactly once per result. The "Prediction details" drawer lists mode, BO, context, model
id, composition method, explanation method (RF = "Saabas-style tree path decomposition" — never
SHAP; XGB = "TreeSHAP (native XGBoost)"), data-through date, the raw snapshot timestamp labeled
"(source timezone unspecified)", and "Causal claims: None". A dev-only full-explanation action
(logs to console, guarded by `NODE_ENV === "development"`) exists; the product path is always
summary (amendment #23).

## I. Freshness / support UX

- **One source of truth** (amendment #19): `useMeta` fetches `/meta` once per page load
  (module-level promise cache); header chip, home footer, and result footer all derive
  "Data through 28 Jun 2026" from `deployment_state_data_through` / the prediction's
  `metadata.state_data_through` — the date string is never hardcoded in a component, so a future
  state release updates the UI without code changes.
- **String-level date formatting** (amendment #20): `formatDataThrough` regex-extracts the date
  components directly from the API string — no `new Date()` / browser timezone conversion, so the
  20:00 cutoff can never shift calendar dates (tested with a 23:59:59 timestamp).
- **Data coverage** (`SupportNotice`): cold starts ("Team Germany has no match history in this
  data snapshot — the model used neutral default inputs for it") and known-map `fallbacks_used`
  (union across maps, phrased via a fixed vocabulary: "Limited map-specific history, recent-form
  history and recent history on the selected map for this matchup — the model used neutral
  defaults where data was missing") in an amber "DATA COVERAGE" aside, visually separate from
  model factors and never phrased as a team weakness (tested + captured live).

## J. Accessibility

Full combobox keyboard operation (tested: ArrowDown+Enter selection, Escape dismissal),
`role=radiogroup/radio` segmented controls, labeled selects for map slots, visible
`:focus-visible` outlines, `aria-live="polite"` result region, `role="alert"` error states,
`aria-current="page"` navigation, winner never indicated by color alone ("Favored: X" text +
percentages), and functional reduced-motion: the probability-bar transition and all
animations/transitions are disabled under `prefers-reduced-motion: reduce` in globals.css
(verified in styles; the animation itself was confirmed working in a mid-transition screenshot).

## K. Responsive design

Verified with real renders (screenshots in `reports/figures/phase10a/`):

- **1440-class desktop** (1536 CSS px): all eight §43 states inspected in detail.
- **820 px**: md-breakpoint layout confirmed (teams side-by-side with "vs" divider, horizontal
  control row) via a same-origin 820-px iframe viewport render (`18_responsive_820_md_layout.jpg`).
- **390 px**: mobile layout confirmed (stacked team panels, single-column map slots, bottom
  navigation, readable result hero) via a 390-px iframe viewport render plus a genuinely resized
  ~328-CSS-px OS window — and programmatically: `document.documentElement.scrollWidth` = 371 ≤
  386 viewport, i.e. **zero horizontal overflow**.
- Method note (amendment #25): no Playwright was added. The Claude-in-Chrome browser tooling was
  used; because the host display runs 250 % scaling and the window resisted small OS resizes,
  the 390/820 checks used same-origin iframes (media queries evaluate against the iframe
  viewport) plus one real narrow-window render — both are true layout-engine renders of the
  target widths, with contract-correct live API data (amendment #26), not fabricated numbers.

## L. PWA setup

`app/manifest.ts` (name/short_name/description from `APP_NAME` constants, standalone display,
theme/background = `--canvas`), placeholder icons (192/512 + `app/icon.png` favicon) — neutral
dark typographic "MI" tiles generated locally, explicitly placeholder, no invented branding
(amendment #24); default Next.js branding assets removed. `public/sw.js` (registered by
`SWRegister` in production builds only), conservative per amendments #4/#5: **network-only for
`/api/` and all cross-origin requests** (never intercepted — a prediction response is never
cached or replayed), **network-first for navigations** with a tiny pre-cached `/offline` fallback
page, cache-first only for content-addressed `/_next/static/` and the manifest/icons, versioned
cache name (`cmi-shell-v1`) with old-cache deletion on activation. Offline, the shell loads but
predictions honestly require connectivity.

## M. Testing

`npm test`: **61 passed, 2 skipped** (the skips are the opt-in live-API smoke suite) across 7
files: format (percent + timezone-safe dates), factors (structured display copy, verbatim
backend prose untouched, within-explanation normalization, RF-never-SHAP labels, fallback
phrasing), API client (typed GET/POST, structured-error parsing, unreachable wrapping, frozen
request contract incl. no `prediction_datetime` and order-preserving `ordered_maps`),
TeamSelector (search, keyboard combobox, Escape, **stale-response protection**, empty-query
no-request, same-team exclusion, selected/cold-start panels), selectors (BO typed values, mode
wording without jargon, 1/3/5 slot counts, `model_supported` filtering with a synthetic
Cache-unsupported entry proving Cache never appears, duplicate prevention, "supported maps"
wording), PredictionResult against REAL fixtures (pre-veto rendering, RF-only factor groups,
known-maps BO3 with reach % and leverage separation, expandable per-map factors, cold-start +
partial-support notices, freshness date, details drawer, no-%-on-factor-bars), and PredictClient
end-to-end with routed fetch mocks (full predict flow, result invalidation + signature restore,
known-maps map-count gating, structured error copy, unavailable+retry state, single URL auto-run,
URL team-unavailable message). Plus `test/integration/api.smoke.test.ts` — run live via
`RUN_API_SMOKE=1`: **2/2 passed** against the real FastAPI service; skipped by default so the
suite never depends on Python availability.

## N. Performance

- Production build: all 9 routes fully static (`○ (Static)`); compile ~2.5 s; **build verified to
  succeed with the FastAPI server stopped** (all data fetching is client-side at runtime).
- First-load JS is the create-next-app baseline plus this app's components; no chart/UI packages
  (the probability bar is pure CSS+DOM).
- Team search: one debounced request per settled query (250 ms), aborted/discarded when stale;
  empty queries make zero requests.
- Prediction request UX: warm pre-veto responses returned in tens of ms locally; the fixed-height
  result region gives zero layout shift between idle → loading → result (the entry animation was
  captured mid-transition, confirming a single smooth reveal).
- Layout shift: navigation is sticky/fixed; result region min-height reserved; no CLS-inducing
  image loads (no images beyond icons).

## O. Limitations

- "CS2 Match Intel" remains a working name; PWA icons are explicit placeholders.
- The loading state is hard to photograph against the local API (~tens of ms); it is covered by
  tests and the fixed-height design rather than a dedicated screenshot.
- The 390/820 responsive verification used iframe viewports + one real narrow window rather than
  exact OS window sizes (250 % display scaling constraint) — media-query-accurate, but not
  device-hardware testing.
- Offline scope is shell-only by design; predictions require the backend (no fake offline
  results).
- `/major` routes are intentionally shells; the tournament UI is Phase 10B.
- The live smoke suite requires a locally running FastAPI service and is opt-in by design.

## Backend regression (spec §46 / amendments #31/#32)

- `git status`: the ONLY new path is `web/` (plus this report + screenshots under the gitignored
  `reports/`); zero modifications to any tracked Python/model/data/state/config file.
- `scripts/validate_phase9d.py`: PASS (61/61) after Phase 10A.
- `scripts/validate_phase9e.py`: PASS (66/66) after Phase 10A.
- No backend contract problem was discovered; nothing in Python was patched from this phase. The
  one integration friction found (port 3000 occupied locally vs. the backend CORS allowlist) was
  solved entirely on the frontend via the same-origin rewrite proxy.

```
PWA FOUNDATION = IMPLEMENTED
/PREDICT = IMPLEMENTED
PRE-VETO UX = IMPLEMENTED
KNOWN-MAPS UX = IMPLEMENTED
MODEL-GROUNDED EXPLANATIONS = INTEGRATED
BACKEND ML LOGIC = UNCHANGED
DEPLOYMENT DATA THROUGH 2026-06-28
/MAJOR = NOT IMPLEMENTED YET
```
