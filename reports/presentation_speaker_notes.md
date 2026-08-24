# Speaker Notes — CS2 Match Prediction (15-minute presentation)

Auto-extracted from `reports/cs2_match_prediction_presentation.pptx` by
`reports/presentation/export_speaker_notes.py` — the same notes are embedded in the
PPTX notes pane. Target pace: ~50-60 seconds per slide => ~13.5 minutes total,
leaving safety margin inside the 15-minute limit.

## Slide 1 — Title — Predicting CS2 Matches with Probabilities

**MAIN MESSAGE**

This project builds and honestly evaluates a probabilistic prediction system for professional Counter-Strike 2 matches.

**SCRIPT (speak naturally)**

Good morning. In this presentation I'll show a machine-learning system that predicts professional Counter-Strike 2 matches. The key word on this slide is 'probabilities' — the system never just says 'Team A will win'; it estimates how likely each team is to win. I'll walk through the data, the methodology, three model families that were compared, the two final frozen models, and then the part I'm most proud of: before the IEM Cologne Major 2026 started, the system was frozen and used to simulate the whole tournament 50,000 times — and afterwards we compared those frozen predictions against what actually happened. Everything ends in a deployed web application.

**WHAT I MUST UNDERSTAND**

- CS2 = Counter-Strike 2, a 5-vs-5 esport; matches are best-of-1/3/5 series of 'maps'.
- The whole talk has one arc: data → models → frozen evaluation → real tournament → app.

**LIKELY PROFESSOR QUESTIONS**

- *Q: Why esports?*
  - A: Rich public match history, frequent events, and a genuinely hard, high-variance prediction problem — ideal for studying probabilistic ML honestly.
- *Q: Is this betting-related?*
  - A: No — it is an academic exercise in probabilistic prediction and honest evaluation.

**TRANSITION**

Let's define the actual prediction problem precisely.

## Slide 2 — Objective — Estimate P(Team A wins) before the match starts

**MAIN MESSAGE**

The task is probability estimation from strictly pre-match information — not just winner classification.

**SCRIPT (speak naturally)**

The problem statement is deliberately strict: given only information available before a match starts — who is playing, the series format, and both teams' history — estimate the probability that Team A wins. Why insist on a probability instead of a predicted winner? Three reasons. First, a label hides confidence: 'A wins' at 51 percent and at 95 percent are completely different statements. Second, probabilities can be properly scored — Log Loss and Brier score punish a model that is confidently wrong, which accuracy can't detect. Third, the downstream goal — simulating a whole tournament — only works with probabilities, because you need to sample upsets with realistic frequency. In an esport where upsets are common, saying '62 percent' honestly is more useful than pretending certainty.

**WHAT I MUST UNDERSTAND**

- 'Pre-match only' is the core constraint that drives all leakage prevention later.
- Log Loss = −log of the probability assigned to what actually happened, averaged; Brier = mean squared error of the probability.
- Both are proper scoring rules: the best long-run strategy is reporting your true belief.

**LIKELY PROFESSOR QUESTIONS**

- *Q: Why not just maximize accuracy?*
  - A: Accuracy only checks which side of 0.5 you were on; it treats 51% and 99% identically. For simulation and honest uncertainty we need probability quality, measured by Log Loss/Brier.
- *Q: What makes this hard?*
  - A: High variance (upsets), rosters change, the meta shifts over time, and public data has identity/label quality problems we had to fix first.

**TRANSITION**

Here is the system that answers this question — it actually has two prediction modes.

## Slide 3 — System overview — One question, two prediction modes

**MAIN MESSAGE**

There are two frozen models because there are two genuinely different prediction tasks — pre-veto series prediction and known-map prediction.

**SCRIPT (speak naturally)**

The system answers the same question in two situations. Mode A: before the map veto, all we know is who plays and the format — a tuned Random Forest, using 17 pre-match team-history features, directly outputs the series win probability. This is also the engine of the tournament simulator, because for future tournament matches maps are never known in advance. Mode B: once the map veto has happened, the ordered maps are legitimate input. There, a tuned XGBoost model predicts each individual map with 131 richer features, and a small dynamic program combines the per-map probabilities into a series probability — it accounts for the fact that map 3 is only played if the series is still alive. Three model families were compared during development; these two are the frozen winners for their respective tasks.

**WHAT I MUST UNDERSTAND**

- Veto = the pick/ban phase where teams alternately eliminate and pick maps before a series.
- RF V2 and XGB V3 solve different tasks (series vs single map) — their accuracies are not comparable numbers.
- The DP is exact composition, not simulation.

**LIKELY PROFESSOR QUESTIONS**

- *Q: Why not one model for both?*
  - A: The information sets differ: knowing the exact maps adds real signal (map-specific strength, per-map rosters). A single model would either waste that input or need fake placeholder maps.
- *Q: Why RF for one task and XGB for the other?*
  - A: Each was selected by frozen, pre-registered criteria on its own task's validation — probability quality first. RF V2 won the series task, XGB V3 the map task.

**TRANSITION**

Both models learn from the same underlying dataset — let's look at it.

## Slide 4 — Data — 3.5 years of professional CS2 matches

**MAIN MESSAGE**

A large public dataset — but it needed a serious audit before it could be trusted; the winner label itself was broken.

**SCRIPT (speak naturally)**

The data covers about nine and a half thousand professional series from January 2023 to late June 2026 — roughly ten and a half thousand played maps, across three tiers of tournaments. Two audit findings shaped everything. First, the dataset's own 'team1 wins' label disagreed with the actual match scores in half the rows — so the target was reconstructed directly from the scores. Second, team IDs turned out to be one-per-match, not persistent identities, so team identity was rebuilt from normalized names with a manual review of ambiguous cases. After conservative cleaning we keep about 9,450 series and 10,300 maps for modelling — and, crucially, the 107 Cologne Major matches were fenced off from day one, reserved as a purely external evaluation event.

**WHAT I MUST UNDERSTAND**

- Target reconstruction: winner = sign(score1 − score2); 5 ties and 1 missing score dropped.
- Every exclusion is logged with a reason — auditable, reproducible cleaning.
- 106 of the 107 Cologne rows are official matches; 1 is a showmatch (explained on the Cologne slide).

**LIKELY PROFESSOR QUESTIONS**

- *Q: Where does the data come from?*
  - A: A public Kaggle export of professional CS2 matches (HLTV-style coverage), audited from scratch with reproducible pandas scripts.
- *Q: How did you find the broken label?*
  - A: A raw-data audit compared the provided team1_win column against the actual scores — 49.9% disagreement, and internally inconsistent between series and map rows. Scores were self-consistent, so they define the target.
- *Q: Is ~10k matches enough?*
  - A: Enough to learn a real signal above baselines, not enough for per-map or BO5 subtleties — that's in the limitations.

**TRANSITION**

With clean data in hand, the pipeline itself is designed around one enemy: information leakage.

## Slide 5 — Methodology — A chronological pipeline built to prevent leakage

**MAIN MESSAGE**

The pipeline replays history in order, so every prediction uses only what was knowable at that moment — and evaluation gets strictly harder from left to right.

**SCRIPT (speak naturally)**

The methodology is one long chronological pipeline. After the audit and cleaning, we replay all matches in time order, maintaining each team's evolving state — ratings, form, map history. When a match arrives, its features are computed from that state first, and only afterwards is its result applied. That single rule makes leakage structurally impossible rather than something we hope we avoided. The split is chronological too: train on 2023 to August 2025, validate on the following five months, and a sealed test partition covering spring 2026. Hyperparameters were tuned with expanding-window cross-validation inside the training period only. And on the far right, the hardest evaluation: the Cologne Major, completely outside development, evaluated only after the models were frozen.

**WHAT I MUST UNDERSTAND**

- Expanding-window CV = each fold trains on an earlier period and validates on the next one — CV that respects time.
- Timestamp batching: matches with identical timestamps get features first, results applied after — none sees another's outcome.
- Same build_features() function serves training rows and live predictions — train/inference code cannot diverge.

**LIKELY PROFESSOR QUESTIONS**

- *Q: Why a chronological split instead of random?*
  - A: Random splits leak the future: a model would train on matches played after its validation matches, inflating scores. Deployment always predicts forward in time, so evaluation must too.
- *Q: What is leakage exactly?*
  - A: Any information in a training feature that would not have been available at prediction time — e.g. using a team's season-end rating for a January match.
- *Q: How do you know there is no leakage?*
  - A: By construction (strict-past feature computation, tested timestamp handling, sealed test, hashed frozen artifacts) — plus the external Cologne evaluation, which is immune to internal leakage by design.

**TRANSITION**

Inside that pipeline, what do the features actually look like?

## Slide 6 — Feature engineering — Team history, summarized into interpretable families

**MAIN MESSAGE**

Features are interpretable summaries of each team's past, expressed as A-minus-B differences — with ELO difference the strongest single signal.

**SCRIPT (speak naturally)**

Every feature summarizes the past. Six families: overall strength — a classic ELO rating updated after every match; recent form over the last five and ten series; opponent strength, so beating strong opponents counts more than farming weak ones; the map families — how deep a team's map pool is and how good they are on the specific selected map; player and roster features — how the current five players have been performing; and activity — is the team rusty or match-sharp. Two design rules matter. Everything enters as a Team-A-minus-Team-B difference, because the raw data had a spurious ordering bias worth about five percentage points that a model would happily learn instead of real skill. And cold starts are explicit: an unknown team gets neutral defaults plus a flag saying 'no history', so the model can treat genuine uncertainty as such. The pre-veto model uses 17 of these; the known-map model 131.

**WHAT I MUST UNDERSTAND**

- ELO: expected score E = 1/(1+10^((R_B−R_A)/400)); update R ← R + 32·(result − E). Simple, interpretable, deliberately untuned.
- Mirrored augmentation: each training row also appears with teams swapped and the label flipped — the model can't learn 'team1 wins more'.
- elo_diff top feature in both models (RF permutation importance 0.071; XGB #1 as well).

**LIKELY PROFESSOR QUESTIONS**

- *Q: Why ELO and not something newer (Glicko/TrueSkill)?*
  - A: Transparent, one-line update, strong baseline; refinements (time decay, per-tier K) are listed as future work — the comparison to a pure-ELO baseline shows how much the ML adds on top.
- *Q: 131 features — overfitting risk?*
  - A: Controlled by chronological CV, strong regularization (final XGB uses depth-2 trees), and verified by the small train→test gap on the sealed test.
- *Q: How are roster changes handled?*
  - A: Player-level features follow the current five players' own histories, and roster-continuity features measure how much of the map experience belongs to the current core.

**TRANSITION**

Three model families competed on these features.

## Slide 7 — Models — Three families, tuned identically, compared honestly

**MAIN MESSAGE**

Three model families with different inductive biases, given exactly the same information and tuning budget — so differences reflect the algorithms, not the setup.

**SCRIPT (speak naturally)**

The three candidates. Logistic regression — a linear model, implemented from scratch — is the transparency baseline; if trees can't beat it, the extra complexity isn't earning its keep. Random forest builds hundreds of decision trees on random subsets and averages them; it captures non-linear structure, but untuned it simply memorized the training set — a train-validation AUC gap of plus 0.37 — and tuning its depth and leaf sizes brought that down to 0.06. XGBoost builds trees sequentially, each correcting the last, with heavy regularization — it had the most stable train-to-validation behaviour of the three. The comparison was kept deliberately fair: same features, same chronological folds for tuning, same augmentation, and one single scoring pass on validation per tuned model.

**WHAT I MUST UNDERSTAND**

- RF = bagging (parallel trees, variance reduction); XGB = boosting (sequential trees, bias reduction) — know this contrast.
- All three landed within ~0.015 AUC of each other → the features, not the algorithm, are the binding constraint.
- LR 'from scratch' = gradient-descent implementation, not sklearn.

**LIKELY PROFESSOR QUESTIONS**

- *Q: What is a Random Forest?*
  - A: An ensemble of decision trees, each trained on a bootstrap sample with random feature subsets at each split; predictions are averaged. Averaging many overfit trees gives a low-variance ensemble.
- *Q: What is XGBoost?*
  - A: Gradient boosting: trees added one at a time, each fit to the current errors (gradient of log loss), with shrinkage and regularization. Typically the strongest family on tabular data.
- *Q: Why no neural networks?*
  - A: ~10k rows of tabular data is where trees dominate; an MLP would add tuning cost and opacity with little expected gain. Listed as future work with more data.

**TRANSITION**

So who won? Here are the validation results — and the selection logic.

## Slide 8 — Results (pre-veto) — All three are close; probabilities decide the winner

**MAIN MESSAGE**

RF V2 was selected by a pre-registered hierarchy that puts probability quality first — even though LR had marginally higher accuracy.

**SCRIPT (speak naturally)**

Here are the tuned models on the held-out validation period — 1,419 series the models never saw. Left panel: accuracy and AUC. Notice logistic regression actually has the best accuracy, by half a point. Right panel: probability quality — log loss and Brier, lower is better — and there random forest wins both, and also wins AUC. The selection rule was fixed before looking: probability quality first, then AUC, then accuracy — because everything downstream, especially the Monte-Carlo simulator, consumes probabilities, not thresholded labels. Under that hierarchy Random Forest V2 wins decisively. Two honest footnotes: all three models sit within one and a half AUC points of each other, which tells us the features, not the algorithm, are the constraint; and XGBoost had a much more stable train-to-validation transition, so RF's win on a single validation period carries some risk — that caveat is documented in the project, not hidden.

**WHAT I MUST UNDERSTAND**

- Numbers: LR/RF/XGB — Acc 0.613/0.607/0.612 · AUC 0.641/0.657/0.650 · LL 0.658/0.651/0.654 · Brier 0.233/0.230/0.231.
- Majority baseline 0.553 = always predicting the more frequent class.
- Selection hierarchy was written down before the comparison — no post-hoc metric shopping.

**LIKELY PROFESSOR QUESTIONS**

- *Q: Why is AUC the right discrimination metric?*
  - A: AUC is threshold-free: the probability a random winner is ranked above a random loser. With a 55/45 class skew, accuracy alone is easy to game; AUC is not.
- *Q: Isn't 61% accuracy low?*
  - A: Upsets are structural in CS2 — even bookmakers sit in the low-to-mid 60s. The value is in calibrated probabilities above baselines, not in certainty.
- *Q: Why not ensemble the three?*
  - A: Considered and deliberately not pursued after freezing — the evaluation protocol forbids post-hoc model construction; listed as future work.

**TRANSITION**

That was maps-unknown. When the maps are known, we can do better — with a different model.

## Slide 9 — Results (known-map) — Predict each map, then compose the series exactly

**MAIN MESSAGE**

The known-map system predicts each map independently and composes the series probability exactly with dynamic programming — and it was validated once, on a sealed test.

**SCRIPT (speak naturally)**

In known-maps mode, the XGBoost model scores each map separately — here Mirage 57, Inferno 61, Nuke 59 for Team A. A small dynamic program then walks through the series states — how many maps each side has won — and sums the probability of every path where Team A reaches two wins first. Map three only counts in branches where the series is one-one, so 'later maps only play if needed' is handled exactly, for any best-of-N. The map model was evaluated exactly once, on a sealed test of 1,427 maps closed until the model was frozen: accuracy 61.3 percent, AUC 0.649 with a bootstrap interval clearly above chance. The right chart gives context: it clearly beats a map-specific ELO baseline and a coin flip, and edges the overall-ELO baseline on probability quality; the confusion matrix shows reasonably balanced errors.

**WHAT I MUST UNDERSTAND**

- DP: dp[maps][wins] accumulates path probabilities; terminal when a side reaches ceil(N/2) wins.
- Composition assumes conditional independence of maps given the pre-match features — a stated simplification.
- 'Team1 rate' in test is 57% — that's why recall on wins (0.71) is higher than on losses.

**LIKELY PROFESSOR QUESTIONS**

- *Q: What is dynamic programming here?*
  - A: Exact expansion of the series over states (maps played, maps won): each map multiplies its win/lose probabilities into the surviving states; the answer is the total mass reaching the required wins. Not a heuristic, not sampling.
- *Q: Are maps really independent?*
  - A: We assume independence given the features — momentum within a series is not modelled; a known limitation.
- *Q: Why is the AUC here (0.649) lower than the series model's Cologne AUC?*
  - A: Different tasks — single maps are noisier than whole series; the numbers are not comparable.

**TRANSITION**

Before the tournament part, one short slide on why probabilities are the bridge.

## Slide 10 — From probabilities to tournaments — One match probability -> 50,000 possible Majors

**MAIN MESSAGE**

Sampling winners from the model's probabilities — winner ~ Bernoulli(p) — is what turns one match model into a tournament forecast with realistic upsets.

**SCRIPT (speak naturally)**

This slide is the hinge of the talk. If we kept only labels, simulating a tournament would be pointless: the favorite wins every simulated match, so every simulation is the same bracket and upsets are impossible. Instead, each simulated match draws its winner like a weighted coin: if the model says 62 percent, Team A wins 62 percent of the simulated runs and loses the other 38. Crucially, we do not repeatedly pick whichever side has p above one half — we sample. Chaining those samples through the real tournament rules, 106 matches per run, and repeating fifty thousand times gives a distribution over entire tournaments: each team ends up with a probability of winning the Major, reaching playoffs, going out early. And that's precisely why the models were selected on probability quality — the simulator is only as honest as the probabilities you feed it.

**WHAT I MUST UNDERSTAND**

- Bernoulli(p) = single biased coin flip; the sampled winner advances and the bracket continues from that state.
- 50,000 runs → Monte-Carlo standard error ≤ 0.22 percentage points on any probability — sampling noise is negligible.
- A deterministic favorite-path was also computed as a contrast — shown two slides ahead.

**LIKELY PROFESSOR QUESTIONS**

- *Q: Why Monte Carlo instead of exact computation?*
  - A: The Swiss format's pairings depend on records and rematch-avoidance — the state space of full tournaments is combinatorially huge, so exact enumeration is infeasible; sampling approximates the distribution to arbitrary precision.
- *Q: Why exactly 50,000?*
  - A: Chosen so Monte-Carlo noise (max SE ~0.22pp) is far below the differences we care about; it's also cheap because all 2,976 possible matchup probabilities are precomputed once.
- *Q: Does the simulation update ratings between rounds?*
  - A: No — team state stays frozen at the pre-event snapshot throughout; simulated results never feed back into features.

**TRANSITION**

Now the real test: we froze all of this before the Cologne Major and let reality grade it.

## Slide 11 — External evaluation — IEM Cologne Major 2026, frozen before, judged after

**MAIN MESSAGE**

The strongest evidence in the project: a genuinely external, pre-registered evaluation — model and state frozen before the event, judged on 106 real Major matches.

**SCRIPT (speak naturally)**

This is the evaluation I trust most, because it is immune to any internal mistake. Before the Major began, everything was frozen — the random forest, its preprocessing, and the team-state snapshot ending three days before the first match — with cryptographic hashes recorded. The full 50,000-run simulation was executed and saved before any result existed; only then were the real results opened. On the 106 official matches, the frozen model scored 64.2 percent accuracy, AUC 0.697, log loss 0.632 against 0.693 for an uninformed baseline, Brier 0.221 against 0.250. Every number is on the favorable side of the model's own development validation — encouraging, but I'll say precisely this much: a strong demonstration on one real event, not statistical proof. One data note: of 107 Cologne rows, one was a Germany-versus-Poland showmatch, excluded with documented evidence.

**WHAT I MUST UNDERSTAND**

- Frozen state = strict pre-Cologne snapshot (last update 30 May 2026); one team (THUNDERdOWNUNDER) was a genuine cold start.
- Dev-validation comparison: LL 0.632 vs 0.651, Brier 0.221 vs 0.230, AUC 0.697 vs 0.657, Acc 0.642 vs 0.607.
- Match orientation: team_a = better tournament seed at pairing time; baseline accuracy 0.528 = team_a prevalence.

**LIKELY PROFESSOR QUESTIONS**

- *Q: Was Cologne in the training data?*
  - A: No. It was fenced off in the evaluation manifest from phase one, structurally absent from all feature tables, and the prediction state ends before the event. The deployed app's snapshot later includes Cologne, but the historical evaluation reads only the frozen artifacts.
- *Q: What does AUC 0.697 mean?*
  - A: Take a random Cologne match the model got a winner and a loser for — with probability ~0.70 the model ranked the actual winner above the actual loser. 0.5 is chance.
- *Q: Why not confidence intervals here?*
  - A: The same teams recur across the 106 matches, so an IID bootstrap would understate uncertainty; rather than report a too-narrow interval, the project reports point estimates and says 'single event' explicitly.

**TRANSITION**

So what did those 50,000 simulated Majors actually predict — and what really happened?

## Slide 12 — Simulation vs reality — The champion was the model's #4

**MAIN MESSAGE**

The Monte-Carlo distribution contained the real outcome — Falcons at 8.9%, rank 4 — while any single deterministic bracket structurally could not.

**SCRIPT (speak naturally)**

Here is the frozen pre-event championship forecast, top eight of thirty-two. Vitality was the clear favorite at about 30 percent — and in orange, Team Falcons at 8.9 percent, rank four. Falcons won the Major. Two readings. Positively: 8.9 percent is nearly three times the uniform one-in-thirty-two reference, Falcons were top-four in the distribution, and five of the eight real playoff teams were in the model's top eight — what happened was a plausible draw from the forecast. Negatively, and I want to be equally clear: the favorite did not win, the single most confident wrong call was Vitality over 9z at 78 percent, and neither actual finalist was in the pre-event top two. That contrast is the argument for distributions: a deterministic 'favorite always wins' bracket can only crown its own favorite — verified: it crowned Vitality — while the Monte-Carlo distribution carried the real champion with real weight.

**WHAT I MUST UNDERSTAND**

- Never claim the prediction was 'right' — claim the distribution was honest and informative.
- 8.9% vs 3.1%: compare to the uniform 32-team reference, NOT to a coin flip.
- Deterministic favorite path: 6/8 correct Stage-1 advancers, but 0/2 finalists and 0/1 champion.

**LIKELY PROFESSOR QUESTIONS**

- *Q: Isn't 8.9% just a miss?*
  - A: Under a proper scoring view, assigning 8.9% to the actual champion (when uniform gives 3.1%) is a better forecast than almost any single-bracket prediction; the point of a distribution is exactly that secondary contenders sometimes win.
- *Q: Could any model have picked Falcons #1?*
  - A: Only by being badly overconfident elsewhere — the three teams above Falcons had 60% combined championship mass; pre-event evidence genuinely favored them.
- *Q: How well were probabilities sized overall?*
  - A: Mean probability assigned to actual winners was 0.547 — modestly above chance, consistent with a competitive field; per-stage accuracy was best early (0.70 Stage 1) and hardest in playoffs (4/7).

**TRANSITION**

A quick look inside the machine that generated those 50,000 tournaments.

## Slide 13 — Tournament engine — The real Major format, replayed 50,000 times

**MAIN MESSAGE**

The engine implements the real Valve Swiss + playoff rules exactly, samples every match from the frozen probability matrix, and is provably faithful — it reproduces the real tournament when given the real results.

**SCRIPT (speak naturally)**

The Cologne format: three Swiss stages of sixteen teams. In a Swiss stage you always play someone with the same record — one-and-oh against one-and-oh — rematches avoided by rule; three wins advance, three losses eliminate, which takes exactly 33 matches per stage. Eight advance, eight join at the next stage, until the final eight play single-elimination playoffs — best-of-three until a best-of-five grand final; 106 matches per tournament. Every simulated match samples its winner from the frozen matrix of all 2,976 team-pair-format probabilities, computed once from the pre-veto model — future map picks don't exist. Seeded random streams make all 5.3 million simulated matches reproducible. And one detail I like: as a validity check the engine was fed the 106 real results — it reproduced the entire actual bracket, every pairing and stage transition, 106 out of 106.

**WHAT I MUST UNDERSTAND**

- Pairings within a stage depend on records AND rematch-avoidance — that's why simulation, not closed-form math.
- Probability matrix: 32×31 ordered pairs × 3 formats = 2,976 entries; the model is never called during simulation.
- Why RF V2 here: the tournament simulator predicts future matches whose map vetoes haven't happened.

**LIKELY PROFESSOR QUESTIONS**

- *Q: How are the Swiss pairings decided exactly?*
  - A: By the Valve rulebook: seed-based first round (1v9…8v16), then same-record pools paired by current seeding with exhaustive rematch-minimizing search; a 'Difficulty Score' (Buchholz-style) orders teams mid-stage.
- *Q: Why not use the known-map model in the simulator?*
  - A: Maps for future tournament matches are unknown — using it would require inventing vetoes. The pre-veto model is the honest tool for that job.
- *Q: What varies between the 50,000 runs?*
  - A: Only the sampled winners; the probabilities, rules and seeds structure stay fixed. Each run's RNG stream is derived from a base seed plus the run index.

**TRANSITION**

All of this is wrapped into an application you can actually use.

## Slide 14 — Application — The models, deployed; ML stays in Python

**MAIN MESSAGE**

The whole system ships as a real product: Python owns every model decision, FastAPI exposes it with verified contracts, and a Next.js PWA renders it — including explanations.

**SCRIPT (speak naturally)**

The application layers mirror the science. At the bottom, the two frozen models. Above them, a Python inference core reusing the exact same feature code as training — so the app cannot drift from the evaluation — plus an explanation core: every prediction ships with factor attributions from the model itself, TreeSHAP for XGBoost and an exact tree-path decomposition for the forest, phrased associationally, never causally. A FastAPI layer exposes this with typed contracts; at startup it re-verifies the hashes of every model and state file and refuses to serve if anything drifted. On top, a Next.js progressive web app: pick teams and format, choose maps-unknown or maps-known, and you get the probability bar, the per-map breakdown — including the chance each map is even played — and the 'why' factors. The Major simulator is exposed through the same API; its dedicated UI page is the next release step. One rule throughout: no ML logic is ever duplicated in the frontend.

**WHAT I MUST UNDERSTAND**

- Explanations: XGB uses exact TreeSHAP (built into XGBoost); RF uses Saabas-style tree-path attribution — exact for this forest, but not Shapley values.
- The deployed snapshot legitimately includes Cologne (data through 28 Jun 2026) — allowed because the historical evaluation was frozen first.
- App predictions are at the deployment snapshot cutoff — it is not live August-2026 data.

**LIKELY PROFESSOR QUESTIONS**

- *Q: Why FastAPI?*
  - A: Thin, typed, async Python layer — the models are already Python; pydantic contracts give validation for free; zero model logic in the transport layer.
- *Q: Frontend predictions match the research code exactly?*
  - A: Yes — the API calls the same frozen pipelines; startup hash checks plus golden-fixture tests pin the numbers.
- *Q: Are the explanations causal?*
  - A: No, and the UI says so — they describe what moved this model's prediction, not why a team will actually win.

**TRANSITION**

To close: what worked, what the limits are, and where this goes next.

## Slide 15 — Conclusion — What worked, what is limited, what comes next

**MAIN MESSAGE**

The project delivers a modest but real, honestly measured predictive signal, end-to-end from raw data to a deployed probabilistic product — with its limits stated as clearly as its wins.

**SCRIPT (speak naturally)**

To conclude. What worked: genuine predictive signal above every baseline, demonstrated twice — on a sealed internal test and on a fully external Major with everything frozen in advance; a system that is probabilistic end to end, selected and judged with proper scoring rules; and a complete shipped application. The limitations are equally explicit: the data ends June 28th 2026, so the app reasons from that snapshot; Cologne is one event and best-of-five is nearly absent; the ELO is deliberately simple; maps are treated as independent within a series; roster dynamics are only partially captured. Next steps follow directly: a second data source such as GRID, time-decayed and opponent-adjusted ratings, deeper roster modelling, and a proper calibration and ensembling study. The takeaway in one sentence: pre-match information carries real, honestly measurable signal about CS2 outcomes — and a probabilistic system can deliver it while being exact about its own uncertainty. Thank you.

**WHAT I MUST UNDERSTAND**

- Keep the three columns in this order if asked to summarize: worked → limited → next.
- If pressed for the single biggest limitation: data freshness (snapshot ends 28 Jun 2026) plus single-event external evidence.

**LIKELY PROFESSOR QUESTIONS**

- *Q: What would you do first with more time?*
  - A: Ingest a second, fresher data source — it simultaneously fixes staleness, strengthens team identity, and enables a second external evaluation event.
- *Q: Could this generalize to other esports?*
  - A: The architecture (chronological state engine → probabilistic model → DP/Monte-Carlo composition) is game-agnostic; only the feature families are CS2-specific.
- *Q: Is the model good enough to bet on?*
  - A: That was never the goal and no market comparison was done; the honest claim is 'meaningfully better than uninformed baselines, with calibrated-looking probabilities'.

**TRANSITION**

— end of talk —
