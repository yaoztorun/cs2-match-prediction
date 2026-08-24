# Presentation Q&A — Likely Questions and Short Answers

Companion to `reports/cs2_match_prediction_presentation.pptx`. Every numeric claim below
comes from the frozen reports cited in `reports/presentation_sources.md`.

---

## Methodology & evaluation

### Why a chronological split instead of a random split?
Because deployment always predicts *forward in time*. A random split would let the model
train on matches played *after* the matches it is evaluated on — it would implicitly know
future team strength, inflating every metric. The project splits 70/15/15 strictly by
datetime (train → Aug 2025, validation → Jan 2026, test → Jun 2026), and even
hyperparameter tuning uses expanding-window CV inside the training period only.

### What is leakage?
Any information reaching a training feature or a model decision that would not have been
available at prediction time. Examples avoided here: using a match's own stats as
features, computing a team's rating over its full career instead of only its past,
tuning on the test set, or letting Cologne results influence anything before the frozen
pre-event simulation. Prevention is structural: features are computed by replaying
history in order, state is updated only *after* a match's features are emitted, and
matches sharing one timestamp never see each other's results.

### How did you prevent future information concretely?
1. Chronological replay: `build_features()` reads only state accumulated strictly before
   the match's datetime. 2. Timestamp batching: same-timestamp matches are all featured
   before any of their results is applied. 3. Chronological split + train-only CV.
4. TEST sealed until both models were frozen. 5. Cologne structurally absent from all
   development tables and states; the pre-event simulation used a state snapshot ending
   2026-05-30, before the 2026-06-02 cutoff. 6. Training and inference share the exact
   same feature code, so there is no separate "live" path that could differ.

### Was Cologne part of training?
No. It was tagged in an evaluation manifest at phase one and excluded from every feature
table, split, CV fold and state used for development. The pre-event simulation used the
frozen model plus a strict pre-Cologne state snapshot. Only *after* those artifacts were
frozen (with recorded hashes) were the real results opened. The later *deployment*
snapshot for the app legitimately includes Cologne — but the historical evaluation always
reads the original frozen artifacts and is never regenerated.

### Why did you evaluate on one external tournament — is that enough?
It is the strongest kind of evidence available (a pre-registered, frozen, genuinely
out-of-sample event), but it is one event with 106 dependent matches (the same teams
recur). The project therefore reports it as a favorable *demonstration*, explicitly not
as statistical proof of generalization, and gives no confidence intervals there because
an IID bootstrap would understate the uncertainty.

## Metrics

### Why use AUC?
ROC-AUC is threshold-free discrimination: the probability that a randomly chosen actual
winner receives a higher predicted probability than a randomly chosen actual loser.
With a ~55/45 class skew, accuracy can be gamed by favoring the majority side; AUC
cannot. 0.5 = chance, 1.0 = perfect ranking.

### What does AUC 0.697 (Cologne) mean?
Pick a random Cologne match the model called correctly-rankably: with probability ~0.70
the model assigned the actual winner a higher probability than the actual loser. It
measures ranking quality, not calibration.

### Why Log Loss and Brier?
Both are *proper scoring rules* — a forecaster minimizes them in expectation only by
reporting its true belief. Log Loss = mean of −log p(actual outcome); it punishes
confident mistakes brutally. Brier = mean squared error of the probability; gentler,
easier to interpret (0.25 = always saying 50%). They measure probability *quality*,
which accuracy cannot see — and the simulator consumes probabilities.

### Why not only accuracy?
Accuracy only checks which side of 0.5 a prediction fell on; 51% and 99% count the same.
Two models with equal accuracy can have wildly different usefulness for simulation or
risk assessment. That is why the pre-registered selection hierarchy was: Log Loss/Brier
first, then AUC, then accuracy.

### What is a confusion matrix?
A 2×2 count table of predictions vs reality at the 0.5 threshold: true positives (win
predicted, win happened), false positives, false negatives, true negatives. On the map
model's sealed test: [[297 TN, 317 FP], [235 FN, 578 TP]] — errors are fairly balanced
with a lean toward predicting "team1 wins", matching the 57% base rate.

## Models

### What is Logistic Regression?
A linear model: it computes a weighted sum of the features and squashes it through a
sigmoid into a probability. Implemented from scratch (NumPy gradient descent) in this
project. Transparent and hard to overfit — the baseline any complex model must beat.

### What is a Random Forest?
An ensemble of decision trees, each trained on a bootstrap resample of the data with
random feature subsets at each split; predictions are averaged. Individual trees overfit
badly; averaging many decorrelated trees cancels that variance. Tuning (depth 8, ≥20
samples per leaf) was essential: untuned, its train-validation AUC gap was +0.37; tuned,
+0.055.

### What is XGBoost?
Gradient boosting: trees are built *sequentially*, each new tree fitted to correct the
current ensemble's errors (the gradient of log loss), with shrinkage and heavy
regularization. The final known-map model uses 118 trees of depth 2 — many tiny
corrections rather than a few deep trees — and had the most stable train→validation
behaviour of all models.

### Why RF instead of XGB for the pre-veto application model?
By the pre-registered hierarchy on the series validation set: RF V2 won Log Loss
(0.6514 vs 0.6542), Brier (0.2298 vs 0.2311) and AUC (0.6566 vs 0.6504); XGB won only
accuracy (0.6117 vs 0.6068), which is criterion 3. A documented caveat remains: XGB's
train→validation gap (+0.007) was much smaller than RF's (+0.055), so RF's win carries
single-split risk — recorded, not hidden.

### Why two separate models?
Two different tasks with different information sets. Pre-veto: no map information exists
(and for future tournament matches it never does) → series-level RF V2 on 17 features.
Known-map: the user supplies the ordered maps, unlocking map-specific and roster-on-map
features → map-level XGB V3 on 131 features, composed by DP. Blending them would either
waste information or fabricate it.

### Why not neural networks?
~10k rows of tabular data is the regime where gradient-boosted trees consistently win;
an MLP would add tuning cost, opacity and overfitting risk with little expected gain.
Reasonable future work if the dataset grows substantially.

## Features

### What is ELO?
A rating system: every team starts at 1500; after each match the winner takes points
from the loser, with the amount scaled by surprise — expected score
E = 1/(1+10^((R_B−R_A)/400)), update R ← R + K·(S−E) with K=32. Deliberately simple
(no time decay, no tier weighting) for interpretability; `elo_diff` is the strongest
single feature in both models.

### How were teams represented?
By canonical team name (after normalization + manual review of ambiguous names), because
the raw `team_id` was a per-match surrogate with zero reuse. Each canonical team carries
an evolving state: ELO, win rates, recent form, margins, activity, map histories and
roster information — features are differences of the two teams' states.

### How are roster changes handled?
Partially. Player-level features are built from the *current five players'* own
histories (their recent K/D, KAST, map experience), and continuity features measure how
much of the team's map history was produced by the current core. A full transfer model
(e.g. rating adjustments on roster swaps) is future work.

## Composition & simulation

### What is dynamic programming here?
The exact expansion of a best-of-N series over states (maps played, maps won by A).
Start with probability mass 1 at (0,0); each map i splits every live state by p_i and
1−p_i; a state is terminal once a side reaches ceil(N/2) wins, so later maps only
contribute along branches where the series is still alive. P(series) = total mass in
"A reached the required wins" states. Exact for any odd N — no sampling, no special
cases for BO1/BO3/BO5.

### Why Monte Carlo for the tournament?
Swiss pairings depend on evolving records, seedings and rematch-avoidance rules, so the
space of possible tournaments is combinatorially huge and cannot be enumerated exactly.
Sampling full tournaments approximates the outcome distribution to arbitrary precision
and is cheap because all 2,976 possible matchup probabilities are precomputed.

### Why Bernoulli sampling? (Why not "p>0.5 wins"?)
The winner of each simulated match is *drawn* with the model's probability —
winner ~ Bernoulli(p) — so upsets occur exactly as often as the model believes they
should. Always advancing the p≥0.5 favorite collapses 50,000 simulations into one
deterministic bracket that can only ever crown the overall favorite (verified: that
bracket crowned Vitality, who did not win). Sampling is what made the actual champion
(Falcons, 8.9%) visible in the forecast at all.

### Why 50,000 simulations?
At N=50,000 the worst-case Monte-Carlo standard error on any probability is
√(0.25/50,000) ≈ 0.22 percentage points — far below the differences that matter — and
each probability is reported with its own MC standard error. Also fully reproducible:
each run has a seeded RNG stream (base seed 42 + run index).

## Results & limits

### What are the biggest limitations?
1. Data ends 2026-06-28 — the app predicts from that snapshot, not live form.
2. One external evaluation event; BO5 nearly absent everywhere (n=1 at Cologne).
3. Simple ELO; maps assumed conditionally independent within a series.
4. Roster dynamics only partially modelled; team identity depends on name resolution.
5. Probabilities are honest but modest — mean p(actual winner) at Cologne was 0.547;
   this is a high-variance domain and predictions remain genuinely uncertain.

### How would you improve the project?
In order: (1) a second, fresher data source; (2) time-decayed form and
opponent-adjusted/tier-weighted ratings; (3) deeper roster/player modelling;
(4) a calibration study (reliability curves, isotonic/Platt — none was fitted, so
Log Loss/Brier gains were deliberately never called "calibration"); (5) ensembling the
model families; (6) a second frozen external-event evaluation to accumulate evidence.

### How could a second dataset be used?
The feature engine namespaces every match by source (`source:match_id`), so a second
provider (e.g. GRID) can be appended to the same chronological stream without ID
collisions. It would bring fresher matches (past 2026-06-28), an independent check on
team identity, richer per-round/per-player detail, and would enable cross-source
validation — train on one source, evaluate on the other.
