# Feature-layer redesign

What changed, why, and what the numbers say. Written against the state of the
pipeline before `src/features/` existed.

## 1. Point-in-time features (the leak)

**Before.** `zone_efficiency`, `fg_pct_*`, `season_fg_pct` and `career_fg_pct`
were whole-season aggregates joined onto every shot in that season. A shot's
own outcome sat inside its own features. `career_stats_backfill.py` added the
current season into the cumulative total before writing it, so even the
"career" figure included the season being predicted.

The inflated metric was the smaller problem. The larger one: those features do
not exist at prediction time. Predicting a November shot required April's
finished splits, so `player_lookup.py` substituted the *prior* season instead —
meaning the model's single most important input meant one thing in training and
a different thing in production.

**After.** `src/features/point_in_time.py` computes shooting counts from games
strictly before each shot, at two horizons (career-to-date, season-to-date).
Rates come from a three-level hierarchy:

```
league (per zone, empirical Bayes)  →  player career-to-date  →  player season-to-date
```

**The fitted priors reproduce a known result**, which is the best evidence the
estimator is doing something real:

| Zone | prior mean | strength *k* (attempts) |
|---|---|---|
| Restricted Area | 0.629 | 64 |
| In The Paint (Non-RA) | 0.416 | 78 |
| Mid-Range | 0.404 | 146 |
| Left Corner 3 | 0.397 | 223 |
| Right Corner 3 | 0.397 | 195 |
| Above the Break 3 | 0.354 | **316** |

Three-point rates regress five times harder than rim rates. That is the
well-established finding that 3P% is mostly noise at small samples while rim
finishing reflects real, quickly-measurable talent — and nothing here was told
to produce it. `k` is estimated by method of moments from the spread of true
talent, after subtracting binomial sampling variance.

This also **retires `PositionPrior`**. That table existed to give a player with
zero NBA history a labelled starting estimate, special-cased through the stack
as `stats_source="prior"`. Shrinkage makes it a continuum: a rookie's first shot
resolves to the league prior, his four-hundredth barely uses it, and no consumer
branches on which case it is.

## 2. Train/serve parity (the silent bug)

**Before.** The training matrix was built in a 150-line SQL string; the
recommender rebuilt the same forty features by hand in a Python dict. Nothing
enforced agreement. Drift would be invisible — offline metrics only ever
exercised the training path.

**After.** `src/features/spec.py::derive_features` is the only place a derived
feature is computed, called by both paths. `tests/test_train_serve_parity.py`
asserts they agree.

**It immediately caught three live defects:**

1. `recent_10_fg` / `recent_20_fg` were never set at serving time. They are
   populated for virtually every training row, so every served shot looked to
   the model like a player's career debut.
2. Omitting a defender left the defender columns NaN — which in the training
   matrix means "this game has no matchup data at all", not "an average
   defender".
3. The court grid emitted restricted-area candidates half a foot from the
   basket, outside the 0–3ft range where such shots are actually attempted.

Together these pushed served rim probabilities to **0.17 against a true rate
near 0.70**. The model was fine the whole time; the serving path was not.

## 3. Defender as a mixture, not an argmax

**Before.** `defender_linker.py` assigned each shot the single defender with the
most matchup minutes across the *entire game*. All eighteen of a player's shots
on a given night carried the same defender.

**After.** `build_defender_mixture` weights every defender who guarded the
shooter by possessions spent doing it. **The primary defender accounts for a
mean of only 37% of possessions** (median 35%), so the old argmax discarded
roughly two-thirds of the matchup signal on the average shot. `def_matchup_dispersion`
reports how spread out the assignment was, letting the model discount defender
features exactly when the shooter was guarded by a committee.

Still unfixed: on a rim attempt the relevant defender is often the help-side
big, not the on-ball defender. That needs lineup reconstruction from
play-by-play.

## 4. Calibration is now a decision, not a reflex

**Before.** The isotonic calibrator was fit on the model's own *training-set*
predictions. Holding out 15% inside `fit_calibrator` did not help — the model
had already seen those rows. A boosted ensemble is sharply overconfident on
memorized rows, so the calibrator learned a near-identity mapping and then
applied it to new shots that needed real correction.

**After.** The mapping is fit on the most recent out-of-sample data, judged on
data used for neither fitting nor early stopping, and **adopted only if it
improves held-out log-loss**.

On this dataset it does not, and that is worth stating plainly:

| | log-loss | ECE |
|---|---|---|
| raw | 0.6312 | **0.0071** |
| calibrated | 0.6323 | 0.0164 |

The raw model is already well calibrated, and a mapping learned from an earlier
scoring environment imports a stale correction. Forcing it on shifted the
per-player residual mean from **−0.013 to −0.597** — a systematic overprediction
introduced entirely by the "fix".

## 5. Evaluation that corresponds to the product

- **Honest baseline.** Lift is reported against zone + distance-spline + season
  logistic, not a zone average. Beating a zone average mostly demonstrates that
  shots get harder with distance.
- **Per-player residuals.** Summed (actual − predicted) per player, z-scored
  against binomial standard error. Currently mean z = −0.013, sd = 1.36, with
  2.2% of players beyond |z| > 3 against ~0.3% expected — so real player-level
  signal remains unexplained.
- **The recommender's actual KPI.** `zone_ranking_correlation` asks whether the
  model's ranking of zones matches what each player actually produced out of
  sample. **ρ = 0.74, positive for 98.8% of 327 players.** Nothing measured this
  before.

## 6. Creation skill — and the honest negative result

Two new tables (`player_tracking_stats`, `player_shot_profile`) and one new
feature module capture handle and passing: dribbles per touch, drives, pull-up
share, potential assists, and — most usefully — the *defender-distance
distribution* of each player's shots, which recovers contest level at the player
level even though it is unobservable per shot.

Face validity is strong. Top `self_creation_index` for 2023-24: Shai
Gilgeous-Alexander, Jalen Brunson, Luka Dončić, Trae Young, Damian Lillard.
Bottom: Lively, Kornet, Mitchell Robinson — rim-runners with ~0% pull-up share.
`creation_retention` (pull-up eFG minus catch-and-shoot eFG) separates them
further: Luka loses 1.7 points of eFG creating his own shot, where a typical
player loses far more.

**But ablating the entire creation group costs essentially nothing on shot-make
prediction** (log-loss 0.6311 without vs 0.6312 with). This is the finding, not
a defect. A player's point-in-time zone rate *already absorbs* the difficulty of
the shots he takes — knowing how he creates them adds little once you know how
well he converts them.

Creation skill earns its place elsewhere. In `attainability.py` it is the
dominant signal, and the attainability model beats a league-average baseline by
**36.3%**, with the top features being `tight_share` (0.37), `open_share`
(0.13), `pullup_share`, `avg_def_dist` and `drives_per_min`. That is the point:
*"can this player generate this look"* is a genuinely different question from
*"will it go in"*, and handle answers the first one.

## 7. The recommender answers the right question

Ranking by expected points alone recommends the restricted area to everyone,
always — true, useless, and unavoidable for a model that ignores whether a shot
is gettable. Expected points are now paired with attainability, and the ranking
uses `EP × √attainability`. The square root is deliberate: full weighting would
collapse every recommendation onto what the player already does most, which is
not advice.

The result behaves as it should. Curry's top-ranked zone is Above the Break 3
(attainability 0.42) over the restricted area (0.17); Jokić's is the rim (0.42).

Every projection now carries a credible interval whose width tracks evidence:
Curry's above-the-break band is ±0.024 on 9,782 attempts, Jokić's left-corner
band ±0.14 on 73.

## 8. Infrastructure

- **DuckDB** for the training-matrix build, reading the existing SQLite file in
  place. Full 10-season matrix (2.1M shots × 80 features) builds in ~90s.
- **Run registry** — `runs/<stamp>__<name>/run.json` with params, metrics,
  feature list, data content hash, and git SHA.
- **Rolling-origin backtest** (`backtest.py`) with feature-group ablations,
  replacing single-split numbers.
- **Calibrators serialize as knot arrays**, not joblib pickles, so they survive
  a sklearn upgrade.
- `config.current_season()` / `seasons_in_database()` replace the hand-edited
  season list.
- Fixed: the hyperparameter search sliced by row position for "temporal" folds
  while the matrix had no `ORDER BY` — the folds were not temporal.

## Known gaps

- **Contest level is unobservable per shot.** No public endpoint exposes it.
  This is the model's hard noise floor and the main reason AUC sits near 0.69.
- **Help-side defenders** are not modelled; needs play-by-play lineups.
- **Team-level creation support** — the largest passing effect is on
  *teammates'* shots, not the passer's. `FEATURE_GROUPS["team_creation"]` is
  reserved for it.
- **Play-by-play is the highest-value uningested source**: assisted vs
  unassisted, transition vs halfcourt, putbacks, and lineups.

---

## Addendum — three serving bugs found by looking at the UI

The offline metrics were fine throughout. These were all in the path between a
trained model and a number on screen, which is exactly the region unit tests and
log-loss do not cover.

### 1. Defender size features were fitted noise (the big one)

Naming a tall defender against a guard moved the prediction by **18 percentage
points** — Curry's above-the-break probability fell from 0.312 to 0.161 against
Wembanyama. Decomposing it:

| varied | effect |
|---|---|
| everything | −17.9 pp |
| defender **physicals** only | −17.8 pp |
| defender **quality** only | −0.9 pp |

The measured effect in the data is ~1 pp. Actual FG% by attacker-minus-defender
height is *flat* across every populated band (0.361 at −6/−3 ft vs 0.351 at
+3/+6 above the break); the extreme bands hold 4–25 rows out of two million,
and the model had fitted them.

The cause is the mixture change: `height_diff` in training is measured against
the possession-weighted *average* defender, which sits near the league mean
almost always. Feeding one specific defender's raw height compares an
individual against a distribution of averages.

`FEATURE_GROUPS["defender_physical"]` is now held out by default. Cost:
**0.0008 log-loss** (0.6312 → 0.6320), exactly what the ablation predicted.
Defender effect is now 0.7–2.9 pp, largest at the rim and in the paint — where
a rim protector should matter.

### 2. Feature bounds, to stop extrapolation generally

The recommender invents feature vectors that never occurred, by design. A
boosted tree outside its fitted region does not degrade gracefully — it applies
the outermost leaf's value at full confidence. Training now writes each
feature's 1st/99th percentile into the model metadata, and the serving path
clamps to it.

### 3. Player ratings were never model output

The "OFF 75 / DEF 87" badges came from a hardcoded frontend formula, not the
model. It read `ast`, `tov` and `ft_pct` — **NULL for all 582 players**, never
ingested — so its entire playmaking term evaluated to zero, which is why the
best passer in the league rated 75. What remained was dominated by raw FG%, so
its top-rated offensive players were end-of-bench centres who only shoot dunks.
Both scales saturated: 41% of the league sat exactly on the offensive floor of
68, and 69 players were tied at the defensive cap of 87.

Replaced by `src/inference/player_ratings.py` — percentile ranks over the
league, from volume-shrunk points per attempt, real tracking playmaking, usage,
and self-creation. Luka 75 → 99. Wembanyama 87 → 99.

### And the heat map itself

- Court geometry was invented (`cw * 0.42`) while the data plotted in real
  coordinates: the three-point arc was drawn at 21.0 ft against a real 23.75,
  so every above-the-break point rendered *outside* the line. The free-throw
  line was 11 ft too far out. All lines now derive from real dimensions through
  the same transform as the data.
- `globalCompositeOperation = "screen"` made overlapping blobs **sum**, so
  brightness encoded sampling density rather than value. Replaced with a
  distance-weighted average (Shepard interpolation) — an average is invariant
  to how many points are nearby.
- The blue→green→red ramp was a rainbow: OKLab lightness ran 0.66 → 0.87 →
  0.65, so the worst and best shots on the floor rendered at nearly identical
  lightness and were indistinguishable in greyscale or to a colourblind viewer.
  Replaced with a monotonic heat ramp (0.27 → 0.95) and a scale legend carrying
  real values.

---

## Round three — what was tried, and what the measurements said

Four items from the "what's left" list were built and measured. **Three of them
did not work**, and that is the useful part of this section: each was a
plausible idea, and the data said no.

### ✗ Hierarchical player / defender effects — no measurable gain

Built in `src/training/player_effects.py`: an L2-penalised logistic model on
player and defender indicators, fit with the boosted model's log-odds as a
fixed offset, so it can only learn what stage 1 got wrong.

Two implementation notes worth keeping. `LogisticRegression` has no offset
parameter, and XGBoost's linear booster (which does, via `base_margin`) is both
deprecated and badly behaved — on a synthetic check with a planted +0.800 logit
effect it recovered ~1e-5. The penalised logistic is now solved directly with
L-BFGS, which recovers the planted effect as +0.858.

The result:

| penalty λ | val log-loss | max &#124;effect&#124; |
|---|---|---|
| 3 | 0.63558 | 1.13 |
| 100 | 0.63538 | 0.40 |
| 1000 | **0.63531** | 0.12 |

The search prefers the *strongest* shrinkage available, and test metrics are
unchanged (log-loss 0.6327 → 0.6327, residual var(z) 1.88 → 1.87).

**What this tells us:** the 83% per-player residual overdispersion is *not a
stable, learnable player trait*. Effects fitted on the training seasons do not
transfer to the test season, which means the excess variance is season-specific
— role changes, health, or the unobserved contest level — rather than a fixed
per-player offset the model is failing to capture. That reframes the
overdispersion from "obvious missing feature" to "genuinely hard".

### ✗ Spatial basis — measurably worse

36 radial basis functions over the half court, to replace axis-aligned splits
on `loc_x`/`loc_y` with a smooth surface.

| | log-loss | AUC | ECE |
|---|---|---|---|
| without (v5) | **0.6327** | **0.6847** | **0.0078** |
| with (v6) | 0.6335 | 0.6836 | 0.0096 |

Worse on every accuracy metric. Distance and the zone indicators already carry
the spatial signal; 36 extra columns is surface area to overfit. And the
visual motivation — a smoother rendered surface — is already delivered by the
canvas interpolation, downstream, at no accuracy cost. Kept behind
`--spatial-basis` for re-measurement.

### ⚠ Play-by-play — ingested, one column quarantined

`src/ingestion/pbp_ingestor.py` pulls PlayByPlayV3 per game (resumable; skips
games already stored). It yields per-shot mechanics that the season-level
creation profile could only approximate: the league's own shot taxonomy
("Step Back Jump shot", "Pullup Jump shot", "Cutting Layup Shot", "Putback"),
putback detection from the preceding rebound, and time since the previous event
as a transition proxy.

**`is_assisted` is target leakage and is quarantined.** Assists are credited
only on *made* baskets, so measured FG% is exactly **1.000** when the flag is
set and 0.230 when it is not — a perfect predictor that does not exist at
prediction time. It is stored (a player's assisted rate is a real quantity) but
excluded from `FEATURE_GROUPS`, with `tests/test_no_leaky_features.py` asserting
it stays excluded. This is the single most dangerous column added all project;
a model given it would have scored beautifully offline and been worthless live.

The remaining context features are gated behind 90% coverage in
`build_matrix`. The ingest walks games in id order, so a partial run covers the
*oldest* seasons only — "context is missing" would be a near-perfect proxy for
"recent season", and the model would learn the proxy. Below the threshold the
columns are dropped entirely.

### Status of the model

`shot-quality-v5` remains production: 73 features, log-loss 0.6327, AUC 0.6847,
ECE 0.0078, zone-rank ρ 0.736. Neither v6 (spatial basis) nor the player-effects
stage improved on it.

The honest summary of round three is that the feature set is at a local
optimum. The things that would move it — genuine per-shot contest level, and
help-side defenders from lineup reconstruction — are data problems, not
modelling ones.

---

## ✓ Play-by-play mechanics — the one that worked

Controlled A/B on an identical window (fit 2023-24, val 2024-25, test 2025-26),
identical split, identical everything except the 13 play-by-play context
features:

| | mechanics-off | mechanics-on | change |
|---|---|---|---|
| features | 73 | 86 | +13 |
| **log-loss** | 0.6302 | **0.6178** | **−0.0124** |
| **AUC** | 0.6871 | **0.7025** | **+0.0154** |
| **lift over strong baseline** | +3.32% | **+5.22%** | **+1.9 pp** |
| ECE | 0.0059 | 0.0077 | slightly worse |
| zone-rank ρ | 0.7362 | 0.7308 | slightly worse |

For scale: every other feature group tested in this project was worth ≤0.002
log-loss, and defender quality — the thing the whole matchup product is built
around — was worth 0.0021. **This is roughly six times larger than anything
else.** It is also the largest AUC movement of the project by a wide margin.

### Why it works

The features are per-shot *mechanics*, and the spread they expose **within a
single zone** is enormous. Restricted-area shots, 2025-26:

| mechanic | n | FG% |
|---|---|---|
| dunk | 1,171 | **0.904** |
| alley-oop | 3,345 | 0.837 |
| cutting | 8,188 | 0.793 |
| driving | 37,249 | 0.656 |
| layup | 2,562 | 0.617 |
| putback | 9,940 | 0.608 |
| hook | 1,301 | 0.541 |
| pull-up | 2,317 | **0.489** |

A dunk and a pull-up from the same square foot of floor differ by **41
percentage points**, and until now the model saw them as the same shot with the
same features. Every creation signal it had — dribbles per touch, pull-up share,
openness — was a *season average describing the player*. This is the first
source that describes the **shot**.

### Verified not to be leakage

The `is_assisted` experience made this check mandatory. No mechanic sits at 0.0
or 1.0, the spread survives conditioning on zone, and the ordering is exactly
what basketball predicts. The genuinely leaky columns (`is_assisted`,
`action_type`) are excluded by `FEATURE_GROUPS` and asserted out by
`tests/test_no_leaky_features.py`.

### Caveats

The A/B ran on a three-season window with a **single fit season** (~230k shots)
because that is what the backfill had covered. The production model trains on
eight. The direction and rough magnitude should hold — the effect is
mechanical, not a small-sample artefact — but the exact numbers will move.
Calibration and zone-ranking both drifted slightly worse, worth re-checking at
full scale.

**This answers the "have we hit a dead end" question: no.** Three ideas from
the same list measured as negative or neutral; this one is worth more than
everything else in the feature set combined, and it only became testable once
the data existed.

### Serving mechanics: the recommender had to change too

A shot that has not been taken has no mechanic — the mechanic is part of what
is being decided. Leaving the indicators at zero was not an option: every
training row has exactly one mechanic set, so an all-zero row is a combination
the model has never seen, and that is the exact shape of the bug that made
naming a tall defender swing the prediction by 18 points.

The serving path marginalises over the player's own mix instead:

```
P(make | player, spot) = Σ_m  P(mech = m | player, zone) · P(make | mech = m, …)
```

Mix estimates are shrunk toward the league's mix for that zone by attempts, on
the same principle as every other rate here. Curry's above-the-break attempts
are 47% catch-and-shoot, 30% pull-up, 19% step-back — the blended answer
reflects that rather than describing an average shot nobody takes.

One subtlety worth recording. The serving path assigns each grid row a mechanic
*name* and runs it through `classify_mechanic`, so the training encoding is
reproduced by construction rather than reimplemented. That only works if every
bucket name classifies back to itself — and two did not, `alley_oop` and
`stepback`, whose rules matched only the spaced spellings. They would have been
silently filed as "other" at serving time while training saw them correctly.
`tests/test_no_leaky_features.py` now asserts the round-trip.

It also made the product better. The engine reports the best mechanic per
location alongside the blended number, which is the actionable half:

| player | zone | blended | best mechanic | that mechanic |
|---|---|---|---|---|
| Curry | Restricted Area | 0.657 | cutting | **0.898** |
| Curry | Above the Break 3 | 0.345 | jumper (catch-and-shoot) | 0.385 |
| Curry | Mid-Range | 0.453 | step-back | 0.483 |
| Jokić | Restricted Area | 0.712 | cutting | **0.902** |

"Your catch-and-shoot above the break is worth more than your pull-up" is
advice. A single blended number is a statistic.

---

## Final: shot-quality-v8

The backfill completed (99% of the 2016-17+ window) and the production model
retrained on all eight fit seasons. **The A/B result held, and improved.**

| | v5 (no mechanics) | **v8 (mechanics)** |
|---|---|---|
| features | 73 | 86 |
| log-loss | 0.6320 | **0.6180** |
| AUC | 0.6853 | **0.7024** |
| zone-rank ρ | 0.7324 | **0.7406** |
| ECE | 0.0085 | 0.0089 |
| lift over strong baseline | +3.13% | **+5.28%** |
| **uncertainty explained** | 11.2% | **13.2%** |

The share of total outcome uncertainty explained rose from 11.2% to 13.2% — an
18% relative gain, from one feature group, on a problem where every other group
tested was worth under 0.002 log-loss.

AUC crosses 0.70: given one made and one missed shot at random, the model now
ranks them correctly **70.2%** of the time.

Calibration survived the change — largest error across ten equal-size buckets
of 21,433 shots is **1.6 percentage points**:

| bucket | predicted | actual |
|---|---|---|
| 1 | 0.199 | 0.197 |
| 5 | 0.421 | 0.413 |
| 9 | 0.693 | 0.677 |
| 10 | 0.853 | 0.845 |

And the zone-ranking regression seen in the small A/B (ρ 0.7362 → 0.7308) was a
single-fit-season artefact: with eight seasons ρ is **0.7406**, better than v5.

### The revised list of what is left

Genuinely closed: player random effects, spatial basis, creation features for
shot-make prediction, defender physicals — all measured, all negative.

Still open, in order:

1. **Lineup reconstruction** from play-by-play substitutions. **Correction to an
   earlier draft of this document:** the data is NOT local. `extract_context()`
   keeps field-goal events and discards everything else, and no raw play-by-play
   is persisted anywhere, so this needs a full re-fetch (~3.5h), not just
   parsing. Unlocks help-side rim protection and the teammate half of the
   passing story.
2. **Hyperparameter search** over the current features — repointed and correct,
   but has still never completed a run.
3. **Per-shot contest level** remains the hard wall: worth 12-18 points within
   shot type, and no public source exposes it.
