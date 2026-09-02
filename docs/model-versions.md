# Model version history

Every training run already writes a machine record to `runs/<stamp>__<name>/run.json`
and a `models/metadata_<name>.json`. Those are complete but not readable — they say
what a model scored, never why it was built or what was rejected. This file is the
human counterpart: one entry per version, what changed, what it measured, and what
was tried and thrown away.

**A note on what "measured" means here.** Feature groups are validated by ablation —
retrain with the group removed, compare on held-out seasons — not by reading XGBoost
feature importances, which are close to meaningless when features are correlated (and
in this matrix they heavily are). Raw importances appear in training output for
orientation only. Where a number below is quoted, an A/B on an identical code path
produced it.

---

## What data actually trains the model

The database holds **3,515,642 shots across 16 seasons** (2010-11 → 2025-26). Training
uses **2,096,585** of them. The gap is deliberate but worth stating, because "we have
16 seasons" and "we train on 16 seasons" are not the same claim.

| stage | shots |
|---|---|
| all seasons in the database | 3,515,642 |
| 2016-17 onward (`--from-season`, the default) | 2,253,739 |
| minus `home_away IS NULL` | −146,623 |
| minus `score_diff IS NULL` | −8,215 |
| minus Backcourt | −4,404 |
| minus `zone IS NULL` | −201 |
| **actually trained on** | **2,096,585** |

**Why 2016-17 and not 2010-11.** The `matchups` table — who guarded whom, for how many
possessions — begins in 2016-17. Everything on the defender side depends on it,
including the whole point-in-time defender-quality layer added in v10. Training on the
six earlier seasons would mean every defender feature is NaN for 36% of the rows, and
"defender data missing" would become a near-perfect proxy for "old season" — the trap
the `MIN_CONTEXT_COVERAGE` gate exists to prevent for play-by-play.

**A caveat on that boundary.** 2016-17 itself has matchup data for only **20 games** out
of roughly 1,300. The first included season is therefore nearly as defender-blind as the
excluded ones, and moving the cutoff to 2017-18 is worth measuring.

**A recoverable loss.** The 146,623 rows dropped for a null `home_away` are 12–18k per
season, evenly spread across all ten included seasons — so this is not a backfill that
stopped partway, it is one that never covered these rows. `home_away` is needed to
identify which team was defending. Recovering it would add roughly 7% more training
data. `src/ingestion/home_away_backfill.py` exists and evidently does not catch them;
undiagnosed.

---

## Shot-quality model

Predicts P(make | a shot is taken here). All versions train on 2016-17 onward with a
single held-out test season (2025-26) and a validation season for early stopping;
those are the numbers in the table below, and they are directly comparable to each
other. The rolling-origin backtest (`--origins 2`, held out on 2024-25 *and* 2025-26)
is a separate, noisier-but-fairer measurement quoted where it was run.

| version | features | log-loss | AUC | ECE | accuracy | what changed |
|---|---|---|---|---|---|---|
| v4 | 80 | 0.6312 | 0.6859 | 0.0071 | 0.6396 | baseline for this sequence |
| v5 | 73 | 0.6320 | 0.6853 | 0.0085 | 0.6396 | **removed** defender physicals (−7) |
| v6 | 109 | 0.6335 | 0.6836 | 0.0096 | 0.6369 | **added** radial spatial basis (+36) — rejected |
| v8 | 86 | 0.6180 | 0.7024 | 0.0089 | 0.6488 | dropped the basis, **added** shot mechanics + PBP context (+13) |
| v9 | 100 | 0.6146 | 0.7057 | 0.0086 | 0.6514 | **added** opponent zone defence (+8) and possession origin (+6) |
| **v10** | 100 | **0.6139** | **0.7063** | 0.0097 | **0.6518** | no feature change — point-in-time defender quality |

There is no v7: no model artifact or run record exists for that number, so it was
either never completed or discarded before it produced one.

### v5 — removing the defender physicals

Dropped `def_height`, `def_weight`, `def_wingspan`, `height_diff`, `weight_diff`,
`wingspan_diff`, `size_mismatch`.

**Log-loss got slightly worse** (0.6312 → 0.6320) and this was still the right call.
The features measured almost nothing real: actual FG% by attacker-minus-defender
height is flat across every populated band, about one point over the whole range,
because `height_diff` is computed against the possession-weighted *average* defender,
which sits near the league mean almost always. The extreme bands hold 4 to 25 rows out
of two million — and the model fitted them anyway. The recommender then drove straight
into them: naming a tall defender against a guard moved the prediction by 18 points,
of which 17.8 came from these features and 0.9 from every genuine defensive-quality
feature combined. A spurious effect twenty times the size of the real one, triggered by
the single most obvious thing a user can do in the UI.

A hundredth of a point of offline log-loss was the correct price. Re-enable with
`--defender-physicals` to reproduce.

### v6 — the radial spatial basis, rejected

Added 36 Gaussian radial-basis columns over the half court. Trees split one axis at a
time, so the fitted surface is a union of axis-aligned rectangles; the basis was meant
to let a single split carve a circular region and smooth the estimate.

**Measured worse on every metric** — log-loss 0.6320 → 0.6335, AUC 0.6853 → 0.6836,
ECE 0.0085 → 0.0096. Thirty-six extra columns is a lot of surface to overfit, and
distance plus the zone indicators already carry the spatial signal that exists. The
visual benefit was moot: the court render interpolates between grid points anyway, so
smoothness was already being delivered downstream at no cost to accuracy.

Dropped in v8. Re-enable with `--spatial-basis`.

### v8 — shot mechanics and play-by-play context

The largest single gain in the sequence: **log-loss 0.6335 → 0.6180, AUC 0.6836 →
0.7024**. Added twelve `mech_*` one-hots (pullup, stepback, cutting, driving, dunk,
layup, hook, alley-oop, putback, jumper, other), plus `is_putback` and
`seconds_since_prev_event`.

This is the per-shot version of a signal that had only ever existed as a season
average. A player's typical dribble count and typical openness describe a *player*;
two above-the-break threes by the same shooter — one a catch-and-shoot off a kick-out,
one a step-back over a set defender — were previously identical rows.

Isolated by a matched A/B on a 2023-24-onward window, where play-by-play coverage is
complete:

| | log-loss | AUC |
|---|---|---|
| `mechanics-off` (73 features) | 0.6302 | 0.6871 |
| `mechanics-on` (86 features) | **0.6178** | **0.7025** |

`is_assisted` was deliberately excluded and remains so — it is a perfect target leak.

### v9 — opponent zone defence and possession origin

Added point-in-time opponent defence per zone (8 columns) and five possession-origin
indicators (after rebound / made shot / turnover / dead ball / stoppage, plus other).

The opponent-defence motivation was measurable before the fact: scoring v8 on its
held-out season and aggregating residuals by defending team gave a z-score standard
deviation of 2.22 overall and **2.82 in the restricted area**, against 1.0 for a
correctly specified model, with eighteen of thirty teams beyond |z| > 2 at the rim.
Actual rim FG% allowed ranges from 0.622 to 0.709 across teams and the model could see
none of it — its entire knowledge of opponent defence was one season-level
`def_rating`. The signal persists year to year (+0.43 to +0.79 correlation season over
season), so prior games genuinely predict the next one.

Possession origin was a transition-versus-set-defence signal sitting unused: the ingest
had been writing `prev_event_type` since the backfill and nothing read it. Measured
within above-the-break threes, which controls for location: 0.362 after a rebound
against 0.335 after a substitution and 0.333 after a timeout.

### v10 — point-in-time defender quality

No feature columns changed. What changed is where the defender numbers come from.

`defender_stats` (LeagueDashPtDefend) is a whole-season aggregate and was joined by
season alone, so a shot contested by a defender contributed to that defender's own
season FG%-allowed, which was then fed back in as a feature describing that same shot.
This is the identical leak `src/features/point_in_time.py` was written to remove for
shooters — it had simply never been applied to the defender side.

`build_defender_category_rates` now rebuilds those figures from `matchups` + `shots`,
strictly prior-games-only, shrunk toward a category-level empirical-Bayes prior. The
possession-weighted mixture over everyone who guarded the shooter is unchanged. The
fitted `category_priors` travel in the model metadata so serving shrinks toward what
training used.

On the rolling-origin backtest, which is the fairer test:

| | log-loss | AUC |
|---|---|---|
| v9 | 0.6171 | 0.7021 |
| **v10** | **0.6166** | **0.7025** |

Both held-out seasons improved on both metrics. The gain is small because the leak only
distorted predictions *within* a season, so a season-level backtest partially sees the
same contaminated signal at test time and understates the fix. The real payoff is
correctness: a November prediction no longer borrows from April, and defender quality
is computable at any date rather than requiring the season to finish.

### What each feature group is actually worth

Group ablation on v9 (`runs/20260824-191347__ablate-v9/ablations.csv`), retraining with
one group removed at a time. Positive delta means removing it *hurt*, so the group was
earning its place. This is the honest alternative to reading XGBoost importances, which
are close to meaningless when features are correlated — and here they heavily are.

| group removed | log-loss | Δ vs full | zone-rank ρ |
|---|---|---|---|
| context | 0.6304 | **+0.0156** | 0.742 |
| shot_context | 0.6241 | **+0.0093** | 0.740 |
| possession_origin | 0.6168 | +0.0020 | 0.743 |
| opponent_defence | 0.6161 | +0.0013 | 0.745 |
| interaction | 0.6150 | +0.0001 | 0.739 |
| creation | 0.6149 | +0.0001 | 0.740 |
| *(nothing removed)* | 0.6148 | — | 0.740 |
| shooter_physical | 0.6148 | −0.0001 | 0.739 |
| shooter_skill | 0.6147 | −0.0001 | **0.727** |
| defender | 0.6144 | −0.0004 | 0.743 |

Three findings worth stating plainly, because they are uncomfortable:

1. **Game context and per-shot play-by-play carry most of the model.** Everything else
   is a rounding error by comparison.
2. **Removing the defender group slightly *improves* log-loss.** The defender features
   are not paying for themselves on make-prediction, which is part of why the v10 leak
   fix was worth doing on correctness grounds rather than expecting a large metric win.
3. **`shooter_skill` looks free to remove by log-loss but is not.** Zone-rank ρ falls
   from 0.740 to 0.727 — the worst in the table. It barely helps predict whether a
   given shot goes in, and it is what makes per-player rankings mean anything. A
   single-metric read would have thrown it out.

The `creation` group costing essentially nothing is not a bug either — it is the
finding. A player's point-in-time zone rate already absorbs the difficulty of the shots
he takes, so knowing *how* he creates them adds little once you know how well he
converts. Creation earns its place in the attainability model instead, where it is the
dominant signal, because "can this player generate this look" is a genuinely different
question from "will it go in".

### Held out of the shot-quality model by measurement

Both are re-enablable by flag, and both are documented at length in
`src/features/spec.py` where they are defined.

| feature group | flag | why it is off |
|---|---|---|
| `defender_physical` | `--defender-physicals` | Removed in **v5**. Near-zero real signal, but produced an **18-point spurious swing** at serving time when a tall defender was named — twenty times the size of the genuine defensive-quality effect, fitted off a handful of extreme rows. Cost ~0.001 log-loss to remove and was still correct. |
| `spatial_basis` | `--spatial-basis` | Tried in **v6**, dropped in v8. 36 radial-basis columns, measured **worse** on every metric (log-loss 0.6320 → 0.6335, AUC 0.6853 → 0.6836). The court render interpolates anyway, so the visual benefit was already free. |
| `score_pressure` | — (reverted) | `score_diff / (seconds_remaining + 30)`, an urgency interaction. Measured a **wash**: log-loss 0.6169 vs 0.6171, AUC 0.7023 vs 0.7021 — inside run-to-run noise (sd ≈ 0.003). The trees already reconstruct it from `score_diff`, `seconds_remaining_in_game` and `clutch_flag`. Reverted rather than left as dead weight. |
| `is_assisted` | never | Perfect target leak — assists are credited only on made baskets, so FG% is exactly 1.000 when set. Guarded by `tests/test_no_leaky_features.py`. Usable as a *team aggregate over prior games*, which is how the supporting-cast features reach it. |

---

## Attainability model

Predicts what share of a player's shot diet comes from a location — "can he get this
shot", as distinct from "will it go in". Metric is MAE against a league-average-share
baseline on the held-out season.

| version | MAE | baseline | lift | change |
|---|---|---|---|---|
| original | 0.0512 | 0.0804 | +36.3% | ordinal `zone_index`, no position |
| v2 | 0.0498 | 0.0804 | +38.0% | one-hot zones, position bucket, position backfill |
| cast | 0.0482 | 0.0804 | **+40.1%** | leave-one-out supporting cast |
| subzone | 0.0375 | 0.0610 | +38.5% | angle-split sub-zones |
| prior | 0.0318 | 0.0610 | +47.9% | the player's own prior-season diet |
| **pit** | **0.0307** | 0.0666 | **+53.9%** | point-in-time: season-to-date + career + last season |

MAE is **not comparable across the taxonomy change** — eight sub-zones mean smaller
shares and mechanically smaller errors. Lift over the matched baseline is the
comparable column, and the sub-zone model trades ~1.6pp of it for a capability the
previous models did not have at all.

### v2 — one-hot zones, position, and a data fix

Replaced the ordinal `zone_index` with one-hot zones. Ordinal encoding forced an
arbitrary ordering on an unordered category and, more practically, made explanations
useless: TreeSHAP attributed the entire zone effect to one opaque `zone_index` term
no reader could interpret.

Added position bucket (G/F/C) — and found that **2023-24 and 2024-25 had 100% NULL
positions** (572/572 and 569/569 rows), which would have left the new features dead
for two seasons. Backfilled 1,510 rows from each player's nearest labelled season;
nulls fell from 1,790 to 280.

### cast — the supporting cast, leave-one-out

Added `cast_ast_rate`, `cast_3p_rate`, `cast_efg`, `cast_att`: the shooter's
*teammates'* ball movement, spacing and efficiency, point-in-time and reset each
season.

**Leave-one-out is the whole feature, not a refinement.** Oklahoma City ranked 27th of
30 in raw assisted-field-goal rate in 2024-25, which reads as a team that does not move
the ball. The figure is depressed almost entirely by Shai Gilgeous-Alexander's own
self-created volume — excluding him, his teammates assist on .712 of their makes, near
the top of the league. The raw team rate would have told the model the opposite of the
truth for exactly the high-usage players whose environment is most worth knowing.

Verified not to skew: sweeping the supporting cast across its entire p5→p95 league
range, holding the player fixed, moves attainability by at most **2.8pp** (mid-range),
against 18pp for the banned `defender_physical` group. A player's own game still
dominates his teammates by roughly 2–3×.

These features are in the **attainability model only** and cannot affect make
probability or shot quality.

### subzone — angle-split location granularity

`Above the Break 3` spanned the entire arc from wing to wing, so a dead-centre pull-up
and a 60° wing spot-up were one bucket and received a byte-identical attainability.
`Mid-Range` had the same problem. Both are now split at 30° off dead centre.

The split is justified by a clean monotonic gradient in self-creation, measured over
2016-17 onward:

| degrees off centre | share of ATB3 volume | self-created |
|---|---|---|
| 0–10 | 15.3% | 26.6% |
| 10–20 | 13.8% | 28.4% |
| 20–30 | 20.1% | 25.5% |
| 30–45 | 30.8% | 19.5% |
| 45–60 | 15.5% | 14.4% |
| 60+ | 4.6% | 9.5% |

Corners are deliberately left unsplit — already angle-specific by construction, and at
96% assisted there is no self-created population inside them to separate.

### prior — the player's own prior-season shot diet

The single largest accuracy gain of any change to this model, and it was a
missing feature rather than a modelling subtlety.

Shot diet is one of the most stable quantities in basketball: lag-1
autocorrelation of a player's sub-zone share is **0.941 pooled**, and 0.765 to
0.920 in every individual sub-zone. The model had no access to it. It was
reconstructing shot diet from dribbles-per-touch and pull-up share while the
answer sat in the player's own previous season.

The cost was not a modest loss of accuracy but a systematic **collapse toward
the league mean**. Predicted standard deviation ran at roughly half the true
spread, so no player could be placed at a realistic extreme:

| sub-zone | pred sd / actual sd, before | after |
|---|---|---|
| In The Paint | 51% | 73% |
| Above the Break 3 (centre) | 58% | 76% |
| Mid-Range (centre) | 66% | 76% |
| Restricted Area | 87% | 91% |
| *median* | **65%** | **76%** |

League-wide, against each player's ACTUAL 2024-25 sub-zone shares (335 players
with 200+ attempts):

| | MAE | correlation |
|---|---|---|
| without prior diet | 0.0384 | 0.881 |
| **with prior diet** | **0.0305** | **0.918** |

Every sub-zone improved on MAE, correlation and dispersion. Jokic genuinely takes
37% of his shots in the paint; the previous model could not emit a number above
about 0.20 for anyone, and now predicts 0.364.

Not leakage: the prior season is complete before the season being predicted
begins, exactly as `zone_rate` uses strictly prior shooting in the shot-quality
model. Rookies get NaN — 27% of rows — and are carried by the creation traits
and position prior alone. Ablate with `--no-prior-diet`.

The explanation was restructured to match. The prior share is deliberately kept
OUT of the ranked factor list: it is not a trait, and at r=0.94 it would be the
top "reason" for every shot on the floor, burying everything about the player's
actual game. It gets its own line — where he starts from, then what moves him
off it.

### pit — point-in-time diet, and why one lag was not enough

Leaning on last season alone was the easy option and the weaker one. It discards
every season before it, cannot see the season being asked about, and lags role
changes.

Each addition measured on its own:

| history available to the model | MAE | corr | dispersion |
|---|---|---|---|
| last season only | 0.0330 | 0.911 | 74% |
| + two seasons ago | 0.0315 | 0.918 | 78% |
| + career-to-date, shrunk | 0.0310 | 0.921 | 77% |
| + explicit trend and evidence terms | 0.0313 | 0.920 | 76% |

Older seasons genuinely carry information. Explicit trend terms did not: the
model cannot anticipate a role change by extrapolating the last two seasons, and
adding those columns made it slightly worse.

**The current season is worth more than all of the previous one.** Measured
directly, predicting a player's full 2024-25 sub-zone shares:

| evidence | MAE |
|---|---|
| first 25% of this season (~20 games) | **0.0222** |
| first 50% of this season | **0.0139** |
| the entire prior season | 0.0294 |

So the model was restructured to be genuinely point-in-time, the same shape
`point_in_time.py` already used for shooting rates. Training rows are now one
per (player, season, **snapshot**, sub-zone) at five points through each season,
with the target being the share over the **remainder** of that season — disjoint
from the evidence, and the question the recommender actually asks. 178,360 rows
against roughly 3,000 per season before.

Feature importances confirm the ordering: `diet_to_date` 0.334, `prior_zone_share`
0.267, `career_diet` 0.066.

Evaluated against each player's ACTUAL rest-of-season shares:

| asked at | MAE | corr |
|---|---|---|
| season start (no current-season evidence) | 0.0248 | 0.963 |
| midseason (half the season known) | **0.0214** | **0.978** |

And the role-change failure is fixed. Wembanyama's above-the-break-centre rate
went from 20% to 30% between seasons:

| asked at | predicted | actual rest-of-season |
|---|---|---|
| season start | 0.194 | 0.302 |
| midseason | **0.277** | **0.277** |

`season_progress` is capped at the largest training snapshot (0.7); serving at a
completed season would otherwise extrapolate past every row the model saw.

### Rejected: a self-creation-weighted target

Replacing the frequency target with "share of the shots a player generates himself"
was tested and **rejected**, despite producing the intuitive ordering (dead-centre
threes 1.4× more attainable than wing threes, where the frequency target says the
opposite).

`is_assisted` exists only on **made** baskets, so an unassisted *attempt* share is not
observable — only an unassisted *make* share. That makes the target partly a
measurement of shooting skill, which is precisely the contamination attainability
exists to avoid:

| target | correlation with the player's zone FG% |
|---|---|
| frequency (kept) | +0.324 |
| self-creation (rejected) | **+0.662** |

Since the ranking score is `EP × √attainability` and EP already contains make
probability, adopting it would have counted shooting skill twice.

The creation dimension was instead moved to the **explanation layer**, where it is
reported as an observed fact rather than fitted as a target, so the contamination
cannot propagate into rankings.

---

## Explanation layer

`src/inference/explain.py`, served by `GET /explain/attainability/{player_id}`.

Decomposes an attainability estimate using exact TreeSHAP contributions — complete by
construction, so no factor can be silently omitted — into:

- **baseline**: bias + zone terms, what any player would get here
- **player effect**: everything else, what this player adds or gives up
- **factors**: ranked traits with the player's value and its league percentile
- **creation**: who generates his shots here, self vs. assisted, shrunk toward a
  sub-zone league prior

Two copy bugs were caught by reading real output rather than trusting the templates,
both of which would have shipped as confidently wrong text: "Restricted area shots are
only 33%" (it is the most common zone on the floor), and "he gets there less than most
forwards: tall enough to work inside" (a trait clause asserting a direction that fought
the sign). Summaries now use neutral labels; vivid trait phrasing is confined to the
factor list, where a signed ± column disambiguates it.

---

## Data corrections

Fixes to the database itself, which affect every model trained afterwards.

| what | before | after |
|---|---|---|
| Games missing from `games` but referenced by shots/matchups | 10 games, 1,815 orphaned shots, 1,978 orphaned matchups | 0 — backfilled individually via BoxScoreSummary V2/V3, since `LeagueGameLog` omits them |
| Orphaned `shot_context` rows | 176 | 0 |
| NULL player positions | 1,790 player-seasons | 280 |
| Missing wingspans | 2,327 player-seasons | 1,293 (2K ceiling reached — the remaining 427 players are not modelled in NBA 2K) |

**Not a duplicate, despite appearances.** 336 shot rows share
`(game_id, player_id, quarter, time_remaining, loc_x, loc_y)`. They are free-throw-adjacent
shots with placeholder `(0, -6)` coordinates that collide on that naive key while being
genuinely different attempts with different outcomes. Deleting them would destroy real
data.

**Known dead code.** `player_zone_stats` (49,781 rows) is populated but reaches no
model — it is the whole-season aggregate whose leak `point_in_time.py` was built to
remove. Still read by `player_lookup.py` for display. `physical_ingestor_bref.py` has
never successfully filled a single row despite being step 2 of the three-tier wingspan
fallback chain; undiagnosed.
