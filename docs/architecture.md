# Architecture — module by module

What this project is, laid out the way you'd walk an interviewer through it: what
each piece does, why it exists, and what the interesting engineering decision was.
Model-specific numbers (which feature helped, which was rejected, why) live in
[model-versions.md](model-versions.md); this document is about the system, not the
results.

The one-line version: a database of 3.5M NBA shots feeds two models (shot quality
and attainability), served through a FastAPI backend to a React frontend, with
every derived feature computed once and shared between training and serving so the
two paths cannot silently disagree.

---

## The pipeline, top to bottom

```
nba_api (stats.nba.com)
        │
        ▼
┌─────────────────┐
│   ingestion/     │  22 scripts — one per data source, each independently rerunnable
└────────┬─────────┘
         ▼
┌─────────────────┐
│   db/            │  SQLite, 12 tables, 3.5M shot rows
└────────┬─────────┘
         ▼
┌─────────────────┐
│   features/      │  raw columns → point-in-time, shrunk, leak-free model inputs
└────────┬─────────┘
         ▼
┌─────────────────┐
│   training/      │  two XGBoost models: shot quality, attainability
└────────┬─────────┘
         ▼
┌─────────────────┐
│   inference/      │  FastAPI — same feature code, one prediction at a time
└────────┬─────────┘
         ▼
┌─────────────────┐
│  shot-vision-    │  React frontend — the interviewer-facing surface
│  engine-main/    │
└──────────────────┘
```

Every arrow above is a real module boundary with its own directory. The two that
matter most in an interview are `features/` (why it exists as a separate layer at
all) and `inference/explain.py` (turning a model into something a person can read).

---

## `src/db/` — schema

**`models.py`** (515 lines, 12 SQLAlchemy tables). The two load-bearing ones:

- **`Shot`** — one row per attempt, 3.5M rows. Composite foreign key to `Player`
  (`player_id`, `season`) rather than a surrogate key, because a player's identity
  for feature purposes genuinely changes season to season (age, team, role).
- **`Matchup`** — one row per (game, offensive player, defensive player), with
  possession counts. This table is what makes point-in-time defender quality
  possible at all; it only exists from 2016-17 onward, which is why the whole
  pipeline's usable window starts there rather than at 2010-11 (see
  [model-versions.md](model-versions.md#what-data-actually-trains-the-model)).

**`database.py`** — a thin SQLAlchemy engine factory. Nothing clever here on
purpose: `config.py` holds the one connection string, every other module imports
`get_engine()` rather than constructing its own, so there is exactly one place that
knows where the database lives.

---

## `src/ingestion/` — 22 independent scripts

One script per NBA API endpoint or data-quality fix, each runnable in isolation and
idempotent (upsert, not insert). The interesting thing here isn't any one script —
it's that there are 22 of them, discovered incrementally as gaps in the data turned
up. A representative sample:

| script | what it fixes |
|---|---|
| `shot_ingestor.py` | the core shot log |
| `matchup_ingestor.py` | who guarded whom, from BoxScoreMatchupsV3 |
| `physical_ingestor_nba.py` → `_bref.py` → `_2k.py` | a 3-tier wingspan fallback chain, because no single source has full coverage |
| `pbp_ingestor.py` | per-shot play-by-play context (mechanics, transition) — the slowest ingest at ~1.2 hours |
| `home_away_backfill.py`, `score_diff_backfill.py`, `position_backfill.py` | targeted repairs for specific known data holes |

**A genuinely interesting bug, worth telling an interviewer about directly:**
`physical_ingestor_2k.py` logs nothing on a successful scrape — only failures print
— so a first read of its output looked like 434 of 447 remaining players were never
attempted. They were; the script is just silent on success. Confirmed by
re-scraping one player directly and finding it worked. The fix was a second pass,
which recovered players the first read wrongly wrote off as absent. The lesson —
verify a script's own logging before trusting what it appears to say — is the kind
of thing worth having a concrete story for.

---

## `src/features/` — the layer that makes the model trustworthy

This is the module to lead with. Two things make it worth a separate directory
rather than inline SQL in the training script:

### 1. One feature definition, two callers

`spec.py::derive_features()` is a pure function: raw columns in, model features
out. The training matrix builder calls it on 2 million rows; the recommender calls
it on a single row at serving time. **Nothing else in the codebase is allowed to
compute a derived feature.** `tests/test_train_serve_parity.py` asserts the two
paths agree on identical inputs — this is the test that catches the failure mode
where training and serving quietly drift apart because someone hand-copied a
feature computation into two places and only updated one.

That test caught three real bugs before they shipped:
- `recent_10_fg`/`recent_20_fg` were never populated at serving time, so every
  live prediction looked like a player's career debut.
- A shot with no named defender left defender columns NaN, which in training
  means "no matchup data exists" — a different population from "an average
  defender," which is what an omitted defender in a live request actually means.
- The court grid emitted restricted-area candidates outside the range shots are
  ever actually attempted from.

### 2. Point-in-time, everywhere

**`point_in_time.py`** (1,195 lines, the largest module in the codebase) is a
single discipline applied repeatedly: every rate a player or team "has" is computed
from **strictly prior games only**, shrunk toward an empirical-Bayes prior so a
30-attempt sample doesn't read as a settled fact.

This module is worth walking through carefully because the same leak was found and
fixed **twice**, in two different places, which is itself the interview-worthy
story:

1. **Shooter rates.** The original pipeline joined `player_zone_stats`, a
   whole-season aggregate, onto every shot in that season — so a shot's own
   outcome was inside its own feature. Rewritten to accumulate makes/attempts
   strictly before each game's date.
2. **Defender rates**, fixed later, independently, for the identical reason:
   `defender_stats` (a season-aggregate NBA API table) was joined by season alone.
   `build_defender_category_rates` rebuilds it from `matchups` + `shots`,
   prior-games-only. Both origins of a rolling-origin backtest improved on the fix
   ([details](model-versions.md#v10--point-in-time-defender-quality)).

The general shape (`build_prior_counts` → shrink toward a fitted prior → hand the
exact same shrink function to both the training builder and the serving lookup) is
reused for shooting rate, defender quality, opponent zone defence, and — the newest
instance — a player's own **shot diet** in the attainability model.

**`creation.py`** — handle/passing features (dribbles per touch, drive rate,
self-creation index), lagged one season, since these come from a separate tracking
endpoint that isn't available in-season.

**`mechanics.py`** — collapses ~40 raw NBA shot-type strings into 8 buckets
(pullup, stepback, cutting, etc.) via ordered substring rules, with a test asserting
every bucket name round-trips through its own classifier — necessary because the
recommender scores hypothetical shots by naming a mechanic and re-classifying it,
so a bucket that doesn't round-trip would silently encode differently at serving
time than in training.

**`shrinkage.py`** — the empirical-Bayes machinery (`fit_beta_prior`, `shrink`,
`shrink_toward`) used by every point-in-time computation above. One implementation,
imported everywhere, rather than five ad hoc "if attempts < 30" guards scattered
across the codebase.

**`build.py`** — orchestrates all of the above into one training matrix. Uses
DuckDB attached read-only over the SQLite file for the heavy joins — a deliberate
choice: the workload (a handful of joins and group-bys over 3.5M rows, read once)
is exactly what a columnar engine is for, and it required no change to how the data
is stored.

---

## `src/training/` — two models, one honest measurement discipline

**`train.py`** — the shot-quality model. XGBoost, monotone constraints on features
where the direction is known a priori (e.g. a defender's zone FG%-allowed should
never *decrease* predicted make probability), so the model can't learn a
counter-intuitive relationship purely from correlation noise.

**`attainability.py`** (892 lines, the largest training module) — answers a
different question: not "will this shot go in" but "how readily can this player
generate this shot at all." Exists because ranking shot locations by expected
points alone always recommends the rim to everyone — true and useless, since the
entire difficulty is *getting* there. The model went through five iterations this
project, documented in full in
[model-versions.md](model-versions.md#attainability-model); the short version is
that it moved from a single season-level snapshot to a genuinely point-in-time
model (season-to-date, career-to-date, and last season, all shrunk and blended),
because the current season's partial evidence turned out to predict a player's
rest-of-season shot diet better than his *entire previous season* did.

**`backtest.py`** — rolling-origin evaluation and per-feature-group ablation.
The reason this exists rather than trusting a single train/test split: XGBoost's
own feature importances are close to meaningless when features are correlated, and
in this matrix they heavily are (`zone_rate`, `overall_rate`, and the creation
features all describe overlapping aspects of the same player). Ablation — retrain
with one group removed, measure the actual delta — is the only honest way to know
what a feature group is worth. This is what caught the counter-intuitive finding
that the `creation` feature group costs almost nothing to remove from *shot
quality* (a player's point-in-time zone rate already absorbs shot difficulty) while
being the dominant signal in *attainability* — a genuinely different question.

**`calibration.py`** — fits a probability-calibration mapping and only adopts it if
it measurably improves held-out calibration error; otherwise the raw model output
ships. The trainer never assumes calibration helps.

**`train_baseline.py`, `feature_engineering.py`, `position_priors.py`** —
deliberately kept, explicitly marked deprecated in their own docstrings. Worth
having in a portfolio repo: it shows the *previous* architecture (whole-season
joins, hand-copied SQL feature code) next to what replaced it, which is a more
concrete way to explain a design decision than describing it in the abstract.

---

## `src/inference/` — serving

**`recommender.py`** (1,029 lines) — loads both trained models once at startup,
scores an ~180-point analytic grid over the court for a given matchup, and ranks by
expected points weighted by `√attainability` (a square root rather than linear
weighting deliberately, so attainability nudges the ranking toward gettable shots
without collapsing every recommendation onto whatever the player already does
most).

Every projection carries a 95% credible interval, derived from how many real
attempts back it — a corner-three estimate built on 9 attempts and one built on 900
no longer render identically.

**`explain.py`** (488 lines) — turns one attainability number into a decomposition
a person can read, via exact TreeSHAP contributions (they sum to the prediction
exactly, so nothing can be silently left out of the explanation). Splits the
output into three parts a reader actually distinguishes between:

1. **baseline** — what any player gets in this zone
2. **history** — what *his own record* says, reported separately from traits
   because his own prior share is not a trait, it's evidence, and at r=0.94
   season-over-season it would otherwise dominate every other reason listed
3. **factors** — ranked, signed, with the player's value and league percentile

This module is worth reading end to end for an interview, because three real
copy bugs were caught here by generating live output and reading it rather than
trusting the template: a sentence that attributed a player's own prior-season
number to "his game," a rate computed *this season* mislabeled as "last season,"
and a 0.2% share that rendered as "0%" beside a non-zero projection in the same
sentence. Each has a regression test pinned to the exact wrong sentence it used to
produce.

**`api.py`** — FastAPI. Loads models once at startup (`lifespan`), tries a
newest-first list of model names so a bad or missing artifact degrades to the
previous version rather than crashing the app.

**`player_lookup.py`, `player_ratings.py`** — display-path lookups (search,
percentile ratings) that deliberately do **not** feed the model; kept separate so
a change to how a player card renders can never accidentally change a prediction.

---

## `shot-vision-engine-main/` — the frontend

React + TanStack Start. Structured as a cinematic splash (real numbers pulled live
from the API — "2.1M shots, 10 seasons" — not hardcoded copy) leading into the
actual engine: a matchup configurator, an analytic-grid heat map rendered on
canvas, and the click-to-expand attainability panel described above.

Two decisions worth naming if asked:
- **The heat map interpolates a court render from ~180 discrete grid points**
  rather than being pixel-native, because the model only ever scores locations on
  that analytic grid — anything smoother would be interpolation dressed up as
  precision.
- **The splash page's stats are fetched, not authored.** They previously read
  "847,000 shots · 12 seasons," which was simply wrong on both numbers — caught by
  comparing it against what `build_matrix()` actually reports at training time.

---

## Tests

13 files, organized by what they guard against rather than by module:

- `test_train_serve_parity.py` — the parity guarantee described above
- `test_no_leaky_features.py` — asserts `is_assisted` (and other perfect-leak
  columns) can never reach the feature list, however a future refactor is written
- `test_explain.py` — pins the exact wrong sentences the explanation used to
  produce, so a regression reads as a specific named failure, not a diff
- `test_combine_defenders.py`, `test_mechanics.py`, `test_pbp_extraction.py` —
  one file per parsing/encoding rule that has to round-trip correctly

---

## The one sentence version, if asked to summarize in an elevator

*"Two XGBoost models over 2.1M NBA shots — one for shot quality, one for whether a
player can actually generate a given shot at all — built around a single shared
feature layer so training and serving can never quietly disagree, with every rate
computed strictly from games before the one being predicted so nothing in the
model has ever seen its own answer."*
