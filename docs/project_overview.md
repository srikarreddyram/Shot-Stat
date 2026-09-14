# Shot Vision — project overview

A point-in-time-correct NBA shot-outcome prediction engine, trained on 16
seasons of play-by-play, tracking, and defensive data, served through an
explainable matchup tool and a full statistics browser.

This is the high-level, end-to-end summary — what the system is and what it's
built from. For the module-by-module walkthrough see
[architecture.md](architecture.md); for what each model version scored and why
a feature was kept or dropped, see [model-versions.md](model-versions.md).

## Scale

| | |
|---|---|
| Shot attempts | 3.5M |
| Games ingested | 20.4K |
| On-court lineup rows | 17.6M |
| Seasons | 16 (2010–11 to 2025–26) |
| Model features (v20) | 120 |
| Shot-quality model versions | 20 |
| Automated tests | 255 |
| NBA Stats API endpoints tapped | 14 |

## What it is

Shot Vision answers one question with real rigor: **given this shooter, this
defender, and this spot on the floor, what is the probability the shot goes
in — and why?** It is not a box-score dashboard with a model bolted on; the
model is the product, and everything else — the ingestion pipeline, the
point-in-time feature discipline, the calibration layer, the SHAP-based
explanation engine — exists to make that one number trustworthy and legible.

Three things layer on top of the core prediction:

- An **attainability model** answering the different question of whether a
  player would ever actually take this shot.
- An **algorithmic archetype system** that labels every rostered player
  (3&D Wing, Pick-and-Roll Hub, Rim Protector, Three-Level Scorer, …) from
  measured percentiles rather than a hand-typed table.
- A **Stat Engine** — a SofaScore-style browser over every stat the pipeline
  collects, for every player and team the league has fielded since 2010.

## Architecture, end to end

Five stages, each owning one job. A shot's features are computed identically
whether the caller is a training run building two million rows or the API
serving one hypothetical location — the same function runs both times, which
is what makes the two paths provably consistent rather than hopefully so.

```
Ingestion  →  Feature Build  →  Modeling  →  Serving  →  Frontend
28 scripts    DuckDB scans;      2 XGBoost    FastAPI      React/TanStack
across 14     every rate is      models,      recommender  matchup engine,
NBA Stats     point-in-time      monotone     rebuilds     stat browser,
endpoints     and Beta-shrunk    constraints, features     player compare
into SQLite                      calibration, the same
                                  backtesting  way training
                                               did, + a
                                               TreeSHAP
                                               explanation
                                               layer
```

## Data & ingestion

Every shot the league has recorded since 2010–11 — location, distance, zone,
shot clock context, game state — joined against play-by-play (assists,
putbacks, transition), SportVU/Synergy tracking (touch time, dribbles,
closest-defender distance, play-type frequency and efficiency), full box
scores, hustle stats, and defender-quality splits by shot category. Physical
measurements come from three independent sources (the NBA's own combine data,
Basketball-Reference, and a 2K-ratings scrape used as a rough independent
sanity check) and are reconciled rather than trusted from one place.

Endpoints: `ShotChartDetail`, `PlayByPlayV3`, `LeagueDashPlayerStats`,
`LeagueHustleStatsPlayer`, `LeagueDashPtDefend`, `LeagueDashPtStats`,
`LeagueDashPlayerPtShot`, `PlayerDashboardByShootingSplits`,
`SynergyPlayTypes`, `BoxScoreMatchupsV3`, `CommonPlayerInfo`,
`CommonTeamRoster`, `DraftCombineStats`, `LeagueGameLog`.

## The feature layer — leakage is the whole design constraint

The project's central engineering discipline: **a shot's own outcome must
never be reachable from its own features.** The original pipeline joined
whole-season aggregates onto every shot in that season — a corner-three
feature built from 20 attempts had roughly 5% of its own answer baked into
the "evidence." Every rate in the current pipeline is instead computed from
games strictly before the one being predicted, at career-to-date and
season-to-date horizons, and regressed toward an empirically-fit Beta prior
so a rookie's third attempt doesn't read as a 100% shooter.

- **Empirical-Bayes shrinkage** — a Beta prior's concentration is fit from
  the population's own true-talent spread (sampling noise subtracted out),
  not hand-picked, so a thin sample regresses hard and a deep one barely
  moves.
- **Defender quality, point-in-time, by shot category** — FG%-allowed vs.
  league-normal, five categories × six zones, computed the same
  prior-games-only way as the shooter's own rates.
- **Lineup context, reconstructed from substitutions** — all ten players on
  the floor at the moment of each shot, replayed from play-by-play
  substitution events, not just the shooter and primary defender.
- **Supporting cast, leave-one-out** — a shooter's own numbers are excluded
  from his team's aggregate before it's used as a feature, so a star's
  gravity is never measured by re-including himself.
- **Contest & creation** — closest-defender-distance bands and
  self-created-vs-assisted shares, accumulated shot-by-shot rather than read
  off a season total.
- **Clutch performance** — a player's career clutch FG% (last 5 minutes,
  margin ≤5 — the league's own definition) vs. his normal baseline, engaged
  only on shots that are themselves clutch.

## Modeling

Two purpose-built XGBoost models answer two different questions that a single
"shot quality" number conflates if left alone: whether a shot is a *good*
shot, and whether this player would ever *take* it.

| Model | Question it answers | Type | Features | Test AUC |
|---|---|---|---|---|
| Shot Quality (v20) | Given the shot is taken, does it go in? | XGBoost, `binary:logistic` | 120 | 0.707 |
| Attainability | How often does this player actually get a look like this? | XGBoost, regressor | — | — |

- **Monotone constraints** — 17 features are locked to a basketball-true
  direction (a better shooter never predicts worse; a better defender never
  predicts better) — both an honesty check and protection against the
  recommender sweeping into a spurious local quirk.
- **Calibration, adopted only when it wins** — a held-out calibration mapping
  is tested against the raw model's own log-loss/ECE on unseen shots and
  rejected outright if it doesn't actually improve it; v20's raw output
  already beat its own calibrated version.
- **Backtesting, ablation, and R** — every feature group can be dropped and
  re-measured in isolation; a separate R script re-derives the
  defender-quality signal's significance via logistic regression + a
  likelihood-ratio test, a check that owes nothing to gradient boosting.
- **20 tracked model versions** — every training run writes a timestamped
  manifest recording exactly what it scored; nothing is overwritten, so a
  regression is always traceable to the run that caused it.

## Explainability

A prediction with no reason attached is a black box wearing a percentage
sign. Every served shot is decomposed with **TreeSHAP** into four buckets —
offense (the shooter's own profile and teammates), defense (the named
defender and help), matchup interaction, and game context (score, clock,
clutch) — each converted from log-odds into an honest marginal probability
effect, then woven into a plain-English narrative naming the actual top
drivers by name and value, not just a bar chart of feature weights.

The same measured-percentile discipline produces the **archetype system**:
21 traits and 26 archetype definitions (Scoring Guard, 3&D Wing, Rim
Protector, Three-Level Scorer, …), each a league-percentile combination of
named stats with position and, where the label makes a physical claim,
height gates — so "why is this player a Rim Protector" is always answerable
by pointing at the same numbers the label came from.

## Serving layer

A **FastAPI** application (27 routes across 13 domain-split router modules)
exposes the recommender, the explanation engine, the stat browser's
leaderboards and profiles, the archetype catalogue, and a team-matchup
endpoint. The recommender rebuilds a shot's feature vector through the exact
same assembly function the training pipeline uses — the single piece of
discipline that keeps a served prediction and an offline metric describing
the same model.

## Frontend

A **React 19 / TanStack Start** single-page app with three surfaces:

- The **Shot Engine** — a court-heatmap matchup tool with team-themed
  dynamic accents and a live SHAP explanation panel.
- The **Stat Engine** — a leaderboard-and-profile browser with a toggleable
  visual (radar chart + compact bar charts) or list density.
- **Compare** — up to three players on one radar.

Styling is hand-built (no component-kit defaults) over Tailwind and Radix
primitives, dark-themed with an NBA-brand accent treatment.

## Engineering discipline

255 automated tests across 26 files, with a whole test file
(`test_train_serve_parity.py`) dedicated to proving the training and serving
code paths compute identical features from identical inputs — the exact
class of bug (a feature populated in training but left null at serving time)
that has caused this project's worst real regressions. A dedicated leakage
test suite asserts that no feature reachable at serving time could see its
own shot's outcome. The codebase is organized module-by-module rather than
as a handful of monoliths — the largest single file, a 1,701-line
point-in-time feature module, was split into eight focused files with 100%
backward-compatible re-exports, specifically so any future change is easy to
locate.

## Technology inventory

**Data & ML** — Python 3, XGBoost, scikit-learn, SciPy, pandas/NumPy, DuckDB, R

**Backend & data** — SQLAlchemy 2 / SQLite, FastAPI, Pydantic, Uvicorn,
`nba_api`, BeautifulSoup/lxml

**Frontend** — React 19, TanStack Start/Router, Vite, Tailwind CSS 4, Radix
UI, Recharts / hand-rolled SVG

**Testing & tooling** — pytest, Playwright, Ruff, Kafka (optional streaming
demo), Apache Spark (parity-checked point-in-time job), joblib/tqdm
