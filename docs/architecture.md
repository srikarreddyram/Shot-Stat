# System Architecture: NBA Shot Quality Engine — Online Recommendation System

**Version:** 2.0

---

## 1. Overview

The system has three runtime modes:

| Mode | Description |
|---|---|
| **Historical Bootstrap** | One-time ingestion of 12–15 seasons of shot, player, and game data |
| **Nightly Update** | Incremental ingestion of yesterday's games + model retraining with warm start |
| **Real-Time Inference** | Recommendation API — scores a grid of court locations for a given matchup |

These are separate pipelines sharing the same database and model registry.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                  DATA SOURCES                       │
│  nba_api (ShotChartDetail, DefenseDashboard,        │
│           Tracking, GameLogs, SynergyPlayTypes)     │
│  Basketball Reference (Height, Weight, Wingspan)    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              INGESTION LAYER                        │
│  bootstrap_ingestor.py  (one-time, 12-15 seasons)  │
│  nightly_ingestor.py    (scheduled, yesterday's     │
│                          games only)                │
│  Rate-limit backoff · Idempotent upserts            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              STORAGE (SQLite → PostgreSQL)          │
│  Games · Players (per season) · Shots + defender_id │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│          FEATURE ENGINEERING PIPELINE               │
│  Spatial: zone, angle, distance_from_center         │
│  Context: score_diff, time_remaining, home_away     │
│  Rolling: recent_10_fg, recent_20_fg (leakage-safe) │
│  Matchup: height_diff, wingspan_diff, reach_adv     │
│  Defender: def_fg_pct_allowed_by_zone, contest_rate │
└────────────────────┬────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐   ┌─────────────────────────────┐
│ TRAINING PIPELINE│   │     INFERENCE PIPELINE      │
│                  │   │                             │
│  Temporal split  │   │  FastAPI endpoint           │
│  Baseline LR     │   │  Input: player_id,          │
│  XGBoost         │   │         defender_id,        │
│  MLflow tracking │   │         game_state          │
│  Log-loss eval   │   │                             │
│  Model registry  │   │  Grid of court locations    │
│  Nightly warm    │   │  → batch feature vectors    │
│  start retrain   │   │  → XGBoost batch inference  │
└────────┬─────────┘   │  → ranked by EP             │
         │             │  → top-N zones returned     │
         └─────────────┤                             │
                       │  Response: make_prob, EP,   │
                       │  shot_quality, difficulty,  │
                       │  recommended_zone           │
                       └─────────────────────────────┘
```

---

## 3. Component Design

### 3.1 Data Ingestion & Storage

**nba_api Integration:**
- Python scripts polling `ShotChartDetail`, `DefenseDashboardPtDefend`, `PlayerDashboardByShootingSplits`, `SynergyPlayTypes`, `PlayerDefenseDashboard`, `PlayerGameLogs`, and tracking endpoints.
- 12–15 seasons of historical data during bootstrap.
- Nightly incremental pulls after bootstrap.
- Rate-limit handling via exponential backoff.

**Web Scraper:**
- `requests` + `BeautifulSoup` for Basketball Reference. No Selenium.
- Targets: height, weight, wingspan (combine data), positional designation.
- Cache all responses aggressively to avoid rate limits.

**Storage:**
- **Dev:** SQLite via `SQLAlchemy`.
- **Prod:** PostgreSQL via `SQLAlchemy` — same schema, swap connection string.
- **Tables:**
  - `Games` — `game_id`, `date`, `home_team`, `away_team`, `playoff_flag`
  - `Players` (per-season) — `player_id`, `season`, `name`, `height`, `weight`, `wingspan`, `position`, `career_fg_pct`, `career_3p_pct`, `season_fg_pct`, `def_rating`, `contest_rate`, `def_fg_pct_allowed`
  - `Shots` — `shot_id`, `game_id`, `player_id`, `defender_id`, `shot_made`, `loc_x`, `loc_y`, `shot_distance`, `shot_type`, `zone`, `shot_angle`, `quarter`, `time_remaining`, `score_diff`, `home_away`, `playoff_flag`, `touch_time`, `dribbles`, `catch_and_shoot`, `closest_defender_dist`

### 3.2 Feature Engineering Pipeline (Pandas)

| Category | Features | Notes |
|---|---|---|
| **Spatial** | `shot_angle`, `distance_from_center`, `zone` | `zone` is rule-based from coordinates |
| **Game Context** | `score_diff`, `time_remaining`, `quarter`, `home_away`, `playoff_flag` | Direct or simple derivations |
| **Player Form** | `recent_10_fg`, `recent_20_fg`, `fatigue_proxy` | Leakage-safe: strictly prior games only |
| **Physical Matchup** | `height_diff`, `wingspan_diff`, `reach_advantage`, `size_mismatch_flag` | Derived at inference time from Players table |
| **Defender Tendencies** | `def_fg_pct_allowed_by_zone`, `contest_rate`, `def_rating` | From `DefenseDashboard` endpoints |

### 3.3 Training Pipeline

- **Temporal split:** Train on seasons 1→N, test on season N+1. No random splits.
- **Baseline:** Zone-level historical FG% averages.
- **Logistic Regression:** Interpretable baseline ML model.
- **XGBoost:** Primary model. Handles non-linear interactions (zone × time × player ability × defender quality).
- **Evaluation:** Log-loss. Must beat zone-average baseline.
- **Experiment tracking:** MLflow — hyperparameters, metrics, model artifacts.
- **Nightly warm start:** Retrain on updated dataset using previous model as starting point. Promote only if log-loss holds or improves.

### 3.4 Inference Pipeline (Recommendation Engine)

**Input:** `player_id`, `defender_id`, `quarter`, `time_remaining`, `score_diff`, `home_away`

**Process:**
1. Generate a grid of candidate shot locations covering all viable court zones.
2. For each candidate, construct the full feature vector (spatial + context + player + defender + matchup).
3. Run XGBoost over all candidates in a single batch.
4. Return top-N recommendations ranked by **Expected Points (EP)**.

**Output per candidate:**

| Field | Description |
|---|---|
| `make_probability` | P(shot goes in), 0.0–1.0 |
| `expected_points` | `make_probability × shot_value` |
| `shot_quality_score` | 0–100 index |
| `difficulty_score` | `100 − shot_quality_score` |
| `recommended_zone` | Top-ranked zone for this matchup |

---

## 4. Key Design Decisions

**Why XGBoost over a neural network?**
Tree models handle tabular sports data extremely well and are interpretable enough to debug. Inference is microseconds per shot, which matters when scoring a grid of 50+ court locations per request. A neural net would offer marginal lift at the cost of much harder debugging and serving.

**Why SQLite → PostgreSQL?**
SQLite is fine for local development and the bootstrap phase. Once nightly writes + concurrent API reads happen simultaneously, PostgreSQL handles locking correctly. The schema is identical — it's a one-flag change in SQLAlchemy.

**Why nightly retraining over true online learning?**
True online learning (updating model weights per shot) is complex to implement correctly for tree models and adds significant engineering risk. Nightly retraining on the full updated dataset with a warm start captures recent form reliably, is easy to test, and is simple to roll back if a bad model gets promoted.

**Why FastAPI over a heavier serving framework?**
The model is XGBoost loaded in memory. Inference for a full court grid takes under 10ms. FastAPI with a single worker is sufficient for this use case and keeps the stack simple.

---

## 5. Nightly Pipeline Sequence

```
11:30 PM  → Ingest yesterday's completed games (shots, game metadata)
11:45 PM  → Upsert new player season stats from nba_api
12:00 AM  → Recompute rolling features for affected players
12:15 AM  → Retrain XGBoost (warm start from current production model)
12:30 AM  → Evaluate on rolling 30-day holdout window
12:35 AM  → If log-loss ≤ current model + tolerance: promote to production
           → Else: keep current model, log alert
12:40 AM  → Done
```

---

## 6. Full Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Data collection | `nba_api` | Primary shot + matchup source |
| Scraping | `requests` + `BeautifulSoup` | Physical attributes only, no Selenium |
| Storage (dev) | SQLite via `SQLAlchemy` | — |
| Storage (prod) | PostgreSQL via `SQLAlchemy` | Same schema, swap connection string |
| Processing | `pandas` | Feature engineering |
| ML | `xgboost`, `scikit-learn` | — |
| Experiment tracking | `MLflow` | Hyperparams, metrics, model artifacts |
| Scheduling | `APScheduler` or cron | Nightly pipeline trigger |
| Serving | `FastAPI` | Recommendation endpoint |
| Visualization | `Plotly` | Shot charts, EP heatmaps |
| Language | Python 3.10+ | — |

---

## 7. Phased Implementation

### Phase 1 — Offline Foundation (MVP)
- Pull 12–15 seasons of shot data via `nba_api`
- Scrape player physical attributes from Basketball Reference
- Build SQLite schema, joins, and core spatial/game features
- Train baseline LR and XGBoost models without defender data
- Validate log-loss beats zone-average baseline

### Phase 2 — Defender Integration
- Add `defender_id` to Shots table
- Ingest defensive matchup data (`DefenseDashboardPtDefend`)
- Add physical matchup features (`height_diff`, `wingspan_diff`, `reach_advantage`)
- Add defender tendency features (`def_fg_pct_allowed_by_zone`, `contest_rate`)
- Retrain and evaluate model lift from defender features

### Phase 3 — Online System
- Build nightly ingestion + retraining pipeline
- Wrap model in FastAPI serving layer
- Implement recommendation engine (grid scoring → top-N by EP)
- Add rolling form features with proper leakage guards

### Phase 4 — Tracking Data (Stretch)
- Ingest closest defender distance, touch time, dribbles from `nba_api` tracking endpoints (2013–14 onward)
- Evaluate lift; backfill historical shots where available

---

## 8. Known Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| No live defender coordinates | Can't know exact defender position in real-time | Use pre-game matchup assignment + defensive tendency features as proxy |
| Wingspan data sparse pre-2010 | Missing physical matchup signal | Impute from position-level averages; flag imputed records |
| `nba_api` rate limiting | Slow historical ingestion | Exponential backoff; incremental daily pulls after bootstrap |
| Rolling feature leakage | Inflated evaluation metrics | Strictly enforce lookback window excludes current game |
| Model staleness mid-season | Recommendations miss recent form | Nightly retraining with warm start addresses this |
| Defender assignment errors | Wrong physical matchup features | Flag low-confidence matchup assignments; fall back to positional average |
