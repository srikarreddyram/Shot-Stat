# PRD Checklist: NBA Shot Quality Engine — Online Recommendation System

**Version:** 2.0  
**Type:** Real-Time ML Recommendation System

---

## 1. Overview & Problem Statement

The system predicts make probability for any NBA shot given **attacker + defender + game state**, scores every viable court location, and returns the highest-EP shot recommendation. It retrains nightly on fresh data to stay current.

**Core question the model answers:** "Given who is guarding them, where they are in the game, and how they've been shooting lately — where should this player attack?"

---

## 2. Goals
- [ ] Ingest 12–15 seasons of NBA shot data enriched with player, defender, and game context
- [ ] Train a model that predicts make probability for any shot given attacker + defender + game state
- [ ] At inference time, score all viable court locations and return a ranked recommendation
- [ ] Retrain the model nightly as new game data arrives, keeping predictions current
- [ ] Serve predictions via a lightweight API

## 3. Non-Goals
- Live optical tracking / real-time defender coordinate ingestion (requires Hawk-Eye / Second Spectrum)
- Team-level play-calling optimization (individual player recommendations only)
- Native mobile or frontend application (API only in v1)
- True online learning / per-shot weight updates (scheduled nightly retraining instead)

---

## 4. Data Sources

### 4.1 NBA Stats API (`nba_api`)
- [x] `ShotChartDetail` — Shot coordinates, outcome, distance, shot type, game context
- [x] `CommonPlayerInfo` — Height, weight (⚠️ **returns current listed values only**, not season-specific; see Known Risks)
- [x] `DraftCombineStats` — Wingspan (one-time measurement at draft)
- [x] `LeagueDashPlayerStats` — Season FG%, position, games played
- [ ] `PlayerDashboardByShootingSplits` — Zone-level FG% splits per player-season (rim, paint, mid-range, corner 3, above-the-break 3)
- [ ] `SynergyPlayTypes` — Pull-up vs. catch-and-shoot vs. post-up tendencies
- [ ] `DefenseDashboardPtDefend` — Defender FG% allowed by zone and shot type
- [ ] `PlayerDefenseDashboard` — Contest rate, defensive rating per defender
- [ ] `PlayerGameLogs` — Minutes played (fatigue proxy computation)
- [ ] Tracking endpoints — Touch time, dribbles per shot, closest defender distance (2013–14 onward)

### 4.2 Basketball Reference (Scraped)
- [x] Height, weight, wingspan fallback (strict mode — NO imputation allowed)
- [x] Tracking physicals not found on `nba_api`

### 4.3 NBA 2K Ratings (Scraped — 2kratings.com)
- [x] Wingspan final fallback for players missing from NBA API and BRef
- [x] Height/weight gap-fill (only updates NULL columns, never overwrites official data)

### 4.4 Pre-Game Matchup Assignment
- [ ] `nba_api` → `DefenseDashboardPtDefend` (which defender guarded which offensive player)
- [ ] Supplemented by play-by-play matchup inference where available

---

## 5. Physical Matchup Features (Derived at Inference Time)

| Feature | Formula | Relevance |
|---|---|---|
| `height_diff` | `attacker_height − defender_height` | Rim attempts, post shots |
| `wingspan_diff` | `attacker_wingspan − defender_wingspan` | Shot release angle, contest effectiveness |
| `reach_advantage` | `(height_diff × 0.6) + (wingspan_diff × 0.4)` | Composite physical edge |
| `size_mismatch_flag` | `1 if abs(height_diff) > 4 inches` | Significant positional mismatches |

> **Strict Data Policy:** We DO NOT impute physical attributes. All data must be exact measurements from `nba_api`, Basketball Reference, or NBA 2K Ratings. If not found in any source, the value remains `NULL` to avoid introducing noise to the model.

---

## 6. Data Schema

### 6.1 Shots Table
- [ ] `shot_id` — string, unique identifier
- [ ] `game_id` — string, FK → Games
- [ ] `player_id` — string, FK → Players (attacker)
- [ ] `defender_id` — string, FK → Players (defender)
- [ ] `shot_made` — int (0/1), outcome
- [ ] `loc_x`, `loc_y` — float, court coordinates
- [ ] `shot_distance` — float, feet from basket
- [ ] `shot_type` — string, 2PT / 3PT
- [ ] `zone` — string, classified court zone
- [ ] `shot_angle` — float, derived from coordinates
- [ ] `quarter` — int, 1–4 or OT
- [ ] `time_remaining` — float, seconds left in period
- [ ] `score_diff` — int, attacker team − opponent
- [ ] `home_away` — int (0/1)
- [ ] `playoff_flag` — int (0/1)
- [ ] `touch_time` — float, seconds holding ball (if available)
- [ ] `dribbles` — int, dribbles before shot (if available)
- [ ] `catch_and_shoot` — int (0/1), (if available)
- [ ] `closest_defender_dist` — float, feet (if available, 2013–14+)

### 6.2 Players Table (Covers Both Attackers and Defenders)
- [ ] `player_id` — string, unique identifier
- [ ] `season` — string, e.g. `2023-24`
- [ ] `name` — string
- [ ] `height` — float, inches
- [ ] `weight` — float, lbs
- [ ] `wingspan` — float, inches
- [ ] `wingspan_source` — string, e.g., "NBA_API", "BREF", or NULL
- [ ] `shots_fetched_reg` — boolean
- [ ] `shots_fetched_ply` — boolean
- [ ] `position` — string, PG / SG / SF / PF / C
- [ ] `career_fg_pct` — float, as of this season
- [ ] `career_3p_pct` — float, as of this season
- [ ] `season_fg_pct` — float, current season
- [ ] `def_rating` — float, defensive rating (defenders)
- [ ] `contest_rate` — float, % of opponent shots contested
- [ ] `def_fg_pct_allowed` — float, opponent FG% when this player defends

### 6.3 Games Table
- [x] `game_id` — string
- [x] `date` — date
- [x] `home_team` — string
- [x] `away_team` — string
- [x] `playoff_flag` — int (0/1)
- [x] `home_team_win` — int (0/1)

### 6.4 PlayerZoneStats Table (NEW — Zone-Level Shooting Efficiency)
One row per player per season per zone. Captures where each player is deadly vs. where they struggle (e.g., Shaun Livingston's elite mid-range, Steph Curry's above-the-break 3).

- [ ] `player_id` — string, PK (part 1)
- [ ] `season` — string, PK (part 2)
- [ ] `zone` — string, PK (part 3) — one of: `Restricted Area`, `In The Paint (Non-RA)`, `Mid-Range`, `Left Corner 3`, `Right Corner 3`, `Above the Break 3`
- [ ] `fgm` — int, field goals made in this zone
- [ ] `fga` — int, field goal attempts in this zone
- [ ] `fg_pct` — float, FG% in this zone
- [ ] `fg3m` — int, 3-point field goals made (non-zero only for 3PT zones)
- [ ] `fg3a` — int, 3-point field goal attempts
- [ ] `fg3_pct` — float, 3PT% (non-zero only for 3PT zones)

**Source:** `PlayerDashboardByShootingSplits` → DataFrame 3 ("Shot Area")

---

## 7. Feature Engineering

### 7.1 Spatial Features
- [ ] `shot_angle` → `atan2(loc_y, loc_x)`
- [ ] `distance_from_center` → Euclidean from basket origin
- [ ] `zone` → Rule-based classification (Restricted Area, Paint Non-RA, Mid-Range, Left Corner 3, Right Corner 3, Above the Break 3)

### 7.2 Game Context Features
- [ ] `score_diff`, `time_remaining`, `quarter`, `home_away`, `playoff_flag`

### 7.3 Player Form (Leakage-Safe Rolling Windows)
- [ ] `recent_10_fg` — rolling FG% across last 10 games (strictly before current game date)
- [ ] `recent_20_fg` — same, 20-game window
- [ ] `fatigue_proxy` — total minutes played in the last 5 calendar days

### 7.4 Physical Matchup
- [ ] `height_diff`, `wingspan_diff`, `reach_advantage`, `size_mismatch_flag`

### 7.5 Defender Tendencies
- [ ] `def_fg_pct_allowed_by_zone` — how this specific defender performs against shots in each zone
- [ ] `contest_rate` — how frequently they close out
- [ ] `def_rating` — overall defensive quality signal

---

## 8. Recommendation Engine (Inference)

**Input:** `player_id, defender_id, quarter, time_remaining, score_diff, home_away`

**Process:**
- [ ] Generate a grid of candidate shot locations covering all viable court zones
- [ ] For each candidate, construct the full feature vector (spatial + context + player + defender + matchup)
- [ ] Run the model over all candidates in a single batch
- [ ] Return the top-N shot recommendations ranked by Expected Points (EP)

**Outputs per candidate shot:**
- [ ] `make_probability` — P(shot goes in), 0.0 to 1.0
- [ ] `expected_points` — `make_probability × shot_value`
- [ ] `shot_quality_score` — Opportunity quality index, 0 to 100
- [ ] `difficulty_score` — `100 − shot_quality_score`
- [ ] `recommended_zone` — Top-ranked zone for this matchup

---

## 9. Model

- [ ] **Training objective:** Binary cross-entropy (log-loss) on shot outcome
- [ ] **Primary model:** XGBoost (gradient boosted trees)
- [ ] **Baseline model:** Logistic Regression (interpretable)
- [ ] **Train/test split:** Strict temporal — train on seasons 1→N, test on season N+1. No random splits.
- [ ] **Evaluation metric:** Log-loss on held-out season. Must beat zone-average FG% baseline.
- [ ] **Nightly retraining:**
  - [ ] Triggered by scheduled job after each game day
  - [ ] New shots ingested → rolling features recomputed → XGBoost retrained with warm start
  - [ ] New model promoted only if log-loss on recent holdout window improves or holds within tolerance

---

## 10. Phased Rollout

### Phase 1 — Data Collection (Strict Mode)
- [x] Pull 15+ seasons (2010–2026) of game metadata via `LeagueGameLog`
- [x] Build base rosters via `LeagueDashPlayerStats`
- [x] **Physicals (Phase 2A):** Pull height, weight, wingspan from `nba_api` (CommonPlayerInfo + DraftCombine)
- [x] **Physicals (Phase 2B):** Scrape missing physicals from Basketball Reference
- [x] **Physicals (Phase 2C):** Scrape remaining wingspans from 2kratings.com (NBA 2K data)
- [ ] **Zone Stats:** Pull per-player zone-level FG% via `PlayerDashboardByShootingSplits` (rim finishing, paint, mid-range, corner 3, above-the-break 3)
- [ ] **Career Stats:** Pull career FG% and 3PT% via `LeagueDashPlayerStats` (CareerTotals)
- [ ] **Verification:** Generate missing data report
- [x] Pull shot data via `ShotChartDetail` (ONLY after physicals are complete)
- [x] Build SQLite schema with idempotent upserts and strict resume flags
- [ ] Train baseline LR and XGBoost models **without** defender data
- [ ] Validate log-loss beats zone-average baseline

> ⚠️ **Data Policy:** No position averages or height-based estimates are allowed. We use exact measurements or `NULL`.

### Phase 2 — Defender Integration

#### 2A. Season-Level Defender Stats (NEW TABLE: `defender_stats`)
Pull season-average defensive metrics for every player. These don't tell us *who guarded a specific shot*, but they quantify how good each player is as a defender overall.

**Source:** `LeagueDashPtDefend` endpoint (nba_api)
**Granularity:** One row per player per season per defense category

| Column | Type | Source |
|---|---|---|
| `player_id` | string, PK | endpoint |
| `season` | string, PK | endpoint |
| `defense_category` | string, PK | "Overall", "3 Pointers", "2 Pointers", "Less Than 6Ft", "Less Than 10Ft", "Greater Than 15Ft" |
| `freq` | float | D_FGA frequency |
| `dfg_pct` | float | FG% allowed when this player defends |
| `fg_pct_diff` | float | FG% diff vs. normal (negative = good defense) |
| `d_fga` | int | Number of field goal attempts defended |

Tasks:
- [ ] Create `DefenderStats` model in `models.py`
- [ ] Build `defender_stats_ingestor.py` using `LeagueDashPtDefend`
- [ ] Pull for all 15 seasons (rate-limited, resumable)
- [ ] Per-season sanity check: verify all active players have defender stats

#### 2B. Game-Level Matchup Data (NEW TABLE: `matchups`)
Figure out *who guarded who* in each game so we can link defenders to shots.

**Source:** `BoxScoreMatchupsV3` endpoint (nba_api)
**Granularity:** One row per offensive-player / defensive-player pair per game
**Availability:** 2016-17 onward (not available for earlier seasons)

| Column | Type | Source |
|---|---|---|
| `game_id` | string, PK | endpoint |
| `offense_player_id` | string, PK | endpoint |
| `defense_player_id` | string, PK | endpoint |
| `matchup_minutes` | float | partial possessions guarded |
| `player_pts` | int | points scored in this matchup |
| `matchup_fg_pct` | float | FG% in this specific matchup |
| `matchup_fga` | int | FGAs in this matchup |

Tasks:
- [ ] Create `Matchup` model in `models.py`
- [ ] Build `matchup_ingestor.py` using `BoxScoreMatchupsV3`
- [ ] Pull for every game from 2016-17 onward (~15k games)
- [ ] Join matchups to shots: for each shot, find the defender who guarded the shooter for the most minutes in that game
- [ ] Populate `defender_id` in the Shots table

#### 2C. Defender Features for Model
- [ ] Join `defender_id` → `defender_stats` to get per-shot defender quality metrics
- [ ] Add physical matchup features (`height_diff`, `wingspan_diff`, `reach_advantage`, `size_mismatch_flag`)
- [ ] Add defender tendency features (`dfg_pct` overall, `dfg_pct` by zone, `fg_pct_diff`)
- [ ] Retrain XGBoost and measure lift vs. Phase 1 attacker-only model
- [ ] For pre-2016-17 seasons (no matchup data): fall back to team-level defensive ratings

### Phase 3 — Online System
- [ ] Build nightly ingestion + retraining pipeline
- [ ] Wrap model in FastAPI serving layer
- [ ] Implement recommendation engine (grid scoring → top-N by EP)
- [ ] Add rolling form features with proper leakage guards

### Phase 4 — Tracking Data (Stretch)
- [ ] Ingest closest defender distance, touch time, dribbles from `nba_api` tracking endpoints (2013–14 onward)
- [ ] Evaluate lift; backfill historical shots where available

---

## 11. Known Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| No live defender coordinates | Can't know exact defender position in real-time | Use pre-game matchup assignment + defensive tendency features as proxy |
| **Weight data not season-specific** | Player weight changes over career (e.g., Giannis 196→243 lbs) but we only have current listed weight from `CommonPlayerInfo` | Accept as reasonable proxy; height is accurate and more predictive of shot mechanics than weight |
| Wingspan data sparse pre-2010 | Missing physical matchup signal | Model must be robust to `NULL` values. We strictly do not impute to avoid noise. |
| `nba_api` rate limiting | Slow historical ingestion | Exponential backoff; incremental daily pulls after bootstrap |
| Rolling feature leakage | Inflated evaluation metrics | Strictly enforce lookback window excludes current game |
| Model staleness mid-season | Recommendations miss recent form | Nightly retraining with warm start addresses this |
| Defender assignment errors | Wrong physical matchup features | Flag low-confidence matchup assignments; fall back to positional average |

---

## 12. Success Criteria
- [ ] Model log-loss meaningfully lower than zone-average baseline on held-out test season
- [ ] Defender features (physical matchup + tendency) provide measurable lift over attacker-only model
- [ ] Recommendation engine returns a ranked shot chart for any valid `(player_id, defender_id, game_state)` input within 200ms
- [ ] Nightly retraining pipeline runs without manual intervention
- [ ] Rolling features computed correctly with zero data leakage verified on test set

---

## 13. Open Questions (Pre-Implementation)

- [ ] **Defender assignment confidence:** Accept aggregate defender tendency as proxy, or cross-reference play-by-play for shot-level defender identity in Phase 2?
- [ ] **Wingspan imputation cutoff:** Train only on 2013–14 onward (cleaner + tracking data), or go back to 2010+ and accept more imputed values?
- [ ] **Court grid resolution:** Coarse grid (~6 zone-level candidates, fast & interpretable) or fine grid (50–100 coordinate points, smoother heatmap)?
