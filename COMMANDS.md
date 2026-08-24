# NBA Shot Quality Engine — Commands Reference

All commands should be run from the project root:
```bash
cd /Users/tejsr/Projects/NBA_Shot_Predictor
```

> **Note on the redesign.** The training and inference pipeline was rebuilt
> around a shared feature layer (`src/features/`). See [docs/redesign.md](docs/redesign.md)
> for what changed and why. The old entry points (`train_baseline.py`,
> `feature_engineering.py`, `position_priors.py`) are deprecated and documented
> as such in their module docstrings.

---

## The current pipeline

```bash
# 1. Creation-skill data (handle + passing). ~1 min and ~10 min respectively.
python -m src.ingestion.tracking_ingestor --full
python -m src.ingestion.shot_profile_ingestor --full

# 2. Per-shot play-by-play context (shot mechanics, putbacks, transition).
#    ~1.2 hours at 6 workers. Resumable — re-run it and it picks up where it
#    stopped, newest seasons first.
python -m src.ingestion.pbp_ingestor --seasons 2016-17+

# 3. Train the shot-quality model. Writes models/ + runs/<stamp>__<name>/
python -m src.training.train --name shot-quality-v5

# 4. Train the attainability model ("can this player GET this shot?")
python -m src.training.attainability --name attainability

# 5. Serve
uvicorn src.inference.api:app --reload --port 8000
```

### Keeping the laptop cool

Training defaults to every core, which on a laptop is a real thermal event.
`XGB_N_JOBS` caps it:

```bash
XGB_N_JOBS=6 python -m src.training.train --name shot-quality-v5
```

The play-by-play ingest is network-bound and barely touches the CPU, so it is
safe to leave running.

### Evaluating honestly

```bash
# The full accuracy argument, with fresh numbers — calibration table, Murphy
# decomposition, baselines, and what the model still misses.
python -m src.training.accuracy_report

# Rolling-origin backtest — several held-out seasons, not one
python -m src.training.backtest --origins 3

# Plus feature-group ablations (what is each group actually worth?)
python -m src.training.backtest --origins 3 --ablate

# Ablate a single group during training
python -m src.training.train --no-creation --name ablation-no-creation
python -m src.training.train --no-defender --name ablation-no-defender

# Train the old interior/perimeter split for comparison
python -m src.training.train --split --name split-model

# Feature groups held out by default because they measured WORSE.
# Both flags re-enable them for re-measurement — see docs/redesign.md.
python -m src.training.train --defender-physicals --name with-physicals
python -m src.training.train --spatial-basis --name with-basis

# Hyperparameter search over the CURRENT features (the old best_params_v3.json
# was tuned against the pre-rewrite leaky features and is unused).
python -m src.training.tune_hyperparams --trials 30 --sample-frac 0.45

# Hierarchical player/defender effects on top of a trained model.
# Measured: no gain — the search prefers maximum shrinkage.
python -m src.training.player_effects --model shot-quality-v5
```

### Calibration modes

```bash
# recent (default) — fit on the latest unseen data, adopt only if it helps
python -m src.training.train --calibration recent

# oof — chronological out-of-fold over the fit window, adopted unconditionally
python -m src.training.train --calibration oof --folds 3

# none — skip entirely
python -m src.training.train --calibration none
```

### Inspecting runs

```bash
# Recent training runs with their headline metrics
python -m src.common.runs

# One run's full record
cat runs/<stamp>__<name>/run.json
```

### Tests

```bash
pytest                      # everything
pytest -m "not slow"        # skip tests needing trained models + the full DB
pytest tests/test_train_serve_parity.py    # the train/serve skew guard
pytest tests/test_no_leaky_features.py    # keeps is_assisted out of the model
pytest tests/test_pbp_extraction.py       # play-by-play parsing, no network
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize the database (creates tables, no data)
python -c "from src.db.database import init_db; init_db()"
```

---

## Data Ingestion

### Single Season Test (2023-24)
```bash
# Run the full pipeline for one season (games → players → shots → validate → CSV export)
python -m src.ingestion.bootstrap
```

### Full 15-Season Bootstrap (overnight run)
```bash
# Run for all seasons 2010-11 through 2024-25 (takes several hours)
python -m src.ingestion.bootstrap --full
```

### Run Individual Steps
```bash
# Games only (fastest — ~1 min per season)
python -m src.ingestion.game_ingestor                # single season
python -m src.ingestion.game_ingestor --full          # all seasons

# Player stats + physical attributes (~10 min per season)
python -m src.ingestion.player_stats_ingestor         # single season
python -m src.ingestion.player_stats_ingestor --full   # all seasons

# Shot chart data (heaviest — ~10-15 min per season)
python -m src.ingestion.shot_ingestor                 # single season
python -m src.ingestion.shot_ingestor --full           # all seasons
```

---

## Data Export & Inspection

```bash
# Export all tables to CSV (output → data/csv/)
python -m src.ingestion.export_csv

# Export a specific table
python -m src.ingestion.export_csv --table shots
python -m src.ingestion.export_csv --table players
python -m src.ingestion.export_csv --table games
```

CSV files are saved to `data/csv/`:
- `games.csv` — All game metadata
- `players.csv` — Player-season records with physical attributes
- `shots.csv` — Every shot attempt
- `shots_sample_500.csv` — 500 shots with player names (for quick look)
- `zone_summary.csv` — FG% breakdown by court zone
- `top_players_by_volume.csv` — Top 50 shooters by volume

---

## Background Logs

If a script is running in the background, you can watch its live progress (including progress bars) using `tail -f`:

```bash
tail -f /Users/tejsr/.gemini/antigravity-ide/brain/d3e080e7-6699-46f9-8bd2-20256d2b9108/.system_generated/tasks/task-162.log
```
*(Press `Ctrl+C` to stop watching the log. The background task will keep running).*

---

## Stopping & Resuming

If you need to stop a massive scraping run (like the 15-season bootstrap), you can safely cancel it at any time (e.g., `Ctrl+C` if running in a visible terminal, or killing the background task). 

When you re-run the same command later, **the script will automatically resume where it left off**. It queries the database to see which players and seasons have already been fully downloaded, and automatically skips them to save time and API rate limits.

---

## Validation

```bash
# Run validation report on existing database (no new data pulled)
python -m src.ingestion.bootstrap --validate
```

---

## Quick Database Queries (via Python one-liners)

```bash
# Count rows in each table
python -c "
from src.db.database import get_engine
import pandas as pd
engine = get_engine()
for t in ['games', 'players', 'shots']:
    df = pd.read_sql(f'SELECT COUNT(*) as count FROM {t}', engine)
    print(f'{t}: {df[\"count\"].values[0]:,}')
"

# Check a specific player
python -c "
from src.db.database import get_engine
import pandas as pd
engine = get_engine()
df = pd.read_sql(\"SELECT * FROM players WHERE name LIKE '%LeBron%'\", engine)
print(df.to_string())
"

# Preview shots for a player
python -c "
from src.db.database import get_engine
import pandas as pd
engine = get_engine()
df = pd.read_sql(\"SELECT * FROM shots WHERE player_id = '2544' LIMIT 10\", engine)
print(df.to_string())
"
```

---

## Project Structure

```
NBA_Shot_Predictor/
├── config.py                           # Seasons, rate limits, DB path, wingspan defaults
├── requirements.txt                    # Python dependencies
├── COMMANDS.md                         # ← You are here
├── data/
│   ├── nba_shots.db                    # SQLite database
│   └── csv/                            # Exported CSV files
├── docs/
│   ├── prd_checklist.md
│   └── architecture.md
└── src/
    ├── db/
    │   ├── models.py                   # Game, Player, Shot ORM models
    │   └── database.py                 # Engine, session, init_db
    └── ingestion/
        ├── game_ingestor.py            # LeagueGameLog → Games table
        ├── player_stats_ingestor.py    # Stats + height/weight/wingspan → Players
        ├── shot_ingestor.py            # ShotChartDetail → Shots table
        ├── export_csv.py              # DB → CSV export
        └── bootstrap.py              # Orchestrator (runs everything)
```
