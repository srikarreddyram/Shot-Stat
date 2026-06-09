# NBA Shot Quality Engine — Commands Reference

All commands should be run from the project root:
```bash
cd /Users/tejsr/Projects/NBA_Shot_Predictor
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
