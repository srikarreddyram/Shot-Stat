"""
Defensive Activity Ingestor — full per-game box score, plus the NBA's
"hustle stats": screen assists, box outs, loose balls recovered, charges
drawn, and total contested shots.

Populates `player_defensive_activity` from two NBA API endpoints, merged into
one row per (player_id, season):

    LeagueDashPlayerStats (Base, PerGame)   → GP, MIN, PTS, shooting volume,
                                               rebounds, fouls, STL, BLK, and
                                               everything else this module
                                               reads from it
    LeagueHustleStatsPlayer (PerGame)       → DEFLECTIONS + everything else
                                               this module reads from it

Two separate endpoints with two separate eligibility windows: Base is
ordinary box-score data, available every season the league has tracked a box
score (this project's full `config.ALL_SEASONS` range, 2010-11 on) — it used
to be skipped for six real seasons of available data because it shared a
single 2016-17 cutoff with Hustle, which really is unavailable before then
(SportVU/hustle tracking's own start date). BASE_START_SEASON and
HUSTLE_START_SEASON are kept separate so a season with real box-score data
but no hustle tracking still gets the box score half of this table.

ast/tov are deliberately absent from what Base contributes here:
`players.ast`/`players.tov` (from roster_ingestor.py, the same underlying
endpoint) already own those — this project has hit the two-sources-of-truth
bug before and does not need a second copy of the same number under a
different name.

The screen/box-out/loose-ball/charge/contested-shot columns, and the fuller
box score (points, rebounds, fouls, double-doubles), were added to this table
well after deflections/STL/BLK — but they come from the SAME two endpoint
calls this ingestor already made, so backfilling them costs zero additional
API calls; the fix is reading more fields off responses this project was
already fetching and mostly discarding.

Cost: 2 API calls per season — both endpoints return the whole league at
once, same shape as tracking_ingestor.py.

Usage:
    python -m src.ingestion.defensive_activity_ingestor                # test season
    python -m src.ingestion.defensive_activity_ingestor --full          # all seasons
    python -m src.ingestion.defensive_activity_ingestor --season 2023-24
"""
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, leaguehustlestatsplayer
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, init_db
from src.db.models import PlayerDefensiveActivity

# ── Configuration ──────────────────────────────────────────────────────────
# Box-score data (Base) exists for the league's full history; hustle-tracking
# data (Hustle) only from 2016-17. Kept as two separate constants on purpose
# — collapsing them into one, as this file used to, is exactly what silently
# dropped six real seasons of box-score data that Hustle simply never had.
BASE_START_SEASON = "2010-11"
HUSTLE_START_SEASON = "2016-17"

REQUEST_DELAY = 1.2
RETRY_DELAY = 10.0
MAX_RETRIES = 3

# Column on the Base response -> column on our model. All PerGame floats
# except gp (int) and dd2/td3 (season-total ints regardless of PerMode —
# "double-doubles per game" is not a real quantity the API computes).
BASE_FIELDS = {
    "PTS": "pts", "FGM": "fgm", "FGA": "fga", "FG3M": "fg3m", "FG3A": "fg3a",
    "FTM": "ftm", "FTA": "fta", "OREB": "oreb", "DREB": "dreb", "REB": "reb",
    "STL": "stl", "BLK": "blk", "PF": "pf", "PFD": "pfd",
    "PLUS_MINUS": "plus_minus",
}
BASE_INT_FIELDS = {"DD2": "dd2", "TD3": "td3"}

# Column on the Hustle response -> column on our model. All PerGame floats,
# all from the one `hustle` frame fetched below.
HUSTLE_FIELDS = {
    "DEFLECTIONS": "deflections",
    "SCREEN_ASSISTS": "screen_ast",
    "SCREEN_AST_PTS": "screen_ast_pts",
    "OFF_BOXOUTS": "off_boxouts",
    "DEF_BOXOUTS": "def_boxouts",
    "BOX_OUTS": "box_outs",
    "OFF_LOOSE_BALLS_RECOVERED": "off_loose_balls_recovered",
    "DEF_LOOSE_BALLS_RECOVERED": "def_loose_balls_recovered",
    "LOOSE_BALLS_RECOVERED": "loose_balls_recovered",
    "CHARGES_DRAWN": "charges_drawn",
    "CONTESTED_SHOTS": "contested_shots",
    "CONTESTED_SHOTS_2PT": "contested_shots_2pt",
    "CONTESTED_SHOTS_3PT": "contested_shots_3pt",
}


def _fetch(endpoint_cls, season: str, label: str, **kwargs) -> pd.DataFrame | None:
    for attempt in range(MAX_RETRIES):
        try:
            result = endpoint_cls(season=season, timeout=45, **kwargs)
            frames = result.get_data_frames()
            if not frames or frames[0].empty:
                return None
            return frames[0]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    ⚠ {label} {season} attempt {attempt+1} failed "
                      f"({type(e).__name__}) — retrying in {RETRY_DELAY}s")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    ✗ {label} {season} failed after "
                      f"{MAX_RETRIES} attempts: {type(e).__name__}: {e}")
                return None


def ingest_season(season: str, engine=None) -> int:
    """Ingest one season's Base box score (if the season has one) and Hustle
    stats (if the season has those too). Returns rows written."""
    if engine is None:
        engine = get_engine()

    players: dict[str, dict] = {}

    if season >= BASE_START_SEASON:
        base = _fetch(leaguedashplayerstats.LeagueDashPlayerStats, season, "Base",
                      per_mode_detailed="PerGame")
        time.sleep(REQUEST_DELAY)
        if base is not None:
            for row in base.itertuples(index=False):
                pid = str(getattr(row, "PLAYER_ID"))
                rec = players.setdefault(pid, {"player_id": pid, "season": season})
                gp = getattr(row, "GP", None)
                mins = getattr(row, "MIN", None)
                rec["gp"] = int(gp) if pd.notna(gp) else None
                rec["min_per_game"] = float(mins) if pd.notna(mins) else None
                for api_col, model_col in BASE_FIELDS.items():
                    val = getattr(row, api_col, None)
                    rec[model_col] = float(val) if pd.notna(val) else None
                for api_col, model_col in BASE_INT_FIELDS.items():
                    val = getattr(row, api_col, None)
                    rec[model_col] = int(val) if pd.notna(val) else None
            print(f"    ✓ Base            {len(base):>4} players")
        else:
            print(f"    ⚠ {season} Base: no data returned")
    else:
        print(f"  → {season}: before box-score era ({BASE_START_SEASON}), skipping Base")

    if season >= HUSTLE_START_SEASON:
        hustle = _fetch(leaguehustlestatsplayer.LeagueHustleStatsPlayer, season,
                         "Hustle", per_mode_time="PerGame")
        time.sleep(REQUEST_DELAY)
        if hustle is not None:
            for row in hustle.itertuples(index=False):
                pid = str(getattr(row, "PLAYER_ID"))
                rec = players.setdefault(pid, {"player_id": pid, "season": season})
                for api_col, model_col in HUSTLE_FIELDS.items():
                    val = getattr(row, api_col, None)
                    rec[model_col] = float(val) if pd.notna(val) else None
            print(f"    ✓ Hustle          {len(hustle):>4} players")
        else:
            print(f"    ⚠ {season} Hustle: no data returned")
    else:
        print(f"  → {season}: before hustle-stats era ({HUSTLE_START_SEASON}), skipping Hustle")

    if not players:
        return 0

    # Normalize every record to the same key set before the bulk insert. A
    # player can appear in one endpoint's frame but not the other's (Base
    # and Hustle don't guarantee identical rosters), which would otherwise
    # leave some dicts missing a key the others have — SQLAlchemy's
    # multi-row VALUES insert requires uniform columns across the whole
    # batch and raises a CompileError rather than silently NULLing the gap.
    all_cols = [c.name for c in PlayerDefensiveActivity.__table__.columns]
    records = [
        {col: rec.get(col) for col in all_cols}
        for rec in players.values()
    ]
    with engine.begin() as conn:
        stmt = sqlite_insert(PlayerDefensiveActivity).values(records)
        update_cols = {
            c.name: stmt.excluded[c.name]
            for c in PlayerDefensiveActivity.__table__.columns
            if c.name not in ("player_id", "season")
        }
        conn.execute(stmt.on_conflict_do_update(
            index_elements=["player_id", "season"],
            set_=update_cols,
        ))

    return len(records)


def ingest(seasons: list[str]) -> int:
    engine = get_engine()
    init_db(engine)

    eligible = [s for s in seasons if s >= BASE_START_SEASON]
    skipped = [s for s in seasons if s < BASE_START_SEASON]
    hustle_eligible = sum(1 for s in eligible if s >= HUSTLE_START_SEASON)

    print(f"\n{'='*60}")
    print("  DEFENSIVE ACTIVITY INGESTOR (box score + hustle stats)")
    print(f"  Seasons: {len(eligible)} with a box score "
          f"({hustle_eligible} of those also have hustle tracking)"
          + (f", {len(skipped)} before {BASE_START_SEASON} skipped entirely" if skipped else ""))
    print(f"{'='*60}")

    total = 0
    for season in tqdm(eligible, desc="  Seasons"):
        print(f"\n  → {season}")
        total += ingest_season(season, engine=engine)

    print(f"\n{'='*60}")
    print(f"  DEFENSIVE ACTIVITY INGEST COMPLETE — {total:,} player-seasons written")
    print(f"{'='*60}\n")
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest per-game box score + hustle stats (blocks, steals, deflections, screen assists, etc.)."
    )
    parser.add_argument("--full", action="store_true",
                        help="All seasons in config.ALL_SEASONS")
    parser.add_argument("--season", type=str, default=None,
                        help="Single season, e.g. 2023-24")
    args = parser.parse_args()

    if args.season:
        target = [args.season]
    elif args.full:
        target = config.ALL_SEASONS
    else:
        target = config.TEST_SEASONS

    ingest(target)
