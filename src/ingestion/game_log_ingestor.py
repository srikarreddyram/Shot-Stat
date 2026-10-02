"""
Game Log Ingestor — every player's and every team's box score for every game.

Uses LeagueGameLog in player and team mode: ONE request per season per
season type per mode returns the whole league, so the full 2016-17 onward
history is ~40 requests. Feeds the season-by-season averages on player and
team profiles (src/inference/season_averages.py).

Idempotent: rows are upserted, so re-running for the current season simply
picks up games played since the last run.

Usage:
    python -m src.ingestion.game_log_ingestor                 # 2016-17 onward
    python -m src.ingestion.game_log_ingestor --season 2025-26
"""
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, get_session_factory
from src.db.models import PlayerGameLog, TeamGameLog

SEASON_TYPES = ("Regular Season", "Playoffs")
BOX = {"PTS": "pts", "REB": "reb", "OREB": "oreb", "DREB": "dreb", "AST": "ast", "STL": "stl", "BLK": "blk",
       "TOV": "tov", "PF": "pf", "FGM": "fgm", "FGA": "fga", "FG3M": "fg3m", "FG3A": "fg3a", "FTM": "ftm", "FTA": "fta"}


def _common(r, season: str, season_type: str) -> dict:
    # MATCHUP reads "LAL vs. GSW" at home and "LAL @ GSW" away.
    return {
        "game_id": str(r.GAME_ID), "season": season, "season_type": season_type,
        "game_date": pd.to_datetime(r.GAME_DATE).date(),
        "team_id": str(r.TEAM_ID), "team_abbrev": r.TEAM_ABBREVIATION,
        "opp_abbrev": r.MATCHUP.replace("vs.", "@").split("@")[-1].strip(),
        "home": 0 if "@" in r.MATCHUP else 1,
        "win": 1 if r.WL == "W" else 0 if r.WL == "L" else None,
        "plus_minus": float(r.PLUS_MINUS) if pd.notna(r.PLUS_MINUS) else None,
        **{dst: (int(getattr(r, src)) if pd.notna(getattr(r, src)) else None) for src, dst in BOX.items()},
    }


def _player_rows(df: pd.DataFrame, season: str, season_type: str) -> list[dict]:
    return [{**_common(r, season, season_type), "player_id": str(r.PLAYER_ID), "player_name": r.PLAYER_NAME,
             "minutes": float(r.MIN) if pd.notna(r.MIN) else None} for r in df.itertuples(index=False)]


def _team_rows(df: pd.DataFrame, season: str, season_type: str) -> list[dict]:
    return [_common(r, season, season_type) for r in df.itertuples(index=False)]


def _upsert(Session, model, rows: list[dict], keys: list[str]) -> None:
    update_cols = [c.name for c in model.__table__.columns if c.name not in keys]
    with Session() as session:
        for i in range(0, len(rows), 500):
            stmt = sqlite_upsert(model.__table__).values(rows[i:i + 500])
            stmt = stmt.on_conflict_do_update(index_elements=keys,
                                              set_={c: getattr(stmt.excluded, c) for c in update_cols})
            session.execute(stmt)
        session.commit()


def ingest_game_logs(seasons: list[str]) -> int:
    engine = get_engine()
    for model in (PlayerGameLog, TeamGameLog):
        model.__table__.create(engine, checkfirst=True)
    Session = get_session_factory(engine)
    total = 0
    for season in seasons:
        for season_type in SEASON_TYPES:
            for mode, model, build, keys in (("P", PlayerGameLog, _player_rows, ["player_id", "game_id"]),
                                             ("T", TeamGameLog, _team_rows, ["team_id", "game_id"])):
                try:
                    df = leaguegamelog.LeagueGameLog(
                        season=season, season_type_all_star=season_type,
                        player_or_team_abbreviation=mode, timeout=60,
                    ).get_data_frames()[0]
                except Exception as e:  # network/API hiccup: report and keep going
                    print(f"  {season} {season_type} {mode}: FAILED ({e})")
                    continue
                rows = build(df, season, season_type)
                if rows:
                    _upsert(Session, model, rows, keys)
                total += len(rows)
                print(f"  {season} {season_type}: {len(rows):,} {'player' if mode == 'P' else 'team'}-games")
                time.sleep(config.REQUEST_DELAY)
    print(f"Done — {total:,} rows upserted.")
    return total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest per-game player and team box scores (LeagueGameLog).")
    parser.add_argument("--season", type=str, default=None)
    args = parser.parse_args()
    all_seasons = [s for s in config.ALL_SEASONS if s >= "2016-17"]
    ingest_game_logs([args.season] if args.season else all_seasons)
