"""
Advanced Stats Ingestor — the NBA's own advanced numbers for every player
and every team: usage, true shooting, assist/rebound/turnover rates,
on-court offensive/defensive/net ratings, pace and PIE.

Uses LeagueDashPlayerStats and LeagueDashTeamStats with MeasureType
"Advanced": ONE request per season per season type per mode returns the
whole league, so every season in the database is ~64 requests. Feeds the
Stat Engine's "Advanced (official NBA)" stats and the advanced season-by-
season table (src/inference/season_averages.py).

Idempotent: rows are upserted, so re-running for the current season simply
refreshes it.

Usage:
    python -m src.ingestion.advanced_stats_ingestor                  # every season with shots
    python -m src.ingestion.advanced_stats_ingestor --seasons 2025-26 2026-27
"""
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, get_session_factory, init_db
from src.db.models import PlayerAdvancedStats, TeamAdvancedStats

SEASON_TYPES = ("Regular Season", "Playoffs")
SHARED = {
    "GP": "gp", "W": "w", "L": "l", "MIN": "min",
    "OFF_RATING": "off_rating", "DEF_RATING": "def_rating", "NET_RATING": "net_rating",
    "AST_PCT": "ast_pct", "AST_TO": "ast_to", "AST_RATIO": "ast_ratio",
    "OREB_PCT": "oreb_pct", "DREB_PCT": "dreb_pct", "REB_PCT": "reb_pct",
    "TM_TOV_PCT": "tov_pct", "EFG_PCT": "efg_pct", "TS_PCT": "ts_pct",
    "PACE": "pace", "PIE": "pie", "POSS": "poss",
}
INTEGER = {"gp", "w", "l", "poss"}


def _value(row, src: str, dst: str, tov_scale: float = 1.0):
    v = getattr(row, src, None)
    if v is None or pd.isna(v):
        return None
    if dst in INTEGER:
        return int(v)
    return float(v) / tov_scale if dst == "tov_pct" else float(v)


def _tov_scale(frame: pd.DataFrame) -> float:
    """The two endpoints disagree on TM_TOV_PCT: the player one reports
    turnovers per 100 plays (median ~10), the team one a fraction (~0.13).
    Every other rate is a fraction, so it is stored as one; the scale is read
    off the data rather than assumed per endpoint."""
    if frame.empty or "TM_TOV_PCT" not in frame:
        return 1.0
    return 100.0 if frame["TM_TOV_PCT"].median() > 1 else 1.0


def _fetch(endpoint, **kwargs) -> pd.DataFrame:
    for attempt in range(3):
        try:
            return endpoint(measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame",
                            timeout=60, **kwargs).get_data_frames()[0]
        except Exception as e:  # network hiccups and rate limits: back off and retry
            if attempt == 2:
                raise
            print(f"    retrying after error: {e}")
            time.sleep(5 * (attempt + 1))
    return pd.DataFrame()


def _upsert(Session, model, rows: list[dict], keys: list[str]) -> None:
    if not rows:
        return
    update_cols = [c.name for c in model.__table__.columns if c.name not in keys]
    with Session() as session:
        for i in range(0, len(rows), 500):
            stmt = sqlite_upsert(model.__table__).values(rows[i:i + 500])
            stmt = stmt.on_conflict_do_update(index_elements=keys,
                                              set_={c: getattr(stmt.excluded, c) for c in update_cols})
            session.execute(stmt)
        session.commit()


def ingest_advanced_stats(seasons: list[str]) -> int:
    engine = get_engine()
    init_db(engine)
    Session = get_session_factory(engine)
    total = 0
    for season in seasons:
        for season_type in SEASON_TYPES:
            players = _fetch(leaguedashplayerstats.LeagueDashPlayerStats,
                             season=season, season_type_all_star=season_type)
            time.sleep(config.REQUEST_DELAY)
            teams = _fetch(leaguedashteamstats.LeagueDashTeamStats,
                           season=season, season_type_all_star=season_type)
            time.sleep(config.REQUEST_DELAY)

            p_scale, t_scale = _tov_scale(players), _tov_scale(teams)
            player_rows = [{
                "player_id": str(r.PLAYER_ID), "season": season, "season_type": season_type,
                "player_name": r.PLAYER_NAME, "team_abbrev": r.TEAM_ABBREVIATION,
                "usg_pct": _value(r, "USG_PCT", "usg_pct"),
                **{dst: _value(r, src, dst, p_scale) for src, dst in SHARED.items()},
            } for r in players.itertuples(index=False)]
            team_rows = [{
                "team_id": str(r.TEAM_ID), "season": season, "season_type": season_type,
                "team_name": r.TEAM_NAME,
                **{dst: _value(r, src, dst, t_scale) for src, dst in SHARED.items()},
            } for r in teams.itertuples(index=False)]

            _upsert(Session, PlayerAdvancedStats, player_rows, ["player_id", "season", "season_type"])
            _upsert(Session, TeamAdvancedStats, team_rows, ["team_id", "season", "season_type"])
            total += len(player_rows) + len(team_rows)
            print(f"  ✓ {season} {season_type}: {len(player_rows)} players, {len(team_rows)} teams")
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest the NBA's advanced stats for players and teams.")
    parser.add_argument("--seasons", nargs="+", default=None,
                        help="Seasons to fetch. Default: every season with shots in the database.")
    args = parser.parse_args()
    target = args.seasons or config.seasons_in_database()
    print(f"Advanced stats for {len(target)} season(s): {target[0]} to {target[-1]}")
    rows = ingest_advanced_stats(target)
    print(f"\n✓ Advanced stats ingestion complete: {rows:,} rows")
