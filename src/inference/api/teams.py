"""Team-level lookups: the static 30-franchise list and one team's roster."""
from typing import Optional

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from src.inference import api as _state
from src.inference.player_ratings import rating_lookup
from .players import _player_card

router = APIRouter()


@router.get("/teams")
def list_teams():
    """
    The 30 NBA franchises — static league data (nba_api.stats.static), not a
    DB query, so this never depends on ingestion coverage. Powers the
    team-vs-team picker: choose two teams first, then a player from each
    roster, instead of free-text searching the whole league.
    """
    from nba_api.stats.static import teams as nba_teams_static

    return [
        {"team_id": str(t["id"]), "abbreviation": t["abbreviation"], "name": t["full_name"]}
        for t in sorted(nba_teams_static.get_teams(), key=lambda t: t["full_name"])
    ]


@router.get("/team/{team_id}/roster")
def get_team_roster(team_id: str, season: Optional[str] = None):
    """
    Every player rostered to `team_id` this season (players.team_id, kept
    fresh by src/ingestion/{roster,current_roster}_ingestor.py — see the
    on-court lineup work in src/features/point_in_time.py for why that
    column exists at all). Same per-player shape as /players/search, so the
    team-vs-team picker can reuse the same player card rendering.
    """
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")

    season = season or _state.latest_season
    with _state.db_engine.connect() as conn:
        id_rows = conn.execute(text("""
            SELECT player_id FROM players
            WHERE team_id = :team_id AND season = :season
            ORDER BY name
        """), {"team_id": team_id, "season": season}).fetchall()

        ratings = rating_lookup(_state.db_engine, season)
        results = [_player_card(conn, pid, season, ratings) for (pid,) in id_rows]

    return [r for r in results if r is not None]
