"""The Stat Engine: player/team stat browser.

Distinct from model_features.py's /stats/features: that endpoint answers
"what does the MODEL use and how much" (model transparency). This answers
"what does this PLAYER or TEAM actually do", independent of any model — a
SofaScore-style browse-everything page. See src/inference/stat_engine.py
and src/inference/career_stats.py for the actual table-building logic;
this module is only the thin HTTP layer over them.
"""
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException

from src.inference import api as _state

router = APIRouter()


def _records(df) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts. NaN is pandas' native missing-
    value marker but is not valid JSON — a bare NaN reaches a browser's
    JSON.parse as a syntax error, not a null, so every leaderboard response
    has to scrub it here rather than trust a client to tolerate it."""
    return df.replace({np.nan: None}).to_dict(orient="records")


@router.get("/stats/players")
def stat_engine_players(season: Optional[str] = None):
    """Every rostered player, every stat this project tracks, one wide row
    each — sortable/filterable client-side. See stat_engine.STAT_GROUPS for
    what each column means and which section it belongs to."""
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.stat_engine import build_player_stat_table, player_column_metadata

    table = build_player_stat_table(_state.db_engine, season)
    return {
        "season": table.iloc[0]["season"] if not table.empty else season,
        "columns": player_column_metadata(table.columns),
        "players": _records(table),
    }


@router.get("/stats/player/{player_id}")
def stat_engine_player(player_id: str, season: Optional[str] = None):
    """One player's full stat profile, organized into basic-to-advanced
    sections (stat_engine.GROUP_ORDER)."""
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.stat_engine import player_full_profile

    profile = player_full_profile(_state.db_engine, player_id, season)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No stats for player {player_id}.")
    return profile


@router.get("/stats/teams")
def stat_engine_teams(season: Optional[str] = None):
    """Every team's real stat (def_rating) plus roster-derived aggregates —
    see stat_engine.build_team_stat_table's docstring for why these are
    labeled roster_avg_* rather than presented as separately-measured team
    numbers we don't actually have."""
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.stat_engine import build_team_stat_table, team_column_metadata

    table = build_team_stat_table(_state.db_engine, season)
    return {
        "season": table.iloc[0]["season"] if not table.empty else season,
        "columns": team_column_metadata(table.columns),
        "teams": _records(table),
    }


@router.get("/stats/team/{team_id}")
def stat_engine_team(team_id: str, season: Optional[str] = None):
    """One team's real stat plus its full roster, each player carrying
    their own off_rating/def_rating_ours for a quick scan."""
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.stat_engine import team_full_profile

    profile = team_full_profile(_state.db_engine, team_id, season)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No stats for team {team_id}.")
    return profile


@router.get("/stats/player/{player_id}/career")
def stat_engine_player_career(player_id: str, season: Optional[str] = None):
    """One player's stats over time: current season, the season before it,
    a career average and a career total, for every stat we track them for.

    "Career" means career within this project's data, and the window differs
    per stat — the response carries first_season/last_season/seasons so a
    client can say how much history is actually behind each number.
    """
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.career_stats import career_panel
    from src.inference.player_ratings import resolve_rating_season

    resolved = resolve_rating_season(_state.db_engine, season or "2026-27")
    return career_panel(_state.db_engine, player_id, resolved)
