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
from fastapi import APIRouter, HTTPException, Query

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


@router.get("/stats/player/{player_id}/seasons")
def stat_engine_player_seasons(player_id: str, stat_set: str = Query("box", alias="set")):
    """Season-by-season averages of the major box-score stats, regular
    season and playoffs, plus a career line — or, with set=advanced, the
    NBA's own advanced stats season by season. See season_averages.py."""
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.season_averages import player_advanced_seasons, player_seasons

    if stat_set not in ("box", "advanced"):
        raise HTTPException(status_code=422, detail="set must be box or advanced.")
    fn = player_advanced_seasons if stat_set == "advanced" else player_seasons
    out = fn(_state.db_engine, player_id)
    if out is None:
        raise HTTPException(status_code=404, detail=f"No box scores for player {player_id}.")
    return out


@router.get("/stats/team/{team_id}/seasons")
def stat_engine_team_seasons(team_id: str, stat_set: str = Query("box", alias="set")):
    """A team's season-by-season averages, its opponents' line, and its
    league rank in each stat that season — or, with set=advanced, its
    official advanced stats and their league ranks."""
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.season_averages import team_advanced_seasons, team_seasons

    if stat_set not in ("box", "advanced"):
        raise HTTPException(status_code=422, detail="set must be box or advanced.")
    fn = team_advanced_seasons if stat_set == "advanced" else team_seasons
    out = fn(_state.db_engine, team_id)
    if out is None:
        raise HTTPException(status_code=404, detail=f"No box scores for team {team_id}.")
    return out


def _shot_map_args(season_type: str, span: int = 1):
    from src.inference.shot_maps import SEASON_TYPES

    if season_type not in SEASON_TYPES:
        raise HTTPException(status_code=422, detail=f"season_type must be one of {sorted(SEASON_TYPES)}.")
    if span not in (1, 3):
        raise HTTPException(status_code=422, detail="span must be 1 or 3 seasons.")


def _latest_shot_season(player_id: str) -> Optional[str]:
    from sqlalchemy import text

    with _state.db_engine.connect() as conn:
        return conn.execute(text("SELECT MAX(season) FROM shots WHERE player_id = :pid"),
                            {"pid": player_id}).scalar()


@router.get("/stats/player/{player_id}/shot-map")
def stat_engine_player_shot_map(player_id: str, season: Optional[str] = None,
                                season_type: str = "regular", span: int = 1):
    """Where a player shoots from (a smooth density on a 1-ft grid) and how
    well vs the league by zone, for one season or the three ending in it.
    See shot_maps.py for why efficiency is by zone, and the shrinkage."""
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.shot_maps import player_shot_map

    _shot_map_args(season_type, span)
    season = season or _latest_shot_season(player_id)
    out = player_shot_map(_state.db_engine, player_id, season, season_type, span) if season else None
    if out is None:
        raise HTTPException(status_code=404, detail=f"No shots for player {player_id} in {season}.")
    return out


@router.get("/stats/player/{player_id}/shots")
def stat_engine_player_shots(player_id: str, season: Optional[str] = None,
                             season_type: str = "regular", span: int = 1):
    """Every shot in the window, in order, for the animated shot replay."""
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.shot_maps import player_shots

    _shot_map_args(season_type, span)
    season = season or _latest_shot_season(player_id)
    out = player_shots(_state.db_engine, player_id, season, season_type, span) if season else None
    if out is None:
        raise HTTPException(status_code=404, detail=f"No shots for player {player_id} in {season}.")
    return out


@router.get("/stats/player/{player_id}/shot-maps")
def stat_engine_player_shot_maps(player_id: str, group: str, season: Optional[str] = None,
                                 season_type: str = "regular", span: int = 1):
    """The advanced view's extra maps: by shot type, clutch vs the rest, or
    the shots his defensive assignments took (group = shot_types | clutch |
    defense). Each subset is compared with the league's rate for the same
    subset."""
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.shot_maps import MAP_GROUPS, player_shot_maps

    _shot_map_args(season_type, span)
    if group not in MAP_GROUPS:
        raise HTTPException(status_code=422, detail=f"group must be one of {list(MAP_GROUPS)}.")
    season = season or _latest_shot_season(player_id)
    out = player_shot_maps(_state.db_engine, player_id, season, group, season_type, span) if season else None
    if out is None:
        raise HTTPException(status_code=404, detail=f"No shots for player {player_id} in {season}.")
    return out


@router.get("/stats/player/{player_id}/shot-map/change")
def stat_engine_player_shot_change(player_id: str, season: Optional[str] = None, season_type: str = "regular"):
    """How a player's shot diet moved from the previous season to this one."""
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.shot_maps import player_shot_change

    _shot_map_args(season_type)
    season = season or _latest_shot_season(player_id)
    out = player_shot_change(_state.db_engine, player_id, season, season_type) if season else None
    if out is None:
        raise HTTPException(status_code=404, detail=f"Need shots in both {season} and the season before.")
    return out


@router.get("/stats/team/{team_id}/shots")
def stat_engine_team_shots(team_id: str, season: str, side: str = "offense",
                           season_type: str = "regular", span: int = 1):
    """A team's shots (side=offense) or its opponents' (side=defense), in
    order, for the team shot replay."""
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.shot_maps import team_shots

    _shot_map_args(season_type, span)
    if side not in ("offense", "defense"):
        raise HTTPException(status_code=422, detail="side must be offense or defense.")
    out = team_shots(_state.db_engine, team_id, season, side, season_type, span)
    if out is None:
        raise HTTPException(status_code=404, detail=f"No shots for team {team_id} in {season}.")
    return out


@router.get("/stats/team/{team_id}/shot-map")
def stat_engine_team_shot_map(team_id: str, season: str, side: str = "offense",
                              season_type: str = "regular", span: int = 1):
    """A team's shots (side=offense) or its opponents' (side=defense): where
    from, and how well vs the league by zone."""
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.shot_maps import team_shot_map

    _shot_map_args(season_type, span)
    if side not in ("offense", "defense"):
        raise HTTPException(status_code=422, detail="side must be offense or defense.")
    out = team_shot_map(_state.db_engine, team_id, season, side, season_type, span)
    if out is None:
        raise HTTPException(status_code=404, detail=f"No shots for team {team_id} in {season}.")
    return out
