"""Player archetypes (Urban NBA vocabulary — 3&D Wing, Pick-and-Roll Hub,
Rim Protector, Chucker, ...), computed from measured stats. See
src/inference/archetypes.py's module docstring for why this is algorithmic
rather than a hand-typed lookup, and for what it deliberately does not
claim to measure.

Not to be confused with shot_archetypes.py's PCA/K-Means shot-LOCATION
clusters — these are player-level labels instead.
"""
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException

from src.inference import api as _state

router = APIRouter()


def _clean_archetype_score(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return int(value)


@router.get("/stats/archetypes")
def stat_engine_archetypes(season: Optional[str] = None):
    """Every rostered player's primary + secondary archetype.

    Merge onto the already-fetched /stats/players rows by player_id rather
    than re-fetching the whole roster here — this endpoint exists so a
    client can add archetype filtering without paying for
    build_player_stat_table's joins a second time on every page load."""
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.archetypes import archetype_catalogue, compute_archetypes

    table = compute_archetypes(_state.db_engine, season)
    if table.empty:
        return {"season": season, "catalogue": archetype_catalogue(), "players": []}

    players = [
        {
            "player_id": row["player_id"],
            "archetype_key": row["archetype_key"],
            "archetype_label": row["archetype_label"],
            "archetype_score": _clean_archetype_score(row["archetype_score"]),
            "archetype_secondary": row["archetype_secondary"],
        }
        for _, row in table.iterrows()
    ]
    return {
        "season": table.iloc[0]["season"],
        "catalogue": archetype_catalogue(),
        "players": players,
    }


@router.get("/stats/player/{player_id}/archetype")
def stat_engine_player_archetype(player_id: str, season: Optional[str] = None):
    """Full transparency for one player: primary + secondary archetypes,
    every trait's league percentile, and every archetype they qualified for
    but didn't win — so "why isn't he a Rim Protector" is answerable from
    the same payload that says he is a Paint Anchor."""
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.archetypes import player_archetype_detail

    detail = player_archetype_detail(_state.db_engine, player_id, season)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"No stats for player {player_id}.")
    return detail
