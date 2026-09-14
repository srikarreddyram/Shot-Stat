"""The two "why" endpoints — TreeSHAP-backed plain-English breakdowns for
one shot, opened on demand from the shot-detail panel. Not to be confused
with src/inference/explain.py, which does the actual decomposition; this
module is only the thin HTTP wrapper around ShotRecommender's own
explain_attainability/explain_matchup methods.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.inference import api as _state

router = APIRouter()


@router.get("/explain/attainability/{player_id}")
def explain_attainability(
    player_id: str,
    zone: str = Query(..., description="One of the six court zones, e.g. 'Left Corner 3'"),
    season: Optional[str] = None,
    loc_x: Optional[float] = Query(None, description="Shot x, NBA chart units. With loc_y, resolves the angle sub-zone (dead-centre vs wing)."),
    loc_y: Optional[float] = Query(None, description="Shot y, NBA chart units."),
    defender_id: Optional[str] = Query(None, description="Adds `defender`: whether this defender's opponents attack this zone more or less than a typical defender's do. Does not change the attainability estimate itself, which has no defender in it by design."),
):
    """
    Why this player can or cannot get a shot in this zone.

    Splits the attainability estimate into the zone's baseline share for any
    player and this player's own deviation from it, with the traits
    responsible ranked by their exact TreeSHAP contribution. See
    src/inference/explain.py for the decomposition.
    """
    if _state.recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        return _state.recommender.explain_attainability(
            player_id=player_id, zone=zone, season=season or _state.latest_season,
            loc_x=loc_x, loc_y=loc_y, defender_id=defender_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/explain/matchup/{player_id}")
def explain_matchup(
    player_id: str,
    zone: str = Query(..., description="One of the six court zones, e.g. 'Left Corner 3'"),
    loc_x: float = Query(..., description="Shot x, NBA chart units."),
    loc_y: float = Query(..., description="Shot y, NBA chart units."),
    season: Optional[str] = None,
    defender_id: Optional[str] = None,
    secondary_defender_id: Optional[str] = None,
    quarter: int = 1,
    time_remaining: float = 600.0,
    score_diff: int = 0,
    home_away: int = 1,
    playoff_flag: int = 0,
    rest_days: int = 1,
    is_back_to_back: int = 0,
    opp_def_rating: float = 112.0,
):
    """
    The full matchup narrative for one specific clicked location: make
    probability broken down offense vs. defense (including who else is on
    the floor — teammates' creation/gravity and the help defenders' shot-
    blocking/disruption, resolved from the most recent real lineup or,
    failing that, current-roster teammates), expected points, and
    attainability woven in. See ShotRecommender.explain_matchup and
    src/inference/explain.build_matchup_narrative.
    """
    if _state.recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        return _state.recommender.explain_matchup(
            player_id=player_id, zone=zone, loc_x=loc_x, loc_y=loc_y,
            season=season or _state.latest_season, defender_id=defender_id,
            secondary_defender_id=secondary_defender_id, quarter=quarter,
            time_remaining=time_remaining, score_diff=score_diff,
            home_away=home_away, playoff_flag=playoff_flag,
            rest_days=rest_days, is_back_to_back=is_back_to_back,
            opp_def_rating=opp_def_rating,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
