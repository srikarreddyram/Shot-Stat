"""The core recommendation endpoints — the matchup engine's own /recommend,
/recommend/summary and /recommend/heatmap.

Route handlers here reach shared app state (the loaded `recommender`) via
`from src.inference import api as _state` and `_state.recommender`, never
`from src.inference.api import recommender` — the latter would freeze the
`None` placeholder from before the app's lifespan startup ran, since a
plain name import snapshots the value at import time while attribute
access on the module object always reads the current one. See
src/inference/api/__init__.py's module docstring for the full explanation.
"""
from fastapi import APIRouter, HTTPException

from src.inference import api as _state
from .models import RecommendRequest, RecommendResponse, SummaryResponse

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if _state.recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    try:
        results = _state.recommender.recommend(
            player_id=req.player_id,
            season=req.season,
            quarter=req.quarter,
            time_remaining=req.time_remaining,
            score_diff=req.score_diff,
            home_away=req.home_away,
            playoff_flag=req.playoff_flag,
            defender_id=req.defender_id,
            secondary_defender_id=req.secondary_defender_id,
            top_n=req.top_n,
            rest_days=req.rest_days,
            is_back_to_back=req.is_back_to_back,
            opp_def_rating=req.opp_def_rating,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Get player name
    player = _state.recommender._player_row(req.player_id, req.season)
    defender = _state.recommender._defender_row(req.defender_id, req.season) if req.defender_id else None

    return RecommendResponse(
        player_id=req.player_id,
        player_name=player["name"],
        season=req.season,
        game_state={
            "quarter": req.quarter,
            "time_remaining": req.time_remaining,
            "score_diff": req.score_diff,
            "home_away": "Home" if req.home_away else "Away",
            "playoff_flag": bool(req.playoff_flag),
            "defender_id": req.defender_id,
            "secondary_defender_id": req.secondary_defender_id,
        },
        recommendations=results.to_dict("records"),
        attacker_stats_source="measured",
        attacker_resolved_season=player.get("_latest_season"),
        defender_stats_source="measured" if defender else None,
        defender_resolved_season=req.season if defender else None,
    )


@router.post("/recommend/summary", response_model=SummaryResponse)
def recommend_summary(req: RecommendRequest):
    if _state.recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        summary = _state.recommender.zone_summary(
            player_id=req.player_id,
            season=req.season,
            quarter=req.quarter,
            time_remaining=req.time_remaining,
            score_diff=req.score_diff,
            home_away=req.home_away,
            playoff_flag=req.playoff_flag,
            defender_id=req.defender_id,
            secondary_defender_id=req.secondary_defender_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    player = _state.recommender._player_row(req.player_id, req.season)
    defender = _state.recommender._defender_row(req.defender_id, req.season) if req.defender_id else None

    return SummaryResponse(
        player_id=req.player_id,
        player_name=player["name"],
        season=req.season,
        game_state={
            "quarter": req.quarter,
            "time_remaining": req.time_remaining,
            "score_diff": req.score_diff,
            "home_away": "Home" if req.home_away else "Away",
            "playoff_flag": bool(req.playoff_flag),
        },
        zone_summary=summary.to_dict("records"),
        attacker_stats_source="measured",
        attacker_resolved_season=player.get("_latest_season"),
        defender_stats_source="measured" if defender else None,
        defender_resolved_season=req.season if defender else None,
    )


@router.post("/recommend/heatmap")
def recommend_heatmap(req: RecommendRequest):
    """Return full dense grid for heatmap visualization."""
    if _state.recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        result = _state.recommender.recommend_heatmap(
            player_id=req.player_id,
            season=req.season,
            quarter=req.quarter,
            time_remaining=req.time_remaining,
            score_diff=req.score_diff,
            home_away=req.home_away,
            playoff_flag=req.playoff_flag,
            defender_id=req.defender_id,
            secondary_defender_id=req.secondary_defender_id,
            rest_days=req.rest_days,
            is_back_to_back=req.is_back_to_back,
            opp_def_rating=req.opp_def_rating,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    player = _state.recommender._player_row(req.player_id, req.season)
    defender = _state.recommender._defender_row(req.defender_id, req.season) if req.defender_id else None

    return {
        "player_id": req.player_id,
        "player_name": player["name"],
        "season": req.season,
        "attacker_stats_source": "measured",
        "attacker_resolved_season": player.get("_latest_season"),
        "defender_stats_source": "measured" if defender else None,
        "defender_resolved_season": req.season if defender else None,
        **result,
    }
