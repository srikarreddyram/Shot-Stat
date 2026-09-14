"""Live scoring demo (src/streaming) — reads `live_shot_scores`, an
append-only table the Kafka consumer writes to. Purely additive: no other
router depends on this one, and these 404 cleanly (empty lists) if the
streaming demo has never been run.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from src.inference import api as _state

router = APIRouter()


@router.get("/live/games")
def live_games(limit: int = Query(10, ge=1, le=50)):
    """Distinct games that have at least one live-scored shot, most recent first."""
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    with _state.db_engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT game_id, COUNT(*) AS n_scored, MAX(scored_at) AS last_scored_at
            FROM live_shot_scores
            GROUP BY game_id
            ORDER BY last_scored_at DESC
            LIMIT :limit
        """), {"limit": limit}).mappings().all()
    return {"games": [dict(r) for r in rows]}


@router.get("/live/scores")
def live_scores(game_id: Optional[str] = None, limit: int = Query(50, ge=1, le=500)):
    """
    Most recently scored live shots, optionally filtered to one game.

    This is a simulated feed (see docs/streaming.md) — `predicted_make_probability`
    comes from the same trained model `/recommend` uses, scored in real time
    as the shot event was consumed off Kafka.
    """
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    clause = "WHERE game_id = :game_id" if game_id else ""
    with _state.db_engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT shot_id, game_id, player_id, defender_id, zone,
                   predicted_make_probability, actual_shot_made,
                   game_date, scored_at, latency_ms
            FROM live_shot_scores
            {clause}
            ORDER BY scored_at DESC
            LIMIT :limit
        """), {"game_id": game_id, "limit": limit}).mappings().all()
    return {"scores": [dict(r) for r in rows]}
