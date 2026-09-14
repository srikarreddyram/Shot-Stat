"""Shot archetypes (src/analysis/shot_archetypes.py) — PCA + K-Means
clusters over the shot-descriptive feature set, joined from
`shot_archetypes` (populated by `python -m src.analysis.shot_archetypes
--fit`) back onto `shots` for court coordinates. Purely additive.

Not to be confused with player archetypes (Urban NBA vocabulary — 3&D Wing,
Rim Protector, ...) in player_archetypes.py: these are shot-LOCATION
clusters, discovered by unsupervised learning, not player labels.
"""
import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from src.inference import api as _state

router = APIRouter()

_archetype_metadata_cache = None


def _load_archetype_metadata() -> dict:
    global _archetype_metadata_cache
    if _archetype_metadata_cache is None:
        from src.analysis.shot_archetypes import ARCHETYPE_METADATA
        if not ARCHETYPE_METADATA.exists():
            raise HTTPException(
                status_code=503,
                detail="Archetype model not fit yet — run "
                       "'python -m src.analysis.shot_archetypes --fit'",
            )
        _archetype_metadata_cache = json.loads(ARCHETYPE_METADATA.read_text())
    return _archetype_metadata_cache


@router.get("/archetypes")
def list_archetypes():
    """Every archetype's label, size, FG%, and creation/finish mix."""
    metadata = _load_archetype_metadata()
    return {
        "model_version": metadata["model_version"],
        "k": metadata["k"],
        "clusters": metadata["clusters"],
    }


@router.get("/archetypes/court")
def archetypes_court(points_per_cluster: int = Query(250, ge=10, le=1000),
                     season: Optional[str] = None):
    """
    A plottable sample of real shots per archetype, for a court overlay —
    same court/coordinate convention the existing heatmap uses (loc_x, loc_y).
    """
    metadata = _load_archetype_metadata()
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not ready")

    from src.analysis.shot_archetypes import MODEL_VERSION
    season_clause = "AND s.season = :season" if season else ""

    out = {}
    with _state.db_engine.connect() as conn:
        for cluster in metadata["clusters"]:
            cid = cluster["cluster_id"]
            rows = conn.execute(text(f"""
                SELECT s.loc_x, s.loc_y, s.zone
                FROM shot_archetypes a
                JOIN shots s ON s.shot_id = a.shot_id
                WHERE a.model_version = :version AND a.cluster_id = :cid
                {season_clause}
                ORDER BY RANDOM()
                LIMIT :limit
            """), {
                "version": MODEL_VERSION, "cid": cid,
                "season": season, "limit": points_per_cluster,
            }).mappings().all()
            out[str(cid)] = {
                "label": cluster["label"],
                "points": [dict(r) for r in rows],
            }
    return {"clusters": out}


@router.get("/player/{player_id}/archetype-mix")
def player_archetype_mix(player_id: str, season: Optional[str] = None):
    """A player's real shot diet broken down by archetype cluster."""
    metadata = _load_archetype_metadata()
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not ready")

    from src.analysis.shot_archetypes import MODEL_VERSION
    season_clause = "AND s.season = :season" if season else ""
    labels_by_id = {c["cluster_id"]: c["label"] for c in metadata["clusters"]}

    with _state.db_engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT a.cluster_id, COUNT(*) AS n, AVG(s.shot_made) AS fg_pct
            FROM shot_archetypes a
            JOIN shots s ON s.shot_id = a.shot_id
            WHERE a.model_version = :version AND s.player_id = :player_id
            {season_clause}
            GROUP BY a.cluster_id
        """), {"version": MODEL_VERSION, "player_id": player_id, "season": season}).mappings().all()

    total = sum(r["n"] for r in rows) or 1
    return {
        "player_id": player_id,
        "season": season,
        "mix": [
            {
                "cluster_id": r["cluster_id"],
                "label": labels_by_id.get(r["cluster_id"], f"archetype {r['cluster_id']}"),
                "n_shots": r["n"],
                "share": r["n"] / total,
                "fg_pct": r["fg_pct"],
            }
            for r in sorted(rows, key=lambda r: -r["n"])
        ],
    }
