"""Single-player lookups: /player/{id}, /players/search, and the legacy
/headshot/{id} proxy (kept alongside the newer /player/{id}/headshot in
media.py — both exist in production, and consolidating them wasn't asked
for as part of this reorganization).
"""
from pathlib import Path
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import text

from src.inference import api as _state
from src.inference.player_lookup import resolve_player_stats, resolve_defender_stats
from src.inference.player_ratings import rating_lookup

router = APIRouter()


@router.get("/player/{player_id}")
def get_player(player_id: str, season: Optional[str] = None):
    """Look up a player's stats for a given season (defaults to the latest ingested season)."""
    if _state.recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    season = season or _state.latest_season
    try:
        player = _state.recommender._player_row(player_id, season)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Surface the shrunk point-in-time rates rather than the raw prior counts.
    # The counts are an internal representation; a client asking about a player
    # wants the rate the model actually reasons with, small samples already
    # regressed toward the league prior.
    import pandas as _pd
    from src.features.point_in_time import ZONE_SUFFIX as _SUFFIX, apply_hierarchy as _apply

    rates = _apply(_pd.DataFrame([player]), _state.recommender.zone_priors).iloc[0]
    zone_rates = {
        zone: float(rates[f"zone_rate_{suffix}"])
        for zone, suffix in _SUFFIX.items()
    }
    zone_attempts = {
        zone: float(player.get(f"pit_car_att_{suffix}", 0.0) or 0.0)
        for zone, suffix in _SUFFIX.items()
    }

    public = {k: v for k, v in player.items() if not k.startswith(("pit_", "_"))}
    return {
        "player_id": player_id,
        "season": season,
        **public,
        "zone_rates": zone_rates,
        # Attempts behind each rate, so a client can tell a measured number
        # from one still leaning on the prior.
        "zone_attempts": zone_attempts,
        "overall_rate": float(rates["overall_rate"]),
        "three_rate": float(rates["three_rate"]),
    }


HEADSHOT_CDN = "https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
HEADSHOT_CACHE = Path(__file__).resolve().parents[3] / "data" / "cache" / "headshots"


@router.get("/headshot/{player_id}")
def get_headshot(player_id: str):
    """Serve a player headshot, proxied and cached from the NBA CDN.

    The frontend used to point <img> straight at cdn.nba.com. That silently
    fails on some networks — Chrome gets ERR_HTTP2_PROTOCOL_ERROR for every
    headshot while curl fetches the same URL fine — and because the avatar
    component falls back to initials on error, the failure looked like a
    design choice rather than a broken image. Going through the backend uses
    the same HTTP stack as the rest of the ingest pipeline, which works, and
    caches each PNG on disk so the CDN is hit once per player, ever.
    """
    if not player_id.isdigit():
        raise HTTPException(status_code=400, detail="player_id must be numeric")

    path = HEADSHOT_CACHE / f"{player_id}.png"

    if not path.exists():
        try:
            resp = requests.get(HEADSHOT_CDN.format(player_id=player_id), timeout=10)
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"headshot fetch failed: {exc}")

        # A missing player returns an HTML error page, not a 404, so the
        # content type is the thing worth trusting here.
        if resp.status_code != 200 or not resp.headers.get("content-type", "").startswith("image/"):
            raise HTTPException(status_code=404, detail="no headshot for that player")

        HEADSHOT_CACHE.mkdir(parents=True, exist_ok=True)
        # Write via a temp file so a killed request can never leave a
        # truncated PNG behind to be served forever after.
        tmp = path.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.replace(path)

    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "public, max-age=604800"})


def _player_card(conn, player_id: str, season: str, ratings: dict) -> dict | None:
    """
    One player's autocomplete/roster-card shape — shared by /players/search
    and /team/{team_id}/roster (see teams.py) so the two endpoints can't
    quietly drift apart on what fields a client gets back.
    """
    player = resolve_player_stats(conn, player_id, season)
    if player is None:
        return None
    defender = resolve_defender_stats(conn, player_id, season)
    overall_def = (defender or {}).get("def_stats", {}).get("Overall", {})
    return {
        "player_id": str(player_id),
        "name": player["name"],
        "position": player["position"],
        "team_id": player.get("team_id"),
        "height": player["height"],
        "weight": player["weight"],
        "career_fg_pct": player["career_fg_pct"],
        "season_fg_pct": player["season_fg_pct"],
        "career_3p_pct": player["career_3p_pct"],
        "def_fg_pct_allowed": overall_def.get("d_fg_pct"),   # opponent FG% allowed (overall) — lower is better defense
        "def_plus_minus": overall_def.get("pct_plusminus"),  # FG% allowed vs. league normal — negative is better defense
        "wingspan": player["wingspan"],
        "ast": player["ast"],
        "tov": player["tov"],
        "ft_pct": player["ft_pct"],
        "rim_pct": player["zone_stats"].get("Restricted Area"),
        "mid_pct": player["zone_stats"].get("Mid-Range"),
        "headshot_url": f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png",
        "stats_source": player["stats_source"],   # "measured" | "prior"
        "resolved_season": player["resolved_season"],
        # Computed server-side from measured data (see
        # src/inference/player_ratings.py). Previously the frontend invented
        # these from a hardcoded formula that read three columns which are
        # NULL for every player in the database.
        #
        # "rating_source" is surfaced un-prefixed (not "rating_rating_source")
        # so the frontend can badge a 2K-fallback rating the same way it
        # already badges "prior" stats — "measured" for a real computed
        # rating, "2k_fallback" when compute_ratings had nothing at all for
        # this player and NBA 2K filled the gap (see
        # player_ratings.two_k_fallback_ratings for why that's a gap-filler
        # and never a silent override of a real one).
        **{
            f"rating_{k}": v
            for k, v in ratings.get(str(player_id), {}).items()
            if k != "rating_source"
        },
        "rating_source": ratings.get(str(player_id), {}).get("rating_source"),
    }


@router.get("/players/search")
def search_players(q: str = Query(..., min_length=2), season: Optional[str] = None, limit: int = 10):
    """
    Autocomplete player search (defaults to the latest ingested season, i.e.
    whoever is currently rostered). The season filter on `players` is the
    "who's active right now" gate; stats for each match are resolved via
    the measured -> prior-season -> position-prior fallback chain (see
    src/inference/player_lookup.py), since a just-started season has no
    real stats yet for anyone, rookies included.
    """
    if _state.db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")

    season = season or _state.latest_season
    with _state.db_engine.connect() as conn:
        id_rows = conn.execute(text("""
            SELECT DISTINCT player_id FROM players
            WHERE LOWER(UNACCENT(name)) LIKE LOWER(UNACCENT(:q)) AND season = :season
            ORDER BY name
            LIMIT :limit
        """), {"q": f"%{q}%", "season": season, "limit": limit}).fetchall()

        # Ratings are percentile ranks over the whole league, so they are
        # computed for the season once (and cached) rather than per player.
        ratings = rating_lookup(_state.db_engine, season)
        results = [_player_card(conn, pid, season, ratings) for (pid,) in id_rows]

    return [r for r in results if r is not None]
