"""
Shot Quality API — FastAPI serving layer for the recommendation engine.

Endpoints:
    GET  /health            — Health check
    POST /recommend         — Get top-N shot recommendations for a player
    POST /recommend/summary — Get zone-level summary
    GET  /player/{id}       — Look up a player's stats

Usage:
    uvicorn src.inference.api:app --reload --port 8000
"""
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.inference.recommender import ShotRecommender, ZONE_TO_DEF_CATEGORY
from src.inference.player_lookup import resolve_player_stats, resolve_defender_stats
from src.inference.player_ratings import rating_lookup
from src.db.database import get_engine

# Load the recommender at startup
recommender: ShotRecommender = None
db_engine = None
latest_season: str = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender, db_engine, latest_season
    db_engine = get_engine()
    # Model names are now descriptive rather than a v1/v2/v3 counter, and each
    # one has a runs/<stamp>__<name>/run.json recording exactly what it scored.
    # The list is ordered newest-first; the first that loads wins.
    for model_name in ["shot-quality-v19", "shot-quality-v13", "shot-quality-v12", "shot-quality-v11", "shot-quality-v10", "shot-quality-v9", "shot-quality-v8", "shot-quality-v5", "shot-quality-v4", "shot-quality", "v3", "v2", "v1"]:
        try:
            recommender = ShotRecommender(model_name=model_name)
            break
        except Exception as e:
            print(f"⚠ Could not load model {model_name}: {e}")
    if recommender is None:
        print("  API will start but /recommend endpoints won't work until a model is trained.")

    # The most recent season with ingested player data. Clients (the UI) should
    # use this instead of hardcoding a season string, which silently goes stale
    # every year and hides that season's rookies/trades from search results.
    with db_engine.connect() as conn:
        row = conn.execute(text("SELECT MAX(season) FROM players")).fetchone()
        latest_season = row[0] if row else None
    print(f"  ✓ Latest ingested season: {latest_season}")

    yield


# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NBA Shot Quality Engine",
    description="Predicts make probability and recommends optimal shot locations based on player + defender + game state.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    player_id: str = Field(..., description="NBA player ID (e.g. '2544' for LeBron)")
    season: str = Field(..., description="Season string (e.g. '2023-24')")
    quarter: int = Field(default=1, ge=1, le=8, description="Game quarter (1-4, 5+ for OT)")
    time_remaining: float = Field(default=600.0, ge=0, description="Seconds left in period")
    score_diff: int = Field(default=0, description="Attacker team score minus opponent")
    home_away: int = Field(default=1, ge=0, le=1, description="0 = away, 1 = home")
    playoff_flag: int = Field(default=0, ge=0, le=1, description="0 = regular season, 1 = playoffs")
    defender_id: Optional[str] = Field(default=None, description="Defender player ID (optional)")
    secondary_defender_id: Optional[str] = Field(default=None, description="Second defender for a double team (optional). Combined with defender_id via a heuristic — see ShotRecommender._combine_defenders.")
    top_n: int = Field(default=10, ge=1, le=30, description="Number of recommendations to return")
    rest_days: int = Field(default=1, ge=0, le=5, description="Days of rest (0 = back-to-back)")
    is_back_to_back: int = Field(default=0, ge=0, le=1, description="1 if second game in two days")
    opp_def_rating: float = Field(default=112.0, description="Opponent team defensive rating")


class ShotRecommendation(BaseModel):
    zone: str
    loc_x: float
    loc_y: float
    shot_distance: float
    make_probability: float
    expected_points: float
    # Uncertainty on the shooting rate behind this projection, from the number
    # of attempts supporting it. A corner-three number built on nine attempts
    # and one built on nine hundred used to render identically.
    ep_low: float
    ep_high: float
    attempts_behind: float
    # How readily this player can generate this look — largely a handle
    # question. Reported separately from expected points so a client can
    # distinguish "this would be a great shot for you" from "this is a shot
    # you can actually get".
    attainability: Optional[float] = None
    # The ranking quantity: expected points tempered by attainability.
    score: float
    # Aliases retained for the existing React client. `shot_quality_score`
    # mirrors `score`; `difficulty_score` is 1 - make_probability.
    shot_type: str
    shot_quality_score: float
    difficulty_score: float
    # Expected points relative to this player's own attainability-weighted
    # average, which is the number worth showing a user. Raw expected points
    # mostly restates that a three is worth more than a two.
    ep_vs_own_average: float


class ZoneSummary(BaseModel):
    zone: str
    make_probability: float
    expected_points: float
    ep_low: float
    ep_high: float
    attainability: Optional[float] = None
    attempts_behind: float
    score: float
    # Aliases retained for the existing React client.
    best_make_prob: float
    avg_make_prob: float
    best_ep: float
    avg_ep: float
    best_quality: float
    shot_type: str
    # Distance (feet) of the specific location this recommendation refers to
    # — the zone's best-scoring spot, same convention as best_mechanic below.
    # A zone spans a range of distances, so "the" distance is this one shot's,
    # not an average across the whole zone.
    shot_distance: Optional[float] = None
    best_mechanic: Optional[str] = None
    best_mechanic_prob: Optional[float] = None


class RecommendResponse(BaseModel):
    player_id: str
    player_name: str
    season: str
    game_state: dict
    recommendations: list[ShotRecommendation]
    attacker_stats_source: str = "measured"           # "measured" | "prior"
    attacker_resolved_season: Optional[str] = None
    defender_stats_source: Optional[str] = None
    defender_resolved_season: Optional[str] = None


class SummaryResponse(BaseModel):
    player_id: str
    player_name: str
    season: str
    game_state: dict
    zone_summary: list[ZoneSummary]
    attacker_stats_source: str = "measured"           # "measured" | "prior"
    attacker_resolved_season: Optional[str] = None
    defender_stats_source: Optional[str] = None
    defender_resolved_season: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": recommender is not None,
        "model_name": recommender.model_name if recommender else None,
        # Alias for the existing React client, which reads `model_version`.
        "model_version": recommender.model_name if recommender else None,
        "features": len(recommender.feature_cols) if recommender else 0,
        "latest_season": latest_season,
        # Whether a calibration mapping was adopted at training time. It is
        # deliberately not always on: the trainer adopts it only when it
        # improves held-out log-loss, and on the current data it does not.
        "calibrated": recommender.calibration_adopted if recommender else None,
        "attainability_model": (
            recommender.attainability is not None if recommender else None
        ),
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    try:
        results = recommender.recommend(
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
    player = recommender._player_row(req.player_id, req.season)
    defender = recommender._defender_row(req.defender_id, req.season) if req.defender_id else None

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


@app.post("/recommend/summary", response_model=SummaryResponse)
def recommend_summary(req: RecommendRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        summary = recommender.zone_summary(
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

    player = recommender._player_row(req.player_id, req.season)
    defender = recommender._defender_row(req.defender_id, req.season) if req.defender_id else None

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


@app.get("/player/{player_id}")
def get_player(player_id: str, season: Optional[str] = None):
    """Look up a player's stats for a given season (defaults to the latest ingested season)."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    season = season or latest_season
    try:
        player = recommender._player_row(player_id, season)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Surface the shrunk point-in-time rates rather than the raw prior counts.
    # The counts are an internal representation; a client asking about a player
    # wants the rate the model actually reasons with, small samples already
    # regressed toward the league prior.
    import pandas as _pd
    from src.features.point_in_time import ZONE_SUFFIX as _SUFFIX, apply_hierarchy as _apply

    rates = _apply(_pd.DataFrame([player]), recommender.zone_priors).iloc[0]
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
HEADSHOT_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "headshots"


@app.get("/headshot/{player_id}")
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
    and /team/{team_id}/roster so the two endpoints can't quietly drift
    apart on what fields a client gets back.
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


@app.get("/players/search")
def search_players(q: str = Query(..., min_length=2), season: Optional[str] = None, limit: int = 10):
    """
    Autocomplete player search (defaults to the latest ingested season, i.e.
    whoever is currently rostered). The season filter on `players` is the
    "who's active right now" gate; stats for each match are resolved via
    the measured -> prior-season -> position-prior fallback chain (see
    src/inference/player_lookup.py), since a just-started season has no
    real stats yet for anyone, rookies included.
    """
    if db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")

    season = season or latest_season
    with db_engine.connect() as conn:
        id_rows = conn.execute(text("""
            SELECT DISTINCT player_id FROM players
            WHERE LOWER(UNACCENT(name)) LIKE LOWER(UNACCENT(:q)) AND season = :season
            ORDER BY name
            LIMIT :limit
        """), {"q": f"%{q}%", "season": season, "limit": limit}).fetchall()

        # Ratings are percentile ranks over the whole league, so they are
        # computed for the season once (and cached) rather than per player.
        ratings = rating_lookup(db_engine, season)
        results = [_player_card(conn, pid, season, ratings) for (pid,) in id_rows]

    return [r for r in results if r is not None]


@app.get("/teams")
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


@app.get("/team/{team_id}/roster")
def get_team_roster(team_id: str, season: Optional[str] = None):
    """
    Every player rostered to `team_id` this season (players.team_id, kept
    fresh by src/ingestion/{roster,current_roster}_ingestor.py — see the
    on-court lineup work in src/features/point_in_time.py for why that
    column exists at all). Same per-player shape as /players/search, so the
    team-vs-team picker can reuse the same player card rendering.
    """
    if db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")

    season = season or latest_season
    with db_engine.connect() as conn:
        id_rows = conn.execute(text("""
            SELECT player_id FROM players
            WHERE team_id = :team_id AND season = :season
            ORDER BY name
        """), {"team_id": team_id, "season": season}).fetchall()

        ratings = rating_lookup(db_engine, season)
        results = [_player_card(conn, pid, season, ratings) for (pid,) in id_rows]

    return [r for r in results if r is not None]


@app.post("/recommend/heatmap")
def recommend_heatmap(req: RecommendRequest):
    """Return full dense grid for heatmap visualization."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        result = recommender.recommend_heatmap(
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

    player = recommender._player_row(req.player_id, req.season)
    defender = recommender._defender_row(req.defender_id, req.season) if req.defender_id else None

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


@app.get("/explain/attainability/{player_id}")
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
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        return recommender.explain_attainability(
            player_id=player_id, zone=zone, season=season or latest_season,
            loc_x=loc_x, loc_y=loc_y, defender_id=defender_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/explain/matchup/{player_id}")
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
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        return recommender.explain_matchup(
            player_id=player_id, zone=zone, loc_x=loc_x, loc_y=loc_y,
            season=season or latest_season, defender_id=defender_id,
            secondary_defender_id=secondary_defender_id, quarter=quarter,
            time_remaining=time_remaining, score_diff=score_diff,
            home_away=home_away, playoff_flag=playoff_flag,
            rest_days=rest_days, is_back_to_back=is_back_to_back,
            opp_def_rating=opp_def_rating,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/matchup/{attacker_id}/{defender_id}")
def get_matchup(attacker_id: str, defender_id: str, season: Optional[str] = None):
    """Get the full physical and statistical mismatch breakdown (defaults to the latest ingested season)."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    season = season or latest_season

    try:
        attacker = recommender._player_row(attacker_id, season)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Attacker {attacker_id} not found in {season}")

    defender = recommender._defender_row(defender_id, season)
    if not defender:
        raise HTTPException(status_code=404, detail=f"Defender {defender_id} not found in {season}")

    # Physical comparison
    physical = {}
    for attr in ["height", "weight", "wingspan"]:
        a_val = attacker.get(attr)
        d_val = defender.get(attr)
        physical[attr] = {
            "attacker": a_val,
            "defender": d_val,
            "diff": round(a_val - d_val, 1) if (a_val and d_val) else None,
        }

    height_diff = physical["height"]["diff"]
    size_mismatch = abs(height_diff) >= 4 if height_diff else False

    # Defender quality
    def_stats = defender.get("def_stats", {})
    def_overall = def_stats.get("Overall", {})

    # Zone-to-defense-category mapping (shared with recommender.py's grid scoring)
    zone_to_def_cat = ZONE_TO_DEF_CATEGORY

    # Zone-by-zone exploit analysis — uses the defender's zone-specific FG% allowed
    # (falling back to overall when the zone category is missing), so an elite rim
    # protector with mediocre overall numbers (e.g. blended with weak perimeter D)
    # is still correctly flagged as tough at the rim specifically.
    zones = ["Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
             "Left Corner 3", "Right Corner 3", "Above the Break 3"]
    # The attacker's shrunk, point-in-time rate per zone — the same quantity
    # the model consumes, rather than the raw season split the old endpoint
    # showed. Small samples are regressed to the league prior here too, so the
    # matchup screen and the recommendation cannot tell different stories.
    import pandas as _pd
    from src.features.point_in_time import ZONE_SUFFIX as _SUFFIX, apply_hierarchy as _apply
    _rates = _apply(_pd.DataFrame([attacker]), recommender.zone_priors).iloc[0]
    attacker_zone_rates = {
        z: float(_rates[f"zone_rate_{_SUFFIX[z]}"]) for z in zones
    }

    exploit_zones = []
    for zone in zones:
        atk_eff = attacker_zone_rates.get(zone)
        def_cat = zone_to_def_cat.get(zone)
        zone_def = def_stats.get(def_cat, {})
        def_fg = zone_def.get("d_fg_pct")
        if def_fg is None:
            def_fg = def_overall.get("d_fg_pct")
        advantage = None
        if atk_eff is not None and def_fg is not None:
            advantage = round(atk_eff - def_fg, 3)
        exploit_zones.append({
            "zone": zone,
            "attacker_fg_pct": atk_eff,
            "defender_fg_pct_allowed": def_fg,
            "matchup_advantage": advantage,
            "exploit": advantage is not None and advantage > 0,
        })

    # Sort by advantage (biggest exploit first)
    exploit_zones.sort(key=lambda z: z["matchup_advantage"] or -999, reverse=True)

    return {
        "attacker": {
            "player_id": attacker_id,
            "name": attacker.get("name"),
            "position": attacker.get("position"),
            "headshot_url": f"https://cdn.nba.com/headshots/nba/latest/1040x760/{attacker_id}.png",
            "stats_source": "measured",
            "resolved_season": attacker.get("_latest_season"),
        },
        "defender": {
            "player_id": defender_id,
            "name": defender.get("name"),
            "position": defender.get("position"),
            "headshot_url": f"https://cdn.nba.com/headshots/nba/latest/1040x760/{defender_id}.png",
            "stats_source": "measured",
            "resolved_season": season,
        },
        "season": season,
        "physical_comparison": physical,
        "size_mismatch": size_mismatch,
        "defender_quality": {
            "fg_pct_allowed": def_overall.get("d_fg_pct"),
            "plus_minus": def_overall.get("pct_plusminus"),
        },
        "exploit_zones": exploit_zones,
    }


# ── Live scoring demo (src/streaming) ───────────────────────────────────────
# Reads `live_shot_scores`, an append-only table the Kafka consumer writes to.
# Purely additive: no existing route above is touched, and these 404 cleanly
# (empty lists) if the streaming demo has never been run — the rest of the
# API has no dependency on it.

@app.get("/live/games")
def live_games(limit: int = Query(10, ge=1, le=50)):
    """Distinct games that have at least one live-scored shot, most recent first."""
    if db_engine is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    with db_engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT game_id, COUNT(*) AS n_scored, MAX(scored_at) AS last_scored_at
            FROM live_shot_scores
            GROUP BY game_id
            ORDER BY last_scored_at DESC
            LIMIT :limit
        """), {"limit": limit}).mappings().all()
    return {"games": [dict(r) for r in rows]}


@app.get("/live/scores")
def live_scores(game_id: Optional[str] = None, limit: int = Query(50, ge=1, le=500)):
    """
    Most recently scored live shots, optionally filtered to one game.

    This is a simulated feed (see docs/streaming.md) — `predicted_make_probability`
    comes from the same trained model `/recommend` uses, scored in real time
    as the shot event was consumed off Kafka.
    """
    if db_engine is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    clause = "WHERE game_id = :game_id" if game_id else ""
    with db_engine.connect() as conn:
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


# ── Shot archetypes (src/analysis/shot_archetypes.py) ───────────────────────
# PCA + K-Means clusters over the shot-descriptive feature set, joined from
# `shot_archetypes` (populated by `python -m src.analysis.shot_archetypes
# --fit`) back onto `shots` for court coordinates. Purely additive.

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


@app.get("/archetypes")
def list_archetypes():
    """Every archetype's label, size, FG%, and creation/finish mix."""
    metadata = _load_archetype_metadata()
    return {
        "model_version": metadata["model_version"],
        "k": metadata["k"],
        "clusters": metadata["clusters"],
    }


@app.get("/archetypes/court")
def archetypes_court(points_per_cluster: int = Query(250, ge=10, le=1000),
                     season: Optional[str] = None):
    """
    A plottable sample of real shots per archetype, for a court overlay —
    same court/coordinate convention the existing heatmap uses (loc_x, loc_y).
    """
    metadata = _load_archetype_metadata()
    if db_engine is None:
        raise HTTPException(status_code=503, detail="Database not ready")

    from src.analysis.shot_archetypes import MODEL_VERSION
    season_clause = "AND s.season = :season" if season else ""

    out = {}
    with db_engine.connect() as conn:
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


@app.get("/player/{player_id}/archetype-mix")
def player_archetype_mix(player_id: str, season: Optional[str] = None):
    """A player's real shot diet broken down by archetype cluster."""
    metadata = _load_archetype_metadata()
    if db_engine is None:
        raise HTTPException(status_code=503, detail="Database not ready")

    from src.analysis.shot_archetypes import MODEL_VERSION
    season_clause = "AND s.season = :season" if season else ""
    labels_by_id = {c["cluster_id"]: c["label"] for c in metadata["clusters"]}

    with db_engine.connect() as conn:
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


# ── Stat engine ────────────────────────────────────────────────────────────
# Every feature the engine actually reasons with, published as data rather
# than buried in source. The whole project is a feature-engineering exercise
# — point-in-time shooting rates, defender mixtures, creation profiles,
# on-court lineup context — and until this endpoint none of that was visible
# anywhere except by reading src/features/spec.py.

# Prefix-matched families. derive_features emits these as one-hot dummies via
# a prefix scan rather than listing them in FEATURE_GROUPS (see
# spec.all_feature_columns), so they need their own grouping here or they'd
# all fall through as "ungrouped".
_PREFIX_GROUPS = {
    "mech_": "shot_mechanic",
    "finish_": "finish",
    "zone_is_": "zone_indicator",
    "pos_": "position_indicator",
    "origin_": "possession_origin",
}

_GROUP_BLURBS = {
    "shooter_skill": "What this player shoots, from strictly-prior games only — never including the shot being predicted.",
    "shooter_physical": "Measured physical attributes. Exact data or NULL; never estimated from position.",
    "creation": "Handle and passing, lagged one full season. Separates shooting skill from shot difficulty.",
    "defender": "The named defender, blended as a possession-weighted mixture — nobody guards one player for a whole game.",
    "defender_physical": "Defender size. Off by default: no measurable signal, and a large spurious effect at serving time.",
    "opponent_defence": "The defending TEAM's zone-level tendencies, distinct from the individual matchup.",
    "interaction": "Explicit offense-versus-defense terms — the matchup itself rather than either side alone.",
    "spatial": "Where on the floor the shot is taken from.",
    "context": "Game state: clock, score, rest, home/away.",
    "shot_context": "Play-by-play context — what happened immediately before the shot.",
    "contest": "Per-game defender-distance bands. Off by default: measured the most harmful group in ablation.",
    "team_creation": "Who else is on the FLOOR with the shooter — teammates' creation, gravity, rim pressure and foul-drawing, mean and peak-threat.",
    "help_defense": "The other four defenders — zone-matched FG% allowed, plus blocks/steals/deflections and the defensive-gravity composite.",
    "shot_mechanic": "How the shot was created — the move itself.",
    "finish": "How the shot was finished: dunk, layup, floater, jumper.",
    "zone_indicator": "Which of the six court zones the shot is in.",
    "position_indicator": "The shooter's position bucket.",
    "possession_origin": "How the possession began — after a rebound, off a turnover, out of a dead ball.",
    "spatial_basis": "Radial spatial basis. Off by default: measured slightly harmful.",
}


def _humanize(column: str) -> str:
    """Fallback label for a feature with no hand-written copy."""
    return column.replace("_", " ").strip().capitalize()


@app.get("/stats/features")
def list_engine_features():
    """
    Every feature in the live shot-quality model, grouped, labelled, and
    ranked by how much the trained model actually leans on it.

    `gain_share` is XGBoost's gain importance normalised to a share of the
    whole model, so the groups are directly comparable to each other —
    that is the number that says a feature is doing real work rather than
    merely being present. `direction` reports a monotone constraint where
    one is bound: basketball facts the model is not permitted to
    contradict however the training data happens to wiggle (see
    src/training/train.py's MONOTONE_CONSTRAINTS).
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    from src.features.spec import FEATURE_GROUPS
    from src.inference.explain import FEATURE_COPY, SQ_FEATURE_COPY
    from src.training.train import MONOTONE_CONSTRAINTS

    feature_cols = recommender.feature_cols
    booster = recommender.model.get_booster()
    booster.feature_names = feature_cols
    gain = booster.get_score(importance_type="gain")
    total_gain = sum(gain.values()) or 1.0

    group_of: dict[str, str] = {}
    for group, columns in FEATURE_GROUPS.items():
        for column in columns:
            group_of[column] = group

    grouped: dict[str, list[dict]] = {}
    for column in feature_cols:
        group = group_of.get(column)
        if group is None:
            for prefix, prefix_group in _PREFIX_GROUPS.items():
                if column.startswith(prefix):
                    group = prefix_group
                    break
        group = group or "other"

        copy = SQ_FEATURE_COPY.get(column) or FEATURE_COPY.get(column) or {}
        grouped.setdefault(group, []).append({
            "name": column,
            "label": copy.get("label") or _humanize(column),
            "high": copy.get("high"),
            "low": copy.get("low"),
            "gain_share": gain.get(column, 0.0) / total_gain,
            "direction": (
                "raises make probability" if MONOTONE_CONSTRAINTS.get(column) == 1
                else "lowers make probability" if MONOTONE_CONSTRAINTS.get(column) == -1
                else None
            ),
        })

    groups = []
    for group, features in grouped.items():
        features.sort(key=lambda f: -f["gain_share"])
        groups.append({
            "group": group,
            "blurb": _GROUP_BLURBS.get(group),
            "n_features": len(features),
            "gain_share": sum(f["gain_share"] for f in features),
            "features": features,
        })
    groups.sort(key=lambda g: -g["gain_share"])

    return {
        "model_name": recommender.metadata.get("model_name", "shot-quality"),
        "n_features": len(feature_cols),
        "metrics": recommender.metadata.get("test_metrics", {}),
        "groups": groups,
    }


# ── Stat Engine: player/team stat browser ──────────────────────────────────
# Distinct from /stats/features above: that endpoint answers "what does the
# MODEL use and how much" (model transparency). This answers "what does
# this PLAYER or TEAM actually do", independent of any model — a SofaScore-
# style browse-everything page. See src/inference/stat_engine.py.

def _records(df) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts. NaN is pandas' native missing-
    value marker but is not valid JSON — a bare NaN reaches a browser's
    JSON.parse as a syntax error, not a null, so every leaderboard response
    has to scrub it here rather than trust a client to tolerate it."""
    return df.replace({np.nan: None}).to_dict(orient="records")


@app.get("/stats/players")
def stat_engine_players(season: Optional[str] = None):
    """Every rostered player, every stat this project tracks, one wide row
    each — sortable/filterable client-side. See stat_engine.STAT_GROUPS for
    what each column means and which section it belongs to."""
    if db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.stat_engine import build_player_stat_table, player_column_metadata

    table = build_player_stat_table(db_engine, season)
    return {
        "season": table.iloc[0]["season"] if not table.empty else season,
        "columns": player_column_metadata(table.columns),
        "players": _records(table),
    }


@app.get("/stats/player/{player_id}")
def stat_engine_player(player_id: str, season: Optional[str] = None):
    """One player's full stat profile, organized into basic-to-advanced
    sections (stat_engine.GROUP_ORDER)."""
    if db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.stat_engine import player_full_profile

    profile = player_full_profile(db_engine, player_id, season)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No stats for player {player_id}.")
    return profile


@app.get("/stats/teams")
def stat_engine_teams(season: Optional[str] = None):
    """Every team's real stat (def_rating) plus roster-derived aggregates —
    see stat_engine.build_team_stat_table's docstring for why these are
    labeled roster_avg_* rather than presented as separately-measured team
    numbers we don't actually have."""
    if db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.stat_engine import build_team_stat_table, team_column_metadata

    table = build_team_stat_table(db_engine, season)
    return {
        "season": table.iloc[0]["season"] if not table.empty else season,
        "columns": team_column_metadata(table.columns),
        "teams": _records(table),
    }


@app.get("/stats/team/{team_id}")
def stat_engine_team(team_id: str, season: Optional[str] = None):
    """One team's real stat plus its full roster, each player carrying
    their own off_rating/def_rating_ours for a quick scan."""
    if db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")
    from src.inference.stat_engine import team_full_profile

    profile = team_full_profile(db_engine, team_id, season)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No stats for team {team_id}.")
    return profile


# ── Team logos ──────────────────────────────────────────────────────────────
# Proxied and cached rather than hot-linked from the page. cdn.nba.com serves
# these happily to a plain HTTP client but rejects at least some browser
# requests with ERR_HTTP2_PROTOCOL_ERROR, so an <img> pointed straight at it
# is not dependable. Going through the API also means the logos keep working
# with no network once cached, and NBA's marks stay out of the repo.
TEAM_LOGO_CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "team_logos"
TEAM_LOGO_URL = "https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"


@app.get("/team/{team_id}/logo")
def team_logo(team_id: str):
    """The team's official logo as SVG, cached on first request."""
    # team_id is interpolated into an outbound URL and a filesystem path, so
    # it is constrained to exactly the shape a real NBA team id has. Without
    # this, a crafted id is both a path-traversal and an SSRF vector.
    if not team_id.isdigit() or len(team_id) != 10:
        raise HTTPException(status_code=404, detail="Unknown team id.")

    TEAM_LOGO_CACHE.mkdir(parents=True, exist_ok=True)
    cached = TEAM_LOGO_CACHE / f"{team_id}.svg"

    if not cached.exists():
        try:
            response = requests.get(TEAM_LOGO_URL.format(team_id=team_id), timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Could not fetch logo: {exc}")
        if "svg" not in response.headers.get("content-type", ""):
            raise HTTPException(status_code=502, detail="Logo endpoint did not return SVG.")
        cached.write_bytes(response.content)

    return FileResponse(
        cached,
        media_type="image/svg+xml",
        # A team logo does not change; let the browser keep it for a day.
        headers={"Cache-Control": "public, max-age=86400"},
    )
