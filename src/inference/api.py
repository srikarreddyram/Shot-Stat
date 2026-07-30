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
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.inference.recommender import ShotRecommender
from src.db.database import get_engine

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NBA Shot Quality Engine",
    description="Predicts make probability and recommends optimal shot locations based on player + defender + game state.",
    version="1.0.0",
)

# CORS — allow the frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the recommender at startup
recommender: ShotRecommender = None
db_engine = None


@app.on_event("startup")
def load_model():
    global recommender, db_engine
    db_engine = get_engine()
    # Try v3 first (tuned + new features), then v2 (calibrated), fall back to v1
    for version in ["v3", "v2", "v1"]:
        try:
            recommender = ShotRecommender(model_version=version)
            break
        except Exception as e:
            print(f"⚠ Could not load model {version}: {e}")
    if recommender is None:
        print("  API will start but /recommend endpoints won't work until a model is trained.")


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
    top_n: int = Field(default=10, ge=1, le=30, description="Number of recommendations to return")
    rest_days: int = Field(default=1, ge=0, le=5, description="Days of rest (0 = back-to-back)")
    is_back_to_back: int = Field(default=0, ge=0, le=1, description="1 if second game in two days")
    opp_def_rating: float = Field(default=112.0, description="Opponent team defensive rating")


class ShotRecommendation(BaseModel):
    zone: str
    loc_x: float
    loc_y: float
    shot_type: str
    shot_distance: float
    make_probability: float
    expected_points: float
    shot_quality_score: float
    difficulty_score: float


class ZoneSummary(BaseModel):
    zone: str
    best_make_prob: float
    best_ep: float
    avg_make_prob: float
    avg_ep: float
    best_quality: float
    shot_type: str


class RecommendResponse(BaseModel):
    player_id: str
    player_name: str
    season: str
    game_state: dict
    recommendations: list[ShotRecommendation]


class SummaryResponse(BaseModel):
    player_id: str
    player_name: str
    season: str
    game_state: dict
    zone_summary: list[ZoneSummary]


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": recommender is not None,
        "model_version": recommender.version if recommender else None,
        "features": len(recommender.feature_cols) if recommender else 0,
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
    player = recommender._get_player_data(req.player_id, req.season)

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
        },
        recommendations=results.to_dict("records"),
    )


@app.post("/recommend/summary", response_model=SummaryResponse)
def recommend_summary(req: RecommendRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        summary = recommender.recommend_summary(
            player_id=req.player_id,
            season=req.season,
            quarter=req.quarter,
            time_remaining=req.time_remaining,
            score_diff=req.score_diff,
            home_away=req.home_away,
            playoff_flag=req.playoff_flag,
            defender_id=req.defender_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    player = recommender._get_player_data(req.player_id, req.season)

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
    )


@app.get("/player/{player_id}")
def get_player(player_id: str, season: str = "2023-24"):
    """Look up a player's stats for a given season."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        player = recommender._get_player_data(player_id, season)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "player_id": player_id,
        "season": season,
        **player,
    }


@app.get("/players/search")
def search_players(q: str = Query(..., min_length=2), season: str = "2024-25", limit: int = 10):
    """Autocomplete player search. Returns matching players with stats."""
    if db_engine is None:
        raise HTTPException(status_code=503, detail="Database not loaded.")

    with db_engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT p.player_id, p.name, p.position, p.height, p.weight,
                   p.career_fg_pct, p.season_fg_pct, p.career_3p_pct, ds.d_fg_pct, ds.pct_plusminus, p.wingspan,
                   p.ast, p.tov, p.ft_pct, z_rim.fg_pct, z_mid.fg_pct
            FROM players p
            LEFT JOIN defender_stats ds ON p.player_id = ds.player_id AND ds.season = p.season AND ds.defense_category = 'Overall'
            LEFT JOIN player_zone_stats z_rim ON p.player_id = z_rim.player_id AND z_rim.season = p.season AND z_rim.zone = 'Restricted Area'
            LEFT JOIN player_zone_stats z_mid ON p.player_id = z_mid.player_id AND z_mid.season = p.season AND z_mid.zone = 'Mid-Range'
            WHERE LOWER(p.name) LIKE LOWER(:q)
              AND p.season = :season
            ORDER BY p.name
            LIMIT :limit
        """), {"q": f"%{q}%", "season": season, "limit": limit}).fetchall()

    return [
        {
            "player_id": str(r[0]),
            "name": r[1],
            "position": r[2],
            "height": r[3],
            "weight": r[4],
            "career_fg_pct": r[5],
            "season_fg_pct": r[6],
            "career_3p_pct": r[7],
            "def_rating": r[8],
            "contest_rate": r[9],
            "wingspan": r[10],
            "ast": r[11],
            "tov": r[12],
            "ft_pct": r[13],
            "rim_pct": r[14],
            "mid_pct": r[15],
            "headshot_url": f"https://cdn.nba.com/headshots/nba/latest/1040x760/{r[0]}.png",
        }
        for r in rows
    ]


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
            rest_days=req.rest_days,
            is_back_to_back=req.is_back_to_back,
            opp_def_rating=req.opp_def_rating,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    player = recommender._get_player_data(req.player_id, req.season)

    return {
        "player_id": req.player_id,
        "player_name": player["name"],
        "season": req.season,
        **result,
    }


@app.get("/matchup/{attacker_id}/{defender_id}")
def get_matchup(attacker_id: str, defender_id: str, season: str = "2024-25"):
    """Get the full physical and statistical mismatch breakdown."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        attacker = recommender._get_player_data(attacker_id, season)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Attacker {attacker_id} not found in {season}")

    defender = recommender._get_defender_data(defender_id, season)
    if defender is None:
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
    def_overall = defender.get("def_stats", {}).get("Overall", {})

    # Zone-by-zone exploit analysis
    zones = ["Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
             "Left Corner 3", "Right Corner 3", "Above the Break 3"]
    exploit_zones = []
    for zone in zones:
        atk_eff = attacker.get("zone_stats", {}).get(zone)
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
        },
        "defender": {
            "player_id": defender_id,
            "name": defender.get("name"),
            "position": defender.get("position"),
            "headshot_url": f"https://cdn.nba.com/headshots/nba/latest/1040x760/{defender_id}.png",
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
