"""Pydantic request/response models for the /recommend* endpoints.

Split out on its own because src/inference/api/recommend.py imports these,
and every other router module that returns a plain dict has no need of them.
"""
from typing import Optional

from pydantic import BaseModel, Field


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
