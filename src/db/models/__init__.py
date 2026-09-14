"""
SQLAlchemy ORM models for the NBA Shot Quality Engine.

This used to be one 878-line models.py holding 21 unrelated table classes.
It is now one file per domain — games.py, players.py, shots.py, defense.py,
tracking.py, priors.py, matchups.py, live.py, ratings.py, play_type.py —
each just the tables that actually belong together, plus this __init__.py
re-exporting all of them so every existing `from src.db.models import X`
across the codebase keeps working unchanged.

All of them share the one `Base` in base.py. That matters beyond style:
SQLAlchemy's declarative mapper resolves string-based `relationship(...)`
references (e.g. Game.shots = relationship("Shot", ...)) by class name
against Base's shared registry, not by import path — so Game and Shot can
live in different files and still find each other, as long as both get
imported (which happens here, unconditionally, before anything queries
either one).
"""
from .base import Base
from .games import Game, TeamStats, TeamSchedule
from .players import Player, PlayerZoneStats
from .shots import Shot, ShotContext, ShotArchetype, ShotOnCourt
from .defense import REGULAR_SEASON, DefenderStats, PlayerDefensiveActivity
from .tracking import PlayerTrackingStats, PlayerGameContest, PlayerShotProfile
from .priors import PositionPrior
from .matchups import Matchup
from .live import LiveShotScore
from .ratings import PlayerTwoKRating
from .play_type import PlayerPlayType

__all__ = [
    "Base",
    "Game", "TeamStats", "TeamSchedule",
    "Player", "PlayerZoneStats",
    "Shot", "ShotContext", "ShotArchetype", "ShotOnCourt",
    "REGULAR_SEASON", "DefenderStats", "PlayerDefensiveActivity",
    "PlayerTrackingStats", "PlayerGameContest", "PlayerShotProfile",
    "PositionPrior",
    "Matchup",
    "LiveShotScore",
    "PlayerTwoKRating",
    "PlayerPlayType",
]
