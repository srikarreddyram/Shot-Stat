"""The NBA's own advanced stats, per player and per team, per season and
season type: usage, true shooting, assist/rebound/turnover rates, on-court
ratings, pace and PIE.

Source: LeagueDashPlayerStats / LeagueDashTeamStats with MeasureType
"Advanced" (src/ingestion/advanced_stats_ingestor.py) — one request per
season per season type returns the whole league. These are official,
possession-based numbers; the project deliberately stores them rather than
re-deriving them, since the NBA's possession and on-court data is not
something the box scores alone can reproduce.

Percentages are stored as fractions (0.25, not 25), including the turnover
rate, which the API reports per 100 plays.
"""
from sqlalchemy import Column, Float, Integer, String

from .base import Base


class _AdvancedColumns:
    gp = Column(Integer, nullable=True)
    w = Column(Integer, nullable=True)
    l = Column(Integer, nullable=True)  # noqa: E741 — the NBA's own column name
    min = Column(Float, nullable=True)           # minutes per game
    off_rating = Column(Float, nullable=True)    # points scored per 100 possessions on court
    def_rating = Column(Float, nullable=True)    # points allowed per 100 possessions on court
    net_rating = Column(Float, nullable=True)
    ast_pct = Column(Float, nullable=True)
    ast_to = Column(Float, nullable=True)
    ast_ratio = Column(Float, nullable=True)     # assists per 100 possessions used
    oreb_pct = Column(Float, nullable=True)
    dreb_pct = Column(Float, nullable=True)
    reb_pct = Column(Float, nullable=True)
    tov_pct = Column(Float, nullable=True)       # turnovers per play
    efg_pct = Column(Float, nullable=True)
    ts_pct = Column(Float, nullable=True)
    pace = Column(Float, nullable=True)
    pie = Column(Float, nullable=True)
    poss = Column(Integer, nullable=True)        # possessions played, season total


class PlayerAdvancedStats(_AdvancedColumns, Base):
    __tablename__ = "player_advanced_stats"

    player_id = Column(String, primary_key=True)
    season = Column(String, primary_key=True)
    season_type = Column(String, primary_key=True)  # "Regular Season" / "Playoffs"
    player_name = Column(String, nullable=True)
    team_abbrev = Column(String, nullable=True)     # the last team he played for
    usg_pct = Column(Float, nullable=True)


class TeamAdvancedStats(_AdvancedColumns, Base):
    __tablename__ = "team_advanced_stats"

    team_id = Column(String, primary_key=True)
    season = Column(String, primary_key=True)
    season_type = Column(String, primary_key=True)
    team_name = Column(String, nullable=True)
