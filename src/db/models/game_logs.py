"""Per-game box scores: one row per player per game played, and one row per
team per game.

Source: the NBA's LeagueGameLog endpoint (player and team modes), which
returns every game of a season in a single call. These back the season-by-
season averages on player and team profiles (src/inference/season_averages.py).
"""
from sqlalchemy import Column, Date, Float, Index, Integer, String

from .base import Base


class PlayerGameLog(Base):
    __tablename__ = "player_game_logs"

    player_id = Column(String, primary_key=True)
    game_id = Column(String, primary_key=True)

    season = Column(String, nullable=False)
    season_type = Column(String, nullable=False)  # "Regular Season" / "Playoffs"
    game_date = Column(Date, nullable=False)
    player_name = Column(String, nullable=True)
    team_id = Column(String, nullable=False)
    team_abbrev = Column(String, nullable=False)
    opp_abbrev = Column(String, nullable=False)
    home = Column(Integer, nullable=False)  # 1 home, 0 away
    win = Column(Integer, nullable=True)

    minutes = Column(Float, nullable=True)
    pts = Column(Integer, nullable=True)
    reb = Column(Integer, nullable=True)
    oreb = Column(Integer, nullable=True)
    dreb = Column(Integer, nullable=True)
    ast = Column(Integer, nullable=True)
    stl = Column(Integer, nullable=True)
    blk = Column(Integer, nullable=True)
    tov = Column(Integer, nullable=True)
    pf = Column(Integer, nullable=True)
    fgm = Column(Integer, nullable=True)
    fga = Column(Integer, nullable=True)
    fg3m = Column(Integer, nullable=True)
    fg3a = Column(Integer, nullable=True)
    ftm = Column(Integer, nullable=True)
    fta = Column(Integer, nullable=True)
    plus_minus = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_pgl_player_date", "player_id", "game_date"),
        Index("ix_pgl_team_game", "team_id", "game_id"),
    )


class TeamGameLog(Base):
    """A team's official line for one game. Opponent points are
    pts − plus_minus, so no second lookup is needed for points allowed."""
    __tablename__ = "team_game_logs"

    team_id = Column(String, primary_key=True)
    game_id = Column(String, primary_key=True)

    season = Column(String, nullable=False)
    season_type = Column(String, nullable=False)
    game_date = Column(Date, nullable=False)
    team_abbrev = Column(String, nullable=False)
    opp_abbrev = Column(String, nullable=False)
    home = Column(Integer, nullable=False)
    win = Column(Integer, nullable=True)

    pts = Column(Integer, nullable=True)
    reb = Column(Integer, nullable=True)
    oreb = Column(Integer, nullable=True)
    dreb = Column(Integer, nullable=True)
    ast = Column(Integer, nullable=True)
    stl = Column(Integer, nullable=True)
    blk = Column(Integer, nullable=True)
    tov = Column(Integer, nullable=True)
    pf = Column(Integer, nullable=True)
    fgm = Column(Integer, nullable=True)
    fga = Column(Integer, nullable=True)
    fg3m = Column(Integer, nullable=True)
    fg3a = Column(Integer, nullable=True)
    ftm = Column(Integer, nullable=True)
    fta = Column(Integer, nullable=True)
    plus_minus = Column(Float, nullable=True)

    __table_args__ = (Index("ix_tgl_team_season", "team_id", "season"),)
