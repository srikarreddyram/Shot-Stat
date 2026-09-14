"""Game- and team-level tables: one game, one team-season, one team's
schedule entry."""
from sqlalchemy import Column, String, Integer, Float, Date, Index
from sqlalchemy.orm import relationship

from .base import Base


class Game(Base):
    __tablename__ = "games"

    game_id = Column(String, primary_key=True)
    date = Column(Date, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    playoff_flag = Column(Integer, nullable=False, default=0)  # 0 = regular, 1 = playoff
    home_team_win = Column(Integer, nullable=True)

    # Relationships
    shots = relationship("Shot", back_populates="game")

    def __repr__(self):
        return f"<Game {self.game_id} {self.home_team} vs {self.away_team} ({self.date})>"


class TeamStats(Base):
    """
    Season-level team statistics (defensive rating, etc.).
    """
    __tablename__ = "team_stats"

    team_id = Column(String, primary_key=True)
    season = Column(String, primary_key=True)  # e.g. "2023-24"

    team_name = Column(String, nullable=True)
    team_abbrev = Column(String, nullable=True)  # e.g. "ATL", "BOS"
    def_rating = Column(Float, nullable=True)  # Defensive rating

    def __repr__(self):
        return f"<TeamStats {self.team_name} {self.season} DefRtg:{self.def_rating}>"


class TeamSchedule(Base):
    """
    Game-level schedule data to compute rest days and back-to-backs.
    """
    __tablename__ = "team_schedule"

    team_id = Column(String, primary_key=True)
    game_id = Column(String, primary_key=True)

    date = Column(Date, nullable=False)
    season = Column(String, nullable=False)

    rest_days = Column(Integer, nullable=True)
    is_back_to_back = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_team_schedule_game", "game_id"),
        Index("ix_team_schedule_team_season", "team_id", "season"),
    )

    def __repr__(self):
        return f"<TeamSchedule {self.team_id} Game:{self.game_id} Date:{self.date}>"
