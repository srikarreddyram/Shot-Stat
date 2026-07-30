"""
SQLAlchemy ORM models for the NBA Shot Quality Engine.

Tables:
  - Game            — one row per NBA game
  - Player          — one row per player per season (stats are season-specific)
  - PlayerZoneStats — one row per player per season per court zone (shooting efficiency)
  - Shot            — one row per shot attempt
"""
from sqlalchemy import (
    Column, String, Integer, Float, Date, ForeignKey, ForeignKeyConstraint,
    UniqueConstraint, Index, Boolean
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


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


class Player(Base):
    """
    One row per player per season. Physical attributes (height, weight, wingspan)
    are relatively stable but stats change each season.
    """
    __tablename__ = "players"

    player_id = Column(String, primary_key=True)
    season = Column(String, primary_key=True)  # e.g. "2023-24"

    name = Column(String, nullable=False)
    height = Column(Float, nullable=True)          # inches
    weight = Column(Float, nullable=True)           # lbs
    wingspan = Column(Float, nullable=True)         # inches
    wingspan_source = Column(String, nullable=True) # e.g. "NBA_API", "BREF"
    shots_fetched_reg = Column(Boolean, default=False)
    shots_fetched_ply = Column(Boolean, default=False)
    position = Column(String, nullable=True)        # PG, SG, SF, PF, C

    # Offensive stats
    career_fg_pct = Column(Float, nullable=True)
    career_3p_pct = Column(Float, nullable=True)
    season_fg_pct = Column(Float, nullable=True)
    ast = Column(Float, nullable=True)
    tov = Column(Float, nullable=True)
    ft_pct = Column(Float, nullable=True)

    # Defensive stats (Phase 2 — nullable for now)
    def_rating = Column(Float, nullable=True)
    contest_rate = Column(Float, nullable=True)
    def_fg_pct_allowed = Column(Float, nullable=True)

    def __repr__(self):
        return f"<Player {self.name} ({self.player_id}) {self.season}>"


class PlayerZoneStats(Base):
    """
    Zone-level shooting efficiency per player per season.

    One row per (player_id, season, zone) — the composite PK ensures idempotent upserts.

    Zones exactly match the NBA API 'Shot Area' breakdown:
        - Restricted Area         (at the rim)
        - In The Paint (Non-RA)   (paint outside the circle)
        - Mid-Range               (all mid-range jumpers)
        - Left Corner 3
        - Right Corner 3
        - Above the Break 3

    Populated by zone_stats_ingestor.py using PlayerDashboardByShootingSplits.
    """
    __tablename__ = "player_zone_stats"

    player_id = Column(String, primary_key=True)
    season    = Column(String, primary_key=True)  # e.g. "2023-24"
    zone      = Column(String, primary_key=True)  # one of the 6 zones above

    # Shot volume
    fgm = Column(Integer, nullable=True)   # field goals made
    fga = Column(Integer, nullable=True)   # field goal attempts
    fg_pct = Column(Float, nullable=True)  # FG%

    # 3-point breakdown (non-zero for the three 3PT zones only)
    fg3m   = Column(Integer, nullable=True)
    fg3a   = Column(Integer, nullable=True)
    fg3_pct = Column(Float, nullable=True)

    __table_args__ = (
        # Explicit composite PK already defined above; add index for fast lookups
        Index("ix_zone_stats_player_season", "player_id", "season"),
    )

    def __repr__(self):
        return f"<PlayerZoneStats {self.player_id} {self.season} | {self.zone}: {self.fg_pct:.3f}>"


class Shot(Base):
    """
    One row per shot attempt. Links to Game and Player (attacker).
    defender_id is nullable — populated in Phase 2.
    Tracking fields (touch_time, dribbles, etc.) are nullable — populated in Phase 4.
    """
    __tablename__ = "shots"

    shot_id = Column(String, primary_key=True)
    game_id = Column(String, ForeignKey("games.game_id"), nullable=False)
    player_id = Column(String, nullable=False)       # attacker
    season = Column(String, nullable=False)           # needed for Player FK
    defender_id = Column(String, nullable=True)       # Phase 2

    # Outcome
    shot_made = Column(Integer, nullable=False)       # 0 or 1

    # Spatial
    loc_x = Column(Float, nullable=True)
    loc_y = Column(Float, nullable=True)
    shot_distance = Column(Float, nullable=True)      # feet from basket
    shot_type = Column(String, nullable=True)          # "2PT Field Goal" / "3PT Field Goal"
    zone = Column(String, nullable=True)               # classified court zone
    shot_angle = Column(Float, nullable=True)          # derived from coordinates

    # Game context
    quarter = Column(Integer, nullable=True)           # 1-4 or 5+ for OT
    time_remaining = Column(Float, nullable=True)      # seconds left in period
    score_diff = Column(Integer, nullable=True)        # attacker team - opponent
    home_away = Column(Integer, nullable=True)         # 0 = away, 1 = home
    playoff_flag = Column(Integer, nullable=True)      # 0 or 1

    # Tracking data (Phase 4 — nullable)
    touch_time = Column(Float, nullable=True)
    dribbles = Column(Integer, nullable=True)
    catch_and_shoot = Column(Integer, nullable=True)   # 0 or 1
    closest_defender_dist = Column(Float, nullable=True)

    # Foreign key to Player (composite: player_id + season)
    __table_args__ = (
        ForeignKeyConstraint(
            ["player_id", "season"],
            ["players.player_id", "players.season"],
        ),
        Index("ix_shots_game_id", "game_id"),
        Index("ix_shots_player_season", "player_id", "season"),
        Index("ix_shots_zone", "zone"),
    )

    # Relationships
    game = relationship("Game", back_populates="shots")
    player = relationship("Player", foreign_keys=[player_id, season])

    def __repr__(self):
        result = "Made" if self.shot_made else "Missed"
        return f"<Shot {self.shot_id} {result} by {self.player_id}>"


class DefenderStats(Base):
    """
    Season-level defensive metrics per player per defense category.

    Tells us how good each player is at defending different shot zones.
    One row per (player_id, season, defense_category).

    Defense categories (from LeagueDashPtDefend):
        - Overall
        - 3 Pointers
        - 2 Pointers
        - Less Than 6Ft
        - Less Than 10Ft
        - Greater Than 15Ft

    Populated by defender_stats_ingestor.py.
    """
    __tablename__ = "defender_stats"

    player_id = Column(String, primary_key=True)
    season = Column(String, primary_key=True)           # e.g. "2023-24"
    defense_category = Column(String, primary_key=True)  # e.g. "Overall"

    gp = Column(Integer, nullable=True)              # games played
    freq = Column(Float, nullable=True)              # frequency of defensive assignments
    d_fgm = Column(Integer, nullable=True)           # FG made against this defender
    d_fga = Column(Integer, nullable=True)           # FG attempted against this defender
    d_fg_pct = Column(Float, nullable=True)          # FG% allowed when defending
    normal_fg_pct = Column(Float, nullable=True)     # expected FG% (league baseline)
    pct_plusminus = Column(Float, nullable=True)     # difference (negative = good D)

    __table_args__ = (
        Index("ix_defender_stats_player_season", "player_id", "season"),
    )

    def __repr__(self):
        return f"<DefenderStats {self.player_id} {self.season} | {self.defense_category}: {self.d_fg_pct}>"


class Matchup(Base):
    """
    Game-level matchup data: who guarded whom for how long.

    One row per (game_id, offense_player_id, defense_player_id).
    Available from 2016-17 onward only.

    Populated by matchup_ingestor.py using BoxScoreMatchupsV3.
    """
    __tablename__ = "matchups"

    game_id = Column(String, primary_key=True)
    offense_player_id = Column(String, primary_key=True)
    defense_player_id = Column(String, primary_key=True)

    matchup_minutes = Column(Float, nullable=True)       # seconds of matchup time
    partial_possessions = Column(Float, nullable=True)   # possessions in matchup
    player_points = Column(Integer, nullable=True)       # points scored in matchup
    matchup_fgm = Column(Integer, nullable=True)         # FGM in matchup
    matchup_fga = Column(Integer, nullable=True)         # FGA in matchup
    matchup_fg_pct = Column(Float, nullable=True)        # FG% in matchup

    __table_args__ = (
        Index("ix_matchups_game", "game_id"),
        Index("ix_matchups_offense", "game_id", "offense_player_id"),
    )

    def __repr__(self):
        return f"<Matchup {self.game_id} OFF:{self.offense_player_id} DEF:{self.defense_player_id}>"


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
