"""The core per-player-per-season row, plus zone-level shooting efficiency
— the two tables nearly everything else in this project joins against."""
from sqlalchemy import Column, String, Integer, Float, Boolean, Index

from .base import Base


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
    team_id = Column(String, nullable=True)          # NBA team id, this season

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
