"""Synergy play-type tendency and efficiency."""
from sqlalchemy import Column, String, Integer, Float, Index

from .base import Base


class PlayerPlayType(Base):
    """
    Season-level offensive tendency and efficiency by Synergy play type —
    isolation, both pick-and-roll roles (ball-handler and roll man), post-up,
    spot-up, hand-off, cut, off-ball screen, transition, and putbacks
    (offensive rebounds put back up).

    One row per (player_id, season, play_type). Data begins 2015-16, the
    season the NBA's stats site began publishing Synergy-sourced play-type
    data; earlier seasons have no rows (never imputed).

    Why this exists
    ----------------
    Everything else this project measures about a player's offense is either
    a court-location zone (player_zone_stats) or a shot-difficulty context
    (player_shot_profile) — neither says what ACTION produced the shot. Two
    players with identical mid-range FG% can get there completely
    differently: one living in the post, the other coming off a wall of
    screens. Play type is that action, and PPP (points per possession) is
    its own efficiency currency — comparable across play types in a way FG%
    is not, since a possession can end in a trip to the line or a turnover
    without ever producing a field-goal attempt at all.

    poss/pts/fgm/fga are TOTALS for the season (per_mode_simple="Totals" at
    ingestion), not per-game rates — deliberately, so a career figure can be
    recomputed from summed totals later rather than needing to be un-averaged
    the way a PerGame pull would require.

    Populated by src/ingestion/playtype_ingestor.py from SynergyPlayTypes,
    offensive grouping only (this project does not yet track play-type
    defense — the same endpoint supports it via type_grouping="defensive"
    if that becomes worth adding later).
    """
    __tablename__ = "player_play_type"

    player_id = Column(String, primary_key=True)
    season = Column(String, primary_key=True)
    play_type = Column(String, primary_key=True)  # e.g. "Isolation", "PRBallHandler"

    gp = Column(Integer, nullable=True)
    poss = Column(Integer, nullable=True)          # total possessions of this play type
    poss_pct = Column(Float, nullable=True)        # share of this player's own offensive possessions
    pts = Column(Integer, nullable=True)
    fgm = Column(Integer, nullable=True)
    fga = Column(Integer, nullable=True)
    fg_pct = Column(Float, nullable=True)
    efg_pct = Column(Float, nullable=True)
    ppp = Column(Float, nullable=True)             # points per possession (NBA's own figure)
    percentile = Column(Float, nullable=True)      # NBA's own percentile rank within this play type

    __table_args__ = (
        Index("ix_player_play_type_player_season", "player_id", "season"),
    )

    def __repr__(self):
        return f"<PlayerPlayType {self.player_id} {self.season} | {self.play_type}: {self.ppp} PPP>"
