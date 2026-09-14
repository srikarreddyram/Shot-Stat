"""Rookie-season position-bucket priors — the one sanctioned exception to
this project's "exact data or NULL" policy, used only as a gap-filler for
players with zero NBA shot history."""
from sqlalchemy import Column, String, Integer, Float

from .base import Base


class PositionPrior(Base):
    """
    Historical rookie-season averages, bucketed by broad position group
    (G / F / C). This is the ONLY sanctioned exception to the project's
    "exact data or NULL, never impute" policy: it exists solely to give a
    labeled, clearly-flagged starting estimate for a player with zero NBA
    shot history (e.g. this year's draft class before they've played a
    single game). Consumers must surface that a value came from here
    (stats_source="prior") rather than presenting it as measured data.

    "Rookie season" = a player's earliest season present in the Shots /
    DefenderStats tables. Priors are computed across ALL historical rookie
    seasons in the dataset, not just the most recent one.

    One row per (position_bucket, stat_type, stat_key):
      stat_type="zone"     stat_key = one of the 6 PlayerZoneStats zones
      stat_type="defense"  stat_key = one of the 6 DefenderStats categories
      stat_type="overall"  stat_key = "career" (season_fg_pct/3p_pct/ft_pct/ast/tov)

    Populated by src/training/position_priors.py.
    """
    __tablename__ = "position_priors"

    position_bucket = Column(String, primary_key=True)  # "G", "F", or "C"
    stat_type = Column(String, primary_key=True)         # "zone" | "defense" | "overall"
    stat_key = Column(String, primary_key=True)           # zone / defense_category / "career"

    fg_pct = Column(Float, nullable=True)
    fg3_pct = Column(Float, nullable=True)
    d_fg_pct = Column(Float, nullable=True)
    pct_plusminus = Column(Float, nullable=True)
    ast = Column(Float, nullable=True)
    tov = Column(Float, nullable=True)
    ft_pct = Column(Float, nullable=True)

    sample_size = Column(Integer, nullable=True)   # rookie-seasons or shots/attempts underlying this row
    computed_through_season = Column(String, nullable=True)  # most recent season included, for reproducibility

    def __repr__(self):
        return f"<PositionPrior {self.position_bucket} {self.stat_type}:{self.stat_key}>"
