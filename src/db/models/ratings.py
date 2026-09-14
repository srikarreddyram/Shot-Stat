"""NBA 2K ratings — a second, independent opinion of player quality."""
from sqlalchemy import Column, String, Integer, Float, Date

from .base import Base


class PlayerTwoKRating(Base):
    """
    NBA 2K video-game ratings, scraped from 2kratings.com — a second,
    independent opinion of a player's quality to check our own computed
    ratings (src/inference/player_ratings.py) against, and a fallback for
    exactly the case that motivated this table: a player our own model has
    little or no fresh evidence for (just traded, a rookie, an
    injury-shortened recent season) still has a real, current 2K rating,
    because 2K's ratings team updates it independent of games actually
    played for this team.

    One row per player_id — NOT per season. 2K re-rates players continuously
    within a game's yearly cycle (`edition`, e.g. "NBA 2K27") rather than
    keeping a queryable historical archive the way our own box-score-derived
    tables do, so this is always "their current read," refreshed by re-
    running the ingestor rather than accumulating a season-keyed history.

    `offense_avg`/`defense_avg` are our own simple rollups (plain mean of a
    fixed attribute subset — see two_k_ratings_ingestor.py) computed at
    ingestion time so they sit on roughly the same footing as our
    off_rating/def_rating for a direct comparison; `raw_attributes` keeps
    every individual attribute 2K publishes (shooting splits, defense,
    playmaking, physicals, ...) as JSON, in case a future comparison wants
    a specific one rather than the rollup.
    """
    __tablename__ = "player_two_k_ratings"

    player_id = Column(String, primary_key=True)

    edition = Column(String, nullable=True)         # e.g. "NBA 2K27 Rating"
    overall = Column(Integer, nullable=True)
    offense_avg = Column(Float, nullable=True)
    defense_avg = Column(Float, nullable=True)
    raw_attributes = Column(String, nullable=True)  # JSON: {attribute_name: value}
    fetched_at = Column(Date, nullable=True)

    def __repr__(self):
        return f"<PlayerTwoKRating {self.player_id} overall:{self.overall}>"
