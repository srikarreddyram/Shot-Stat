"""The Kafka streaming demo's append-only score log."""
from sqlalchemy import Column, String, Integer, Float, Index

from .base import Base


class LiveShotScore(Base):
    """
    Append-only log written by the Kafka streaming consumer
    (src/streaming/consumer.py) — one row per replayed shot event it scored.

    This is additive: no existing table's schema or contents change to
    support it, and nothing outside src/streaming and the /live/* API routes
    reads or writes it. `shot_id` is not a foreign key to `shots` on purpose
    — a real live feed's shot_id would not exist in that table yet.
    """
    __tablename__ = "live_shot_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    shot_id = Column(String, nullable=False)
    game_id = Column(String, nullable=False)
    player_id = Column(String, nullable=False)
    defender_id = Column(String, nullable=True)
    zone = Column(String, nullable=True)

    predicted_make_probability = Column(Float, nullable=False)
    actual_shot_made = Column(Integer, nullable=True)  # 0/1, if known

    game_date = Column(String, nullable=True)
    scored_at = Column(Float, nullable=False)     # unix timestamp
    latency_ms = Column(Float, nullable=True)      # event emit -> scored

    __table_args__ = (
        Index("ix_live_shot_scores_game", "game_id"),
    )

    def __repr__(self):
        return (f"<LiveShotScore {self.shot_id} game={self.game_id} "
                f"p={self.predicted_make_probability:.3f}>")
