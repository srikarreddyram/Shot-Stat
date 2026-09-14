"""Game-level who-guarded-whom matchup minutes."""
from sqlalchemy import Column, String, Integer, Float, Index

from .base import Base


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
        # point_in_time.lookup_defender_category_rates filters by
        # defense_player_id on every /recommend, /matchup, and (as of the
        # on-court lineup work) /explain/matchup call — up to 5 times per
        # request for the latter, once per help defender. There was no
        # index for that filter at all: measured at ~3.2s PER CALL on the
        # live 2.2M-shot / matchups table, a full scan every time. This is
        # what made /explain/matchup take 22s end to end.
        Index("ix_matchups_defense", "defense_player_id", "game_id"),
    )

    def __repr__(self):
        return f"<Matchup {self.game_id} OFF:{self.offense_player_id} DEF:{self.defense_player_id}>"
