"""Shot-level tables: the shot itself, its play-by-play context, its
PCA/K-Means archetype cluster, and the full on-court lineup at the time it
went up."""
from sqlalchemy import (
    Column, String, Integer, Float, ForeignKey, ForeignKeyConstraint, Index,
)
from sqlalchemy.orm import relationship

from .base import Base


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


class ShotContext(Base):
    """
    Per-shot context derived from play-by-play.

    One row per shot, keyed by the same `shot_id` the shots table uses
    (game_id + "_" + play-by-play actionNumber), so it joins directly.

    Why this table exists
    ---------------------
    Everything the model knew about *how* a shot came about was previously a
    season-level average: a player's typical dribble count, his typical
    openness. That describes a player, not a shot. Two above-the-break threes
    by the same shooter — one a catch-and-shoot off a kick-out, one a step-back
    over a set defender — were identical rows.

    Play-by-play resolves them individually:

      is_assisted    ⚠ NOT A MODEL FEATURE — this is target leakage. Assists
                     are credited only on MADE baskets, so the column predicts
                     the label perfectly (FG% is exactly 1.000 when set). It is
                     stored for descriptive use only: a player's assisted rate
                     is a genuine quantity, and it is the right denominator for
                     self-creation work. See the warning in pbp_ingestor.py.
      subtype        the league's own shot taxonomy — "Step Back Jump shot",
                     "Pullup Jump shot", "Driving Floating Jump Shot", "Dunk
                     Shot". This is the per-shot version of the self-creation
                     signal that `player_shot_profile` could only supply as a
                     season aggregate.
      is_putback     a shot immediately following an offensive rebound is a
                     different event from a halfcourt possession, and they are
                     common enough at the rim to matter.
      seconds_since_prev_event
                     a transition proxy. Early-clock shots after a live-ball
                     event convert differently from set-defense halfcourt looks,
                     and nothing else in the pipeline could see the difference.

    Populated by src/ingestion/pbp_ingestor.py (one API call per game).
    """
    __tablename__ = "shot_context"

    shot_id = Column(String, primary_key=True)
    game_id = Column(String, nullable=False)

    is_assisted = Column(Integer, nullable=True)          # 0/1
    shot_subtype = Column(String, nullable=True)          # league shot taxonomy
    action_type = Column(String, nullable=True)           # "Made Shot"/"Missed Shot"
    is_putback = Column(Integer, nullable=True)           # 0/1
    seconds_since_prev_event = Column(Float, nullable=True)
    prev_event_type = Column(String, nullable=True)
    period = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_shot_context_game", "game_id"),
    )

    def __repr__(self):
        return f"<ShotContext {self.shot_id} {self.shot_subtype} ast={self.is_assisted}>"


class ShotArchetype(Base):
    """
    One row per shot: which PCA+K-Means archetype cluster it was assigned
    to (src/analysis/shot_archetypes.py). Additive — no existing table
    changes to support it, and `shots` carries no foreign key toward it, so
    a shot with no archetype row (e.g. one taken before the model was last
    fit) is simply absent here rather than null-filled.

    `model_version` ties every row to the fit that produced it, so a re-fit
    with a different k or feature set does not silently mix cluster ids
    from two different clusterings — callers filter by the current version
    rather than assuming there is only ever one.
    """
    __tablename__ = "shot_archetypes"

    shot_id = Column(String, primary_key=True)
    cluster_id = Column(Integer, nullable=False)
    model_version = Column(String, primary_key=True)

    def __repr__(self):
        return f"<ShotArchetype {self.shot_id} cluster={self.cluster_id}>"


class ShotOnCourt(Base):
    """
    One row per (shot, player) for every one of the 10 players on the floor
    when that shot went up — the on-court lineup, not just the shooter and
    the primary defender the rest of this project already tracks.

    Reconstructed from PlayByPlayV3's substitution events (src/ingestion/
    lineup_ingestor.py), not a new data source: the same endpoint
    pbp_ingestor.py already calls per game carries `actionType ==
    "Substitution"` rows with the incoming/outgoing player ids, which is
    enough to replay who was on the floor at any point in the game without
    an extra API call.

    A tidy long table (one row per player, not five fixed columns) so a
    query for "the other four teammates" or "the four help defenders" is a
    plain filter + exclude-the-shooter, with no assumption baked in about
    which of five hardcoded slots a player occupies.
    """
    __tablename__ = "shot_on_court"

    shot_id = Column(String, primary_key=True)
    player_id = Column(String, primary_key=True)
    team_id = Column(String, nullable=False)
    # "offense" (the shooting team) or "defense" (the other team). Derived
    # from team_id at write time so a reader never has to know which side
    # shot the ball to filter by role.
    role = Column(String, nullable=False)

    __table_args__ = (
        Index("ix_shot_on_court_shot", "shot_id"),
        # The serving path (point_in_time._resolve_recent_lineup_ids) looks a
        # player up by (player_id, role) to find the last lineup he shared the
        # floor with. Without this index that is a full scan of ~17.5M rows —
        # measured at 4.25s PER CALL, twice per /explain/matchup request,
        # which was enough on its own to push that endpoint past a timeout.
        Index("ix_shot_on_court_player_role", "player_id", "role"),
    )

    def __repr__(self):
        return f"<ShotOnCourt {self.shot_id} {self.player_id} ({self.role})>"
