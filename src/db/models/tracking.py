"""SportVU-tracking-derived tables: season-level handle/passing/rim-pressure
skill, per-game contest-distance distribution, and per-season shot-
difficulty-context splits."""
from sqlalchemy import Column, String, Integer, Float, Date, Index

from .base import Base


class PlayerTrackingStats(Base):
    """
    Season-level player tracking ("SportVU") stats describing CREATION SKILL —
    how well a player handles the ball and passes, as distinct from how well
    they shoot.

    One row per (player_id, season). Available 2013-14 onward only; earlier
    seasons have no tracking cameras and stay absent (never imputed).

    Why this table exists
    ---------------------
    Two players with an identical Mid-Range FG% are not equally good shooters
    if one gets there off a screen with four feet of space and the other
    creates the look himself against a set defender. Shooting splits alone
    confound *shooting skill* with *shot difficulty*, and shot difficulty is
    largely a function of handle. These columns let the model separate them,
    and — more importantly — let the recommender estimate whether a player can
    actually GENERATE a given shot, not just whether they'd make it.

    Populated by src/ingestion/tracking_ingestor.py from LeagueDashPtStats
    measure types Drives, Passing, and Possessions.
    """
    __tablename__ = "player_tracking_stats"

    player_id = Column(String, primary_key=True)
    season = Column(String, primary_key=True)  # e.g. "2023-24"

    gp = Column(Integer, nullable=True)
    min_per_game = Column(Float, nullable=True)

    # ── Handle / on-ball load (measure type: Possessions) ──────────────────
    touches = Column(Float, nullable=True)              # per game
    front_ct_touches = Column(Float, nullable=True)     # frontcourt touches per game
    time_of_poss = Column(Float, nullable=True)         # minutes of possession per game
    avg_sec_per_touch = Column(Float, nullable=True)    # how long they hold it
    avg_drib_per_touch = Column(Float, nullable=True)   # the core "handle usage" signal
    pts_per_touch = Column(Float, nullable=True)
    elbow_touches = Column(Float, nullable=True)
    post_touches = Column(Float, nullable=True)
    paint_touches = Column(Float, nullable=True)

    # ── Rim pressure (measure type: Drives) ───────────────────────────────
    drives = Column(Float, nullable=True)               # per game
    drive_fg_pct = Column(Float, nullable=True)         # finishing off the bounce
    drive_pts = Column(Float, nullable=True)
    drive_passes_pct = Column(Float, nullable=True)     # share of drives kicked out
    drive_ast_pct = Column(Float, nullable=True)        # share of drives → assist
    drive_tov_pct = Column(Float, nullable=True)        # share of drives → turnover
    drive_pf_pct = Column(Float, nullable=True)         # share of drives drawing a foul

    # ── Playmaking / gravity (measure type: Passing) ──────────────────────
    passes_made = Column(Float, nullable=True)          # per game
    passes_received = Column(Float, nullable=True)
    ast = Column(Float, nullable=True)
    secondary_ast = Column(Float, nullable=True)        # hockey assists
    potential_ast = Column(Float, nullable=True)        # passes that WOULD be assists if made
    ast_points_created = Column(Float, nullable=True)
    ast_to_pass_pct = Column(Float, nullable=True)
    ast_to_pass_pct_adj = Column(Float, nullable=True)  # includes potential assists

    __table_args__ = (
        Index("ix_tracking_player_season", "player_id", "season"),
    )

    def __repr__(self):
        return f"<PlayerTrackingStats {self.player_id} {self.season} drib/touch:{self.avg_drib_per_touch}>"


class PlayerGameContest(Base):
    """
    Per-player-per-GAME shot distribution across closest-defender distance
    bands — how contested a player's looks actually were on a given night.

    One row per (player_id, game_date, def_dist_range).

    Why this table exists
    ---------------------
    Contest level is the single largest missing input to shot quality. Per-SHOT
    defender distance is not published by any public endpoint — that remains
    true, and it is the model's hard noise floor. What IS available, and what
    this table captures, is the distribution one level up.

    `player_shot_profile` already stores the same four bands, but only as a
    SEASON aggregate: "Luka takes 30% of his shots wide open" across a whole
    year. That describes a player, not a night, and it cannot be made
    point-in-time. Pulled per game date instead, the same figures become
    accumulable over strictly prior games, which is the construction every
    other feature in `point_in_time.py` uses.

    Provenance: LeagueDashPlayerPtShot with DateFrom == DateTo, one call per
    (date, band). The endpoint honours the date filter — verified by
    reconciliation, not assumption: Doncic on 2024-01-26 returns 2 + 8 + 23 + 0
    = 33 FGA against exactly 33 shots for that game in the `shots` table.

    Note `shotchartdetail` does NOT honour a CloseDefDistRange parameter; it
    silently ignores it and returns the unfiltered set, so per-shot bands
    cannot be recovered that way.

    Populated by src/ingestion/contest_ingestor.py.
    """
    __tablename__ = "player_game_contest"

    player_id = Column(String, primary_key=True)
    game_date = Column(Date, primary_key=True)
    # "0-2 Feet - Very Tight" | "2-4 Feet - Tight"
    # | "4-6 Feet - Open"     | "6+ Feet - Wide Open"
    def_dist_range = Column(String, primary_key=True)

    season = Column(String, nullable=True)
    fga = Column(Integer, nullable=True)
    fgm = Column(Integer, nullable=True)
    fg_pct = Column(Float, nullable=True)
    # Share of that player's attempts that night falling in this band, as the
    # endpoint reports it.
    fga_frequency = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_contest_player_date", "player_id", "game_date"),
        Index("ix_contest_date", "game_date"),
    )

    def __repr__(self):
        return (f"<PlayerGameContest {self.player_id} {self.game_date} "
                f"{self.def_dist_range}: {self.fgm}/{self.fga}>")


class PlayerShotProfile(Base):
    """
    Per-player-season shooting splits broken out by SHOT DIFFICULTY CONTEXT —
    how many dribbles preceded the shot, how long the ball was held, and how
    close the nearest defender was.

    One row per (player_id, season, split_type, split_value).

    split_type / split_value pairs:
        "dribbles"   → "0 Dribbles" | "1 Dribble" | "2 Dribbles"
                       | "3-6 Dribbles" | "7+ Dribbles"
        "def_dist"   → "0-2 Feet - Very Tight" | "2-4 Feet - Tight"
                       | "4-6 Feet - Open" | "6+ Feet - Wide Open"
        "touch_time" → "Touch < 2 Seconds" | "Touch 2-6 Seconds"
                       | "Touch 6+ Seconds"
        "general"    → "Catch and Shoot" | "Pull Ups" | "Less Than 10 ft"

    Why this table exists
    ---------------------
    Two things the rest of the pipeline cannot see anywhere else:

      1. **Openness propensity.** Per-shot defender distance is not available
         from any public endpoint, so contest level is unobservable at the
         shot level — that is the model's hard noise floor. The `def_dist`
         split recovers it at the PLAYER level: the share of a player's shots
         that come wide open is a stable, measurable trait, and it is exactly
         the trait that separates a self-creator from a spot-up specialist.

      2. **Skill net of difficulty.** The gap between a player's Pull Ups
         FG% and their Catch and Shoot FG% measures how much efficiency they
         retain when they have to make the shot themselves.

    FGA_FREQUENCY is the share of that player's total attempts falling in the
    split, so the splits within a split_type form a distribution summing to ~1.

    Populated by src/ingestion/shot_profile_ingestor.py from
    LeagueDashPlayerPtShot (one league-wide call per season per split value).
    """
    __tablename__ = "player_shot_profile"

    player_id = Column(String, primary_key=True)
    season = Column(String, primary_key=True)
    split_type = Column(String, primary_key=True)   # dribbles | def_dist | touch_time | general
    split_value = Column(String, primary_key=True)  # e.g. "3-6 Dribbles"

    gp = Column(Integer, nullable=True)
    fga_frequency = Column(Float, nullable=True)  # share of the player's total FGA
    fgm = Column(Integer, nullable=True)
    fga = Column(Integer, nullable=True)
    fg_pct = Column(Float, nullable=True)
    efg_pct = Column(Float, nullable=True)
    fg3m = Column(Integer, nullable=True)
    fg3a = Column(Integer, nullable=True)
    fg3_pct = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_shot_profile_player_season", "player_id", "season"),
        Index("ix_shot_profile_split", "split_type", "split_value"),
    )

    def __repr__(self):
        return (f"<PlayerShotProfile {self.player_id} {self.season} "
                f"{self.split_type}={self.split_value}: {self.fg_pct}>")
