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
    Index, Boolean
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
