"""Defensive quality tables: season-level FG%-allowed by category, and
season-level activity (blocks/steals/deflections/hustle) that FG%-allowed
alone can't see."""
from sqlalchemy import Column, String, Integer, Float, Index

from .base import Base

#: The season type every consumer should filter to unless it explicitly wants
#: playoff defence. Defined once here so the string is not retyped per query.
REGULAR_SEASON = "Regular Season"


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

    Populated by defender_stats_ingestor.py, which ingests BOTH regular season
    and playoffs. season_type is therefore part of the primary key: without it
    the playoff pass overwrote the regular-season row for every player whose
    team made the postseason, replacing a full season with a handful of games
    (Gobert's 2024-25 read 15 games and 223 shots defended — Minnesota's
    playoff run — instead of his 72-game regular season). Roughly 45% of
    rotation players were affected, and the corrupted rows fed the model's
    defender features. Consumers that want a full season must filter
    season_type explicitly; `REGULAR_SEASON` above is the default everywhere.
    """
    __tablename__ = "defender_stats"

    player_id = Column(String, primary_key=True)
    season = Column(String, primary_key=True)           # e.g. "2023-24"
    defense_category = Column(String, primary_key=True)  # e.g. "Overall"
    season_type = Column(String, primary_key=True, default="Regular Season")

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


class PlayerDefensiveActivity(Base):
    """
    Season-level defensive activity — how much a player disrupts a possession
    that isn't already captured by FG%-allowed: blocks, steals, deflections.

    One row per (player_id, season). `stl`/`blk`/`gp`/`min_per_game` come from
    LeagueDashPlayerStats (Base, PerGame); `deflections` comes from the
    separate LeagueHustleStatsPlayer endpoint (PerGame) — hustle stats have
    tracked deflections league-wide since 2016-17, matching this project's
    training window with no gap to backfill around.

    Why this exists: def_fg_pct_zone and friends only describe a shot that
    was ALREADY TAKEN — they say nothing about whether a shot was attempted
    at all. A dominant rim protector suppresses opponents' willingness to
    even attack the paint, not just their odds of finishing there once they
    do (the case that motivated this table: opponents visibly stop
    attempting layups with a shot-blocker of Victor Wembanyama's caliber
    patrolling the restricted area, whether or not he is the shot's primary
    defender). That's a floor-presence effect, so these columns feed the
    on-court lineup context (src/features/point_in_time.build_lineup_context)
    rather than the primary-defender feature group.

    Populated by src/ingestion/defensive_activity_ingestor.py.
    """
    __tablename__ = "player_defensive_activity"

    player_id = Column(String, primary_key=True)
    season = Column(String, primary_key=True)  # e.g. "2023-24"

    gp = Column(Integer, nullable=True)
    min_per_game = Column(Float, nullable=True)

    stl = Column(Float, nullable=True)          # per game
    blk = Column(Float, nullable=True)          # per game
    deflections = Column(Float, nullable=True)  # per game

    # The rest of the Base (LeagueDashPlayerStats) response, added alongside
    # stl/blk — same call that already fetches this table's gp/min_per_game,
    # already returns full box-score volume including points and rebounds,
    # and neither had ever been captured anywhere in this project until now.
    # ast/tov are deliberately absent: `players.ast`/`players.tov` (from
    # roster_ingestor.py, the same underlying endpoint) already own those —
    # this project has hit the two-sources-of-truth bug before and does not
    # need a second copy of the same number under a different name.
    pts = Column(Float, nullable=True)          # per game
    fgm = Column(Float, nullable=True)          # per game
    fga = Column(Float, nullable=True)          # per game
    fg3m = Column(Float, nullable=True)         # per game
    fg3a = Column(Float, nullable=True)         # per game
    ftm = Column(Float, nullable=True)          # per game
    fta = Column(Float, nullable=True)          # per game
    oreb = Column(Float, nullable=True)         # per game
    dreb = Column(Float, nullable=True)         # per game
    reb = Column(Float, nullable=True)          # per game
    pf = Column(Float, nullable=True)           # personal fouls per game
    pfd = Column(Float, nullable=True)          # fouls drawn per game
    plus_minus = Column(Float, nullable=True)   # on-court net rating, NBA's own
    dd2 = Column(Integer, nullable=True)        # double-doubles, season count
    td3 = Column(Integer, nullable=True)        # triple-doubles, season count

    # The rest of LeagueHustleStatsPlayer, added alongside deflections —
    # same endpoint, same call, no extra API cost. All per game, all
    # nullable, all 2016-17 onward like deflections.
    screen_ast = Column(Float, nullable=True)               # SCREEN_ASSISTS
    screen_ast_pts = Column(Float, nullable=True)            # SCREEN_AST_PTS
    off_boxouts = Column(Float, nullable=True)               # OFF_BOXOUTS
    def_boxouts = Column(Float, nullable=True)               # DEF_BOXOUTS
    box_outs = Column(Float, nullable=True)                  # BOX_OUTS (total)
    off_loose_balls_recovered = Column(Float, nullable=True)  # OFF_LOOSE_BALLS_RECOVERED
    def_loose_balls_recovered = Column(Float, nullable=True)  # DEF_LOOSE_BALLS_RECOVERED
    loose_balls_recovered = Column(Float, nullable=True)      # LOOSE_BALLS_RECOVERED (total)
    charges_drawn = Column(Float, nullable=True)              # CHARGES_DRAWN
    contested_shots = Column(Float, nullable=True)            # CONTESTED_SHOTS (total)
    contested_shots_2pt = Column(Float, nullable=True)        # CONTESTED_SHOTS_2PT
    contested_shots_3pt = Column(Float, nullable=True)        # CONTESTED_SHOTS_3PT

    __table_args__ = (
        Index("ix_defensive_activity_player_season", "player_id", "season"),
    )

    def __repr__(self):
        return f"<PlayerDefensiveActivity {self.player_id} {self.season} blk:{self.blk}>"
