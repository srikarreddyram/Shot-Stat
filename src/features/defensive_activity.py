"""
Defensive activity — blocks, steals, deflections, and the composite built
from them: defensive gravity.

The question this answers
--------------------------
def_fg_pct_zone and its relatives describe a shot that was ALREADY TAKEN —
how likely it was to go in given who contested it. None of them describe
whether a shot was attempted at all. A dominant rim protector's biggest
effect on a possession is often invisible to that entire feature group: the
ball-handler sees him patrolling the paint and simply doesn't attack the
rim. That's a floor-presence effect — it applies to a shot no matter who the
shooter's PRIMARY defender is — so these features are built to feed the
on-court lineup context (point_in_time.build_lineup_context) as a peak-threat
signal among the four help defenders, the same way playmaking_gravity's
teammate effect does on offense.

Composite: defensive_gravity
-----------------------------
A weighted, within-season z-scored blend of three per-minute rates:

    blk_per_min          0.50   — direct rim-deterrence proxy. The
                                   motivating case (a shot-blocker of Victor
                                   Wembanyama's caliber suppressing opponents'
                                   willingness to attack the paint) is
                                   fundamentally about blocks, so it carries
                                   the most weight.
    deflections_per_min  0.30   — general disruption/havoc: a hand in every
                                   passing lane forces hastier, worse shot
                                   decisions than a possession with a quiet
                                   defense, independent of shot location.
    stl_per_min          0.20   — perimeter disruption. Weighted lightest
                                   because a gambling, high-steal defender
                                   is not always the same thing as a stout
                                   one — steals correlate with risk-taking
                                   that occasionally costs the defense a shot
                                   at the other end.

Same weighted-blend, z-score-within-season, missing-component-as-neutral
construction as creation.py's _COMPOSITES — see that module's docstring for
why blend-then-shrink beats a single raw ratio.

Availability and the one-season lag
------------------------------------
Sourced from src/ingestion/defensive_activity_ingestor.py, which pulls
LeagueDashPlayerStats (STL/BLK) and LeagueHustleStatsPlayer (DEFLECTIONS).
Hustle stats are published league-wide from 2016-17, matching this project's
whole training window — no gap to lag around on that account. Still lagged
one full season for the same reason creation profiles are: the current
season's totals do not exist yet at serving time, so training and serving
must see the same kind of number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Minimum volume before a defensive-activity profile is trusted. Same
# reasoning as creation.py's MIN_GAMES_FOR_PROFILE/MIN_MINUTES_PER_GAME_FOR_
# PROFILE: without a floor, a player with a handful of minutes can lead the
# league in blk_per_min on the strength of one lucky block.
MIN_GAMES_FOR_DEFENSIVE_PROFILE = 15
MIN_MINUTES_PER_GAME_FOR_DEFENSIVE_PROFILE = 8.0

_COMPOSITE_WEIGHTS = {
    "blk_per_min": 0.50,
    "deflections_per_min": 0.30,
    "stl_per_min": 0.20,
}

DEFENSIVE_ACTIVITY_FEATURE_COLS = [
    "stl_per_min", "blk_per_min", "deflections_per_min", "defensive_gravity",
]


def _zscore_within_season(df: pd.DataFrame, col: str) -> pd.Series:
    """Z-score a column within each season — same rationale as creation.py's
    _zscore_within: league-wide activity rates drift year over year (rule
    changes, pace-of-play shifts), so an absolute threshold would drift with
    them. Uses no outcome data, so this introduces no leakage."""
    grouped = df.groupby("season")[col]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    return (df[col] - mean) / std.replace(0, np.nan)


def load_defensive_activity_profiles(engine) -> pd.DataFrame:
    """
    Build one defensive-activity profile per (player_id, season).

    Returned `season` is the season the stats were MEASURED in — lagging is
    applied by the caller (build_lineup_context), matching how creation
    profiles are lagged by attach_creation_features.
    """
    df = pd.read_sql("""
        SELECT player_id, season, gp, min_per_game, stl, blk, deflections
        FROM player_defensive_activity
    """, engine)

    if df.empty:
        return pd.DataFrame(columns=["player_id", "season"] + DEFENSIVE_ACTIVITY_FEATURE_COLS)

    mins = df["min_per_game"].replace(0, np.nan)
    df["stl_per_min"] = df["stl"] / mins
    df["blk_per_min"] = df["blk"] / mins
    df["deflections_per_min"] = df["deflections"] / mins

    plays_enough = (
        (df["gp"].fillna(0) >= MIN_GAMES_FOR_DEFENSIVE_PROFILE)
        & (df["min_per_game"].fillna(0) >= MIN_MINUTES_PER_GAME_FOR_DEFENSIVE_PROFILE)
    )
    rate_cols = ["stl_per_min", "blk_per_min", "deflections_per_min"]
    df.loc[~plays_enough, rate_cols] = np.nan

    # Composite: weighted blend of within-season z-scores. A missing
    # component counts as league-average for this index rather than voiding
    # it outright, but a player with NONE of the inputs must come out NaN —
    # see creation.py's _COMPOSITES construction for why 0.0 there would be a
    # false "exactly average" claim rather than "unknown".
    acc = pd.Series(0.0, index=df.index)
    any_component = pd.Series(False, index=df.index)
    total_weight = 0.0
    for col, weight in _COMPOSITE_WEIGHTS.items():
        z = _zscore_within_season(df, col)
        acc = acc + weight * z.fillna(0.0)
        any_component = any_component | z.notna()
        total_weight += weight
    df["defensive_gravity"] = (acc / total_weight).where(any_component)

    keep = ["player_id", "season"] + DEFENSIVE_ACTIVITY_FEATURE_COLS
    return df[keep]
