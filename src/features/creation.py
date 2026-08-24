"""
Creation skill — handle, passing, and the shot difficulty they buy.

The question this answers
-------------------------
Two players both shoot 38% from above the break. One of them gets there by
running off two screens and catching a pass with four feet of daylight. The
other pulls up over a set defender after seven dribbles. Their zone rates are
identical and their shooting ability plainly is not.

Nothing in the previous feature set could tell them apart. `zone_efficiency`
conflates *shooting skill* with *shot difficulty*, and shot difficulty is
mostly bought with the dribble. That conflation costs accuracy in the model
and, worse, makes the recommender wrong in a specific and misleading way:
it will happily tell a spot-up shooter to take step-back threes, because the
only thing it knows about that location is a rate somebody else earned there.

Three distinct mechanisms
-------------------------
1. **Difficulty adjustment.** Openness is the largest single driver of whether
   a shot goes in, and it is unobservable per-shot from public data. But a
   player's *typical* openness is measurable and stable (`player_shot_profile`
   def_dist splits). Handing the model both a player's zone rate and the
   contest level that rate was earned against lets it read the rate correctly
   instead of taking it at face value.

2. **Attainability.** This is the one that matters most for the product. The
   recommender's real question is not "would this player make a shot here"
   but "can this player GET a shot here". That is almost entirely a function
   of handle. Luka can generate a step-back above the break essentially at
   will; a standstill shooter can only get that look if someone creates it for
   him. `self_creation_index` is the primary input to the attainability model
   in src/training/attainability.py.

3. **Gravity.** Passing mostly does not make the passer's own shot easier —
   it makes his teammates' shots easier, by forcing help defenders to honor
   the kick-out. The direct self-effect is real but second-order: a defender
   who must respect the drive-and-kick cannot fully commit to contesting.
   `playmaking_gravity` captures the self-effect here. The (larger) teammate
   effect needs on-floor lineup data and is deferred — see the module note in
   src/features/spec.py.

Availability and the one-season lag
-----------------------------------
Tracking data begins in 2013-14, which fully covers the 2016-17+ window the
defender-aware model trains on.

Creation features are deliberately LAGGED one season: a shot in 2023-24 sees
the player's 2022-23 creation profile. Two reasons, and they are the same two
reasons the shooting rates became point-in-time:

  - `player_tracking_stats` is a whole-season aggregate, so using the current
    season would put a sliver of the current shot inside its own features.
  - More decisively, the current season's totals do not exist in November.
    Lagging makes the training-time and serving-time feature the same object.

The cost is low because creation skill is highly stable year over year — a
player's dribbles-per-touch is far more persistent than his shooting
percentages. The cost that remains is real and specific: a rookie has no prior
season, and a genuine second-year leap in handle is seen a year late. Rookies
fall back to a position-bucket mean, shrunk by how thin the bucket is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.position_bucket import position_bucket

# Defender-distance splits, with the midpoint (in feet) used to collapse the
# distribution into a single "average daylight" number. The open-ended 6+
# bucket is assigned 7.0 rather than something larger — most shots in it sit
# just past the boundary, and a generous midpoint would overstate the space
# that high-volume spot-up shooters get.
DEF_DIST_MIDPOINTS = {
    "0-2 Feet - Very Tight": 1.0,
    "2-4 Feet - Tight": 3.0,
    "4-6 Feet - Open": 5.0,
    "6+ Feet - Wide Open": 7.0,
}

OPEN_SPLITS = ["4-6 Feet - Open", "6+ Feet - Wide Open"]
TIGHT_SPLITS = ["0-2 Feet - Very Tight", "2-4 Feet - Tight"]

# Dribble splits that indicate the player made the shot for himself. One
# dribble is excluded: a one-dribble side-step off a catch is a rhythm move,
# not self-creation, and lumping it in blurs the distinction this index exists
# to draw.
SELF_CREATED_DRIBBLE_SPLITS = ["3-6 Dribbles", "7+ Dribbles"]

# Raw tracking columns that get normalized per-minute before use. Their raw
# per-game form confounds skill with playing time — a bench creator with 18
# minutes should not read as less skilled than a starter doing the same thing
# for 34.
_PER_MINUTE_COLS = {
    "touches": "touches_per_min",
    "drives": "drives_per_min",
    "time_of_poss": "poss_time_per_min",
    "potential_ast": "potential_ast_per_min",
    "ast_points_created": "ast_pts_created_per_min",
    "passes_made": "passes_per_min",
}

# Composite index definitions: (output column, [(input column, weight)]).
# Inputs are z-scored within season first, so weights are on comparable
# scales and the composite is era-neutral — league-wide dribbling has risen
# steadily since 2013 and an absolute threshold would drift.
_COMPOSITES = {
    "self_creation_index": [
        ("pullup_share", 0.30),
        ("self_created_dribble_share", 0.30),
        ("avg_drib_per_touch", 0.20),
        ("drives_per_min", 0.20),
    ],
    "playmaking_gravity": [
        ("potential_ast_per_min", 0.35),
        ("ast_pts_created_per_min", 0.30),
        ("ast_to_pass_pct_adj", 0.20),
        ("drive_ast_pct", 0.15),
    ],
    "rim_pressure": [
        ("drives_per_min", 0.45),
        ("drive_fg_pct", 0.30),
        ("drive_pf_pct", 0.25),
    ],
}

# Minimum volume before a creation profile is trusted.
#
# Without these, a player with 40 total minutes lands at the very top of
# `self_creation_index` on the strength of four pull-up attempts, and a
# defender-distance "distribution" built from a single non-empty split yields
# an avg_def_dist of exactly 3.0 feet. Both are noise wearing the costume of a
# measurement. Profiles below these thresholds are blanked and fall through to
# the position-bucket fallback in `attach_creation_features`, which is flagged
# via `creation_is_prior` — the same treatment any other missing measurement
# gets, rather than a number nobody should act on.
MIN_GAMES_FOR_PROFILE = 15
MIN_MINUTES_PER_GAME_FOR_PROFILE = 8.0
MIN_FGA_FOR_SHOT_PROFILE = 100

# Feature columns this module contributes to the model.
CREATION_FEATURE_COLS = [
    # Raw handle / usage
    "avg_drib_per_touch", "avg_sec_per_touch", "touches_per_min",
    "poss_time_per_min", "drives_per_min", "drive_fg_pct",
    "drive_tov_pct", "drive_pf_pct",
    # Shot diet and the space it comes with
    "avg_def_dist", "open_share", "tight_share",
    "pullup_share", "catch_shoot_share", "self_created_dribble_share",
    "zero_dribble_share",
    # Skill net of difficulty
    "creation_retention",
    # Composites
    "self_creation_index", "playmaking_gravity", "rim_pressure",
]


def _zscore_within(df: pd.DataFrame, col: str, by: str = "season") -> pd.Series:
    """
    Z-score a column within each season.

    Uses no outcome data — this is a population rescaling, not a fit against
    the label — so computing it across all seasons including the test season
    introduces no leakage. It is in fact necessary: league-average dribbles
    per touch in 2024 is not what it was in 2014, and an index anchored to
    absolute values would rate every modern role player as a self-creator.
    """
    grouped = df.groupby(by)[col]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    return (df[col] - mean) / std.replace(0, np.nan)


def load_creation_profiles(engine) -> pd.DataFrame:
    """
    Build one creation profile per (player_id, season) from the tracking and
    shot-profile tables.

    The returned `season` is the season the stats were MEASURED in. Lagging is
    applied later by `attach_creation_features`, which is what actually joins
    them onto shots.
    """
    tracking = pd.read_sql("""
        SELECT player_id, season, gp, min_per_game,
               touches, front_ct_touches, time_of_poss,
               avg_sec_per_touch, avg_drib_per_touch, pts_per_touch,
               drives, drive_fg_pct, drive_pts, drive_passes_pct,
               drive_ast_pct, drive_tov_pct, drive_pf_pct,
               passes_made, passes_received, secondary_ast,
               potential_ast, ast_points_created,
               ast_to_pass_pct, ast_to_pass_pct_adj
        FROM player_tracking_stats
    """, engine)

    profile = pd.read_sql("""
        SELECT player_id, season, split_type, split_value,
               fga_frequency, fgm, fga, fg_pct, efg_pct, fg3a, fg3_pct
        FROM player_shot_profile
    """, engine)

    if tracking.empty:
        return pd.DataFrame(columns=["player_id", "season"] + CREATION_FEATURE_COLS)

    df = tracking.copy()

    # ── Per-minute normalization ─────────────────────────────────────────
    mins = df["min_per_game"].replace(0, np.nan)
    for raw_col, out_col in _PER_MINUTE_COLS.items():
        df[out_col] = df[raw_col] / mins

    # ── Openness, from the defender-distance distribution ────────────────
    dd = profile[profile["split_type"] == "def_dist"].copy()
    if not dd.empty:
        dd["midpoint"] = dd["split_value"].map(DEF_DIST_MIDPOINTS)
        dd["freq"] = dd["fga_frequency"].fillna(0.0)

        # Weighted average daylight. Normalizing by the frequency sum rather
        # than assuming it reaches 1.0 protects against a season where one
        # split call failed — the average stays correct over the splits that
        # did land instead of silently biasing toward zero.
        freq_sum = dd.groupby(["player_id", "season"])["freq"].transform("sum")
        dd["weighted"] = dd["midpoint"] * dd["freq"] / freq_sum.replace(0, np.nan)

        openness = dd.groupby(["player_id", "season"]).agg(
            avg_def_dist=("weighted", "sum"),
        ).reset_index()

        shares = dd.pivot_table(
            index=["player_id", "season"], columns="split_value",
            values="freq", aggfunc="first",
        ).reset_index()
        for split in DEF_DIST_MIDPOINTS:
            if split not in shares.columns:
                shares[split] = np.nan
        shares["open_share"] = shares[OPEN_SPLITS].sum(axis=1, min_count=1)
        shares["tight_share"] = shares[TIGHT_SPLITS].sum(axis=1, min_count=1)

        openness = openness.merge(
            shares[["player_id", "season", "open_share", "tight_share"]],
            on=["player_id", "season"], how="left",
        )
        df = df.merge(openness, on=["player_id", "season"], how="left")
    else:
        df["avg_def_dist"] = np.nan
        df["open_share"] = np.nan
        df["tight_share"] = np.nan

    # ── Self-creation, from the dribble distribution ─────────────────────
    dr = profile[profile["split_type"] == "dribbles"].copy()
    if not dr.empty:
        dr["freq"] = dr["fga_frequency"].fillna(0.0)
        dr_wide = dr.pivot_table(
            index=["player_id", "season"], columns="split_value",
            values="freq", aggfunc="first",
        ).reset_index()
        for split in SELF_CREATED_DRIBBLE_SPLITS + ["0 Dribbles"]:
            if split not in dr_wide.columns:
                dr_wide[split] = np.nan
        dr_wide["self_created_dribble_share"] = dr_wide[
            SELF_CREATED_DRIBBLE_SPLITS
        ].sum(axis=1, min_count=1)
        dr_wide["zero_dribble_share"] = dr_wide["0 Dribbles"]

        df = df.merge(
            dr_wide[["player_id", "season",
                     "self_created_dribble_share", "zero_dribble_share"]],
            on=["player_id", "season"], how="left",
        )
    else:
        df["self_created_dribble_share"] = np.nan
        df["zero_dribble_share"] = np.nan

    # ── Pull-up vs catch-and-shoot: share, and efficiency retained ───────
    gen = profile[profile["split_type"] == "general"].copy()
    if not gen.empty:
        gen_wide = gen.pivot_table(
            index=["player_id", "season"], columns="split_value",
            values=["fga", "efg_pct", "fga_frequency"], aggfunc="first",
        )
        gen_wide.columns = [f"{a}__{b}" for a, b in gen_wide.columns]
        gen_wide = gen_wide.reset_index()

        pullup_fga = gen_wide.get("fga__Pull Ups")
        cns_freq = gen_wide.get("fga_frequency__Catch and Shoot")
        pullup_efg = gen_wide.get("efg_pct__Pull Ups")
        cns_efg = gen_wide.get("efg_pct__Catch and Shoot")

        gen_wide["catch_shoot_share"] = cns_freq if cns_freq is not None else np.nan

        # Pull-up share needs a denominator, and the pull-up endpoint supplies
        # none (see shot_profile_ingestor._fetch_pullups). Total attempts come
        # from the dribble splits, whose frequencies partition the player's
        # whole shot diet by construction.
        if pullup_fga is not None and not dr.empty:
            totals = dr.groupby(["player_id", "season"])["fga"].sum().reset_index(
                name="total_fga"
            )
            gen_wide = gen_wide.merge(totals, on=["player_id", "season"], how="left")
            gen_wide["pullup_share"] = (
                gen_wide["fga__Pull Ups"] / gen_wide["total_fga"].replace(0, np.nan)
            )
        else:
            gen_wide["pullup_share"] = np.nan

        # The retention gap. Almost every player shoots worse off the dribble,
        # so this is negative for nearly everyone; what distinguishes an elite
        # creator is how SMALL the penalty is. Kept as a signed difference
        # rather than a ratio so a player with a near-zero catch-and-shoot
        # sample cannot produce an explosive value.
        if pullup_efg is not None and cns_efg is not None:
            gen_wide["creation_retention"] = pullup_efg - cns_efg
        else:
            gen_wide["creation_retention"] = np.nan

        df = df.merge(
            gen_wide[["player_id", "season", "catch_shoot_share",
                      "pullup_share", "creation_retention"]],
            on=["player_id", "season"], how="left",
        )
    else:
        df["catch_shoot_share"] = np.nan
        df["pullup_share"] = np.nan
        df["creation_retention"] = np.nan

    # ── Volume guard ─────────────────────────────────────────────────────
    # Applied BEFORE the composites are built, for two reasons: the composites
    # would otherwise be assembled from noise, and — less obviously — the
    # within-season z-scores those composites depend on would be computed
    # against a population padded with garbage values, shifting the mean and
    # standard deviation for every legitimate player too.
    plays_enough = (
        (df["gp"].fillna(0) >= MIN_GAMES_FOR_PROFILE)
        & (df["min_per_game"].fillna(0) >= MIN_MINUTES_PER_GAME_FOR_PROFILE)
    )
    tracking_cols = [c for c in _PER_MINUTE_COLS.values() if c in df.columns] + [
        c for c in ("avg_drib_per_touch", "avg_sec_per_touch", "drive_fg_pct",
                    "drive_tov_pct", "drive_pf_pct", "drive_ast_pct",
                    "ast_to_pass_pct_adj")
        if c in df.columns
    ]
    df.loc[~plays_enough, tracking_cols] = np.nan

    if not profile.empty:
        shot_volume = (
            profile[profile["split_type"] == "dribbles"]
            .groupby(["player_id", "season"])["fga"].sum()
            .rename("_profile_fga").reset_index()
        )
        df = df.merge(shot_volume, on=["player_id", "season"], how="left")
        shoots_enough = df["_profile_fga"].fillna(0) >= MIN_FGA_FOR_SHOT_PROFILE
        profile_cols = [
            c for c in ("avg_def_dist", "open_share", "tight_share",
                        "pullup_share", "catch_shoot_share",
                        "self_created_dribble_share", "zero_dribble_share",
                        "creation_retention")
            if c in df.columns
        ]
        df.loc[~shoots_enough, profile_cols] = np.nan
        df = df.drop(columns=["_profile_fga"])

    # ── Composite indices ────────────────────────────────────────────────
    for out_col, components in _COMPOSITES.items():
        total_weight = 0.0
        acc = pd.Series(0.0, index=df.index)
        any_component = pd.Series(False, index=df.index)
        for col, weight in components:
            if col not in df.columns:
                continue
            z = _zscore_within(df, col)
            # A missing component counts as league average for this index
            # rather than voiding it — a player with three of four real signals
            # still has a meaningful composite. But a player with NONE of them
            # must come out NaN, not 0.0: zero on a z-scored index reads as
            # "exactly league average", which is a confident claim, and the
            # volume guard above just finished establishing we know nothing
            # about this player at all.
            acc = acc + weight * z.fillna(0.0)
            any_component = any_component | z.notna()
            total_weight += weight
        df[out_col] = (acc / total_weight).where(any_component) if total_weight > 0 else np.nan

    keep = ["player_id", "season"] + [c for c in CREATION_FEATURE_COLS if c in df.columns]
    return df[keep]


def _season_start_year(season: str) -> int:
    """'2023-24' → 2023. Used to compute the one-season lag."""
    return int(season.split("-")[0])


def _previous_season(season: str) -> str:
    """'2023-24' → '2022-23'."""
    start = _season_start_year(season) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def attach_creation_features(
    df: pd.DataFrame,
    profiles: pd.DataFrame,
    positions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Join lagged creation features onto a frame keyed by (player_id, season).

    A row in season S receives the profile measured in season S-1, for the
    availability and leakage reasons in the module docstring.

    Players with no prior-season profile — rookies, and anyone returning from
    a season missed entirely — fall back to the mean profile of their position
    bucket in the lagged season. That fallback is flagged in
    `creation_is_prior`, so the model can learn to discount it and the API can
    tell a user which numbers are measured and which are assumed.
    """
    out = df.copy()
    out["_lag_season"] = out["season"].map(_previous_season)

    feature_cols = [c for c in CREATION_FEATURE_COLS if c in profiles.columns]

    lagged = profiles.rename(columns={"season": "_lag_season"})
    out = out.merge(
        lagged[["player_id", "_lag_season"] + feature_cols],
        on=["player_id", "_lag_season"],
        how="left",
    )

    measured = out[feature_cols].notna().any(axis=1)
    out["creation_is_prior"] = (~measured).astype(int)

    # ── Position-bucket fallback for players with no lagged profile ──────
    if positions is not None and not positions.empty and (~measured).any():
        pos = positions.copy()
        pos["position_bucket"] = pos["position"].map(position_bucket)

        prof_pos = profiles.merge(
            pos[["player_id", "season", "position_bucket"]],
            on=["player_id", "season"], how="left",
        )
        bucket_means = (
            prof_pos.groupby(["position_bucket", "season"])[feature_cols]
            .mean()
            .reset_index()
            .rename(columns={"season": "_lag_season"})
        )

        out = out.merge(
            pos[["player_id", "season", "position_bucket"]],
            on=["player_id", "season"], how="left",
        )
        out = out.merge(
            bucket_means, on=["position_bucket", "_lag_season"],
            how="left", suffixes=("", "_bucket"),
        )
        for col in feature_cols:
            bucket_col = f"{col}_bucket"
            if bucket_col in out.columns:
                out[col] = out[col].fillna(out[bucket_col])
                out = out.drop(columns=[bucket_col])

    out = out.drop(columns=["_lag_season"], errors="ignore")
    return out
