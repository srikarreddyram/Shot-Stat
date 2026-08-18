"""
Feature Engineering Pipeline — Builds the training matrix from SQLite.

Joins shots + players + player_zone_stats into a single DataFrame with
all features needed for model training. No API calls — pure SQL + pandas.

Usage:
    python -m src.training.feature_engineering
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine


# ── Feature column definitions ──────────────────────────────────────────────
# These are the columns XGBoost / LR will train on.
# Zone and position dummies are added dynamically after one-hot encoding.
BASE_FEATURE_COLS = [
    # Spatial
    "loc_x", "loc_y", "shot_distance", "shot_angle",
    "distance_from_center", "abs_loc_x", "is_three",

    # Game context
    "quarter", "time_remaining", "score_diff", "home_away", "playoff_flag",
    "clutch_flag",

    # Team / schedule context (Phase D)
    "rest_days", "is_back_to_back", "opp_def_rating",

    # Player ability
    "height", "weight", "wingspan",
    "career_fg_pct", "career_3p_pct", "season_fg_pct",
    "zone_efficiency",

    # Zone-level shooting efficiency (per player-season)
    "fg_pct_restricted_area", "fg_pct_paint", "fg_pct_midrange",
    "fg_pct_left_corner_3", "fg_pct_right_corner_3", "fg_pct_above_break_3",

    # Defender physicals
    "def_height", "def_weight", "def_wingspan",
    "height_diff", "weight_diff", "wingspan_diff", "size_mismatch",

    # Defender quality
    "def_fg_pct_overall", "def_pct_plusminus",
    "def_freq_zone",  # how often this defender defends this zone (specialization)

    # Zone-specific defender quality (the key fix for rim protection)
    "def_fg_pct_zone",       # defender's FG% allowed in THIS zone
    "def_pct_plusminus_zone", # defender's +/- in THIS zone

    # Composite
    "matchup_advantage",  # attacker zone FG% minus defender zone FG% allowed
]

TARGET_COL = "shot_made"


# ── Zone efficiency mapping ─────────────────────────────────────────────────
_ZONE_TO_EFFICIENCY_COL = {
    "Restricted Area":        "fg_pct_restricted_area",
    "In The Paint (Non-RA)":  "fg_pct_paint",
    "Mid-Range":              "fg_pct_midrange",
    "Left Corner 3":          "fg_pct_left_corner_3",
    "Right Corner 3":         "fg_pct_right_corner_3",
    "Above the Break 3":      "fg_pct_above_break_3",
}


def build_training_matrix(seasons: list[str], use_defender: bool = False) -> pd.DataFrame:
    """
    Build the full training matrix by joining shots + players + zone stats.

    Returns a DataFrame with all features, metadata (season, shot_id, player_id),
    and the target column (shot_made).
    """
    engine = get_engine()
    season_list = ", ".join(f"'{s}'" for s in seasons)

    print(f"\n{'='*60}")
    print(f"  FEATURE ENGINEERING")
    print(f"  Seasons: {seasons[0]} → {seasons[-1]} ({len(seasons)} seasons)")
    if use_defender:
        print(f"  Mode:    With Defender Features")
    print(f"{'='*60}")

    # ── Step 1: Load everything with one big SQL join ────────────────────────
    print("  → Loading shots + players + zone stats from SQLite...")

    query_select_def = ""
    query_join_def = ""

    if use_defender:
        query_select_def = """,
        -- Defender physicals
        def_p.height AS def_height,
        def_p.weight AS def_weight,
        def_p.wingspan AS def_wingspan,

        -- Defender overall stats
        ds_overall.d_fg_pct AS def_fg_pct_overall,
        ds_overall.pct_plusminus AS def_pct_plusminus,

        -- Defender zone-specific stats (now populated with correct column mapping)
        ds_zone.freq AS def_freq_zone,
        ds_zone.d_fg_pct AS def_fg_pct_zone,
        ds_zone.pct_plusminus AS def_pct_plusminus_zone
        """
        
        query_join_def = """
    LEFT JOIN players def_p
        ON s.defender_id = def_p.player_id AND s.season = def_p.season

    LEFT JOIN defender_stats ds_overall
        ON s.defender_id = ds_overall.player_id AND s.season = ds_overall.season
        AND ds_overall.defense_category = 'Overall'

    LEFT JOIN defender_stats ds_zone
        ON s.defender_id = ds_zone.player_id AND s.season = ds_zone.season
        AND ds_zone.defense_category = (CASE
            WHEN s.zone = 'Restricted Area' THEN 'Less Than 6Ft'
            WHEN s.zone IN ('In The Paint (Non-RA)') THEN 'Less Than 10Ft'
            WHEN s.zone = 'Mid-Range' THEN 'Greater Than 15Ft'
            WHEN s.zone IN ('Left Corner 3', 'Right Corner 3', 'Above the Break 3') THEN '3 Pointers'
        END)
        """

    query = f"""
    SELECT
        s.shot_id,
        s.season,
        s.player_id,
        s.shot_made,

        -- Spatial (from shots table)
        s.loc_x,
        s.loc_y,
        s.shot_distance,
        s.shot_type,
        s.zone,
        s.shot_angle,

        -- Game context (from shots table)
        s.quarter,
        s.time_remaining,
        s.score_diff,
        s.home_away,
        s.playoff_flag,

        -- Player attributes (from players table)
        p.height,
        p.weight,
        p.wingspan,
        p.position,
        p.career_fg_pct,
        p.career_3p_pct,
        p.season_fg_pct,

        -- Zone efficiency (pivoted from player_zone_stats)
        zs_ra.fg_pct    AS fg_pct_restricted_area,
        zs_paint.fg_pct AS fg_pct_paint,
        zs_mid.fg_pct   AS fg_pct_midrange,
        zs_lc3.fg_pct   AS fg_pct_left_corner_3,
        zs_rc3.fg_pct   AS fg_pct_right_corner_3,
        zs_atb3.fg_pct  AS fg_pct_above_break_3,

        -- Schedule context (Phase D)
        -- Shooter's team rest days
        CASE WHEN s.home_away = 1 THEN ts_home.rest_days ELSE ts_away.rest_days END AS rest_days,
        CASE WHEN s.home_away = 1 THEN ts_home.is_back_to_back ELSE ts_away.is_back_to_back END AS is_back_to_back,

        -- Opponent's defensive rating
        CASE WHEN s.home_away = 1 THEN opp_away.def_rating ELSE opp_home.def_rating END AS opp_def_rating
        {query_select_def}
    FROM shots s

    JOIN players p
        ON s.player_id = p.player_id AND s.season = p.season

    JOIN games g
        ON s.game_id = g.game_id

    LEFT JOIN player_zone_stats zs_ra
        ON s.player_id = zs_ra.player_id AND s.season = zs_ra.season
        AND zs_ra.zone = 'Restricted Area'

    LEFT JOIN player_zone_stats zs_paint
        ON s.player_id = zs_paint.player_id AND s.season = zs_paint.season
        AND zs_paint.zone = 'In The Paint (Non-RA)'

    LEFT JOIN player_zone_stats zs_mid
        ON s.player_id = zs_mid.player_id AND s.season = zs_mid.season
        AND zs_mid.zone = 'Mid-Range'

    LEFT JOIN player_zone_stats zs_lc3
        ON s.player_id = zs_lc3.player_id AND s.season = zs_lc3.season
        AND zs_lc3.zone = 'Left Corner 3'

    LEFT JOIN player_zone_stats zs_rc3
        ON s.player_id = zs_rc3.player_id AND s.season = zs_rc3.season
        AND zs_rc3.zone = 'Right Corner 3'

    LEFT JOIN player_zone_stats zs_atb3
        ON s.player_id = zs_atb3.player_id AND s.season = zs_atb3.season
        AND zs_atb3.zone = 'Above the Break 3'

    -- Schedule: home team rest
    LEFT JOIN team_schedule ts_home
        ON g.game_id = ts_home.game_id AND g.home_team = ts_home.team_id

    -- Schedule: away team rest
    LEFT JOIN team_schedule ts_away
        ON g.game_id = ts_away.game_id AND g.away_team = ts_away.team_id

    -- Opponent def rating (if shooter is home, opponent is away team)
    LEFT JOIN team_stats opp_away
        ON opp_away.team_abbrev = g.away_team AND opp_away.season = s.season

    -- Opponent def rating (if shooter is away, opponent is home team)
    LEFT JOIN team_stats opp_home
        ON opp_home.team_abbrev = g.home_team AND opp_home.season = s.season
    {query_join_def}
    WHERE s.score_diff IS NOT NULL
      AND s.home_away IS NOT NULL
      AND s.zone IS NOT NULL
      AND s.zone != 'Backcourt'
      AND s.season IN ({season_list})
    """

    df = pd.read_sql(query, engine)
    print(f"  ✓ Loaded {len(df):,} shots × {len(df.columns)} columns")

    # ── Step 2: Engineer derived features ────────────────────────────────────
    print("  → Engineering features...")

    # is_three: binary flag (1 if 3PT attempt, 0 if 2PT)
    df["is_three"] = (df["shot_type"] == "3PT Field Goal").astype(int)

    # distance_from_center: Euclidean distance from the basket (0,0)
    df["distance_from_center"] = np.sqrt(df["loc_x"] ** 2 + df["loc_y"] ** 2)

    # abs_loc_x: left/right symmetry — distance from the center lane
    df["abs_loc_x"] = df["loc_x"].abs()

    # clutch_flag: last 2 min of Q4/OT with score within 5
    df["clutch_flag"] = (
        (df["quarter"] >= 4)
        & (df["time_remaining"] <= 120)
        & (df["score_diff"].abs() <= 5)
    ).astype(int)

    # zone_efficiency: the shooter's FG% in the zone they're shooting from
    # This is the KEY feature — "how good is this player from THIS spot?"
    def _get_zone_efficiency(row):
        col = _ZONE_TO_EFFICIENCY_COL.get(row["zone"])
        if col and pd.notna(row.get(col)):
            return row[col]
        return np.nan

    df["zone_efficiency"] = df.apply(_get_zone_efficiency, axis=1)

    if use_defender:
        # Physical matchup features
        df['height_diff'] = df['height'] - df['def_height']
        df['weight_diff'] = df['weight'] - df['def_weight']
        df['wingspan_diff'] = df['wingspan'] - df['def_wingspan']
        df['size_mismatch'] = (df['height_diff'].abs() >= 4).astype(int)

        # Composite feature: attacker's zone efficiency vs defender's zone-level allowed FG%
        # Falls back to overall if zone-level is unavailable
        def_fg_zone = df['def_fg_pct_zone'].fillna(df['def_fg_pct_overall'])
        df['matchup_advantage'] = df['zone_efficiency'] - def_fg_zone

    # ── Step 3: One-hot encode categoricals ──────────────────────────────────
    print("  → Encoding categoricals...")

    # Zone → 5 dummy columns (drop_first avoids multicollinearity)
    zone_dummies = pd.get_dummies(df["zone"], prefix="zone", drop_first=True)
    df = pd.concat([df, zone_dummies], axis=1)

    # Position → dummy columns
    pos_dummies = pd.get_dummies(df["position"], prefix="pos", drop_first=True)
    df = pd.concat([df, pos_dummies], axis=1)

    # ── Step 4: Summary ─────────────────────────────────────────────────────
    feature_cols = get_feature_columns(df)
    null_features = {
        col: df[col].isnull().sum()
        for col in feature_cols
        if df[col].isnull().sum() > 0
    }

    print(f"\n  ✓ Final matrix: {len(df):,} rows × {len(feature_cols)} features")
    print(f"  ✓ Target (FG%): {df[TARGET_COL].mean():.3f}")
    print(f"  ✓ Clutch shots: {df['clutch_flag'].sum():,} ({df['clutch_flag'].mean()*100:.1f}%)")

    if null_features:
        print(f"\n  ⚠ Features with NULLs (XGBoost handles these, LR will median-fill):")
        for col, n in sorted(null_features.items(), key=lambda x: -x[1]):
            print(f"    {col:<35} {n:>8,} ({n/len(df)*100:.1f}%)")

    print(f"\n{'='*60}\n")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the final list of feature columns for model training.
    Includes base features + dynamically generated dummy columns.
    """
    cols = BASE_FEATURE_COLS.copy()

    # Add one-hot encoded zone columns (zone_Mid-Range, zone_Left Corner 3, etc.)
    # Exclude 'zone' (raw string) and 'zone_efficiency' (already in base list)
    cols += [c for c in df.columns if c.startswith("zone_") and c not in ("zone", "zone_efficiency")]

    # Add one-hot encoded position columns (pos_F, pos_C, etc.)
    cols += [c for c in df.columns if c.startswith("pos_")]

    # Only return columns that actually exist in the DataFrame
    return [c for c in cols if c in df.columns]


if __name__ == "__main__":
    df = build_training_matrix(config.ALL_SEASONS)
    feature_cols = get_feature_columns(df)
    print(f"Matrix shape: {df.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target mean (overall FG%): {df[TARGET_COL].mean():.3f}")
