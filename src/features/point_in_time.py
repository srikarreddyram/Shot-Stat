"""
Point-in-time player shooting rates — the leakage fix.

What was wrong
--------------
The previous pipeline joined whole-season aggregates (`player_zone_stats`,
`season_fg_pct`, `career_fg_pct`) onto every shot in that season. A shot's
own outcome was therefore inside its own features. For a corner-3 feature
built from 20 attempts, roughly 5% of the "evidence" was the answer.

The inflated test metric was the smaller problem. The larger one: those
features are unavailable at prediction time. Predicting a shot in November
required April's finished splits, so `player_lookup.py` quietly substituted
the *prior* season instead — meaning `zone_efficiency` denoted one thing in
training and a different thing in production, on the model's single most
important input.

What this module does
---------------------
For every (player, game) it computes shooting counts from games strictly
BEFORE that game's date, at two horizons:

    career-to-date — every prior game, across seasons
    season-to-date — prior games within the same season only

Counts, not rates. Rates are produced by `shrinkage.shrink*`, which is shared
with the serving path, so the two paths cannot drift.

The three-level hierarchy
-------------------------
Rates are built by regressing each level toward the one above it:

    league (zone × season)  →  player career-to-date  →  player season-to-date

A rookie's first shot resolves to the league prior. A veteran in game 3 of a
new season leans on his career rate. An established player in March is
essentially his own season numbers. All three cases come out of one formula
with no branching, no sentinel values, and no separate priors table.

`as_of` semantics are strict: a game played on the same date is excluded, not
just games before the season. Two games on one date are rare but real, and
including the same-day game would reintroduce the leak in miniature.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.features.shrinkage import BetaPrior, fit_priors, shrink, shrink_toward

# The six shot areas, matching PlayerZoneStats.zone and Shot.zone exactly.
ZONES = [
    "Restricted Area",
    "In The Paint (Non-RA)",
    "Mid-Range",
    "Left Corner 3",
    "Right Corner 3",
    "Above the Break 3",
]

THREE_POINT_ZONES = ["Left Corner 3", "Right Corner 3", "Above the Break 3"]

# Zone name → the suffix used in feature column names. Kept explicit rather
# than slugified on the fly so a zone rename upstream breaks loudly here
# instead of silently producing a new, unmatched feature column.
ZONE_SUFFIX = {
    "Restricted Area": "restricted_area",
    "In The Paint (Non-RA)": "paint",
    "Mid-Range": "midrange",
    "Left Corner 3": "left_corner_3",
    "Right Corner 3": "right_corner_3",
    "Above the Break 3": "above_break_3",
}

# Zone → the LeagueDashPtDefend category whose FG%-allowed best describes
# defending that zone. Single source of truth: the recommender, the API's
# /matchup endpoint, and the training join all read this. Defined here rather
# than in `spec.py` (which imports it back) so `build_defender_category_rates`
# below can use it without spec.py importing point_in_time importing spec.
ZONE_TO_DEF_CATEGORY = {
    "Restricted Area": "Less Than 6Ft",
    "In The Paint (Non-RA)": "Less Than 10Ft",
    "Mid-Range": "Greater Than 15Ft",
    "Left Corner 3": "3 Pointers",
    "Right Corner 3": "3 Pointers",
    "Above the Break 3": "3 Pointers",
}

DEFENSE_CATEGORIES = sorted(set(ZONE_TO_DEF_CATEGORY.values()))

# How many attempts of season-specific evidence it takes to move a player off
# his career rate. Lower than a league prior's strength because a player's own
# career mean is a far better starting guess than the league's, so less
# evidence is needed to justify departing from it. Tuned by backtest; see
# src/training/backtest.py.
SEASON_TO_CAREER_STRENGTH = 60.0

# Teammate attempts required this season before a supporting-cast rate is
# reported at all (see `build_supporting_cast`). Roughly two games' worth of
# team shooting — enough that the rate describes a cast rather than a night.
MIN_CAST_ATTEMPTS = 150


def _load_player_game_zone_counts(engine, seasons: list[str] | None = None) -> pd.DataFrame:
    """
    Per (player, game, zone) makes and attempts, with the game date attached.

    Deliberately NOT filtered to the training seasons. A player's career-to-
    date rate entering 2016-17 must include his 2013-15 shots, and his rolling
    form in October must reflect last April. Filtering here would silently
    reset every player's history at the training window's left edge.
    """
    df = pd.read_sql("""
        SELECT s.player_id,
               s.game_id,
               s.season,
               g.date        AS game_date,
               s.zone,
               SUM(s.shot_made) AS makes,
               COUNT(*)         AS attempts
        FROM shots s
        JOIN games g ON s.game_id = g.game_id
        WHERE s.zone IS NOT NULL
          AND s.zone != 'Backcourt'
        GROUP BY s.player_id, s.game_id, s.season, g.date, s.zone
    """, engine)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def build_prior_counts(engine, seasons: list[str] | None = None) -> pd.DataFrame:
    """
    Build the strictly-prior count table: one row per (player_id, game_id)
    with career-to-date and season-to-date makes/attempts for each of the six
    zones, plus overall and three-point totals.

    Returned columns (per zone suffix S):
        pit_car_mk_S, pit_car_att_S   career-to-date
        pit_ssn_mk_S, pit_ssn_att_S   season-to-date
    plus pit_car_mk_all / pit_car_att_all / pit_ssn_mk_all / pit_ssn_att_all
    and  pit_car_mk_3pt / pit_car_att_3pt.
    """
    counts = _load_player_game_zone_counts(engine, seasons)

    # Every player-game the player appeared in (took at least one shot).
    player_games = (
        counts[["player_id", "game_id", "season", "game_date"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Expand to the full (player-game × zone) grid. A player who took no
    # mid-range shots in a game still needs his running mid-range totals
    # carried forward at that game, otherwise the cumulative sum would only
    # advance on games where he happened to shoot from there.
    zone_frame = pd.DataFrame({"zone": ZONES})
    grid = player_games.merge(zone_frame, how="cross")

    grid = grid.merge(
        counts[["player_id", "game_id", "zone", "makes", "attempts"]],
        on=["player_id", "game_id", "zone"],
        how="left",
    )
    grid["makes"] = grid["makes"].fillna(0.0)
    grid["attempts"] = grid["attempts"].fillna(0.0)

    # Chronological order within each (player, zone). game_id breaks ties on
    # doubleheader dates deterministically — arbitrary but stable, and both
    # same-day games are excluded from each other's priors either way.
    grid = grid.sort_values(
        ["player_id", "zone", "game_date", "game_id"]
    ).reset_index(drop=True)

    g_career = grid.groupby(["player_id", "zone"], sort=False)
    g_season = grid.groupby(["player_id", "zone", "season"], sort=False)

    # shift(1) after cumsum is what makes this strictly prior: row i holds the
    # total through row i-1, so the current game never contributes to its own
    # features.
    grid["pit_car_mk"] = g_career["makes"].cumsum() - grid["makes"]
    grid["pit_car_att"] = g_career["attempts"].cumsum() - grid["attempts"]
    grid["pit_ssn_mk"] = g_season["makes"].cumsum() - grid["makes"]
    grid["pit_ssn_att"] = g_season["attempts"].cumsum() - grid["attempts"]

    # Pivot to one row per player-game, one column set per zone.
    wide = grid.pivot_table(
        index=["player_id", "game_id"],
        columns="zone",
        values=["pit_car_mk", "pit_car_att", "pit_ssn_mk", "pit_ssn_att"],
        aggfunc="first",
    )
    wide.columns = [f"{stat}_{ZONE_SUFFIX[zone]}" for stat, zone in wide.columns]
    wide = wide.reset_index()

    # Overall and three-point totals, summed from the per-zone columns so they
    # are guaranteed consistent with them.
    for stat in ("pit_car_mk", "pit_car_att", "pit_ssn_mk", "pit_ssn_att"):
        zone_cols = [f"{stat}_{ZONE_SUFFIX[z]}" for z in ZONES]
        wide[f"{stat}_all"] = wide[zone_cols].sum(axis=1)

    for stat in ("pit_car_mk", "pit_car_att"):
        three_cols = [f"{stat}_{ZONE_SUFFIX[z]}" for z in THREE_POINT_ZONES]
        wide[f"{stat}_3pt"] = wide[three_cols].sum(axis=1)

    return wide


def build_rolling_form(engine, windows: tuple[int, ...] = (10, 20)) -> pd.DataFrame:
    """
    Shot-weighted rolling FG% over a player's last N games, strictly before the
    current one.

    Carried over from the previous pipeline (which already got this right) and
    moved here so every point-in-time computation lives in one module. Uses
    total makes over total attempts across the window rather than an average
    of per-game percentages, so a 1-for-1 night does not weigh as much as a
    12-for-20 night.
    """
    per_game = pd.read_sql("""
        SELECT s.player_id, s.game_id, g.date AS game_date,
               SUM(s.shot_made) AS makes, COUNT(*) AS attempts
        FROM shots s
        JOIN games g ON s.game_id = g.game_id
        GROUP BY s.player_id, s.game_id, g.date
    """, engine)
    per_game["game_date"] = pd.to_datetime(per_game["game_date"])
    per_game = per_game.sort_values(
        ["player_id", "game_date", "game_id"]
    ).reset_index(drop=True)

    out_cols = ["player_id", "game_id"]
    for window in windows:
        label = f"recent_{window}_fg"
        makes_roll = per_game.groupby("player_id")["makes"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )
        att_roll = per_game.groupby("player_id")["attempts"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )
        per_game[label] = makes_roll / att_roll.replace(0, np.nan)
        out_cols.append(label)

    return per_game[out_cols]


def fit_league_zone_priors(engine, through_season: str) -> dict[str, BetaPrior]:
    """
    Fit one Beta prior per zone from completed player-seasons up to and
    including `through_season`.

    `through_season` must be the last season in the TRAINING window, never the
    test season. A prior fit on data the model is about to be scored against
    is the same leak this module exists to remove, just laundered through an
    aggregate.
    """
    df = pd.read_sql(f"""
        SELECT s.zone,
               s.player_id,
               s.season,
               SUM(s.shot_made) AS makes,
               COUNT(*)         AS attempts
        FROM shots s
        WHERE s.zone IS NOT NULL
          AND s.zone != 'Backcourt'
          AND s.season <= '{through_season}'
        GROUP BY s.zone, s.player_id, s.season
    """, engine)

    priors = fit_priors(df, ["zone"], makes_col="makes", attempts_col="attempts")
    return {zone: priors[(zone,)] for zone in df["zone"].unique() if (zone,) in priors}


def apply_hierarchy(
    df: pd.DataFrame,
    zone_priors: dict[str, BetaPrior],
    season_strength: float = SEASON_TO_CAREER_STRENGTH,
) -> pd.DataFrame:
    """
    Turn prior counts into shrunk rate features via the three-level hierarchy.

    Adds, per zone suffix S:
        zone_rate_S       the player's shrunk rate in that zone
        zone_att_S        prior attempts behind it (the model's confidence cue)

    plus `overall_rate` and `three_rate`, the shrunk replacements for the old
    leaky `season_fg_pct` / `career_fg_pct` / `career_3p_pct`.

    Both the training builder and the serving path call this function on
    identically-shaped input, which is what makes train/serve parity testable
    rather than aspirational.
    """
    out = df.copy()

    for zone in ZONES:
        suffix = ZONE_SUFFIX[zone]
        prior = zone_priors.get(zone)
        if prior is None:
            out[f"zone_rate_{suffix}"] = np.nan
            out[f"zone_att_{suffix}"] = 0.0
            continue

        car_mk = out.get(f"pit_car_mk_{suffix}", 0.0)
        car_att = out.get(f"pit_car_att_{suffix}", 0.0)
        ssn_mk = out.get(f"pit_ssn_mk_{suffix}", 0.0)
        ssn_att = out.get(f"pit_ssn_att_{suffix}", 0.0)

        # Level 2: career-to-date regressed toward the league zone prior.
        career_rate = shrink(car_mk, car_att, prior)

        # Level 3: season-to-date regressed toward this player's own career
        # rate. Note career counts INCLUDE the season-to-date counts, so the
        # season evidence is not double-counted as independent — it is being
        # asked whether this season deviates from the player's established
        # baseline, which is exactly the question a hot/cold read should pose.
        out[f"zone_rate_{suffix}"] = shrink_toward(
            ssn_mk, ssn_att, career_rate, season_strength
        )
        out[f"zone_att_{suffix}"] = np.asarray(car_att, dtype=float)

    # Overall and three-point rates use a pooled prior across zones — these
    # are coarse "is this player a shooter at all" signals, and a zone-
    # specific prior would be the wrong population for them.
    pooled_mean = float(np.mean([p.mean for p in zone_priors.values()])) if zone_priors else 0.45
    pooled_strength = float(np.mean([p.strength for p in zone_priors.values()])) if zone_priors else 100.0
    pooled = BetaPrior(mean=pooled_mean, strength=pooled_strength)

    out["overall_rate"] = shrink_toward(
        out.get("pit_ssn_mk_all", 0.0),
        out.get("pit_ssn_att_all", 0.0),
        shrink(out.get("pit_car_mk_all", 0.0), out.get("pit_car_att_all", 0.0), pooled),
        season_strength,
    )
    out["overall_att"] = np.asarray(out.get("pit_car_att_all", 0.0), dtype=float)

    three_prior_means = [zone_priors[z].mean for z in THREE_POINT_ZONES if z in zone_priors]
    three_prior = BetaPrior(
        mean=float(np.mean(three_prior_means)) if three_prior_means else 0.35,
        strength=pooled_strength,
    )
    out["three_rate"] = shrink(
        out.get("pit_car_mk_3pt", 0.0), out.get("pit_car_att_3pt", 0.0), three_prior
    )
    out["three_att"] = np.asarray(out.get("pit_car_att_3pt", 0.0), dtype=float)

    return out


def lookup_prior_counts(conn, player_id: str, as_of_date=None) -> dict:
    """
    Serving-path equivalent of `build_prior_counts` for a single player.

    Returns the same `pit_*` keys the training builder produces, so the caller
    can wrap them in a one-row DataFrame and hand them to `apply_hierarchy`
    unchanged. That shared final step is the parity guarantee: the two paths
    disagree about how to gather counts (a cumulative sum over millions of
    rows versus one indexed query) and agree about everything after.

    `as_of_date` defaults to "everything in the database". Passing an explicit
    date reproduces what the model would have seen on that day, which is what
    the backtest harness uses to score historical recommendations.
    """
    from sqlalchemy import text

    date_clause = "AND g.date < :as_of" if as_of_date is not None else ""
    params = {"pid": str(player_id)}
    if as_of_date is not None:
        params["as_of"] = str(as_of_date)

    rows = conn.execute(text(f"""
        SELECT s.zone, s.season,
               SUM(s.shot_made) AS makes,
               COUNT(*)         AS attempts
        FROM shots s
        JOIN games g ON s.game_id = g.game_id
        WHERE s.player_id = :pid
          AND s.zone IS NOT NULL
          AND s.zone != 'Backcourt'
          {date_clause}
        GROUP BY s.zone, s.season
    """), params).fetchall()

    # Which season counts as "current" is the latest season the player has
    # any shots in as of this date — not the calendar season, which would be
    # wrong for a player who has not yet debuted this year.
    latest_season = max((r[1] for r in rows), default=None)

    counts = {}
    for zone in ZONES:
        suffix = ZONE_SUFFIX[zone]
        counts[f"pit_car_mk_{suffix}"] = 0.0
        counts[f"pit_car_att_{suffix}"] = 0.0
        counts[f"pit_ssn_mk_{suffix}"] = 0.0
        counts[f"pit_ssn_att_{suffix}"] = 0.0

    for zone, season, makes, attempts in rows:
        if zone not in ZONE_SUFFIX:
            continue
        suffix = ZONE_SUFFIX[zone]
        counts[f"pit_car_mk_{suffix}"] += float(makes or 0)
        counts[f"pit_car_att_{suffix}"] += float(attempts or 0)
        if season == latest_season:
            counts[f"pit_ssn_mk_{suffix}"] += float(makes or 0)
            counts[f"pit_ssn_att_{suffix}"] += float(attempts or 0)

    for stat in ("pit_car_mk", "pit_car_att", "pit_ssn_mk", "pit_ssn_att"):
        counts[f"{stat}_all"] = sum(
            counts[f"{stat}_{ZONE_SUFFIX[z]}"] for z in ZONES
        )
    for stat in ("pit_car_mk", "pit_car_att"):
        counts[f"{stat}_3pt"] = sum(
            counts[f"{stat}_{ZONE_SUFFIX[z]}"] for z in THREE_POINT_ZONES
        )

    counts["_latest_season"] = latest_season
    return counts


def lookup_recent_form(conn, player_id: str, as_of_date=None,
                       windows: tuple[int, ...] = (10, 20)) -> dict:
    """
    Serving-path equivalent of `build_rolling_form` for a single player.

    Its absence was a live training/serving skew: `recent_10_fg` and
    `recent_20_fg` are populated for virtually every training row (min_periods=1
    means only a player's literal first career game is missing), but the
    recommender never set them at all. Every served prediction therefore landed
    in a corner of feature space the model associates with a debut game, which
    dragged predictions well below where they belonged.

    Returns {"recent_10_fg": float|None, "recent_20_fg": float|None}.
    """
    from sqlalchemy import text

    date_clause = "AND g.date < :as_of" if as_of_date is not None else ""
    params = {"pid": str(player_id), "limit": max(windows)}
    if as_of_date is not None:
        params["as_of"] = str(as_of_date)

    rows = conn.execute(text(f"""
        SELECT SUM(s.shot_made) AS makes, COUNT(*) AS attempts
        FROM shots s
        JOIN games g ON s.game_id = g.game_id
        WHERE s.player_id = :pid
          {date_clause}
        GROUP BY s.game_id, g.date
        ORDER BY g.date DESC, s.game_id DESC
        LIMIT :limit
    """), params).fetchall()

    out: dict[str, float | None] = {}
    for window in windows:
        window_rows = rows[:window]
        makes = sum(float(r[0] or 0) for r in window_rows)
        attempts = sum(float(r[1] or 0) for r in window_rows)
        out[f"recent_{window}_fg"] = (makes / attempts) if attempts > 0 else None
    return out


def league_average_defender(conn, season: str) -> dict:
    """
    The average defender, per defense category.

    Used when the caller names no defender. Leaving those features NaN was
    wrong in a specific way: in the training matrix NaN defender columns mean
    "this shot has no matchup data at all" — an entire class of older or
    unlinked games with its own scoring characteristics — whereas an API caller
    who omits a defender means "against a typical defender". Filling with league
    means says the second thing, which is what the user asked.
    """
    from sqlalchemy import text

    rows = conn.execute(text("""
        SELECT defense_category,
               AVG(d_fg_pct), AVG(pct_plusminus), AVG(freq)
        FROM defender_stats
        WHERE season <= :season
        GROUP BY defense_category
    """), {"season": season}).fetchall()

    physicals = conn.execute(text("""
        SELECT AVG(height), AVG(weight), AVG(wingspan)
        FROM players WHERE season <= :season
    """), {"season": season}).fetchone()

    return {
        "by_category": {
            r[0]: {"d_fg_pct": r[1], "pct_plusminus": r[2], "freq": r[3]}
            for r in rows
        },
        "height": physicals[0] if physicals else None,
        "weight": physicals[1] if physicals else None,
        "wingspan": physicals[2] if physicals else None,
    }


def build_opponent_zone_defence(engine) -> pd.DataFrame:
    """
    Per (game, defending team) shooting allowed per zone, from strictly prior
    games.

    Why this exists
    ---------------
    The model's entire knowledge of opponent defence was one season-level
    number, `team_stats.def_rating` — no zone split, not point-in-time. Scoring
    v8 on its held-out season and aggregating residuals by defending team shows
    how much that misses: the standard deviation of team z-scores is 2.22
    overall and **2.82 in the restricted area**, against 1.0 for a correctly
    specified model. Eighteen of thirty teams sit beyond |z| > 2 at the rim.

    Actual rim FG% allowed ranges from 0.622 to 0.709 across teams, and the
    model cannot see any of it.

    The rate is worth carrying because it persists: team rim defence correlates
    +0.43 to +0.79 season over season, so prior games genuinely predict the next
    one rather than describing noise that has already passed.

    Defending team needs no new data — it is `games.home_team`/`away_team`
    selected by `shots.home_away`.

    Returns one row per (game_id, def_team) with, per zone suffix S:
        opp_def_mk_S, opp_def_att_S   makes and attempts allowed before this game
    """
    counts = pd.read_sql("""
        SELECT g.game_id,
               g.date AS game_date,
               CASE WHEN s.home_away = 1 THEN g.away_team ELSE g.home_team END
                   AS def_team,
               s.zone,
               SUM(s.shot_made) AS makes,
               COUNT(*)         AS attempts
        FROM shots s
        JOIN games g ON s.game_id = g.game_id
        WHERE s.zone IS NOT NULL
          AND s.zone != 'Backcourt'
          AND s.home_away IS NOT NULL
        GROUP BY g.game_id, g.date, def_team, s.zone
    """, engine)

    if counts.empty:
        return pd.DataFrame(columns=["game_id", "def_team"])

    counts["game_date"] = pd.to_datetime(counts["game_date"])

    team_games = (
        counts[["game_id", "def_team", "game_date"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    grid = team_games.merge(pd.DataFrame({"zone": ZONES}), how="cross")
    grid = grid.merge(
        counts[["game_id", "def_team", "zone", "makes", "attempts"]],
        on=["game_id", "def_team", "zone"], how="left",
    )
    grid["makes"] = grid["makes"].fillna(0.0)
    grid["attempts"] = grid["attempts"].fillna(0.0)

    grid = grid.sort_values(
        ["def_team", "zone", "game_date", "game_id"]
    ).reset_index(drop=True)

    grouped = grid.groupby(["def_team", "zone"], sort=False)
    # Same strictly-prior discipline as build_prior_counts: the current game
    # never contributes to the rate used to predict it.
    grid["opp_def_mk"] = grouped["makes"].cumsum() - grid["makes"]
    grid["opp_def_att"] = grouped["attempts"].cumsum() - grid["attempts"]

    wide = grid.pivot_table(
        index=["game_id", "def_team"],
        columns="zone",
        values=["opp_def_mk", "opp_def_att"],
        aggfunc="first",
    )
    wide.columns = [f"{stat}_{ZONE_SUFFIX[zone]}" for stat, zone in wide.columns]
    return wide.reset_index()


# Closest-defender bands, and the distance each is taken to represent when
# collapsing the four shares into one expected-separation number. Midpoints of
# the stated ranges; the open-ended top band is treated as 7ft rather than
# something larger, so the scalar stays conservative about how open "wide open"
# really is.
CONTEST_BANDS = {
    "0-2 Feet - Very Tight": 1.0,
    "2-4 Feet - Tight": 3.0,
    "4-6 Feet - Open": 5.0,
    "6+ Feet - Wide Open": 7.0,
}

CONTEST_BAND_SUFFIX = {
    "0-2 Feet - Very Tight": "very_tight",
    "2-4 Feet - Tight": "tight",
    "4-6 Feet - Open": "open",
    "6+ Feet - Wide Open": "wide_open",
}

# Tracked attempts a player needs before his contest profile is reported at
# all. Below this the shares are a handful of shots and would assert a
# tendency that is not there; NaN is the honest answer and XGBoost handles it.
MIN_CONTEST_ATTEMPTS = 60


def build_contest_history(engine) -> pd.DataFrame:
    """
    How contested a player's shots have been, accumulated over strictly PRIOR
    games.

    Why this is not what `player_shot_profile` already has
    ------------------------------------------------------
    That table stores the same four closest-defender bands, but only as a
    season aggregate, and the creation features built from it are lagged a
    whole season. So the model's entire knowledge of how open a shooter gets is
    a number describing last year. It cannot say that a player has been run off
    the line for the past three weeks.

    `player_game_contest` records the bands per GAME (see
    `ingestion/contest_ingestor.py`), which makes the same quantity
    accumulable: for every (player, game) this returns his band shares over
    every game he played BEFORE it, career-to-date and season-to-date.

    Returned columns, per band suffix S:
        contest_car_S      career-to-date share of tracked attempts in band S
        contest_ssn_S      season-to-date share
    plus:
        contest_car_sep    career-to-date expected separation, feet
        contest_ssn_sep    season-to-date expected separation, feet
        contest_att        tracked attempts behind the career figure

    A note on the denominator: these are shares of TRACKED attempts, which run
    slightly below true FGA because a minority of shots carry no defender
    assignment. That is fine for a share and wrong for a volume, so no attempt
    count from this table is used as a volume anywhere.
    """
    rows = pd.read_sql("""
        SELECT c.player_id, c.game_date, c.season, c.def_dist_range, c.fga
        FROM player_game_contest c
        WHERE c.fga > 0
    """, engine)
    if rows.empty:
        return pd.DataFrame(columns=["player_id", "game_id"])

    rows["game_date"] = pd.to_datetime(rows["game_date"])
    rows["suffix"] = rows["def_dist_range"].map(CONTEST_BAND_SUFFIX)
    rows = rows.dropna(subset=["suffix"])

    # One row per (player, game_date) with a column per band.
    wide = rows.pivot_table(
        index=["player_id", "season", "game_date"],
        columns="suffix", values="fga", aggfunc="sum",
    ).fillna(0.0).reset_index()

    suffixes = list(CONTEST_BAND_SUFFIX.values())
    for s in suffixes:
        if s not in wide.columns:
            wide[s] = 0.0

    wide = wide.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # Strictly prior: cumulative sum then subtract this game's own row, the
    # same construction `build_prior_counts` uses for shooting.
    car = wide.groupby("player_id", sort=False)
    ssn = wide.groupby(["player_id", "season"], sort=False)
    for s in suffixes:
        wide[f"_car_{s}"] = car[s].cumsum() - wide[s]
        wide[f"_ssn_{s}"] = ssn[s].cumsum() - wide[s]

    for scope in ("car", "ssn"):
        total = wide[[f"_{scope}_{s}" for s in suffixes]].sum(axis=1)
        gated = total.where(total >= MIN_CONTEST_ATTEMPTS)
        for s in suffixes:
            wide[f"contest_{scope}_{s}"] = wide[f"_{scope}_{s}"] / gated
        # One scalar: the separation an average attempt came with, in feet.
        weighted = sum(
            (wide[f"_{scope}_{s}"] * CONTEST_BANDS[band]
             for band, s in CONTEST_BAND_SUFFIX.items()),
            start=pd.Series(0.0, index=wide.index),
        )
        wide[f"contest_{scope}_sep"] = weighted / gated
        if scope == "car":
            wide["contest_att"] = total

    # Attach the game_id the shot rows key on. `player_game_contest` is keyed
    # by date because that is the granularity the endpoint filters at; a player
    # plays at most one game per date, so the join is unambiguous.
    games = pd.read_sql("""
        SELECT DISTINCT s.player_id, s.game_id, g.date AS game_date
        FROM shots s JOIN games g ON g.game_id = s.game_id
    """, engine)
    games["game_date"] = pd.to_datetime(games["game_date"])

    out_cols = (
        [f"contest_car_{s}" for s in suffixes]
        + [f"contest_ssn_{s}" for s in suffixes]
        + ["contest_car_sep", "contest_ssn_sep", "contest_att"]
    )
    merged = games.merge(
        wide[["player_id", "game_date"] + out_cols],
        on=["player_id", "game_date"], how="inner",
    )
    return merged[["player_id", "game_id"] + out_cols]


def build_supporting_cast(engine) -> pd.DataFrame:
    """
    What the shooter's OWN team gives him, computed leave-one-out and
    point-in-time.

    Why this exists
    ---------------
    The model knows a great deal about who is defending a shot and nothing at
    all about who is playing alongside it. `team_stats` carries exactly one
    column, `def_rating`, and it is consumed as `opp_def_rating` — the
    opponent's. The shooter's supporting cast has never been a feature.

    Why leave-one-out is not optional
    ---------------------------------
    A star's team stats are mostly the star. Denver's assisted-FG rate ranked
    6th of 30 in 2024-25 and Oklahoma City's 27th, which reads as "Jokic plays
    with better passers than Shai" — but Denver's number IS Jokic. Feeding the
    raw team rate into a Jokic shot re-encodes Jokic, and the model would
    happily learn a circular relationship.

    Every quantity here therefore EXCLUDES the shooter's own contribution:
    these are his teammates' numbers, not his team's. That is the quantity
    that answers "does this player get help", which is the question worth
    asking.

    Strictly prior, same as everything else in this module: a shot never
    contributes to the features describing it.

    Returns one row per (player_id, game_id) with:
        cast_ast_rate   teammates' assisted share of made field goals —
                        ball movement the shooter is not himself producing
        cast_3p_rate    teammates' three-point percentage — the spacing that
                        determines how much help defence he draws
        cast_efg        teammates' effective field-goal percentage — how much
                        attention the rest of the lineup commands
        cast_att        teammate attempts behind those rates, the model's
                        confidence cue
    """
    rows = pd.read_sql("""
        SELECT s.player_id,
               s.game_id,
               s.season,
               g.date AS game_date,
               CASE WHEN s.home_away = 1 THEN g.home_team ELSE g.away_team END
                   AS off_team,
               s.shot_made,
               CASE WHEN s.zone IN ('Left Corner 3', 'Right Corner 3',
                                    'Above the Break 3') THEN 1 ELSE 0 END AS is_three,
               CASE WHEN sc.shot_id IS NULL THEN 0 ELSE 1 END AS has_context,
               COALESCE(sc.is_assisted, 0) AS is_assisted
        FROM shots s
        JOIN games g ON g.game_id = s.game_id
        LEFT JOIN shot_context sc ON sc.shot_id = s.shot_id
        WHERE s.zone IS NOT NULL
          AND s.zone != 'Backcourt'
          AND s.home_away IS NOT NULL
    """, engine)

    if rows.empty:
        return pd.DataFrame(columns=["player_id", "game_id"])

    rows["game_date"] = pd.to_datetime(rows["game_date"])
    rows["made_three"] = rows["shot_made"] * rows["is_three"]
    # `is_assisted` is a per-shot label leak (assists are credited only on
    # makes) and is banned as a shot feature by tests/test_no_leaky_features.py.
    # Aggregated over a team's PRIOR games it leaks nothing about tonight's
    # shot, which is what makes the ball-movement signal reachable at all.
    #
    # The assisted rate is computed over makes WITH play-by-play context only.
    # Coverage starts in 2016-17, and counting an uncovered make as unassisted
    # would not read as missing data — it would read as a team that never
    # passes, which is a confident wrong answer rather than an absent one.
    rows["assisted_make"] = rows["shot_made"] * rows["is_assisted"]
    rows["make_with_context"] = rows["shot_made"] * rows["has_context"]
    # Effective FG% weights a made three at 1.5, the standard adjustment for
    # its extra point.
    rows["efg_num"] = rows["shot_made"] + 0.5 * rows["made_three"]

    stat_cols = ["shot_made", "made_three", "is_three", "assisted_make",
                 "make_with_context", "efg_num"]

    # Team totals per game, then the same totals attributed to each player, so
    # subtracting one from the other leaves the teammates' contribution alone.
    agg = {c: "sum" for c in stat_cols}
    team_game = rows.groupby(
        ["off_team", "season", "game_id", "game_date"], as_index=False
    ).agg({**agg, "player_id": "size"}).rename(columns={"player_id": "fga"})

    player_game = rows.groupby(
        ["player_id", "off_team", "season", "game_id", "game_date"], as_index=False
    ).agg({**agg, "shot_made": "sum"})
    player_game["fga"] = rows.groupby(
        ["player_id", "off_team", "season", "game_id", "game_date"]
    ).size().values

    # Cumulative totals reset EACH SEASON. A roster turns over completely; a
    # team's ball movement three years ago describes a different set of
    # players and is not evidence about the cast around this shooter tonight.
    team_game = team_game.sort_values(
        ["off_team", "season", "game_date", "game_id"]
    ).reset_index(drop=True)
    g_team = team_game.groupby(["off_team", "season"], sort=False)
    for col in stat_cols + ["fga"]:
        team_game[f"team_{col}"] = g_team[col].cumsum() - team_game[col]

    # Same for the player, within (team, season) so a midseason trade does not
    # carry his old team's contribution into the subtraction.
    player_game = player_game.sort_values(
        ["player_id", "off_team", "season", "game_date", "game_id"]
    ).reset_index(drop=True)
    g_player = player_game.groupby(["player_id", "off_team", "season"], sort=False)
    for col in stat_cols + ["fga"]:
        player_game[f"self_{col}"] = g_player[col].cumsum() - player_game[col]

    merged = player_game.merge(
        team_game[["off_team", "season", "game_id"]
                  + [f"team_{c}" for c in stat_cols + ["fga"]]],
        on=["off_team", "season", "game_id"], how="left",
    )

    # The leave-one-out step.
    for col in stat_cols + ["fga"]:
        merged[f"cast_{col}"] = merged[f"team_{col}"] - merged[f"self_{col}"]

    def _rate(num: str, den: str) -> pd.Series:
        """
        A rate, or NaN when the teammates have not yet accumulated enough
        evidence this season to support one.

        Below the gate the honest answer is "not known yet", which XGBoost
        handles natively as a missing value. Emitting 0.667 off three attempts
        would instead assert an elite shooting cast on no evidence — the same
        low-sample trap `shrink()` exists to avoid elsewhere, handled here by
        abstention because these rates have no fitted prior to regress toward.
        """
        denom = merged[den]
        out = merged[num] / denom.replace(0, np.nan)
        return out.where(denom >= MIN_CAST_ATTEMPTS)

    merged["cast_ast_rate"] = _rate("cast_assisted_make", "cast_make_with_context")
    merged["cast_3p_rate"] = _rate("cast_made_three", "cast_is_three")
    merged["cast_efg"] = _rate("cast_efg_num", "cast_fga")
    merged["cast_att"] = merged["cast_fga"]

    return merged[[
        "player_id", "game_id",
        "cast_ast_rate", "cast_3p_rate", "cast_efg", "cast_att",
    ]]


def fit_league_creation_priors(engine, through_season: str) -> dict[str, BetaPrior]:
    """
    Per zone, a Beta prior over "what share of a player's makes here did he
    create himself".

    The quantity the recommender's attainability number cannot express on its
    own. Attainability is a frequency — what share of a player's shots come
    from a spot — and frequency conflates two very different situations: a
    shot he can manufacture whenever he wants, and a shot that only exists
    when a teammate finds him. League-wide those separate sharply: 44% of
    restricted-area makes are unassisted against 4% of corner threes. A corner
    three is not a shot anybody goes and gets.

    Keyed by SUB-zone, so a dead-centre three and a wing three get separate
    priors — the split is sharpest exactly here (27% self-created within ten
    degrees of dead centre against 9.5% beyond sixty), and reporting one
    blended above-the-break figure for both would hide the distinction the
    number exists to make.

    Fit through the last TRAINING season only, same rule as every other prior
    in this module.
    """
    from src.training.attainability import ANGLE_SPLIT_ZONES, attach_sub_zone

    df = pd.read_sql(f"""
        SELECT s.zone, s.loc_x, s.loc_y, s.player_id, s.season, s.shot_made,
               CASE WHEN sc.is_assisted = 1 THEN 0 ELSE 1 END AS self_make
        FROM shots s
        JOIN shot_context sc ON sc.shot_id = s.shot_id
        WHERE s.shot_made = 1
          AND s.zone IS NOT NULL AND s.zone != 'Backcourt'
          AND s.season <= '{through_season}'
    """, engine)
    if df.empty:
        return {}

    df = attach_sub_zone(df)
    grouped = df.groupby(["sub_zone", "player_id", "season"], as_index=False).agg(
        self_makes=("self_make", "sum"), makes=("shot_made", "size")
    )
    priors = fit_priors(
        grouped, ["sub_zone"], makes_col="self_makes", attempts_col="makes"
    )
    out = {z: priors[(z,)] for z in grouped["sub_zone"].unique() if (z,) in priors}

    # Bare-zone aggregate priors for the two angle-split zones, keyed by the
    # UNSPLIT name — same bug, same fix as lookup_diet_history's missing
    # bare-zone entries. `lookup_zone_creation` resolves to the bare zone
    # name whenever it isn't given exact coordinates (attach_sub_zone's
    # documented behaviour), and `creation_priors` had no such key for
    # either angle-split zone, only its "(centre)"/"(wing)" splits. The
    # lookup's own fallback (`.get(zone) or .get(parent)`) cannot rescue
    # this, because zone == parent in exactly the case that needs rescuing.
    #
    # The visible symptom: asking for Jamal Murray's self-created share at
    # Above the Break 3 came back with no prior at all, so `share` and
    # `league_self_created_share` were both None and `creation_note`
    # returned nothing — despite `self_creation_index` correctly rating him
    # an elite self-creator and his supporting cast (Jokić) showing a 71%
    # teammate assist rate. The gap read as "we don't know," not as the
    # (false) "nobody creates this for him" it was mistaken for, but the
    # missing prior is the same class of silent gap either way.
    parent_grouped = df.groupby(["zone", "player_id", "season"], as_index=False).agg(
        self_makes=("self_make", "sum"), makes=("shot_made", "size")
    )
    parent_priors = fit_priors(
        parent_grouped, ["zone"], makes_col="self_makes", attempts_col="makes"
    )
    for zone in ANGLE_SPLIT_ZONES:
        if (zone,) in parent_priors:
            out[zone] = parent_priors[(zone,)]
    return out


def lookup_zone_creation(conn, player_id: str, zone: str,
                         creation_priors: dict[str, BetaPrior],
                         season: str | None = None, as_of_date=None) -> dict:
    """
    How this player's makes in one zone were generated: by himself, or by a
    teammate finding him.

    Shrunk toward the zone's league rate, so a player with four makes in a
    corner does not read as a 100% self-creator. Career-to-date rather than
    season-to-date: shot creation is a stable trait and the season-only sample
    per zone is thin for everyone but high-volume starters.

    Returns the player's shrunk self-created share, the league rate for the
    same zone, and the raw counts behind it so a caller can say how much
    evidence there is.
    """
    from sqlalchemy import text

    from src.training.attainability import ANGLE_SPLIT_ZONES, attach_sub_zone

    # `zone` may arrive as either a plain zone or an already-split sub-zone.
    # The parent zone is what the shots table stores, so query on that and
    # narrow to the sub-zone in pandas afterwards.
    parent = re.sub(r" \((centre|wing)\)$", "", zone)

    clauses = []
    params = {"pid": str(player_id), "zone": parent}
    if season is not None:
        clauses.append("AND s.season <= :season")
        params["season"] = season
    if as_of_date is not None:
        clauses.append("AND g.date < :as_of")
        params["as_of"] = str(as_of_date)

    rows = conn.execute(text(f"""
        SELECT s.zone, s.loc_x, s.loc_y,
               CASE WHEN sc.is_assisted = 1 THEN 0 ELSE 1 END AS self_make
        FROM shots s
        JOIN shot_context sc ON sc.shot_id = s.shot_id
        JOIN games g ON g.game_id = s.game_id
        WHERE s.player_id = :pid
          AND s.zone = :zone
          AND s.shot_made = 1
          {' '.join(clauses)}
    """), params).fetchall()

    frame = pd.DataFrame(rows, columns=["zone", "loc_x", "loc_y", "self_make"])
    if parent in ANGLE_SPLIT_ZONES and not frame.empty:
        frame = attach_sub_zone(frame)
        # An unsplit parent name means "either half"; a split name narrows.
        if zone != parent:
            frame = frame[frame["sub_zone"] == zone]

    prior = creation_priors.get(zone) or creation_priors.get(parent)
    league = float(prior.mean) if prior is not None else None

    self_makes = float(frame["self_make"].sum()) if not frame.empty else 0.0
    makes = float(len(frame))

    if prior is None:
        share = None
    else:
        share = float(shrink(self_makes, makes, prior))

    return {
        "self_created_share": share,
        "league_self_created_share": league,
        "makes": int(makes),
        "self_makes": int(self_makes),
    }


def lookup_supporting_cast(conn, player_id: str, season: str,
                           as_of_date=None) -> dict:
    """
    Serving-path equivalent of `build_supporting_cast` for one player.

    Returns the same `cast_*` keys the training builder produces. Without this
    the attainability model would receive NaN for every cast feature at
    serving time while seeing them populated for virtually every training row
    — the precise skew that made `recent_10_fg` look like a career debut on
    every served shot, and which tests/test_train_serve_parity.py exists to
    catch.

    The player's team is taken as the one he took the most shots for this
    season, so a midseason trade resolves to where he actually plays now.
    """
    from sqlalchemy import text

    date_clause = "AND g.date < :as_of" if as_of_date is not None else ""
    params = {"pid": str(player_id), "season": season}
    if as_of_date is not None:
        params["as_of"] = str(as_of_date)

    team_row = conn.execute(text(f"""
        SELECT CASE WHEN s.home_away = 1 THEN g.home_team ELSE g.away_team END AS team,
               COUNT(*) AS n
        FROM shots s
        JOIN games g ON g.game_id = s.game_id
        WHERE s.player_id = :pid AND s.season = :season
          AND s.home_away IS NOT NULL
          {date_clause}
        GROUP BY team ORDER BY n DESC LIMIT 1
    """), params).fetchone()

    empty = {c: None for c in
             ("cast_ast_rate", "cast_3p_rate", "cast_efg", "cast_att")}
    if team_row is None:
        return empty

    params["team"] = team_row[0]
    row = conn.execute(text(f"""
        SELECT SUM(s.shot_made)                                        AS makes,
               COUNT(*)                                                AS fga,
               SUM(CASE WHEN s.zone IN ('Left Corner 3','Right Corner 3',
                                        'Above the Break 3')
                        THEN 1 ELSE 0 END)                             AS threes,
               SUM(CASE WHEN s.zone IN ('Left Corner 3','Right Corner 3',
                                        'Above the Break 3')
                        THEN s.shot_made ELSE 0 END)                   AS made_threes,
               SUM(CASE WHEN sc.shot_id IS NOT NULL AND s.shot_made = 1
                        THEN 1 ELSE 0 END)                             AS makes_with_ctx,
               SUM(CASE WHEN sc.is_assisted = 1 AND s.shot_made = 1
                        THEN 1 ELSE 0 END)                             AS assisted
        FROM shots s
        JOIN games g ON g.game_id = s.game_id
        LEFT JOIN shot_context sc ON sc.shot_id = s.shot_id
        WHERE s.season = :season
          AND s.player_id != :pid
          AND s.home_away IS NOT NULL
          AND s.zone IS NOT NULL AND s.zone != 'Backcourt'
          AND CASE WHEN s.home_away = 1 THEN g.home_team ELSE g.away_team END = :team
          {date_clause}
    """), params).fetchone()

    if row is None or not row[1]:
        return empty

    makes, fga, threes, made_threes, makes_ctx, assisted = (
        float(v or 0) for v in row
    )
    if fga < MIN_CAST_ATTEMPTS:
        return empty

    return {
        "cast_ast_rate": (assisted / makes_ctx) if makes_ctx else None,
        "cast_3p_rate": (made_threes / threes) if threes else None,
        "cast_efg": ((makes + 0.5 * made_threes) / fga) if fga else None,
        "cast_att": fga,
    }


def apply_opponent_defence(df: pd.DataFrame,
                           zone_priors: dict[str, BetaPrior]) -> pd.DataFrame:
    """
    Turn opponent prior counts into shrunk rates, and select the one matching
    each shot's own zone.

    Shrinks toward the same league zone priors the shooter rates use, so a team
    twelve games into a season is not credited with an elite rim defence on
    forty possessions.
    """
    out = df.copy()
    for zone in ZONES:
        suffix = ZONE_SUFFIX[zone]
        prior = zone_priors.get(zone)
        mk = out.get(f"opp_def_mk_{suffix}", 0.0)
        att = out.get(f"opp_def_att_{suffix}", 0.0)
        if prior is None:
            out[f"opp_def_rate_{suffix}"] = np.nan
        else:
            out[f"opp_def_rate_{suffix}"] = shrink(mk, att, prior)
    return out


# ── Point-in-time defender quality ───────────────────────────────────────────
#
# `defender_stats` (LeagueDashPtDefend) is a season aggregate — the NBA API
# never published a per-game or as-of-date version of it. Every defender
# feature that reads it (`def_fg_pct_zone`, `def_pct_plusminus_zone`,
# `def_fg_pct_overall`, `def_pct_plusminus`, `def_freq_zone`) was therefore
# joined by season alone, exactly the leak this module exists to remove for
# shooters: a shot contested by a defender contributes to that defender's own
# season FG%-allowed, which is then used as a feature describing that same
# shot. What follows rebuilds the same (defender, category) -> FG%-allowed
# figures from `matchups` + `shots` instead, strictly prior-games-only, using
# the same possession-weighted mixture `build.build_defender_mixture` already
# uses for physicals — so the fix only touches where the QUALITY numbers come
# from, not how they get blended into a shot's feature vector on either path.


def _load_defender_exposure(engine) -> pd.DataFrame:
    """
    Per (shot, defender who guarded that shooter that game) fractional
    credit for the shot's outcome, weighted by possession share within that
    game.

    A shot is not linked to its own contesting defender in this data — only
    "who guarded this shooter how much, this game" is known (`matchups`).
    Every shot a shooter took in a game therefore inherits the SAME defender
    weight distribution, which is the same simplification
    `build_defender_mixture` already makes and accepts; this does not make it
    worse, only reuses it for a second purpose.
    """
    weights = pd.read_sql("""
        SELECT game_id, offense_player_id, defense_player_id,
               COALESCE(NULLIF(partial_possessions, 0), matchup_minutes, 0) AS raw_weight
        FROM matchups
        WHERE COALESCE(NULLIF(partial_possessions, 0), matchup_minutes, 0) > 0
    """, engine)
    total = weights.groupby(["game_id", "offense_player_id"])["raw_weight"].transform("sum")
    weights["w"] = weights["raw_weight"] / total.replace(0, np.nan)
    weights = weights.dropna(subset=["w"])

    shots = pd.read_sql("""
        SELECT s.game_id, s.player_id AS offense_player_id, s.zone,
               s.shot_made, g.date AS game_date
        FROM shots s
        JOIN games g ON g.game_id = s.game_id
        WHERE s.zone IS NOT NULL AND s.zone != 'Backcourt'
    """, engine)
    shots["game_date"] = pd.to_datetime(shots["game_date"])
    shots["category"] = shots["zone"].map(ZONE_TO_DEF_CATEGORY)

    exposure = shots.merge(
        weights[["game_id", "offense_player_id", "defense_player_id", "w"]],
        on=["game_id", "offense_player_id"], how="inner",
    )
    return exposure


def fit_league_category_priors(engine, through_season: str) -> dict[str, BetaPrior]:
    """
    One Beta prior per defense category, pooling the zones that map to it —
    the defender-side equivalent of `fit_league_zone_priors`. A category like
    "3 Pointers" spans zones with different true league rates (corners run
    cooler than above-the-break), so this refits from scratch by category
    rather than averaging the zone priors after the fact.
    """
    df = pd.read_sql(f"""
        SELECT s.zone, s.player_id, s.season,
               SUM(s.shot_made) AS makes, COUNT(*) AS attempts
        FROM shots s
        WHERE s.zone IS NOT NULL
          AND s.zone != 'Backcourt'
          AND s.season <= '{through_season}'
        GROUP BY s.zone, s.player_id, s.season
    """, engine)
    df["category"] = df["zone"].map(ZONE_TO_DEF_CATEGORY)
    priors = fit_priors(df, ["category"], makes_col="makes", attempts_col="attempts")
    return {cat: priors[(cat,)] for cat in df["category"].unique() if (cat,) in priors}


def build_defender_category_rates(engine, through_season: str) -> pd.DataFrame:
    """
    Point-in-time FG%-allowed per (defender, game, defense category) — the
    leak-free replacement for reading `defender_stats` by season.

    Same strictly-prior discipline as `build_prior_counts`: cumulative sum
    then shift, so a game's own shots never contribute to that game's own
    defender features.

    Returns one row per (defense_player_id, game_id, defense_category) with
    d_fg_pct / pct_plusminus / freq, plus a pooled "Overall" row per
    (defense_player_id, game_id) — the same shape `defender_stats` provided,
    so `build.build_defender_mixture` only needs its join key changed to use
    this instead, not its downstream blending logic.
    """
    exposure = _load_defender_exposure(engine)
    exposure["w_mk"] = exposure["w"] * exposure["shot_made"]

    per_game = exposure.groupby(
        ["defense_player_id", "game_id", "game_date", "category"], as_index=False
    ).agg(w_att=("w", "sum"), w_mk=("w_mk", "sum"))

    defender_games = per_game[["defense_player_id", "game_id", "game_date"]].drop_duplicates()
    grid = defender_games.merge(pd.DataFrame({"category": DEFENSE_CATEGORIES}), how="cross")
    grid = grid.merge(
        per_game, on=["defense_player_id", "game_id", "game_date", "category"], how="left"
    )
    grid["w_att"] = grid["w_att"].fillna(0.0)
    grid["w_mk"] = grid["w_mk"].fillna(0.0)

    grid = grid.sort_values(
        ["defense_player_id", "category", "game_date", "game_id"]
    ).reset_index(drop=True)
    g = grid.groupby(["defense_player_id", "category"], sort=False)
    # Strictly prior: row i holds the total through row i-1.
    grid["pit_mk"] = g["w_mk"].cumsum() - grid["w_mk"]
    grid["pit_att"] = g["w_att"].cumsum() - grid["w_att"]

    priors = fit_league_category_priors(engine, through_season=through_season)
    pooled_mean = float(np.mean([p.mean for p in priors.values()])) if priors else 0.45
    pooled_strength = float(np.mean([p.strength for p in priors.values()])) if priors else 100.0

    # Single-level shrinkage (career-to-date only, toward the league category
    # rate) rather than the two-level career->season hierarchy shooting rates
    # use: a defender's possession-weighted evidence in one category is far
    # sparser per season than a shooter's own attempts in a zone, so a second,
    # noisier season-only layer looked more likely to overfit than to track a
    # real hot/cold defensive stretch. Worth re-measuring with `--ablate` if
    # the point-in-time feature earns its place at all.
    rate = pd.Series(np.nan, index=grid.index)
    category_mean = pd.Series(np.nan, index=grid.index)
    for cat in DEFENSE_CATEGORIES:
        mask = (grid["category"] == cat).to_numpy()
        prior = priors.get(cat, BetaPrior(mean=pooled_mean, strength=pooled_strength))
        rate.loc[mask] = shrink(grid.loc[mask, "pit_mk"], grid.loc[mask, "pit_att"], prior)
        category_mean.loc[mask] = prior.mean
    grid["d_fg_pct"] = rate
    grid["pct_plusminus"] = grid["d_fg_pct"] - category_mean

    total_att = grid.groupby(["defense_player_id", "game_id"])["pit_att"].transform("sum")
    grid["freq"] = grid["pit_att"] / total_att.replace(0, np.nan)

    long_table = grid[
        ["defense_player_id", "game_id", "category", "d_fg_pct", "pct_plusminus", "freq"]
    ].rename(columns={"category": "defense_category"})

    # Pooled "Overall" row, matching the shape `defender_stats` already had —
    # `build_defender_mixture` filters `defense_category == "Overall"` for it.
    overall_prior = BetaPrior(mean=pooled_mean, strength=pooled_strength)
    overall = grid.groupby(["defense_player_id", "game_id"], as_index=False).agg(
        pit_mk=("pit_mk", "sum"), pit_att=("pit_att", "sum")
    )
    overall["d_fg_pct"] = shrink(overall["pit_mk"], overall["pit_att"], overall_prior)
    overall["pct_plusminus"] = overall["d_fg_pct"] - pooled_mean
    overall["freq"] = 1.0
    overall["defense_category"] = "Overall"
    overall = overall[
        ["defense_player_id", "game_id", "defense_category", "d_fg_pct", "pct_plusminus", "freq"]
    ]

    return pd.concat([long_table, overall], ignore_index=True)


def lookup_defender_category_rates(
    conn, defender_id: str, category_priors: dict[str, BetaPrior], as_of_date=None
) -> dict:
    """
    Serving-path equivalent of `build_defender_category_rates` for a single
    named defender — the parity guarantee that keeps the recommender from
    reading the leaky `defender_stats` table the way it used to.

    `category_priors` must be the SAME fitted priors the model was trained
    against (loaded from run metadata, same as `zone_priors` is), not
    refitted from whatever is in the database at request time — the same
    discipline `apply_hierarchy`'s `zone_priors` argument follows.

    Returns `{"by_category": {category: {"d_fg_pct", "pct_plusminus", "freq"}}}`,
    matching the shape `_defender_row` in `recommender.py` already expects
    from `defender_stats`, so only the data source changes there, not the
    blending logic downstream of it.
    """
    from sqlalchemy import text

    date_clause = "AND g.date < :as_of" if as_of_date is not None else ""
    params = {"pid": str(defender_id)}
    if as_of_date is not None:
        params["as_of"] = str(as_of_date)

    rows = conn.execute(text(f"""
        SELECT s.zone, SUM(s.shot_made) AS makes, COUNT(*) AS attempts
        FROM matchups m
        JOIN shots s ON s.game_id = m.game_id AND s.player_id = m.offense_player_id
        JOIN games g ON g.game_id = m.game_id
        WHERE m.defense_player_id = :pid
          AND COALESCE(NULLIF(m.partial_possessions, 0), m.matchup_minutes, 0) > 0
          AND s.zone IS NOT NULL AND s.zone != 'Backcourt'
          {date_clause}
        GROUP BY s.zone
    """), params).fetchall()

    by_cat: dict[str, dict] = {cat: {"mk": 0.0, "att": 0.0} for cat in DEFENSE_CATEGORIES}
    for zone, makes, attempts in rows:
        cat = ZONE_TO_DEF_CATEGORY.get(zone)
        if cat is None:
            continue
        by_cat[cat]["mk"] += float(makes or 0)
        by_cat[cat]["att"] += float(attempts or 0)

    priors = category_priors
    if not priors:
        return {"by_category": {}}

    pooled_mean = float(np.mean([p.mean for p in priors.values()]))
    pooled_strength = float(np.mean([p.strength for p in priors.values()]))

    def _to_native(value) -> float | None:
        # `shrink()` returns a numpy scalar even for plain-float inputs, which
        # FastAPI's encoder cannot always serialize on its own. A 0/0 divide
        # (no exposure and, in tests, a strength=0 prior) yields NaN, which
        # means the same thing None does here — no basis for a rate — so both
        # collapse to the one JSON-safe representation of "unknown".
        f = float(value)
        return None if np.isnan(f) else f

    result = {}
    total_mk = total_att = 0.0
    for cat, counts in by_cat.items():
        prior = priors.get(cat, BetaPrior(mean=pooled_mean, strength=pooled_strength))
        rate = _to_native(shrink(counts["mk"], counts["att"], prior))
        result[cat] = {
            "d_fg_pct": rate,
            "pct_plusminus": (rate - prior.mean) if rate is not None else None,
            "freq": None,  # filled in below once the total is known
        }
        total_mk += counts["mk"]
        total_att += counts["att"]

    for cat, counts in by_cat.items():
        result[cat]["freq"] = (counts["att"] / total_att) if total_att > 0 else None

    overall_prior = BetaPrior(mean=pooled_mean, strength=pooled_strength)
    overall_rate = _to_native(shrink(total_mk, total_att, overall_prior))
    result["Overall"] = {
        "d_fg_pct": overall_rate,
        "pct_plusminus": (overall_rate - pooled_mean) if overall_rate is not None else None,
        "freq": 1.0,
    }
    return {"by_category": result}


LINEUP_FEATURE_COLS = [
    "oncourt_off_creation", "oncourt_off_gravity", "oncourt_off_rim_pressure",
    "oncourt_off_n", "oncourt_def_fg_pct", "oncourt_def_n",
]


def build_lineup_context(engine, through_season: str) -> pd.DataFrame:
    """
    Per shot, what the REST of the on-court lineup looks like — not the
    shooter (already fully described elsewhere) and not the primary
    defender (`defender_id`, already its own feature group), but the other
    four offensive teammates and the other four defenders who were also on
    the floor for that specific shot.

    This is the experiment the "gravity" and "team/roster" questions this
    session kept circling back to actually need: `playmaking_gravity`
    already exists but only as a SELF-effect (a player's own passing making
    HIS OWN shot marginally easier), and the supporting-cast features
    (cast_ast_rate etc.) are a whole-SEASON roster aggregate, not who was
    literally on the floor for this possession. `shot_on_court` (see
    src/ingestion/lineup_ingestor.py) is what makes the real, per-shot
    version of both questions answerable at all.

    Offense side: the mean self_creation_index / playmaking_gravity /
    rim_pressure of the other four, from the same lagged-one-season
    creation profile the shooter's own features already use (so a shot in
    2023-24 sees teammates' 2022-23 profiles — same availability and
    leakage reasoning as attach_creation_features).

    Defense side: the mean point-in-time "Overall" FG%-allowed of the
    other four defenders, from the exact same build_defender_category_rates
    the primary defender's own features already use — not a second,
    differently-built defender-quality number.

    Coverage is bounded by shot_on_court's own coverage (see
    lineup_ingestor.py — real substitution data gaps mean some shots have
    no reconstructed lineup at all), so these columns are NaN for a real
    share of rows. XGBoost treats that as an ordinary missing feature, the
    same way it already does for contest and defender coverage gaps.
    """
    from src.features.creation import _previous_season, load_creation_profiles

    onc = pd.read_sql("""
        SELECT soc.shot_id, soc.player_id, soc.role, soc.team_id,
               s.season, s.game_id, s.player_id AS shooter_id,
               s.defender_id AS primary_defender_id
        FROM shot_on_court soc
        JOIN shots s ON s.shot_id = soc.shot_id
    """, engine)
    if onc.empty:
        return pd.DataFrame(columns=["shot_id"] + LINEUP_FEATURE_COLS)

    # ── Offense: the other four teammates' creation profile ────────────────
    offense = onc[(onc["role"] == "offense") & (onc["player_id"] != onc["shooter_id"])].copy()
    profiles = load_creation_profiles(engine)
    offense["_lag_season"] = offense["season"].map(_previous_season)
    lagged = profiles.rename(columns={"season": "_lag_season"})
    offense = offense.merge(
        lagged[["player_id", "_lag_season", "self_creation_index",
               "playmaking_gravity", "rim_pressure"]],
        on=["player_id", "_lag_season"], how="left",
    )
    off_agg = offense.groupby("shot_id").agg(
        oncourt_off_creation=("self_creation_index", "mean"),
        oncourt_off_gravity=("playmaking_gravity", "mean"),
        oncourt_off_rim_pressure=("rim_pressure", "mean"),
        oncourt_off_n=("player_id", "count"),
    ).reset_index()

    # ── Defense: the other four defenders' point-in-time quality ───────────
    defense = onc[
        (onc["role"] == "defense")
        & (onc["primary_defender_id"].notna())
        & (onc["player_id"] != onc["primary_defender_id"])
    ].copy()
    rates = build_defender_category_rates(engine, through_season=through_season)
    overall = rates[rates["defense_category"] == "Overall"][
        ["defense_player_id", "game_id", "d_fg_pct"]
    ].rename(columns={"defense_player_id": "player_id"})
    defense = defense.merge(overall, on=["player_id", "game_id"], how="left")
    def_agg = defense.groupby("shot_id").agg(
        oncourt_def_fg_pct=("d_fg_pct", "mean"),
        oncourt_def_n=("player_id", "count"),
    ).reset_index()

    return off_agg.merge(def_agg, on="shot_id", how="outer")
