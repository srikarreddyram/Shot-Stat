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

import numpy as np
import pandas as pd

from src.features.shrinkage import BetaPrior, fit_priors, shrink, shrink_toward
from .zones import THREE_POINT_ZONES, ZONE_SUFFIX, ZONES

# How many attempts of season-specific evidence it takes to move a player off
# his career rate. Lower than a league prior's strength because a player's own
# career mean is a far better starting guess than the league's, so less
# evidence is needed to justify departing from it. Tuned by backtest; see
# src/training/backtest.py.
SEASON_TO_CAREER_STRENGTH = 60.0


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
        JOIN games g ON g.game_id = s.game_id
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
