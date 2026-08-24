"""
Per-player shot-mechanic mixes, and how the recommender uses them.

The problem
-----------
Play-by-play mechanics are the most valuable features in the model — a dunk and
a pull-up from the same spot on the floor differ by 41 percentage points. But
they describe a shot that has already happened. The recommender asks about a
shot that has not: "should this player shoot from here?" has no mechanic
attached, because the mechanic is part of what is being decided.

Leaving the indicators at zero is not an option. Every training row has exactly
one mechanic set, so an all-zero row is a combination the model has never seen,
and gradient-boosted trees do not extrapolate gracefully — that is precisely
how naming a tall defender once produced an 18-point swing.

The approach
------------
Marginalise. A player's mechanic mix in a zone is a measurable, fairly stable
trait, so the honest prediction is the expectation over it:

    P(make | player, spot) = SUM_m  P(mech = m | player, zone)
                                    * P(make | mech = m, player, spot, ...)

Every scored row then carries a real mechanic, matching how the model was
trained, and the answer accounts for the fact that Curry's above-the-break
attempts are 47% catch-and-shoot, 30% pull-up and 19% step-back rather than
some average shot that does not exist.

It also makes the engine more useful, not less: the per-mechanic predictions are
worth surfacing on their own. "From the left wing, your catch-and-shoot is worth
1.14 expected points and your step-back 0.96" is advice; a single blended number
is a statistic.

Shrinkage
---------
Mix estimates are shrunk toward the league's mix for that zone, by attempts, on
the same principle as every other rate in this pipeline. A player with nine
above-the-break attempts does not get a confident claim that he is a 100%
step-back shooter from there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.spec import SHOT_MECHANICS, classify_mechanic

# Attempts-equivalent weight given to the league mix. A player needs roughly
# this many attempts in a zone before his own mix dominates the estimate.
MIX_PRIOR_ATTEMPTS = 40.0

# Mechanics below this share are dropped before scoring. They contribute almost
# nothing to the marginal and each one costs a full pass over the grid, so the
# cutoff is a straight accuracy-for-latency trade with a negligible accuracy
# side.
MIN_MECHANIC_SHARE = 0.03


def league_zone_mix(engine, through_season: str | None = None) -> pd.DataFrame:
    """
    League-wide mechanic distribution per zone — the prior thin players regress
    toward.
    """
    clause = f"AND s.season <= '{through_season}'" if through_season else ""
    raw = pd.read_sql(f"""
        SELECT s.zone, x.shot_subtype, COUNT(*) AS n
        FROM shots s
        JOIN shot_context x ON s.shot_id = x.shot_id
        WHERE s.zone IS NOT NULL AND s.zone != 'Backcourt'
        {clause}
        GROUP BY s.zone, x.shot_subtype
    """, engine)
    if raw.empty:
        return pd.DataFrame(columns=["zone", "mechanic", "share"])

    raw["mechanic"] = raw["shot_subtype"].map(classify_mechanic)
    grouped = raw.groupby(["zone", "mechanic"], as_index=False)["n"].sum()
    total = grouped.groupby("zone")["n"].transform("sum")
    grouped["share"] = grouped["n"] / total
    return grouped[["zone", "mechanic", "share"]]


def player_zone_mix(engine, player_id: str, season: str,
                    league: pd.DataFrame | None = None,
                    lookback_seasons: int = 2) -> dict[str, dict[str, float]]:
    """
    `{zone: {mechanic: share}}` for one player, shrunk toward the league mix.

    Uses the current season plus a short lookback, because a mechanic mix is a
    stable trait and one season alone leaves the rarer zones (corner threes for
    most players) far too thin to estimate.
    """
    if league is None:
        league = league_zone_mix(engine)

    start_year = int(season.split("-")[0]) - lookback_seasons
    floor = f"{start_year}-{str(start_year + 1)[-2:]}"

    raw = pd.read_sql(f"""
        SELECT s.zone, x.shot_subtype, COUNT(*) AS n
        FROM shots s
        JOIN shot_context x ON s.shot_id = x.shot_id
        WHERE s.player_id = '{player_id}'
          AND s.season >= '{floor}' AND s.season <= '{season}'
          AND s.zone IS NOT NULL AND s.zone != 'Backcourt'
        GROUP BY s.zone, x.shot_subtype
    """, engine)

    league_by_zone: dict[str, dict[str, float]] = {}
    for zone, grp in league.groupby("zone"):
        league_by_zone[zone] = dict(zip(grp["mechanic"], grp["share"]))

    if raw.empty:
        return league_by_zone

    raw["mechanic"] = raw["shot_subtype"].map(classify_mechanic)
    player = raw.groupby(["zone", "mechanic"], as_index=False)["n"].sum()

    out: dict[str, dict[str, float]] = {}
    for zone, prior in league_by_zone.items():
        rows = player[player["zone"] == zone]
        attempts = float(rows["n"].sum())
        counts = dict(zip(rows["mechanic"], rows["n"]))

        mix = {}
        for mechanic in SHOT_MECHANICS:
            observed = float(counts.get(mechanic, 0.0))
            prior_share = float(prior.get(mechanic, 0.0))
            mix[mechanic] = (
                (observed + MIX_PRIOR_ATTEMPTS * prior_share)
                / (attempts + MIX_PRIOR_ATTEMPTS)
            )

        total = sum(mix.values())
        if total > 0:
            mix = {k: v / total for k, v in mix.items()}
        out[zone] = {k: v for k, v in mix.items() if v >= MIN_MECHANIC_SHARE}

        # Renormalise after the cutoff so the weights still form a distribution;
        # otherwise the marginal is scaled down by whatever the tail held.
        kept = sum(out[zone].values())
        if kept > 0:
            out[zone] = {k: v / kept for k, v in out[zone].items()}

    return out


def expand_grid_over_mechanics(grid: pd.DataFrame,
                               mix: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    Replicate each grid row once per plausible mechanic, carrying the weight.

    Returns the expanded frame with `_mechanic` and `_mech_weight` columns. The
    caller predicts on it and aggregates back with `marginalize`.
    """
    frames = []
    for zone, weights in mix.items():
        zone_rows = grid[grid["zone"] == zone]
        if zone_rows.empty or not weights:
            continue
        for mechanic, weight in weights.items():
            copy = zone_rows.copy()
            copy["_mechanic"] = mechanic
            copy["_mech_weight"] = weight
            frames.append(copy)

    if not frames:
        out = grid.copy()
        out["_mechanic"] = "other"
        out["_mech_weight"] = 1.0
        return out

    return pd.concat(frames, ignore_index=True)


def marginalize(expanded: pd.DataFrame, prediction_col: str = "make_probability",
                key_cols: tuple[str, ...] = ("loc_x", "loc_y", "zone")) -> pd.DataFrame:
    """
    Collapse the expanded frame back to one row per location.

    `make_probability` becomes the weighted expectation over mechanics, and the
    single best mechanic is reported alongside — the number the engine should
    lead with is the realistic one, but the actionable detail is which shot type
    to look for.
    """
    work = expanded.copy()
    work["_weighted"] = work[prediction_col] * work["_mech_weight"]

    aggregated = work.groupby(list(key_cols), as_index=False).agg(
        **{prediction_col: ("_weighted", "sum")},
        _weight_total=("_mech_weight", "sum"),
    )
    # Guard against a zone whose weights did not sum to one.
    aggregated[prediction_col] = (
        aggregated[prediction_col] / aggregated["_weight_total"].replace(0, np.nan)
    )

    best = work.loc[work.groupby(list(key_cols))[prediction_col].idxmax()]
    best = best[list(key_cols) + ["_mechanic", prediction_col]].rename(
        columns={"_mechanic": "best_mechanic", prediction_col: "best_mechanic_prob"}
    )

    return aggregated.drop(columns=["_weight_total"]).merge(best, on=list(key_cols))
