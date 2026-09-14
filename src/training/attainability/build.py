"""Training-matrix construction: shrunk zone-frequency targets, the
point-in-time diet-snapshot rows the model actually trains on, and the
leave-one-out supporting-cast join."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.point_in_time import build_supporting_cast
from src.features.shrinkage import fit_beta_prior
from .features import (
    ANGLE_SPLIT_ZONES,
    CAST_FEATURE_COLS,
    DIET_SNAPSHOTS,
    SUB_ZONES,
    attach_sub_zone,
    encode_zone_and_position,
)


def build_zone_frequency_targets(engine, min_attempts: int = 50) -> pd.DataFrame:
    """
    Observed shot-diet shares per (player, season, zone), shrunk.

    Raw shares are noisy for low-volume players in exactly the way raw
    shooting percentages are, and for the same reason: a player with 60 total
    attempts who happened to take four corner threes reads as a 6.7% corner-
    three shooter with no evidence behind it. Each zone's share is shrunk
    toward the league distribution using a Beta prior fit per zone.

    Shares are per SUB-zone (see `features.attach_sub_zone`), so the two wide
    zones are split by angle and the model can distinguish a dead-centre
    three from a wing one. `zone` is carried alongside for callers that still
    reason in the six-zone taxonomy.

    Returns one row per (player_id, season, sub_zone) with `zone_share`.
    """
    shots = pd.read_sql("""
        SELECT s.player_id, s.season, s.zone, s.loc_x, s.loc_y
        FROM shots s
        WHERE s.zone IS NOT NULL AND s.zone != 'Backcourt'
    """, engine)
    shots = attach_sub_zone(shots)
    counts = (
        shots.groupby(["player_id", "season", "sub_zone"])
        .size().rename("attempts").reset_index()
    )

    totals = counts.groupby(["player_id", "season"])["attempts"].sum().rename("total")
    counts = counts.merge(totals, on=["player_id", "season"])
    counts = counts[counts["total"] >= min_attempts]

    # Complete the grid: a player who took zero mid-range shots has a
    # meaningful zero, and dropping the row would leave the model to infer
    # absence from missingness.
    grid = (
        counts[["player_id", "season", "total"]].drop_duplicates()
        .merge(pd.DataFrame({"sub_zone": SUB_ZONES}), how="cross")
    )
    grid = grid.merge(
        counts[["player_id", "season", "sub_zone", "attempts"]],
        on=["player_id", "season", "sub_zone"], how="left",
    )
    grid["attempts"] = grid["attempts"].fillna(0.0)

    out = []
    for _, group in grid.groupby("sub_zone"):
        prior = fit_beta_prior(
            group["attempts"].to_numpy(), group["total"].to_numpy()
        )
        g = group.copy()
        g["zone_share"] = (
            (g["attempts"] + prior.alpha) / (g["total"] + prior.strength)
        )
        out.append(g)

    result = pd.concat(out, ignore_index=True)

    # Renormalize so a player's shrunk shares sum to one. Shrinking each
    # sub-zone independently does not preserve the simplex, and a "share"
    # vector summing to 1.04 would quietly inflate every expected-points
    # figure the recommender derives from it.
    share_sum = result.groupby(["player_id", "season"])["zone_share"].transform("sum")
    result["zone_share"] = result["zone_share"] / share_sum

    # The parent zone, for callers and joins that still use the six-zone view.
    result["zone"] = result["sub_zone"].str.replace(
        r" \((centre|wing)\)$", "", regex=True
    )
    return result[["player_id", "season", "zone", "sub_zone",
                   "attempts", "total", "zone_share"]]


def build_attainability_matrix(engine, seasons: list[str]) -> pd.DataFrame:
    """
    Join shrunk zone shares to lagged creation profiles and physicals.

    Creation features are lagged exactly as they are for the shot-quality
    model — the profile from season S-1 predicts the shot diet in season S.
    That is what makes this usable prospectively: it answers "given how this
    player creates, what shots will he be able to get", not "given the shots
    he took, what shots did he take".
    """
    from src.features.creation import attach_creation_features, load_creation_profiles

    targets = build_diet_snapshots(engine)
    targets = targets[targets["season"].isin(seasons)]
    targets["zone"] = targets["sub_zone"].str.replace(
        r" \((centre|wing)\)$", "", regex=True
    )

    players = pd.read_sql(
        "SELECT player_id, season, height, weight, wingspan, position FROM players",
        engine,
    )
    df = targets.merge(players, on=["player_id", "season"], how="left")

    profiles = load_creation_profiles(engine)
    df = attach_creation_features(df, profiles, positions=players)

    df = df.merge(build_season_cast(engine), on=["player_id", "season"], how="left")

    df = encode_zone_and_position(df)

    return df


def build_diet_snapshots(engine, min_attempts: int = 50) -> pd.DataFrame:
    """
    Training rows for a point-in-time attainability model.

    One row per (player, season, snapshot, sub_zone). At each snapshot the
    features describe what is KNOWN — his diet so far this season, his career
    to date, his last completed season — and the target is what he actually did
    over the REST of that season.

    Why snapshots rather than one row per player-season
    ---------------------------------------------------
    Season-to-date is the strongest evidence available and a model trained only
    on completed seasons cannot use it. Measured directly: the first 25% of a
    player's own season predicts his full-season diet at MAE 0.0222 and the
    first 50% at 0.0139, against 0.0294 for his entire prior season. Twenty
    games of the current year beat eighty-two of the last one.

    The model therefore has to learn how much to trust twelve games of evidence
    versus sixty, which means it has to see both — hence one row per snapshot.

    Why rest-of-season rather than full-season
    -------------------------------------------
    With season-to-date as a feature, a full-season target would contain the
    feature inside itself and the correlation would be partly mechanical. The
    remainder is disjoint from the evidence, and it is also the question the
    recommender is actually asking: not "what did he do this year" but "what
    will he be able to get from here".
    """
    shots = pd.read_sql("""
        SELECT s.player_id, s.season, s.zone, s.loc_x, s.loc_y, g.date AS game_date
        FROM shots s
        JOIN games g ON g.game_id = s.game_id
        WHERE s.zone IS NOT NULL AND s.zone != 'Backcourt'
    """, engine)
    shots["game_date"] = pd.to_datetime(shots["game_date"])
    shots = attach_sub_zone(shots)
    shots = shots.sort_values(["player_id", "season", "game_date"]).reset_index(drop=True)

    grouped = shots.groupby(["player_id", "season"], sort=False)
    shots["rank"] = grouped.cumcount()
    shots["n_season"] = grouped["rank"].transform("max") + 1
    shots = shots[shots["n_season"] >= min_attempts]
    if shots.empty:
        return pd.DataFrame()

    zone_frame = pd.DataFrame({"sub_zone": SUB_ZONES})
    league = _league_sub_zone_priors(shots)

    frames = []
    for cut in DIET_SNAPSHOTS:
        seen = shots[shots["rank"] < cut * shots["n_season"]]
        rest = shots[shots["rank"] >= cut * shots["n_season"]]

        # Target: his share over the remainder.
        after = rest.groupby(["player_id", "season", "sub_zone"]).size().rename("n_rest")
        after = after.reset_index()
        totals = after.groupby(["player_id", "season"])["n_rest"].sum().rename("rest_total")
        after = after.merge(totals, on=["player_id", "season"])

        keys = after[["player_id", "season", "rest_total"]].drop_duplicates()
        grid = keys.merge(zone_frame, how="cross").merge(
            after[["player_id", "season", "sub_zone", "n_rest"]],
            on=["player_id", "season", "sub_zone"], how="left",
        )
        grid["n_rest"] = grid["n_rest"].fillna(0.0)
        grid["zone_share"] = grid["n_rest"] / grid["rest_total"]

        # Feature: his share so far, and how much of it there is.
        if seen.empty:
            grid["diet_to_date"] = np.nan
            grid["diet_att_to_date"] = 0.0
        else:
            before = seen.groupby(["player_id", "season", "sub_zone"]).size().rename("n_seen")
            before = before.reset_index()
            seen_tot = before.groupby(["player_id", "season"])["n_seen"].sum().rename("seen_total")
            before = before.merge(seen_tot, on=["player_id", "season"])
            grid = grid.merge(
                before[["player_id", "season", "sub_zone", "n_seen", "seen_total"]],
                on=["player_id", "season", "sub_zone"], how="left",
            )
            grid["n_seen"] = grid["n_seen"].fillna(0.0)
            grid["seen_total"] = grid.groupby(["player_id", "season"])["seen_total"].transform("max")
            # Shrunk toward the league share, so eight attempts do not read as
            # a settled preference.
            alpha = grid["sub_zone"].map({z: p.alpha for z, p in league.items()})
            strength = grid["sub_zone"].map({z: p.strength for z, p in league.items()})
            grid["diet_to_date"] = (grid["n_seen"] + alpha) / (grid["seen_total"] + strength)
            grid["diet_to_date"] = grid["diet_to_date"].where(grid["seen_total"] > 0)
            grid["diet_att_to_date"] = grid["seen_total"].fillna(0.0)
            grid = grid.drop(columns=["n_seen", "seen_total"])

        grid["season_progress"] = cut
        frames.append(grid)

    out = pd.concat(frames, ignore_index=True)
    return _attach_long_history(out, shots, league)


def _league_sub_zone_priors(shots: pd.DataFrame) -> dict:
    """One Beta prior per sub-zone, over players' season shares."""
    per = shots.groupby(["player_id", "season", "sub_zone"]).size().rename("n").reset_index()
    tot = per.groupby(["player_id", "season"])["n"].sum().rename("tot")
    per = per.merge(tot, on=["player_id", "season"])
    return {
        z: fit_beta_prior(g["n"].to_numpy(), g["tot"].to_numpy())
        for z, g in per.groupby("sub_zone")
    }


def _attach_long_history(grid: pd.DataFrame, shots: pd.DataFrame,
                         league: dict) -> pd.DataFrame:
    """
    Career-to-date and last-completed-season diet, both strictly prior.

    Career-to-date is shrunk toward the league share for the sub-zone; a
    veteran therefore arrives with more evidence than a second-year player, and
    a rookie with none falls back to the prior rather than to a guess.
    """
    per = shots.groupby(["player_id", "season", "sub_zone"]).size().rename("n").reset_index()
    tot = per.groupby(["player_id", "season"])["n"].sum().rename("tot").reset_index()
    per = per.merge(tot, on=["player_id", "season"])

    seasons = sorted(shots["season"].unique())
    index = {s: i for i, s in enumerate(seasons)}
    per["si"] = per["season"].map(index)
    per = per.sort_values(["player_id", "sub_zone", "si"])

    g = per.groupby(["player_id", "sub_zone"], sort=False)
    per["car_n"] = g["n"].cumsum() - per["n"]
    per["car_tot"] = g["tot"].cumsum() - per["tot"]

    alpha = per["sub_zone"].map({z: p.alpha for z, p in league.items()})
    strength = per["sub_zone"].map({z: p.strength for z, p in league.items()})
    per["career_diet"] = ((per["car_n"] + alpha) / (per["car_tot"] + strength)).where(
        per["car_tot"] > 0
    )
    per["career_diet_att"] = per["car_tot"]
    per["prior_zone_share"] = (per["n"] / per["tot"]).groupby(
        [per["player_id"], per["sub_zone"]]
    ).shift(1)

    cols = ["player_id", "season", "sub_zone",
            "career_diet", "career_diet_att", "prior_zone_share"]
    return grid.merge(per[cols], on=["player_id", "season", "sub_zone"], how="left")


def attach_prior_diet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join each (player, season, sub_zone) row to that player's share in the same
    sub-zone the PREVIOUS season.

    See `features.PRIOR_DIET_COLS` for why this matters more than anything
    else in the feature list. Derived from the target frame itself rather
    than requeried, so the lagged value is the same shrunk share the model is
    fit against and cannot drift from it.

    A player with no prior season — every rookie, and anyone returning after a
    year out — gets NaN, which XGBoost handles natively and which is the honest
    encoding: there is no history to lean on, so the creation traits and
    position prior carry those rows alone.
    """
    seasons = sorted(df["season"].dropna().unique())
    following = {s: seasons[i + 1] for i, s in enumerate(seasons[:-1])}

    prior = df[["player_id", "season", "sub_zone", "zone_share"]].copy()
    prior["season"] = prior["season"].map(following)
    prior = prior.dropna(subset=["season"]).rename(
        columns={"zone_share": "prior_zone_share"}
    )
    return df.merge(prior, on=["player_id", "season", "sub_zone"], how="left")


def build_season_cast(engine) -> pd.DataFrame:
    """
    One supporting-cast row per (player, season): his teammates' full-season
    numbers, with his own contribution excluded.

    `point_in_time.build_supporting_cast` produces these per game, running
    season-to-date. The attainability target is a whole-season shot share, so
    the matching predictor is the whole season — taken as the last game's
    running value, which by construction is the season total minus this
    player.

    Contemporaneous with the target rather than lagged, unlike the creation
    profiles beside it. Lagging was the wrong call here: rosters turn over,
    and last season's teammates are frequently not the ones setting the
    screens this season, so the lag would substitute a different team's
    numbers rather than an older estimate of the same one. It is defensible
    because the leave-one-out removes the player himself, so his own shot diet
    — which IS the target — never enters the feature. The serving path stays
    honest by reading the same quantity season-to-date as of the request.
    """
    per_game = build_supporting_cast(engine)
    if per_game.empty:
        return pd.DataFrame(columns=["player_id", "season"] + CAST_FEATURE_COLS)

    seasons = pd.read_sql("SELECT DISTINCT game_id, season FROM shots", engine)
    per_game = per_game.merge(seasons, on="game_id", how="left")

    # `cast_att` is monotone within a season, so the maximum is the final
    # running value — ordering by game_id would rely on ids sorting by date,
    # which they do not across season types.
    per_game = per_game.sort_values("cast_att")
    return (
        per_game.groupby(["player_id", "season"], as_index=False)
        .tail(1)[["player_id", "season"] + CAST_FEATURE_COLS]
    )
