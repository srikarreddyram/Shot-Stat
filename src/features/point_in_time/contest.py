"""How contested a player's shots have been, accumulated point-in-time from
per-game closest-defender-distance bands."""
from __future__ import annotations

import pandas as pd

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
    # same construction `shooting_rates.build_prior_counts` uses for shooting.
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
