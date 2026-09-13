"""
Attainability — can this player actually GET this shot?

The problem
-----------
The shot-quality model estimates P(make | a shot was taken from here). The
recommender was reading that as P(make | you go take a shot here). Those are
different quantities, and the gap between them is selection: the shots that
exist at a location are the ones somebody chose to take, usually because they
were open enough to be worth taking. A player who only shoots from the elbow
when he is wide open posts a fine elbow percentage, and "shoot more from the
elbow" does not inherit the wide-open part.

The practical symptom is that ranking purely by expected points recommends the
restricted area to everyone, always. That is true, useless, and exactly what a
model that ignores attainability must conclude — the rim is the highest-value
spot on the floor for every player alive, and the entire difficulty is getting
there.

What this model estimates
-------------------------
Given a player and a location, what share of that player's shot diet would
plausibly come from there — i.e. how readily can they generate this look? It
is fit on observed shot-frequency distributions: for each (player, season,
zone), the fraction of their attempts taken in that zone, with the same
empirical-Bayes shrinkage the shooting rates get, so a player with 40 attempts
does not register as a 100%-corner-three specialist.

Why creation skill is the backbone
----------------------------------
Attainability is mostly a handle question, which is where the creation
features earn their place. Whether a player can generate an above-the-break
three off the dribble is not really about his shooting — it is about whether
he can create the separation. `self_creation_index`, `pullup_share`,
`avg_drib_per_touch` and `drives_per_min` are the features that carry it, and
they are the reason an elite creator and a standstill shooter with identical
shooting percentages receive different advice.

How the recommender uses it
---------------------------
Not as a hard filter. A rim-running center's attainability for a corner three
is low but not zero, and zeroing it out would hide a real, if rare, option.
The recommender ranks by expected points weighted by attainability and reports
both numbers, so the interface can distinguish "this would be a great shot for
you" from "this is a shot you can actually get".

Usage:
    python -m src.training.attainability --name attainability-v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.common import runs
from src.features.creation import CREATION_FEATURE_COLS
from src.common.position_bucket import position_bucket
from src.features.point_in_time import (
    ZONE_SUFFIX, ZONES, build_supporting_cast, fit_league_creation_priors,
)
from src.features.shrinkage import fit_beta_prior

MODEL_DIR = Path(config.PROJECT_ROOT) / "models"

# Features the attainability model is allowed to see. Deliberately excludes
# everything about whether the shot GOES IN — this model answers a question
# about shot generation, and letting it peek at shooting skill would blur the
# two quantities the recommender needs kept separate.
POSITION_BUCKETS = ["G", "F", "C"]

# ── Sub-zones: the six court zones, with the two wide ones split by angle ────
#
# "Above the Break 3" spans the entire arc from one wing, across the top of the
# key, to the other. A dead-centre pull-up and a 60-degree wing spot-up were
# one indistinguishable bucket, so attainability returned a byte-identical
# number for both and could not express the difference at all.
#
# They are genuinely different shots. Measured over 2016-17 onward, the share
# of above-the-break threes a player creates for himself falls monotonically
# with angle off centre: 27% within 10 degrees, 19% at 30-45, 9.5% beyond 60.
# The top of the key is where pull-up threes live; the wing is where you catch
# and shoot. Mid-Range is split on the same boundary for the same reason.
#
# Corners are left alone: they are already angle-specific by construction, and
# at 96% assisted there is no self-created population inside them to separate.
ANGLE_SPLIT_ZONES = ("Above the Break 3", "Mid-Range")


def with_bare_zone_entries(zone_means: pd.Series) -> pd.Series:
    """
    Add a bare-zone entry for each of ANGLE_SPLIT_ZONES, summing its two
    sub-zone entries — same fix pattern as lookup_diet_history and
    fit_league_creation_priors elsewhere in this file, applied to
    league_zone_shares.

    `zone_means` is keyed ONLY by sub-zone ("Above the Break 3 (centre)"/
    "(wing)") for the two angle-split zones, so a caller holding the bare
    zone name — anyone who hasn't resolved a specific court location, which
    is the common case for a zone-level attainability display — gets a
    silent None on exactly the two zones with the most volume. A player's
    bare-zone share is the SUM of his two sub-zone shares (they partition
    the same attempts), so the league mean of the sum equals the sum of the
    league means, by linearity.
    """
    zone_means = zone_means.copy()
    for zone in ANGLE_SPLIT_ZONES:
        parts = [f"{zone} (centre)", f"{zone} (wing)"]
        present = [p for p in parts if p in zone_means.index]
        if present:
            zone_means[zone] = zone_means[present].sum()
    return zone_means

# Degrees off dead centre dividing "centre" from "wing". Chosen at the elbow of
# the measured self-creation gradient, where it drops from ~25% to ~19%.
SUB_ZONE_ANGLE_BOUNDARY = 30.0

SUB_ZONES = [
    "Restricted Area",
    "In The Paint (Non-RA)",
    "Mid-Range (centre)",
    "Mid-Range (wing)",
    "Left Corner 3",
    "Right Corner 3",
    "Above the Break 3 (centre)",
    "Above the Break 3 (wing)",
]

SUB_ZONE_SUFFIX = {
    "Restricted Area": "restricted_area",
    "In The Paint (Non-RA)": "paint",
    "Mid-Range (centre)": "midrange_centre",
    "Mid-Range (wing)": "midrange_wing",
    "Left Corner 3": "left_corner_3",
    "Right Corner 3": "right_corner_3",
    "Above the Break 3 (centre)": "above_break_3_centre",
    "Above the Break 3 (wing)": "above_break_3_wing",
}


def attach_sub_zone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `sub_zone` column from `zone` plus coordinates.

    Vectorized and pure, and shared by the training matrix and the serving
    grid — the same single-definition rule `spec.derive_features` follows, so
    a shot cannot be filed under one sub-zone in training and another at
    prediction time.

    Rows without coordinates keep their unsplit zone name rather than being
    guessed into a half; for the two split zones that means they land in
    neither bucket and the one-hot encoding reads all-zero, which is an honest
    "unknown" the trees can isolate.
    """
    out = df.copy()
    if "sub_zone" in out.columns:
        return out

    zone = out["zone"]
    if "loc_x" not in out.columns or "loc_y" not in out.columns:
        out["sub_zone"] = zone
        return out

    # Angle from the right baseline measured at the rim, matching
    # spec.derive_features; 90 degrees is straight on.
    deviation = np.abs(
        np.degrees(np.arctan2(
            out["loc_y"].astype(float), out["loc_x"].astype(float)
        )) - 90.0
    )
    is_centre = deviation <= SUB_ZONE_ANGLE_BOUNDARY
    splittable = zone.isin(ANGLE_SPLIT_ZONES) & deviation.notna()

    out["sub_zone"] = np.where(
        splittable,
        zone + np.where(is_centre, " (centre)", " (wing)"),
        zone,
    )
    return out

ATTAINABILITY_FEATURE_COLS = [
    # Who the player is, physically
    "height", "weight", "wingspan",
    # How they get their shots — the substance of the model
    *CREATION_FEATURE_COLS,
    # Where the shot is. Zones are ONE-HOT, not the ordinal `zone_index` this
    # used to carry. Ordinal encoding forced an arbitrary ordering on an
    # unordered category (why is Mid-Range "between" the paint and a corner
    # three?), so the trees had to spend splits reconstructing zone identity —
    # and, more practically, it made explanations useless: TreeSHAP attributed
    # the entire zone effect to one opaque `zone_index` term that no user can
    # read. One-hot keeps each zone's effect its own named quantity.
    *[f"zone_is_{z}" for z in SUB_ZONE_SUFFIX.values()],
    "is_three",
    # Position bucket (G/F/C), one-hot. Shot diet is strongly positional in a
    # way the creation features only partly capture: they describe how a player
    # handles the ball, not how tall the people guarding him are or where his
    # team stations him.
    *[f"pos_is_{b}" for b in POSITION_BUCKETS],
]

# The supporting cast: what the four other players on the floor give this
# shooter. Held separate from the list above so `--no-cast` can measure it as
# a unit rather than leaving it as an untested assumption.
#
# Every one of these is LEAVE-ONE-OUT — the shooter's own contribution is
# subtracted out, so they describe his teammates rather than his team. That is
# not a refinement, it is the whole feature: Oklahoma City ranked 27th of 30
# in assisted-field-goal rate in 2024-25, which reads as a team that does not
# move the ball, but the figure is depressed almost entirely by Shai
# Gilgeous-Alexander's own self-created volume. Excluding him, his teammates
# assist on .712 of their makes, near the top of the league. The raw team rate
# would have told the model the opposite of the truth for exactly the
# high-usage players whose environment is most worth knowing.
CAST_FEATURE_COLS = ["cast_ast_rate", "cast_3p_rate", "cast_efg", "cast_att"]

# The player's own shot diet last season — by a wide margin the strongest
# predictor of his shot diet this season, and absent from every version of this
# model before it was measured.
#
# Shot diet is one of the most stable quantities in basketball. Lag-1
# autocorrelation of a player's sub-zone share is 0.941 pooled, and between
# 0.765 (mid-range centre) and 0.920 (restricted area) in every individual
# sub-zone. The model was trying to reconstruct that from dribbles per touch
# and pull-up share while the answer sat in the player's own prior season.
#
# The cost of omitting it was not a small loss of accuracy but a systematic
# collapse toward the league mean: predicted standard deviation ran at 49% of
# the true spread in the paint and 59% above the break, so no player could ever
# be placed at a realistic extreme. Jokic genuinely takes 37% of his shots in
# the paint; the model could not emit a number above roughly 0.20 for anyone.
# Adding this lifts the dispersion ratio to 71-78% everywhere.
#
# Not leakage: the prior season is complete and known before the season being
# predicted begins, exactly as `zone_rate` uses strictly prior shooting in the
# shot-quality model, and exactly as the creation profiles beside it are lagged.
PRIOR_DIET_COLS = [
    # Season-to-date: what he has actually done in THIS season so far. By a
    # wide margin the most informative of the three, and the one a model
    # trained only on completed prior seasons cannot see at all.
    "diet_to_date", "diet_att_to_date", "season_progress",
    # Career-to-date, shrunk toward the league share for the sub-zone: all
    # completed prior seasons, so a veteran brings more evidence than one year.
    "career_diet", "career_diet_att",
    # Last completed season on its own, because recency is not the same thing
    # as volume and the model should be free to weight it separately.
    "prior_zone_share",
]

# How many points through a season to snapshot each player when building
# training rows. The model has to learn how much to trust twelve games of
# evidence versus sixty, so it must see both. Fractions of a player's own shot
# count rather than calendar dates, since players miss time.
DIET_SNAPSHOTS = (0.0, 0.15, 0.3, 0.5, 0.7)


def encode_zone_and_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot the zone and position-bucket columns.

    Shared by the training matrix and the recommender's serving path so the
    two cannot disagree about an encoding — the same train/serve parity rule
    `src/features/spec.py` follows, and for the same reason: a mismatch here
    would be silent and would corrupt every served attainability number.

    Every zone and bucket column is always emitted, present in the frame or
    not, so a single-row serving call produces the identical column set a full
    training matrix does.
    """
    out = attach_sub_zone(df)

    for sub, suffix in SUB_ZONE_SUFFIX.items():
        out[f"zone_is_{suffix}"] = (out["sub_zone"] == sub).astype(int)

    out["is_three"] = out["zone"].isin(
        ["Left Corner 3", "Right Corner 3", "Above the Break 3"]
    ).astype(int)

    buckets = out["position"].map(position_bucket) if "position" in out.columns else None
    for bucket in POSITION_BUCKETS:
        out[f"pos_is_{bucket}"] = (
            (buckets == bucket).astype(int) if buckets is not None else 0
        )

    return out


def build_zone_frequency_targets(engine, min_attempts: int = 50) -> pd.DataFrame:
    """
    Observed shot-diet shares per (player, season, zone), shrunk.

    Raw shares are noisy for low-volume players in exactly the way raw
    shooting percentages are, and for the same reason: a player with 60 total
    attempts who happened to take four corner threes reads as a 6.7% corner-
    three shooter with no evidence behind it. Each zone's share is shrunk
    toward the league distribution using a Beta prior fit per zone.

    Shares are per SUB-zone (see `attach_sub_zone`), so the two wide zones are
    split by angle and the model can distinguish a dead-centre three from a
    wing one. `zone` is carried alongside for callers that still reason in the
    six-zone taxonomy.

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


def lookup_diet_history(conn, player_id: str, season: str,
                        sub_zone_priors: dict, as_of_date=None,
                        max_progress: float = 0.7) -> dict:
    """
    Serving-path equivalent of `build_diet_snapshots` for one player: what is
    known about his shot diet right now.

    Returns `{sub_zone: {diet_to_date, career_diet, prior_zone_share, ...}}`
    plus a shared `season_progress`, matching the feature names the training
    frame carries.

    `as_of_date` makes this genuinely point-in-time: the diet-to-date counts
    only games before that date, so a request in December sees December's
    evidence and no more. Without one it uses everything recorded for `season`,
    which is the right reading of "as of now" for a season still in progress.

    `season_progress` is capped at the largest snapshot the model was trained
    on. Serving at 1.0 would ask it to extrapolate past every training row, and
    the honest answer at a completed season is the same one it learned at the
    furthest point it actually saw.
    """
    from sqlalchemy import text

    params = {"pid": str(player_id), "season": season}
    date_clause = ""
    if as_of_date is not None:
        date_clause = "AND g.date < :as_of"
        params["as_of"] = str(as_of_date)

    current = conn.execute(text(f"""
        SELECT s.zone, s.loc_x, s.loc_y
        FROM shots s JOIN games g ON g.game_id = s.game_id
        WHERE s.player_id = :pid AND s.season = :season
          AND s.zone IS NOT NULL AND s.zone != 'Backcourt' {date_clause}
    """), params).fetchall()

    season_total = conn.execute(text("""
        SELECT COUNT(*) FROM shots
        WHERE player_id = :pid AND season = :season
          AND zone IS NOT NULL AND zone != 'Backcourt'
    """), {"pid": str(player_id), "season": season}).scalar() or 0

    history = conn.execute(text("""
        SELECT s.season, s.zone, s.loc_x, s.loc_y
        FROM shots s
        WHERE s.player_id = :pid AND s.season < :season
          AND s.zone IS NOT NULL AND s.zone != 'Backcourt'
    """), {"pid": str(player_id), "season": season}).fetchall()

    def _shares(rows, cols):
        if not rows:
            return {}, 0.0
        frame = attach_sub_zone(pd.DataFrame(rows, columns=cols))
        counts = frame["sub_zone"].value_counts()
        return counts.to_dict(), float(counts.sum())

    seen, seen_total = _shares(current, ["zone", "loc_x", "loc_y"])
    hist_frame = (
        attach_sub_zone(pd.DataFrame(history, columns=["season", "zone", "loc_x", "loc_y"]))
        if history else pd.DataFrame(columns=["season", "sub_zone"])
    )
    career, career_total = ({}, 0.0)
    if not hist_frame.empty:
        c = hist_frame["sub_zone"].value_counts()
        career, career_total = c.to_dict(), float(c.sum())

    prior_shares = {}
    if not hist_frame.empty:
        last = hist_frame["season"].max()
        lf = hist_frame[hist_frame["season"] == last]["sub_zone"].value_counts()
        if lf.sum() > 0:
            prior_shares = (lf / lf.sum()).to_dict()

    progress = min(
        (seen_total / season_total) if season_total else 0.0, max_progress
    )

    out = {"season_progress": progress, "zones": {}}
    for zone in SUB_ZONES:
        prior = sub_zone_priors.get(zone)
        alpha = prior.alpha if prior is not None else 0.0
        strength = prior.strength if prior is not None else 0.0
        out["zones"][zone] = {
            "diet_to_date": (
                (seen.get(zone, 0) + alpha) / (seen_total + strength)
                if seen_total > 0 else None
            ),
            "diet_att_to_date": seen_total,
            "career_diet": (
                (career.get(zone, 0) + alpha) / (career_total + strength)
                if career_total > 0 else None
            ),
            "career_diet_att": career_total,
            "prior_zone_share": prior_shares.get(zone),
        }

    # Bare-zone aggregates for the two angle-split zones, keyed by the
    # UNSPLIT name (e.g. "Above the Break 3", not "... (centre)"/"(wing)").
    #
    # `attach_sub_zone` — the same function this file uses to build the keys
    # above — resolves `sub_zone` to the bare zone name whenever loc_x/loc_y
    # are not supplied (see its own docstring: "rows without coordinates
    # keep their unsplit zone name"). A caller of explain_attainability
    # without exact coordinates therefore looks up `out["zones"]["Above the
    # Break 3"]`, a key that never existed here — every diet-history figure
    # silently came back None regardless of how much real data the player
    # had, and the model's missing-value handling then applied whatever
    # default it learned for that gap, which is not "no effect": querying
    # Stephen Curry's Above-the-Break-3 attainability without coordinates
    # returned 2%, driven almost entirely by this None history, while the
    # visible explanation never mentioned history at all (the summary logic
    # only narrates it when a share IS known) — so the reason shown to the
    # user did not match the reason for the number.
    #
    # These aggregates are plain ratios, not the shrunk sub-zone estimates
    # above: the fitted Beta priors are per SUB-ZONE, and there is no prior
    # fit for the unsplit zone to shrink toward. A real, high-volume rate is
    # far more informative than another None, even unshrunk.
    for zone in ANGLE_SPLIT_ZONES:
        centre, wing = f"{zone} (centre)", f"{zone} (wing)"
        zone_seen = seen.get(centre, 0) + seen.get(wing, 0)
        zone_career = career.get(centre, 0) + career.get(wing, 0)
        # None (not 0.0) when there is no prior season at all — the same
        # "no prior season" vs "a real, measured zero" distinction the
        # per-sub-zone entries above make via `prior_shares.get(zone)`
        # returning None on an empty dict.
        zone_prior_share = (
            None if not prior_shares
            else prior_shares.get(centre, 0.0) + prior_shares.get(wing, 0.0)
        )
        out["zones"][zone] = {
            "diet_to_date": (zone_seen / seen_total) if seen_total > 0 else None,
            "diet_att_to_date": seen_total,
            "career_diet": (zone_career / career_total) if career_total > 0 else None,
            "career_diet_att": career_total,
            "prior_zone_share": zone_prior_share,
        }
    return out


def lookup_prior_diet(conn, player_id: str, season: str) -> dict:
    """
    Serving-path equivalent of `attach_prior_diet` for one player: his shot
    diet in the season BEFORE `season`, keyed by sub-zone.

    Raw shares here rather than the shrunk ones the training frame carries.
    The shrinkage only matters below roughly 50 attempts, and a served player
    with fewer than that in a whole season is not someone the recommender is
    being asked about; matching the estimator exactly would mean refitting the
    per-sub-zone Beta priors at request time for a difference in the third
    decimal.

    Returns {} when there is no prior season, which leaves the feature NaN —
    the same encoding a rookie gets in training.
    """
    from sqlalchemy import text

    row = conn.execute(text("""
        SELECT MAX(season) FROM shots WHERE player_id = :pid AND season < :season
    """), {"pid": str(player_id), "season": season}).fetchone()
    if row is None or row[0] is None:
        return {}

    rows = conn.execute(text("""
        SELECT zone, loc_x, loc_y FROM shots
        WHERE player_id = :pid AND season = :prior
          AND zone IS NOT NULL AND zone != 'Backcourt'
    """), {"pid": str(player_id), "prior": row[0]}).fetchall()
    if not rows:
        return {}

    frame = attach_sub_zone(
        pd.DataFrame(rows, columns=["zone", "loc_x", "loc_y"])
    )
    counts = frame["sub_zone"].value_counts()
    total = float(counts.sum())
    return {z: float(counts.get(z, 0)) / total for z in SUB_ZONES}


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

    See `PRIOR_DIET_COLS` for why this matters more than anything else in the
    feature list. Derived from the target frame itself rather than requeried,
    so the lagged value is the same shrunk share the model is fit against and
    cannot drift from it.

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


def train(seasons: list[str] | None = None, name: str = "attainability",
          use_cast: bool = True, use_prior_diet: bool = True) -> dict:
    """
    Fit the attainability model.

    Regression on the shrunk share with a squared-error objective rather than
    a classifier: the target is a proportion in [0, 1], not an event. Trained
    per (player, season, zone) row — roughly three thousand rows a season, so
    the model is deliberately small and heavily regularized.
    """
    from src.db.database import get_engine

    seasons = seasons or [s for s in config.ALL_SEASONS if s >= "2014-15"]
    engine = get_engine()

    run = runs.start_run(f"{name}", seasons=seasons)

    print(f"\n{'='*62}")
    print("  ATTAINABILITY MODEL")
    print("  Estimating: what share of a player's shots come from each zone")
    print(f"{'='*62}")

    df = build_attainability_matrix(engine, seasons)
    print(f"  {len(df):,} (player, season, zone) rows")

    test_season = seasons[-1]
    val_season = seasons[-2]
    fit_df = df[~df["season"].isin([val_season, test_season])]
    val_df = df[df["season"] == val_season]
    test_df = df[df["season"] == test_season]

    candidate_cols = list(ATTAINABILITY_FEATURE_COLS)
    if use_cast:
        candidate_cols += CAST_FEATURE_COLS
    if use_prior_diet:
        candidate_cols += PRIOR_DIET_COLS
    feature_cols = [c for c in candidate_cols if c in df.columns]
    print(f"  fit {len(fit_df):,} / val {len(val_df):,} / test {len(test_df):,}")
    ablated = [lbl for lbl, on in
               (("supporting cast", use_cast), ("prior diet", use_prior_diet))
               if not on]
    print(f"  features: {len(feature_cols)}"
          f"{'  (ablated: ' + ', '.join(ablated) + ')' if ablated else ''}")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=20,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=40,
    )
    model.fit(
        fit_df[feature_cols], fit_df["zone_share"],
        eval_set=[(val_df[feature_cols], val_df["zone_share"])],
        verbose=False,
    )

    preds = np.clip(model.predict(test_df[feature_cols]), 0.0, 1.0)
    actual = test_df["zone_share"].values

    mae = float(np.mean(np.abs(preds - actual)))
    rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))

    # The baseline that matters: predict each zone's league-average share and
    # ignore the player entirely. If the model cannot beat that, it has learned
    # nothing about individual players' shot diets and the whole exercise is
    # just a lookup table of zone frequencies.
    zone_means = fit_df.groupby("sub_zone")["zone_share"].mean()
    zone_means = with_bare_zone_entries(zone_means)
    base_preds = test_df["sub_zone"].map(zone_means).values
    base_mae = float(np.mean(np.abs(base_preds - actual)))

    print(f"\n  MAE           {mae:.4f}")
    print(f"  RMSE          {rmse:.4f}")
    print(f"  league-avg MAE {base_mae:.4f}")
    print(f"  improvement   {(base_mae - mae) / base_mae:+.1%}")

    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda kv: -kv[1],
    )
    print("\n  Top features (what determines which shots you can get):")
    for feature, importance in importances[:12]:
        print(f"    {feature:<32} {importance:.4f}")

    # League distribution of every model input, for explanations. A SHAP
    # contribution says a feature pushed the estimate down; it cannot say
    # whether the underlying value was unusual. Percentiles turn "pullup_share
    # = 0.05 lowered this" into "he pulls up on 5% of touches, 12th percentile
    # league-wide" — which is the half a reader can actually act on. Computed
    # on the fit window only, so the reference distribution never includes the
    # held-out season.
    # Quantiles are stored as a grid rather than a few landmarks so the
    # explainer can interpolate an approximate percentile for any value
    # without shipping the full training distribution alongside the model.
    quantile_grid = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    reference = {}
    for col in feature_cols:
        values = fit_df[col].dropna()
        if values.empty:
            continue
        reference[col] = {
            "mean": float(values.mean()),
            "quantiles": {str(q): float(values.quantile(q)) for q in quantile_grid},
        }

    model.save_model(str(MODEL_DIR / f"xgb_{name}.json"))
    metadata = {
        "name": name,
        "feature_cols": feature_cols,
        "zones": ZONES,
        "sub_zones": SUB_ZONES,
        "position_buckets": POSITION_BUCKETS,
        "league_zone_shares": {k: float(v) for k, v in zone_means.items()},
        "feature_reference": reference,
        # Per-zone priors over "did the player create this himself". Not a
        # model input — attainability's target is frequency, and this is the
        # separate creation question the explanation reports alongside it.
        # Travels with the model so serving shrinks toward the same priors,
        # fit on the fit window only.
        # League sub-zone priors, so the serving path shrinks a partial season
        # toward exactly what training shrank toward.
        "sub_zone_priors": {
            z: {"mean": p.mean, "strength": p.strength}
            for z, p in _league_sub_zone_priors(
                attach_sub_zone(pd.read_sql(
                    "SELECT player_id, season, zone, loc_x, loc_y FROM shots "
                    "WHERE zone IS NOT NULL AND zone != 'Backcourt'", engine))
            ).items()
        },
        "max_season_progress": max(DIET_SNAPSHOTS),
        "creation_priors": {
            zone: {"mean": p.mean, "strength": p.strength,
                   "n_players": p.n_players, "n_attempts": p.n_attempts}
            for zone, p in fit_league_creation_priors(
                engine, through_season=fit_df["season"].max()
            ).items()
        },
        "test_season": test_season,
        "metrics": {"mae": mae, "rmse": rmse, "baseline_mae": base_mae},
    }
    (MODEL_DIR / f"metadata_{name}.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )

    run.log_metrics(mae=mae, rmse=rmse, baseline_mae=base_mae,
                    improvement=(base_mae - mae) / base_mae)
    run.log_data(n_rows=len(df), n_features=len(feature_cols))
    run.finish()

    print(f"\n  ✓ saved models/xgb_{name}.json")
    print(f"{'='*62}\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the attainability model.")
    parser.add_argument("--name", default="attainability")
    parser.add_argument("--no-cast", action="store_true",
                        help="Ablate the leave-one-out supporting-cast features")
    parser.add_argument("--no-prior-diet", action="store_true",
                        help="Ablate the player's own prior-season shot diet")
    args = parser.parse_args()
    train(name=args.name, use_cast=not args.no_cast,
          use_prior_diet=not args.no_prior_diet)
