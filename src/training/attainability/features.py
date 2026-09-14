"""
Feature encoding shared by training and serving: sub-zones (the two wide
court zones split by shot angle), the one-hot zone/position encoding, and
the fixed feature-column lists both the training matrix builder and
ShotRecommender's serving path read.

Splitting "Above the Break 3" and "Mid-Range" by angle
-------------------------------------------------------
"Above the Break 3" spans the entire arc from one wing, across the top of the
key, to the other. A dead-centre pull-up and a 60-degree wing spot-up were
one indistinguishable bucket, so attainability returned a byte-identical
number for both and could not express the difference at all.

They are genuinely different shots. Measured over 2016-17 onward, the share
of above-the-break threes a player creates for himself falls monotonically
with angle off centre: 27% within 10 degrees, 19% at 30-45, 9.5% beyond 60.
The top of the key is where pull-up threes live; the wing is where you catch
and shoot. Mid-Range is split on the same boundary for the same reason.

Corners are left alone: they are already angle-specific by construction, and
at 96% assisted there is no self-created population inside them to separate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.creation import CREATION_FEATURE_COLS
from src.common.position_bucket import position_bucket

# Features the attainability model is allowed to see. Deliberately excludes
# everything about whether the shot GOES IN — this model answers a question
# about shot generation, and letting it peek at shooting skill would blur the
# two quantities the recommender needs kept separate.
POSITION_BUCKETS = ["G", "F", "C"]

ANGLE_SPLIT_ZONES = ("Above the Break 3", "Mid-Range")


def with_bare_zone_entries(zone_means: pd.Series) -> pd.Series:
    """
    Add a bare-zone entry for each of ANGLE_SPLIT_ZONES, summing its two
    sub-zone entries — same fix pattern as lookup_diet_history and
    fit_league_creation_priors elsewhere in this package, applied to
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
