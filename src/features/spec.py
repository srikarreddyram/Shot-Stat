"""
The feature specification — one definition, both paths.

`derive_features` is a pure, vectorized function from raw columns to model
features. The training matrix builder calls it on two million rows; the
recommender calls it on a 180-point court grid. Neither one computes a
derived feature on its own, which is what makes training/serving skew a test
failure (tests/test_train_serve_parity.py) instead of a silent accuracy leak.

"Raw" here means: spatial coordinates, game state, physical attributes,
point-in-time shooting counts already resolved to rates by
`point_in_time.apply_hierarchy`, defender aggregates, and lagged creation
features. Assembling those differs between paths. Everything downstream of
them does not.

Deferred: team-level creation support
-------------------------------------
The largest remaining passing effect is not on the passer's own shot but on
his teammates' — a shooter standing next to an elite creator gets cleaner
looks than the same shooter on a team without one. Capturing that needs the
set of five offensive players on the floor at the moment of the shot, which
requires reconstructing lineups from play-by-play substitution events. The
`FEATURE_GROUPS["team_creation"]` slot below is where those columns attach
once that ingestor exists; nothing else in the pipeline needs to change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.creation import CREATION_FEATURE_COLS
# ZONE_TO_DEF_CATEGORY lives in point_in_time.py (not defined here) so that
# module can use it too without an import cycle; re-exported here since
# build.py, recommender.py, and api.py all read it as `spec.ZONE_TO_DEF_CATEGORY`.
from src.features.point_in_time import ZONE_SUFFIX, ZONES, ZONE_TO_DEF_CATEGORY

INTERIOR_ZONES = ["Restricted Area", "In The Paint (Non-RA)"]

# Play-by-play shot CREATION, collapsed from the league's 52 raw labels into
# nine buckets describing how the shot came about — not what it looked like.
#
# This is the per-SHOT version of what `player_shot_profile` could only give as
# a season average. "Pullup Jump shot" and "Step Back Jump shot" are the shooter
# making the shot for himself; "Cutting Layup Shot" and "Alley Oop" are somebody
# else creating it; "Putback" is a second-chance possession. The season-level
# creation profile could say a player takes 60% pull-ups — it could never say
# THIS shot was one.
#
# Matching is by substring against the lowercased label, most specific first,
# because the raw labels compose and a first-match-wins scan over an ordered
# list is easier to reason about than an exhaustive enumeration upstream can
# extend.
#
# Every raw label decomposes as [creation modifiers] + [finish descriptors]:
# "Driving Floating Bank Jump Shot" is created by a drive and finished with a
# banked floater. These rules must match ONLY the creation half — `finish_*`
# below carries the rest.
#
# The previous version could not respect that, because one bucket had to carry
# both, so it matched on finish words and mislabelled roughly one shot in seven
# over 2016-17 onward (2,252,349 shots):
#
#   "floating"            -> pullup     170,479 shots. A floater is not a
#                                       pull-up, and "Driving Floating Jump
#                                       Shot" is a DRIVE, wrong on both axes.
#   "turnaround"/"fadeaway" -> stepback 107,234 shots. A turnaround fadeaway is
#                                       a post move; a step-back is a perimeter
#                                       move off the dribble. Different shots.
#   "running" (+ jumper)  -> driving     36,977 shots. "Running Jump Shot" is a
#                                       player relocating, not a drive — which
#                                       is how a CORNER THREE was rendering as
#                                       "driving" in the UI.
#
# "finger roll", "reverse" and "bank" are likewise finish descriptors and no
# longer appear here at all.
#
# Bucket names round-trip through the classifier (`classify_mechanic("post_up")`
# returns "post_up") because the recommender assigns a name to a hypothetical
# grid row and re-classifies it; the name must survive that trip.
SHOT_MECHANIC_RULES = [
    # Second-chance possessions first: a putback is a putback however finished.
    ("putback", ["putback", "tip "]),
    # Created by the passer, not the shooter.
    ("alley_oop", ["alley oop", "alley_oop"]),
    ("cutting", ["cutting"]),
    # Perimeter move off the dribble. Strictly step-backs now.
    ("stepback", ["step back", "stepback"]),
    # Turning on a defender — a post or face-up move, distinct from a step-back.
    ("post_up", ["turnaround", "fadeaway", "post_up"]),
    # Off the dribble, pulling up. Checked before `driving` and `transition` so
    # "Running Pull-Up Jump Shot" reads as the pull-up it is.
    ("pullup", ["pullup", "pull-up"]),
    ("driving", ["driving"]),
    # On the move without a drive — transition and relocation.
    ("transition", ["running", "transition"]),
]

# The fallback: no creation modifier at all. For the single largest label in the
# data, a bare "Jump Shot" (756,877 shots), that is exactly a catch-and-shoot
# spot-up. It is a looser fit for a bare "Hook Shot" or "Dunk Shot", but the
# finish axis disambiguates those — (spot_up, hook) reads as a post hook.
SHOT_MECHANIC_FALLBACK = "spot_up"

SHOT_MECHANICS = [name for name, _ in SHOT_MECHANIC_RULES] + [SHOT_MECHANIC_FALLBACK]

# ── Finish type ──────────────────────────────────────────────────────────────
# What the shot physically WAS, as distinct from how it was created.
#
# These are two orthogonal dimensions and the single-bucket scheme above forces
# them into one. "Driving Dunk Shot" is creation=driving AND finish=dunk;
# first-match-wins gives creation the win, so `dunk` above only ever catches
# the bare "Dunk Shot" label. The damage is not theoretical: measured over
# 2024-25 restricted-area shots, `driving` absorbed 54.1% while `dunk` was left
# with 1.7% — under `mechanics.MIN_MECHANIC_SHARE`, so it was filtered out
# entirely and Giannis Antetokounmpo was offered no dunk at the rim at all.
#
# Ordered most-specific-first for the same reason as the creation rules, and
# checked BEFORE the generic jumper so "Driving Floating Jump Shot" reads as a
# floater rather than a jumper.
FINISH_RULES = [
    ("dunk", ["dunk"]),
    ("layup", ["layup", "finger roll", "tip "]),
    ("hook", ["hook"]),
    # "floater" is not an NBA label — the raw feed always says "Floating". It is
    # accepted so that the bucket NAME round-trips through this classifier, the
    # property the serving path relies on (see src/features/mechanics.py).
    ("floater", ["floating", "floater"]),
    ("jumper", ["jump shot", "jump"]),
]

FINISH_TYPES = [name for name, _ in FINISH_RULES] + ["other"]


def classify_finish(subtype) -> str:
    """
    Map a raw play-by-play subType to one of FINISH_TYPES.

    Independent of `classify_mechanic`: the same label yields a creation bucket
    there and a finish bucket here, so "Driving Dunk Shot" is (driving, dunk)
    rather than being forced to choose.
    """
    if not isinstance(subtype, str) or not subtype.strip():
        return "other"
    text = subtype.lower()
    for name, needles in FINISH_RULES:
        if any(needle in text for needle in needles):
            return name
    return "other"

# ── Possession origin ────────────────────────────────────────────────────────
# What happened immediately before the shot, collapsed from the 14 raw
# play-by-play event types into five buckets describing how the possession
# started.
#
# This is a transition-versus-set-defence signal, and it was sitting unused: the
# ingest has been writing `prev_event_type` to the database since the backfill,
# and nothing ever read it back. Measured within above-the-break threes, which
# controls for location: 0.362 after a rebound against 0.335 after a
# substitution and 0.333 after a timeout. Marginally the spread is wider still
# (0.492 versus 0.407).
#
# `seconds_since_prev_event` does not capture this — it carries roughly 0.013 of
# total feature importance against `is_putback`'s 0.053. Elapsed time says how
# fast the shot came; it does not say whether the defence had time to set.
#
# Same first-match-wins ordered scan as the mechanics rules, and the bucket
# names round-trip through the classifier for the same reason: the serving path
# assigns a name and re-classifies it, rather than reimplementing the encoding.
POSSESSION_ORIGIN_RULES = [
    ("after_rebound", ["rebound", "after_rebound"]),
    ("after_made_shot", ["made shot", "after_made_shot"]),
    ("after_turnover", ["turnover", "steal", "after_turnover"]),
    # Dead ball that stops play but not the clock-and-scheme reset of a timeout.
    ("after_deadball", ["free throw", "foul", "violation", "jump ball",
                        "after_deadball"]),
    # A full reset: the defence is set and the offence is running a called play.
    ("after_stoppage", ["timeout", "period", "substitution", "replay",
                        "ejection", "after_stoppage"]),
]

POSSESSION_ORIGINS = [name for name, _ in POSSESSION_ORIGIN_RULES] + ["other"]

# The modal category league-wide (38.3% of shots), used as the serving default.
DEFAULT_POSSESSION_ORIGIN = "after_rebound"


def classify_possession_origin(prev_event) -> str:
    """Map a raw play-by-play `prev_event_type` to one of POSSESSION_ORIGINS."""
    if not isinstance(prev_event, str) or not prev_event.strip():
        return "other"
    text = prev_event.lower()
    for name, needles in POSSESSION_ORIGIN_RULES:
        if any(needle in text for needle in needles):
            return name
    return "other"


def classify_mechanic(subtype) -> str:
    """Map a raw play-by-play subType to one of SHOT_MECHANICS (creation only)."""
    if not isinstance(subtype, str) or not subtype.strip():
        return SHOT_MECHANIC_FALLBACK
    text = subtype.lower()
    for name, needles in SHOT_MECHANIC_RULES:
        if any(needle in text for needle in needles):
            return name
    return SHOT_MECHANIC_FALLBACK

# ── Spatial basis ────────────────────────────────────────────────────────────
# Radial basis functions over the half court, in FEET from the basket.
#
# Trees split on one axis at a time, so given only `loc_x` and `loc_y` the
# fitted surface is a union of axis-aligned rectangles. That is why the court
# render needed interpolation to look continuous — the underlying estimate
# genuinely is blocky, and no amount of smoothing in the canvas changes what
# the model believes.
#
# Each basis function is a smooth bump centred somewhere on the floor, so a
# single split on one of them carves out a circular region instead of a
# rectangle, and sums of them approximate a smooth surface directly. The
# centres are laid out in polar coordinates because shot difficulty varies far
# more with distance from the rim than with left-right position, and because
# the court is symmetric about the centre line.
_BASIS_RADII = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 23.5, 27.0]
_BASIS_ANGLES = [20.0, 60.0, 90.0, 120.0, 160.0]
SPATIAL_SIGMA_FT = 5.5


def _basis_centers() -> list[tuple[float, float]]:
    """Centres in feet, (x, y), basket at the origin."""
    centers = [(0.0, 0.0)]
    for radius in _BASIS_RADII[1:]:
        for angle in _BASIS_ANGLES:
            rad = np.radians(angle)
            centers.append((radius * np.cos(rad), radius * np.sin(rad)))
    return centers


SPATIAL_CENTERS = _basis_centers()
SPATIAL_FEATURES = [f"rbf_{i}" for i in range(len(SPATIAL_CENTERS))]


# Feature columns grouped by what they describe. The grouping is not cosmetic:
# `backtest.py` ablates whole groups to report what each one is actually worth,
# and `train.py` uses it to build a player-identity-free model for the
# hierarchical residual stage.
FEATURE_GROUPS: dict[str, list[str]] = {
    "spatial": [
        "loc_x", "loc_y", "shot_distance", "shot_angle",
        "distance_from_center", "abs_loc_x", "is_three",
    ],
    # Smooth radial basis over the floor. Held out by default: measured, it
    # makes the model slightly WORSE (log-loss 0.6327 -> 0.6335, AUC 0.6847 ->
    # 0.6836, ECE 0.0078 -> 0.0096 on the 2025-26 hold-out). Thirty-six extra
    # columns is a lot of surface for the trees to overfit, and distance plus
    # the zone indicators already carry the spatial signal that exists.
    #
    # It was added to smooth the fitted surface, since axis-aligned splits make
    # a blocky one — but the court render interpolates between grid points
    # anyway, so the visual benefit was already being delivered downstream at
    # no cost to accuracy. Enable with `--spatial-basis` to re-measure.
    "spatial_basis": SPATIAL_FEATURES,
    "context": [
        "quarter", "time_remaining", "score_diff", "home_away",
        "playoff_flag", "clutch_flag", "seconds_remaining_in_game",
        "rest_days", "is_back_to_back", "opp_def_rating",
    ],
    "shooter_physical": [
        "height", "weight", "wingspan",
    ],
    "shooter_skill": [
        "zone_rate", "zone_att", "overall_rate", "three_rate",
        "recent_10_fg", "recent_20_fg", "zone_rate_vs_league",
    ] + [f"zone_rate_{s}" for s in ZONE_SUFFIX.values()],
    "creation": CREATION_FEATURE_COLS + ["creation_is_prior"],
    "defender": [
        "def_fg_pct_overall", "def_pct_plusminus",
        "def_freq_zone", "def_fg_pct_zone", "def_pct_plusminus_zone",
        "def_matchup_share", "def_matchup_dispersion",
    ],
    # Defender size, held out of the model by default.
    #
    # These measure nothing. Actual FG% by attacker-minus-defender height is
    # flat across every populated band — 0.361 at -6ft to -3ft against 0.351 at
    # +3 to +6 above the break, about one point over the whole range — because
    # `height_diff` is computed against the possession-weighted AVERAGE
    # defender, which sits near the league mean almost always. The extreme
    # bands hold 4 to 25 rows out of two million.
    #
    # The model fitted those handful of rows anyway, and the recommender drove
    # straight into them: naming a tall defender against a guard moved the
    # prediction by 18 points, of which 17.8 came from these features and 0.9
    # from every genuine defensive-quality feature combined. A spurious effect
    # twenty times the size of the real one, triggered by the single most
    # obvious thing a user can do in the UI.
    #
    # Enable with `--defender-physicals` to reproduce the old behaviour.
    "defender_physical": [
        "def_height", "def_weight", "def_wingspan",
        "height_diff", "weight_diff", "wingspan_diff", "size_mismatch",
    ],
    "interaction": [
        "matchup_advantage", "expected_contest", "creation_edge",
        "openness_vs_defender",
    ],
    # Per-shot play-by-play context. `is_assisted` is deliberately ABSENT —
    # assists are credited only on made baskets, so it predicts the label
    # perfectly and is useless at prediction time. See pbp_ingestor.py and
    # tests/test_no_leaky_features.py.
    "shot_context": [
        "is_putback", "seconds_since_prev_event",
    ],
    # Possession origin indicators, grouped separately so `--no-origin` style
    # ablations can measure them on their own.
    "possession_origin": [f"origin_{o}" for o in POSSESSION_ORIGINS],
    # What the shot physically was — dunk, layup, hook, floater, jumper —
    # kept on its own axis from how it was created. See FINISH_RULES.
    "finish": [f"finish_{f}" for f in FINISH_TYPES],
    # Point-in-time opponent defence per zone. The model previously knew about
    # opponent defence only through one season-level `def_rating`, and residuals
    # aggregated by defending team showed it explaining essentially none of it
    # (z-score sd 2.82 at the rim against 1.0 for a correct model).
    "opponent_defence": [
        "opp_zone_def_rate", "opp_zone_def_att",
    ] + [f"opp_def_rate_{s}" for s in ZONE_SUFFIX.values()],
    # How contested this shooter's looks have actually been, accumulated over
    # strictly prior games (see point_in_time.build_contest_history).
    #
    # Contest level is the largest single thing the model cannot see. Per-SHOT
    # closest-defender distance is not published anywhere — `shotchartdetail`
    # accepts a CloseDefDistRange parameter and silently ignores it — and that
    # remains the hard noise floor on whether a given shot goes in.
    #
    # What these carry is the tendency one level up: LeagueDashPlayerPtShot
    # does honour a date filter, so the four defender-distance bands are
    # available per GAME and can therefore be accumulated point-in-time. The
    # creation group already has `avg_def_dist` / `open_share` / `tight_share`
    # from the same bands, but only as a season aggregate lagged a full year —
    # it cannot say a player has been run off the line for the past month.
    "contest": [
        "contest_car_very_tight", "contest_car_tight",
        "contest_car_open", "contest_car_wide_open",
        "contest_ssn_very_tight", "contest_ssn_tight",
        "contest_ssn_open", "contest_ssn_wide_open",
        "contest_car_sep", "contest_ssn_sep", "contest_att",
    ],
    "team_creation": [],  # reserved — see module docstring
}

# Columns that must exist before derive_features runs. Anything else it needs
# is optional and degrades to NaN, which XGBoost handles natively.
REQUIRED_RAW_COLS = ["loc_x", "loc_y", "zone"]



def all_feature_columns(df: pd.DataFrame | None = None) -> list[str]:
    """
    The model's feature list, in stable order.

    When `df` is given, only columns actually present are returned, plus any
    one-hot zone/position dummies found on the frame. Stable ordering matters:
    XGBoost binds predictions to column position, so a reordering between
    training and serving silently scrambles every feature.
    """
    cols: list[str] = []
    for group in FEATURE_GROUPS.values():
        cols.extend(group)

    if df is None:
        return cols

    present = [c for c in cols if c in df.columns]
    dummies = sorted(
        c for c in df.columns
        if (c.startswith("zone_is_") or c.startswith("pos_")
            or c.startswith("mech_") or c.startswith("origin_")
            or c.startswith("finish_"))
    )

    # De-duplicate, preserving order. A column can legitimately arrive from both
    # sources — `origin_*` are listed in FEATURE_GROUPS so ablations can drop
    # them as a unit, AND matched by the prefix scan — and XGBoost rejects a
    # frame with repeated column names outright.
    seen = set()
    ordered = []
    for col in present + dummies:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered


def _select_by_zone(df: pd.DataFrame, template: str) -> np.ndarray:
    """
    Pick, for each row, the value of the column matching that row's zone.

    e.g. template "zone_rate_{}" reads `zone_rate_midrange` for a mid-range
    shot. Implemented with np.select rather than the DataFrame.apply the old
    pipeline used — same result, roughly two orders of magnitude faster over
    two million rows, and it was squarely in the hot path.
    """
    conditions, choices = [], []
    for zone in ZONES:
        col = template.format(ZONE_SUFFIX[zone])
        if col not in df.columns:
            continue
        conditions.append((df["zone"] == zone).values)
        choices.append(df[col].values)

    if not conditions:
        return np.full(len(df), np.nan)
    return np.select(conditions, choices, default=np.nan)


def derive_features(
    df: pd.DataFrame,
    league_zone_rates: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute every derived feature. Pure and vectorized; no I/O, no global
    state, no dependence on row count.

    `league_zone_rates` maps zone → league average rate, used to express a
    player's skill relative to the zone's baseline. Passed in rather than
    computed here so it can be fit on training seasons only and then reused
    verbatim at serving time.
    """
    missing = [c for c in REQUIRED_RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"derive_features missing required raw columns: {missing}")

    out = df.copy()

    # ── Spatial ──────────────────────────────────────────────────────────
    if "shot_type" in out.columns:
        out["is_three"] = (out["shot_type"] == "3PT Field Goal").astype(int)
    elif "is_three" not in out.columns:
        out["is_three"] = out["zone"].isin(
            ["Left Corner 3", "Right Corner 3", "Above the Break 3"]
        ).astype(int)

    out["distance_from_center"] = np.sqrt(out["loc_x"] ** 2 + out["loc_y"] ** 2)
    out["abs_loc_x"] = out["loc_x"].abs()

    # Angle from the right baseline, measured at the rim. Computed here rather
    # than trusted from the shots table so the grid path and the historical
    # path use one definition — they previously disagreed, with the recommender
    # special-casing loc_x == 0 to 90° and the ingestor using its own formula.
    out["shot_angle"] = np.degrees(np.arctan2(out["loc_y"], out["loc_x"]))

    # ── Spatial basis ────────────────────────────────────────────────────
    # Coordinates arrive in tenths of a foot; the centres are in feet.
    x_ft = out["loc_x"].to_numpy(dtype=float) / 10.0
    y_ft = out["loc_y"].to_numpy(dtype=float) / 10.0
    two_sigma_sq = 2.0 * SPATIAL_SIGMA_FT ** 2
    for i, (cx, cy) in enumerate(SPATIAL_CENTERS):
        out[f"rbf_{i}"] = np.exp(-(((x_ft - cx) ** 2 + (y_ft - cy) ** 2) / two_sigma_sq))

    # ── Game context ─────────────────────────────────────────────────────
    if {"quarter", "time_remaining"}.issubset(out.columns):
        # Regulation periods are 12 minutes; overtime is 5. Expressing time as
        # seconds left in the GAME lets one feature carry what quarter plus
        # time-remaining otherwise force the trees to reconstruct through
        # repeated splits.
        periods_left = (4 - out["quarter"]).clip(lower=0)
        out["seconds_remaining_in_game"] = np.where(
            out["quarter"] <= 4,
            out["time_remaining"] + periods_left * 720.0,
            out["time_remaining"],
        )
    if {"quarter", "time_remaining", "score_diff"}.issubset(out.columns):
        out["clutch_flag"] = (
            (out["quarter"] >= 4)
            & (out["time_remaining"] <= 120)
            & (out["score_diff"].abs() <= 5)
        ).astype(int)

    # ── Shooter skill at this location ───────────────────────────────────
    out["zone_rate"] = _select_by_zone(out, "zone_rate_{}")
    out["zone_att"] = _select_by_zone(out, "zone_att_{}")

    # Opponent defence in the zone actually being shot from.
    out["opp_zone_def_rate"] = _select_by_zone(out, "opp_def_rate_{}")
    out["opp_zone_def_att"] = _select_by_zone(out, "opp_def_att_{}")

    if league_zone_rates:
        baseline = out["zone"].map(league_zone_rates).astype(float)
        out["zone_rate_vs_league"] = out["zone_rate"] - baseline
    else:
        out["zone_rate_vs_league"] = np.nan

    # ── Physical matchup ─────────────────────────────────────────────────
    for attr, diff_col in (
        ("height", "height_diff"),
        ("weight", "weight_diff"),
        ("wingspan", "wingspan_diff"),
    ):
        def_col = f"def_{attr}"
        if attr in out.columns and def_col in out.columns:
            out[diff_col] = out[attr] - out[def_col]
        else:
            out[diff_col] = np.nan

    out["size_mismatch"] = (out["height_diff"].abs() >= 4).fillna(False).astype(int)

    # ── Matchup quality ──────────────────────────────────────────────────
    if "def_fg_pct_zone" in out.columns or "def_fg_pct_overall" in out.columns:
        def_zone = out.get("def_fg_pct_zone", pd.Series(np.nan, index=out.index))
        def_overall = out.get("def_fg_pct_overall", pd.Series(np.nan, index=out.index))
        effective_def = def_zone.fillna(def_overall)
        out["matchup_advantage"] = out["zone_rate"] - effective_def
    else:
        out["matchup_advantage"] = np.nan

    # ── Creation interactions ────────────────────────────────────────────
    # These three are the mechanism by which handle and passing change a shot's
    # difficulty. Each is a documented construction rather than a fitted term,
    # so the model gets a usable starting signal and can still learn to weight
    # or ignore it.

    # 1. Expected contest: how much daylight this shot is likely to have.
    #    Starts from the space the shooter typically gets and subtracts the
    #    defender's ability to take it away. `def_pct_plusminus` is how much
    #    better or worse opponents shoot against this defender than league
    #    average, so a strong defender (negative) widens the deduction. The
    #    scale factor converts a percentage-point figure into feet; it is a
    #    unit bridge, and the model rescales it as needed.
    if "avg_def_dist" in out.columns:
        defender_pressure = out.get(
            "def_pct_plusminus_zone",
            out.get("def_pct_plusminus", pd.Series(0.0, index=out.index)),
        ).fillna(0.0)
        # The scale converts percentage points of defensive impact into feet.
        # It is a unit bridge, not a fitted coefficient — the model is free to
        # rescale it — so it is kept deliberately modest: pct_plusminus for a
        # single defender is noisy (sd ≈ 0.05, tails past ±0.6), and a scale
        # large enough to reflect the true open/contested efficiency gap would
        # let those tails dominate a feature whose own spread is under half a
        # foot.
        raw = out["avg_def_dist"] + defender_pressure * 10.0
        # Clipped to a physically real range. Negative daylight is not a
        # quantity, and letting an outlier defender push this to -1.9 feet
        # hands the trees a split point that corresponds to nothing on a
        # basketball court.
        out["expected_contest"] = raw.clip(lower=0.5, upper=12.0)
    else:
        out["expected_contest"] = np.nan

    # 2. Creation edge: can this shooter make his own shot against a defender
    #    who is good enough to deny a catch? Self-creation matters far more
    #    against a strong individual defender than a weak one, because against
    #    a weak defender the easy look is available anyway.
    if "self_creation_index" in out.columns:
        def_quality = out.get(
            "def_pct_plusminus", pd.Series(0.0, index=out.index)
        ).fillna(0.0)
        out["creation_edge"] = out["self_creation_index"] * (-def_quality)
    else:
        out["creation_edge"] = np.nan

    # 3. Openness vs defender: a spot-up shooter facing a defender who never
    #    lets anyone get open is in far more trouble than a creator facing the
    #    same defender, since the creator has a second way to get the shot off.
    if "open_share" in out.columns:
        out["openness_vs_defender"] = out["open_share"] - out.get(
            "def_freq_zone", pd.Series(0.0, index=out.index)
        ).fillna(0.0)
    else:
        out["openness_vs_defender"] = np.nan

    # ── Categorical encodings ────────────────────────────────────────────
    # Explicit indicator columns rather than pd.get_dummies(drop_first=True).
    # The old pipeline's dummies depended on which zones happened to appear in
    # the frame, so the serving grid and the training matrix could produce
    # different column sets — and with drop_first, a different reference level.
    # Enumerating from a fixed list makes the encoding identical everywhere.
    for zone in ZONES:
        out[f"zone_is_{ZONE_SUFFIX[zone]}"] = (out["zone"] == zone).astype(int)

    # Shot mechanics, one indicator per bucket. Enumerated from the fixed list
    # so the encoding is identical whether the frame holds two million shots or
    # one hypothetical grid point.
    if "shot_subtype" in out.columns:
        mechanic = out["shot_subtype"].map(classify_mechanic)
        for name in SHOT_MECHANICS:
            out[f"mech_{name}"] = (mechanic == name).astype(int)

        # Finish type, on its own axis. A driving dunk and a driving layup are
        # the same creation and wildly different shots: measured over 2024-25
        # restricted-area attempts, 88.8% on 4,994 driving dunks against 60.5%
        # on 30,075 driving layups. Both encoded as mech_driving and nothing
        # else, so those 35,069 shots were identical rows to the model.
        finish = out["shot_subtype"].map(classify_finish)
        for name in FINISH_TYPES:
            out[f"finish_{name}"] = (finish == name).astype(int)

    # Possession origin, one indicator per bucket.
    if "prev_event_type" in out.columns:
        origin = out["prev_event_type"].map(classify_possession_origin)
        for name in POSSESSION_ORIGINS:
            out[f"origin_{name}"] = (origin == name).astype(int)

    if "position" in out.columns:
        from src.common.position_bucket import position_bucket
        bucket = out["position"].map(position_bucket)
        for value in ("G", "F", "C"):
            out[f"pos_{value}"] = (bucket == value).astype(int)

    return out


def as_model_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Select the feature columns in order and coerce them to float.

    Both paths call this immediately before predict, for two reasons.

    Order: XGBoost binds a model to column POSITION, not name. A frame whose
    columns are the right set in the wrong order predicts silently and wrongly.

    Dtype: the training matrix arrives from DuckDB with clean numeric types,
    but the serving path assembles columns from SQLAlchemy rows where a NULL
    makes the whole column `object` — XGBoost then refuses the frame outright.
    Coercing in one shared place means the two paths cannot end up handing the
    booster differently-typed versions of the same feature.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns at predict time: {missing}")
    out = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return out.astype("float32")


def interior_mask(df: pd.DataFrame) -> pd.Series:
    """Rows shot from inside the paint. Shared by training and serving."""
    return df["zone"].isin(INTERIOR_ZONES)
