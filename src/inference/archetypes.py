"""
Player archetypes — the "Urban NBA" vocabulary (3&D Wing, Pick-and-Roll Hub,
Rim Protector, Chucker, ...) computed from the stats this project actually
measures, rather than hand-assigned by an editor.

Why algorithmic, not a lookup table
------------------------------------
A hand-typed `{"Luka Dončić": "Big Guard"}` mapping is an opinion frozen at
whenever someone wrote it — it does not update when a player's game changes,
it cannot be computed for the other 580 players nobody bothered to label, and
it cannot be checked against anything. Every archetype here is instead a
named combination of TRAITS, and every trait is an unweighted mean of
league-percentile ranks over a short, fixed list of measured stats — the
exact pattern `src/lib/stat-engine.ts`'s RADAR_AXES already established for
the frontend radar chart, applied here so it can also drive a `WHERE` clause.
The full trait breakdown ships with every player's result, so "why is this
player a Rim-Running Big" is always answerable by looking at the numbers,
never by taking the label on faith.

Curated, not exhaustive
------------------------
The Urban NBA vocabulary this module draws its NAMES from runs to well over
fifty labels once every guard/wing/big variant is counted, but going further
than a well-defined ~25 here would not add insight — a dozen of the extra
names are near-duplicates of ones already covered (e.g. "True Floor General"
vs. "Pass-First Guard" differ only in degree, which the match score already
expresses), and several more describe things this project has no data for at
all: screen-navigation quality, ICE-coverage success, help-defense rotation
speed, and similar Second-Spectrum-internal or scouting-department concepts
are not published anywhere and were already ruled out earlier in this
project for exactly that reason (see the Stat Engine's own "not real, not
building it" list). Building a plausible-sounding archetype from a made-up
formula over stats that don't measure what the name claims would be the same
mistake this project has already fixed twice: the original invented player
rating, and the mislabeled radar axis that called self-creation "Creation".

Gap this module is honest about: rebounding traits use OREB/DREB per game
(added to this project specifically to support this), but rebound CHANCES,
contested-vs-uncontested rebounds and box-out conversion — real NBA hustle
stats — are not ingested, so no archetype here claims to measure positioning
or effort on the boards, only outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.inference.stat_engine import build_player_stat_table

MIN_GAMES = 20  # below this, a hot two-week stretch can lead the league in anything


def _pctile(series: pd.Series) -> pd.Series:
    """Rank-to-percentile in [0, 1]; NaN stays NaN rather than reading as
    average. Mirrors player_ratings._percentile — kept local rather than
    imported since that one is module-private by convention."""
    return series.rank(pct=True, na_option="keep")


# Stats where a LOWER raw value is the better outcome — percentile needs
# inverting for these before they feed a trait, same convention
# lib/stat-engine.ts's LOWER_IS_BETTER already uses for the frontend.
LOWER_IS_BETTER = {"tov", "def_plus_minus", "def_rim_pm", "def_perimeter_pm",
                   "def_two_pt_pm", "def_paint_pm", "def_mid_long_pm",
                   "def_fg_pct_allowed", "pf"}


@dataclass
class Trait:
    key: str
    label: str
    stats: list[str]


# Each trait is an unweighted mean of the league percentile of every listed
# stat the player has a real value for (missing stats are skipped, not
# zeroed — see axisScore's identical reasoning in stat-engine.ts). A trait
# with NONE of its stats measured for a player is None, not a confident 0.
TRAITS: list[Trait] = [
    Trait("self_creation", "Self-Creation", ["self_creation_index", "avg_drib_per_touch", "pullup_share"]),
    Trait("playmaking", "Playmaking", ["playmaking_gravity", "ast", "potential_ast", "ast_to_pass_pct"]),
    Trait("rim_pressure", "Rim Pressure", ["rim_pressure", "drives_per_min", "zone_fga_rim"]),
    Trait("catch_and_shoot", "Catch-and-Shoot", ["catch_shoot_share", "shotctx_catch_shoot_fg_pct", "zone_fg_pct_above_break3"]),
    Trait("movement_shooting", "Movement Shooting", ["playtype_off_screen_poss_pct", "playtype_off_screen_ppp"]),
    Trait("shot_creation", "Off-the-Dribble Creation", ["playtype_isolation_ppp", "playtype_pnr_ball_handler_ppp", "shotctx_pullup_fg_pct"]),
    Trait("pnr_hub", "Pick-and-Roll Volume", ["playtype_pnr_ball_handler_poss_pct", "playtype_pnr_ball_handler_ppp"]),
    Trait("post_game", "Post Game", ["playtype_postup_poss_pct", "playtype_postup_ppp"]),
    Trait("rim_finishing", "Rim Finishing", ["zone_fg_pct_rim", "playtype_pnr_roll_man_ppp", "playtype_cut_ppp"]),
    Trait("passing_hub", "Offensive Hub Passing", ["playmaking_gravity", "potential_ast", "ast_points_created"]),
    Trait("connective", "Connective Passing", ["secondary_ast", "ast_to_pass_pct"]),
    Trait("defensive_activity", "Defensive Activity", ["stl_per_min", "blk_per_min", "deflections_per_min"]),
    # Deliberately separate from defensive_activity above. Activity (steals,
    # blocks, deflections) measures event RATE, not defensive quality — a
    # positionally sound defender who rarely gambles for a steal or contests
    # a shot into a block can have a low activity number while still being a
    # plus defender by the outcome that actually matters: does the man he
    # guards shoot worse than normal. def_plus_minus (FG% allowed vs. league
    # normal, across every zone, already computed in player_ratings.py) is
    # that outcome measure, so this is the trait any archetype claiming
    # "defends well" (as opposed to "generates events") should gate on.
    # Confirmed live: Aaron Nesmith graded a 90 def_rating_ours this season —
    # every zone plus-minus negative (good) — yet defensive_activity sat at
    # the 24th percentile, which is what put "3 & No-D Wing" on him instead
    # of "3&D Wing".
    Trait("defensive_quality", "Defensive Quality (FG% allowed)", ["def_plus_minus"]),
    Trait("rim_protection", "Rim Protection", ["def_rim_pm", "blk_per_min"]),
    Trait("perimeter_defense", "Perimeter Defense", ["def_perimeter_pm", "stl_per_min"]),
    Trait("interior_defense", "Interior Defense (overall)", ["def_paint_pm", "def_two_pt_pm"]),
    Trait("rim_efficiency", "Finishing Efficiency (Rim)", ["zone_fg_pct_rim"]),
    Trait("midrange_efficiency", "Mid-Range Efficiency", ["zone_fg_pct_midrange"]),
    Trait("three_point_efficiency", "Three-Point Efficiency", ["zone_fg_pct_above_break3"]),
    Trait("hustle", "Hustle Activity", ["screen_ast", "box_outs", "charges_drawn", "loose_balls_recovered"]),
    Trait("rebounding", "Rebounding", ["oreb", "dreb"]),
    Trait("ball_security", "Ball Security", ["tov", "ast_to_pass_pct"]),
    Trait("floor_spacing", "Floor Spacing (volume)", ["zone_fga_above_break3", "catch_shoot_share"]),
    Trait("shooting_efficiency", "Overall Shooting Efficiency", ["season_fg_pct", "career_3p_pct", "ft_pct"]),
    Trait("scoring_load", "Scoring Load", ["pts", "fga"]),
]

TRAIT_BY_KEY = {t.key: t for t in TRAITS}


def compute_traits(table: pd.DataFrame) -> pd.DataFrame:
    """One percentile column per trait, added to a copy of `table`. Built
    once per league table and reused for every archetype's scoring, so a
    28-archetype pass over 580 players costs one percentile rank per stat,
    not one per (archetype, stat) pair."""
    out = table.copy()
    stat_pctile: dict[str, pd.Series] = {}

    def pctile_of(stat: str) -> pd.Series:
        if stat not in stat_pctile:
            if stat not in out.columns:
                stat_pctile[stat] = pd.Series(np.nan, index=out.index)
            else:
                raw = pd.to_numeric(out[stat], errors="coerce")
                p = _pctile(raw)
                stat_pctile[stat] = (1 - p) if stat in LOWER_IS_BETTER else p
        return stat_pctile[stat]

    for trait in TRAITS:
        cols = [pctile_of(s) for s in trait.stats]
        stacked = pd.concat(cols, axis=1)
        out[f"trait_{trait.key}"] = stacked.mean(axis=1, skipna=True)
        # mean(skipna=True) over an all-NaN row returns NaN already in
        # pandas, but made explicit here since a silent 0.0 for "we have
        # none of this player's inputs" is exactly the bug this project's
        # other composites (creation.py, defensive_activity.py) already
        # guard against.
        out.loc[stacked.isna().all(axis=1), f"trait_{trait.key}"] = np.nan

    return out


@dataclass
class ArchetypeDef:
    key: str
    label: str
    category: str
    blurb: str
    # (trait_key, min_percentile) that must ALL hold for a player to qualify.
    requires: list[tuple[str, float]]
    # Traits averaged to rank players who qualify against each other.
    rank_traits: list[str]
    # Extra stat-based gate beyond traits, e.g. volume. Optional.
    min_stat: tuple[str, float] | None = None
    # Which position group a player must be in, if the archetype's name
    # itself names a body type ("...Big", "...Guard", "...Wing"). Without
    # this, "Passing Big" is just "elite playmaker" with no size requirement
    # at all, and gets awarded to Curry and Luka — confirmed live before
    # this field existed. Shooting-profile and "Notable patterns" archetypes
    # are deliberately position-agnostic (any body type can be a Chucker).
    positions: frozenset[str] | None = None
    # Minimum height (inches), enforced ONLY against a bare "G" — never
    # against a "G-F"/"F-G" combo, which already self-declares wing/forward
    # eligibility regardless of measured height. A bare "G" is kept inside
    # WING_POS because this data cannot split point guard from shooting
    # guard, but that ambiguity cuts both ways: it also let a 6'0" pure point
    # guard (confirmed real case: Aaron Holiday, 72in) qualify for "3&D Wing"
    # off the same trait numbers a legitimate 6'5"+ wing would need. Height is
    # the one measured fact that actually distinguishes the two.
    min_height: float | None = None


# Position groups, from the `position` string already carried on every
# player row. "-" combo positions (e.g. "G-F") count toward both halves they
# name, since that IS what the combo notation means.
# Built against the position strings this project's data ACTUALLY contains
# — confirmed live to be exactly {G, F, C, G-F, F-G, F-C, C-F}, never the
# finer PG/SG/SF/PF breakdown. Gating against PG/SG/SF/PF (an earlier version
# of this file did) silently excluded every plain "G" or "F" player from
# ever matching a Wing archetype — the two single most common values in the
# whole table (212 and 162 of ~570 rostered players respectively) — which is
# how Klay Thompson's 3-and-no-D shooting profile failed to match ANY
# archetype at all despite clearly qualifying on the underlying trait
# numbers. A bare "G" or "F" is genuinely ambiguous (this data cannot tell a
# lead ball-handler from a two-guard, or a small forward from a power
# forward), so both bare labels are deliberately included in the two groups
# they could plausibly mean, rather than picking one and guessing wrong for
# the other half of the players who hold it.
GUARD_POS = {"G", "G-F", "F-G"}
WING_POS = {"G", "G-F", "F-G", "F", "F-C", "C-F"}
BIG_POS = {"F", "F-C", "C-F", "C"}
# A stricter big-man gate for archetypes whose NAME specifically claims
# "Big" as in "traditional back-to-the-basket size" (Stretch Big) rather than
# "any frontcourt player" (which BIG_POS above correctly covers for
# rim-running/post/passing archetypes). A bare "F" is kept in BIG_POS
# generally because this data cannot split small forward from power forward
# — but "not every wing that can shoot is a Stretch Big" (confirmed
# complaint: bare-"F" wings were qualifying for it off floor-spacing volume
# alone). Excluding bare "F" here still lets every actual power forward
# through, since a true PF/C tweener in this dataset is always recorded as a
# combo ("F-C"/"C-F") or plain "C", never a bare "F".
TRUE_BIG_POS = frozenset({"F-C", "C-F", "C"})

# The median height among bare-"G" players in this dataset (measured live:
# 76in) — "at least a typical guard's height" rather than "short even for a
# guard", which is the real physical claim "3&D Wing"/"Two-Way Wing"/
# "Slasher" make about a bare "G". A 75th-percentile bar (77in) was tried
# first and excluded Anthony Edwards (76in, a player routinely sized up
# against small forwards) from "Two-Way Wing" by one inch — too strict for a
# gate that only needs to rule out point guards clearly too small to guard
# a wing (confirmed real case: Aaron Holiday, 72in), not to also exclude
# every guard who is merely average-or-a-bit-above for the position.
MIN_WING_HEIGHT_BARE_G = 76.0

# The curated archetype set. Every "requires" trait threshold is a league
# PERCENTILE (0.70 = better than 70% of the league on that trait), not a raw
# stat threshold — this is what keeps the thresholds meaningful across
# positions and across seasons where league-wide numbers drift.
ARCHETYPES: list[ArchetypeDef] = [
    # ── Guards ──────────────────────────────────────────────────────────
    ArchetypeDef("pass_first_guard", "Pass-First Guard", "Guards",
                 "Elite playmaking with a self-creation rate that stays modest — generates offense for teammates before himself.",
                 requires=[("playmaking", 0.80), ("self_creation", -0.50)],
                 rank_traits=["playmaking", "ball_security"],
                 positions=GUARD_POS),
    ArchetypeDef("scoring_guard", "Scoring Guard", "Guards",
                 "Runs the offense but hunts his own shot first — high self-creation alongside real, if secondary, playmaking.",
                 requires=[("self_creation", 0.75), ("playmaking", 0.50)],
                 # scoring_load added so a guard whose whole offensive
                 # identity IS scoring volume (Anthony Edwards: self_creation
                 # and scoring_load both effectively maxed) actually ranks as
                 # a Scoring Guard rather than losing the primary slot to a
                 # narrower stylistic archetype like Rim-Pressure Guard by a
                 # one-point rounding margin despite scoring being the far
                 # more defining trait.
                 rank_traits=["self_creation", "shot_creation", "scoring_load"],
                 positions=GUARD_POS),
    ArchetypeDef("pnr_hub", "Pick-and-Roll Hub", "Guards",
                 "A high share of his offense — and a lot of his team's — runs through the pick-and-roll.",
                 requires=[("pnr_hub", 0.80)],
                 rank_traits=["pnr_hub", "playmaking"],
                 positions=GUARD_POS),
    ArchetypeDef("rim_pressure_guard", "Rim-Pressure Guard", "Guards",
                 "Lives downhill: high drive volume and rim-attempt share for a guard-sized player.",
                 requires=[("rim_pressure", 0.80)],
                 rank_traits=["rim_pressure"],
                 positions=GUARD_POS),

    # ── Shooting (position-agnostic on purpose — any body type can be
    #     one of these) ────────────────────────────────────────────────
    ArchetypeDef("movement_shooter", "Movement Shooter", "Shooting",
                 "Gets his shots off screens and relocation rather than standing in one spot.",
                 requires=[("movement_shooting", 0.80)],
                 rank_traits=["movement_shooting", "catch_and_shoot"]),
    ArchetypeDef("spot_up_shooter", "Spot-Up Shooter", "Shooting",
                 "High catch-and-shoot volume and efficiency with little of his own shot creation.",
                 requires=[("catch_and_shoot", 0.80), ("self_creation", -0.50)],
                 rank_traits=["catch_and_shoot"]),
    ArchetypeDef("shot_creator", "Shot Creator", "Shooting",
                 "Manufactures his own difficult looks — isolation and pull-up heavy, and scores well doing it.",
                 requires=[("shot_creation", 0.80), ("self_creation", 0.70)],
                 rank_traits=["shot_creation", "self_creation"]),
    ArchetypeDef("three_level_scorer", "Three-Level Scorer", "Shooting",
                 "Real efficiency at the rim, in the mid-range, and from three — not just one of the three.",
                 # Gates each zone INDIVIDUALLY rather than an average of two
                 # blended traits (season_fg_pct/3P%/FT% and a rim-finishing
                 # trait that mixes in PnR-roll and cut playtypes). The old
                 # version let a below-average zone hide inside an average —
                 # Dončić's high-volume, high-difficulty shot diet drags his
                 # blended season_fg_pct down even though he is comfortably
                 # above-average at all three individual levels, which is
                 # exactly what "not just one of the three" is supposed to
                 # require, and the blended version failed to enforce.
                 requires=[("rim_efficiency", 0.55), ("midrange_efficiency", 0.55),
                           ("three_point_efficiency", 0.55)],
                 rank_traits=["rim_efficiency", "midrange_efficiency", "three_point_efficiency"]),

    # ── Wings & Forwards ─────────────────────────────────────────────────
    ArchetypeDef("three_and_d_wing", "3&D Wing", "Wings",
                 "Knocks down catch-and-shoot threes and defends his position — the archetypal role player.",
                 # Gated on defensive_quality (FG% allowed vs. league normal),
                 # not defensive_activity (steal/block/deflection rate) — a
                 # low-event, positionally sound defender is exactly what
                 # "defends his position" means, and activity was gating out
                 # real 3&D wings whose defense doesn't show up as events.
                 requires=[("catch_and_shoot", 0.65), ("defensive_quality", 0.55)],
                 rank_traits=["catch_and_shoot", "defensive_quality"],
                 positions=WING_POS, min_height=MIN_WING_HEIGHT_BARE_G),
    ArchetypeDef("three_no_d_wing", "3 & No-D Wing", "Wings",
                 "Shoots it well enough to stay on the floor; the defensive end is where his value ends.",
                 requires=[("catch_and_shoot", 0.65), ("defensive_quality", -0.35)],
                 rank_traits=["catch_and_shoot"],
                 positions=WING_POS, min_height=MIN_WING_HEIGHT_BARE_G),
    ArchetypeDef("connector", "Connector", "Wings",
                 "Doesn't dominate the ball — moves it, cuts, and keeps possessions flowing rather than stopping them.",
                 requires=[("connective", 0.70), ("self_creation", -0.50)],
                 rank_traits=["connective"]),
    ArchetypeDef("slasher", "Slasher", "Wings",
                 "Attacks off movement and closeouts rather than a live dribble — cuts and transition over pull-ups.",
                 requires=[("rim_finishing", 0.70), ("shot_creation", -0.50)],
                 rank_traits=["rim_finishing"],
                 positions=WING_POS, min_height=MIN_WING_HEIGHT_BARE_G),
    ArchetypeDef("two_way_wing", "Two-Way Wing", "Wings",
                 "Genuinely productive on both ends — above-average offense AND defense, neither carrying the other.",
                 requires=[("self_creation", 0.55), ("defensive_quality", 0.55)],
                 rank_traits=["self_creation", "defensive_quality"],
                 positions=WING_POS, min_height=MIN_WING_HEIGHT_BARE_G),
    ArchetypeDef("glue_guy", "Glue Guy", "Wings",
                 "Screens, boxes out, dives for loose balls — value that shows up everywhere except his own shot total.",
                 requires=[("hustle", 0.75), ("self_creation", -0.50)],
                 rank_traits=["hustle", "connective"]),

    # ── Bigs ────────────────────────────────────────────────────────────
    ArchetypeDef("rim_running_big", "Rim-Running Big", "Bigs",
                 "Rolls to the rim and finishes there — the lob-threat archetype.",
                 requires=[("rim_finishing", 0.70), ("post_game", -0.50)],
                 rank_traits=["rim_finishing"],
                 positions=BIG_POS),
    ArchetypeDef("stretch_big", "Stretch Big", "Bigs",
                 "Spaces the floor from three at a size that would traditionally camp in the paint.",
                 requires=[("floor_spacing", 0.70)],
                 rank_traits=["floor_spacing", "shooting_efficiency"],
                 # TRUE_BIG_POS, not BIG_POS: this archetype's name is a claim
                 # about SIZE ("...at a size that would traditionally camp in
                 # the paint"), and a bare "F" in this dataset is just as
                 # likely a small forward as a power forward. A shooting wing
                 # clearing floor_spacing was qualifying here purely because
                 # BIG_POS admits every bare "F" — confirmed complaint.
                 positions=TRUE_BIG_POS),
    ArchetypeDef("post_scorer", "Post Scorer", "Bigs",
                 "Real, efficient volume operating with his back to the basket.",
                 requires=[("post_game", 0.75)],
                 rank_traits=["post_game"],
                 positions=BIG_POS),
    ArchetypeDef("passing_big", "Passing Big / Offensive Hub", "Bigs",
                 "Runs real offense through a big man's hands — elbow and post touches turned into shots for others.",
                 requires=[("passing_hub", 0.75)],
                 rank_traits=["passing_hub"],
                 positions=BIG_POS),
    ArchetypeDef("paint_anchor", "Paint Anchor", "Bigs",
                 "Lives in the restricted area and the paint on offense, defends the rim, controls the defensive glass.",
                 requires=[("rim_protection", 0.65), ("rebounding", 0.65)],
                 rank_traits=["rim_protection", "rebounding"],
                 positions=BIG_POS),
    ArchetypeDef("rebounding_specialist", "Rebounding Specialist", "Bigs",
                 "Elite on the glass on both ends, independent of everything else his box score shows.",
                 requires=[("rebounding", 0.85)],
                 rank_traits=["rebounding"],
                 positions=BIG_POS),

    # ── Defense ─────────────────────────────────────────────────────────
    ArchetypeDef("rim_protector", "Rim Protector", "Defense",
                 "Elite shot-blocking and rim-area FG% suppression.",
                 requires=[("rim_protection", 0.80)],
                 rank_traits=["rim_protection"],
                 positions=BIG_POS),
    ArchetypeDef("poa_defender", "Point-of-Attack Defender", "Defense",
                 "Guards the ball with real steal and perimeter-defense numbers, at a guard/wing size.",
                 requires=[("perimeter_defense", 0.75)],
                 rank_traits=["perimeter_defense", "defensive_quality"],
                 positions=GUARD_POS | WING_POS),
    ArchetypeDef("defensive_anchor", "Defensive Anchor", "Defense",
                 "The defense is built around him — elite across rim protection, activity, and rebounding together.",
                 requires=[("rim_protection", 0.70), ("defensive_activity", 0.65), ("rebounding", 0.55)],
                 rank_traits=["rim_protection", "defensive_activity", "rebounding"],
                 positions=BIG_POS),

    # ── The honest negatives (position-agnostic — these describe a
    #     PATTERN, not a body type) ─────────────────────────────────────
    # These exist because the user's own vocabulary calls for them, and
    # because a purely positive taxonomy would quietly flatter every player
    # into their best-sounding label. Each requires the SAME data the
    # positive archetypes use, just on the other side of the threshold.
    ArchetypeDef("chucker", "Chucker", "Notable patterns",
                 "High shot volume without the efficiency or the playmaking to justify it.",
                 requires=[("scoring_load", 0.75), ("shooting_efficiency", -0.35), ("playmaking", -0.40)],
                 rank_traits=["scoring_load"]),
    ArchetypeDef("black_hole", "Black Hole", "Notable patterns",
                 "The ball goes in, and a pass rarely comes back out — high usage, low ball movement.",
                 requires=[("self_creation", 0.70), ("connective", -0.35), ("ball_security", -0.35)],
                 rank_traits=["self_creation"]),
    ArchetypeDef("foul_machine", "Foul Machine", "Notable patterns",
                 "Racks up personal fouls at a rate that puts him in constant foul trouble.",
                 requires=[],
                 rank_traits=[],
                 min_stat=("pf", 0.85)),
    ArchetypeDef("traffic_cone", "Traffic Cone", "Notable patterns",
                 "Bottom quarter of the league on both defensive quality and perimeter defense — genuinely hunted, not just unremarkable.",
                 # rim_protection is deliberately NOT one of these gates: it
                 # is near-zero for almost every guard/wing regardless of
                 # real defensive quality (they simply do not block shots),
                 # so including it would flag most of the league's wings as
                 # "bad defenders" purely for not being centers.
                 # defensive_quality (FG% allowed), not defensive_activity —
                 # same fix as the wing archetypes above: a low-event
                 # defender who is ALSO genuinely bad by outcome is a Traffic
                 # Cone, but a low-event defender who is fine by outcome
                 # (Nesmith's case) is not, and activity alone can't tell
                 # those two apart.
                 requires=[("defensive_quality", -0.25), ("perimeter_defense", -0.25)],
                 rank_traits=["defensive_quality"]),
]

ARCHETYPE_BY_KEY = {a.key: a for a in ARCHETYPES}
CATEGORY_ORDER = ["Guards", "Shooting", "Wings", "Bigs", "Defense", "Notable patterns"]


def _qualifies(row: pd.Series, arch: ArchetypeDef) -> bool:
    if arch.positions is not None:
        position = row.get("position")
        # A player with no known position cannot clear a body-type gate —
        # "must be a Big" is not satisfiable by an unknown, and defaulting
        # an unknown position to "qualifies anyway" would let a missing-data
        # gap quietly turn into a free pass instead of a hard "no".
        if position is None or position not in arch.positions:
            return False
        # A bare "G" (never a "G-F"/"F-G" combo, which already self-declares
        # wing/forward eligibility) additionally needs real wing size for a
        # Wing archetype — confirmed real bug: a 6'0" pure point guard (Aaron
        # Holiday) was qualifying for "3&D Wing" purely because the position
        # string alone can't tell a point guard from a big guard.
        if position == "G" and arch.min_height is not None:
            height = row.get("height")
            if height is None or (isinstance(height, float) and np.isnan(height)) or height < arch.min_height:
                return False
    for trait_key, threshold in arch.requires:
        val = row.get(f"trait_{trait_key}")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return False
        # A POSITIVE threshold is a floor ("must be at least this good"). A
        # NEGATIVE threshold is a CEILING at its absolute value ("must be no
        # better than the bottom |threshold| of the league") — used both for
        # a lenient exclusion (self_creation <= -0.50, "not primarily a
        # scorer", for an archetype whose real definition is something
        # else) and for a genuinely damning claim (defensive_activity <=
        # -0.25, the bottom quarter of the league, for "Traffic Cone").
        # There is no shared "0.0 means below-median" case any more — every
        # ceiling states its own bar explicitly, because collapsing every
        # exclusion onto the same league-median line is how a merely
        # unremarkable defender (below median on three different measures,
        # elite at none) ended up qualifying as a "Traffic Cone" instead of
        # a genuinely bad one.
        if threshold >= 0:
            if val < threshold:
                return False
        elif val > -threshold:
            return False
    if arch.min_stat is not None:
        stat_key, min_pct = arch.min_stat
        pct_col = f"__gate_pctile_{stat_key}"
        if pct_col not in row.index:
            return False
        val = row.get(pct_col)
        if val is None or (isinstance(val, float) and np.isnan(val)) or val < min_pct:
            return False
    return True


def _match_score(row: pd.Series, arch: ArchetypeDef) -> float:
    if not arch.rank_traits:
        # An archetype defined purely by min_stat (Foul Machine) ranks by
        # that same gate stat rather than an empty trait list.
        if arch.min_stat:
            return float(row.get(f"__gate_pctile_{arch.min_stat[0]}", 0.0) or 0.0)
        return 0.0
    vals = [row.get(f"trait_{t}") for t in arch.rank_traits]
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else 0.0


MAX_SECONDARY = 3
# Up to this many archetypes may share one category across primary+secondary.
# A hard 1-per-category cap (an earlier version of this rule) hid genuinely
# distinct, informative labels behind whichever same-category archetype
# happened to score highest — confirmed real case: Dončić and Durant both
# clearly qualify for Three-Level Scorer (a genuinely different claim from
# Shot Creator: efficiency at every level, not shot-manufacturing method),
# but it never appeared anywhere because Shot Creator already occupied
# "Shooting"'s only slot. 2 still prevents the list from being ALL one
# category, and works together with the trait-footprint check below.
MAX_PER_CATEGORY = 2


def _trait_footprint(arch: ArchetypeDef) -> set[str]:
    """Which traits (or the min_stat gate) an archetype's `requires` actually
    tests, used to detect when a second archetype adds zero new information
    rather than a genuinely different fact about the player."""
    footprint = {trait_key for trait_key, _ in arch.requires}
    if arch.min_stat is not None:
        footprint.add(f"__stat_{arch.min_stat[0]}")
    return footprint


def compute_archetypes(engine, season: str | None = None) -> pd.DataFrame:
    """One row per player rostered in `season`, with every trait percentile,
    a primary archetype, and up to MAX_SECONDARY secondary archetypes (at
    most MAX_PER_CATEGORY per category, primary included). Players under
    MIN_GAMES games are traited (so their numbers are visible) but never
    assigned an archetype at all — a two-week call-up should not get crowned
    "Rim Protector" off a hot stretch."""
    table = build_player_stat_table(engine, season)
    if table.empty:
        return table

    table = compute_traits(table)

    # The one non-trait gate (Foul Machine) needs its own percentile column,
    # built the same way a trait's inputs are.
    for stat_key in {a.min_stat[0] for a in ARCHETYPES if a.min_stat}:
        raw = pd.to_numeric(table.get(stat_key), errors="coerce")
        p = _pctile(raw)
        table[f"__gate_pctile_{stat_key}"] = (1 - p) if stat_key in LOWER_IS_BETTER else p

    games = pd.to_numeric(table.get("games_played"), errors="coerce")
    eligible_for_any = games.fillna(0) >= MIN_GAMES

    primary_keys, primary_labels, primary_scores = [], [], []
    secondary_lists = []

    for idx, row in table.iterrows():
        if not eligible_for_any.loc[idx]:
            primary_keys.append(None); primary_labels.append(None); primary_scores.append(None)
            secondary_lists.append([])
            continue

        matches = []
        for arch in ARCHETYPES:
            if _qualifies(row, arch):
                matches.append((arch, _match_score(row, arch)))
        matches.sort(key=lambda m: m[1], reverse=True)

        if not matches:
            primary_keys.append(None); primary_labels.append(None); primary_scores.append(None)
            secondary_lists.append([])
            continue

        primary_arch, primary_score = matches[0]
        primary_keys.append(primary_arch.key)
        primary_labels.append(primary_arch.label)
        primary_scores.append(round(primary_score * 100))

        secondary = []
        category_counts: dict[str, int] = {primary_arch.category: 1}
        seen_traits: set[str] = set(_trait_footprint(primary_arch))
        for arch, score in matches[1:]:
            if category_counts.get(arch.category, 0) >= MAX_PER_CATEGORY:
                continue
            footprint = _trait_footprint(arch)
            # Skip only when EVERY trait this archetype tests is already
            # covered by an archetype already shown — real bug this fixes:
            # a dominant two-way big (confirmed real case: Victor
            # Wembanyama) cleared Rim Protector, Paint Anchor, Rebounding
            # Specialist AND Defensive Anchor at once, but Rebounding
            # Specialist's entire gate (rebounding alone) was already
            # implied by Paint Anchor's (rim_protection + rebounding) —
            # showing it added a fourth badge with zero new information.
            # A candidate that shares SOME traits with what's shown but also
            # tests something new (e.g. Two-Way Wing sharing defensive_
            # quality with 3&D Wing but adding self_creation) still passes.
            if footprint and footprint <= seen_traits:
                continue
            secondary.append({
                "key": arch.key, "label": arch.label, "category": arch.category,
                "blurb": arch.blurb, "score": round(score * 100),
            })
            category_counts[arch.category] = category_counts.get(arch.category, 0) + 1
            seen_traits |= footprint
            if len(secondary) == MAX_SECONDARY:
                break
        secondary_lists.append(secondary)

    table["archetype_key"] = primary_keys
    table["archetype_label"] = primary_labels
    table["archetype_score"] = primary_scores
    table["archetype_secondary"] = secondary_lists
    return table


def player_archetype_detail(engine, player_id: str, season: str | None = None) -> dict | None:
    """Full transparency for one player: primary + secondary archetypes,
    every trait's percentile, and which archetypes they qualified for but
    didn't win — so "why isn't he a Rim Protector" is answerable too."""
    table = compute_archetypes(engine, season)
    if table.empty:
        return None
    row = table[table["player_id"] == str(player_id)]
    if row.empty:
        return None
    row = row.iloc[0]

    traits = []
    for t in TRAITS:
        val = row.get(f"trait_{t.key}")
        traits.append({
            "key": t.key, "label": t.label,
            "percentile": None if val is None or (isinstance(val, float) and np.isnan(val)) else round(float(val) * 100),
            "stats": t.stats,
        })

    qualified = []
    if row.get("archetype_key") is not None:
        for arch in ARCHETYPES:
            if _qualifies(row, arch):
                qualified.append({
                    "key": arch.key, "label": arch.label, "category": arch.category,
                    "blurb": arch.blurb, "score": round(_match_score(row, arch) * 100),
                })
        qualified.sort(key=lambda q: q["score"], reverse=True)

    primary_key = row.get("archetype_key")
    primary_arch = ARCHETYPE_BY_KEY.get(primary_key) if primary_key is not None else None

    return {
        "player_id": str(player_id),
        "primary": (
            {
                "key": row["archetype_key"], "label": row["archetype_label"],
                "category": primary_arch.category if primary_arch else None,
                "blurb": primary_arch.blurb if primary_arch else None,
                "score": row["archetype_score"],
            }
            if primary_key is not None else None
        ),
        "secondary": row.get("archetype_secondary") or [],
        "qualified": qualified,
        "traits": traits,
        "eligible": bool(pd.to_numeric(pd.Series([row.get("games_played")]), errors="coerce").fillna(0).iloc[0] >= MIN_GAMES),
    }


def archetype_catalogue() -> list[dict]:
    """Every archetype this module can assign, for a frontend filter list —
    grouped and ordered the same way TRAIT/ARCHETYPE definitions are."""
    return [
        {"key": a.key, "label": a.label, "category": a.category, "blurb": a.blurb}
        for a in ARCHETYPES
    ]
