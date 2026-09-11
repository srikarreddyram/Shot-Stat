"""
Why is this shot hard for this player to get?

The attainability model answers "what share of this player's shot diet would
plausibly come from here" as a single number. That number is the start of a
question, not the end of one: a user who sees 4% wants to know whether the
shot is rare for everybody, rare for this player, and which part of his game
is responsible.

How the decomposition works
---------------------------
XGBoost exposes exact per-feature contributions for a single prediction
(TreeSHAP, via `pred_contribs=True`). They sum to the prediction, so an
explanation built from them is complete by construction — no feature's
influence can be silently omitted, which is the failure mode of explaining a
model by its global feature importances instead.

Those contributions are then grouped into the two things a reader actually
distinguishes between:

    baseline  = bias + zone terms      what ANY player would get here
    player    = everything else        what THIS player adds or gives up

So a 4% corner three splits into "corner threes are 5% of a typical player's
diet" and "and he is a rim-runner, which costs another point". Reporting only
the total conflates a fact about the floor with a fact about the person.

Percentiles, not just directions
--------------------------------
A contribution says a feature pushed the estimate down. It cannot say whether
the underlying value was unusual — a feature can push hard simply because the
model is sensitive there. Each factor is therefore reported with the player's
own value and its approximate league percentile, interpolated from the
quantile grid stored in the model metadata at training time. "Pulls up on 5%
of his touches, 12th percentile" is the half a reader can act on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.spec import FEATURE_GROUPS

# How each model input reads in English.
#
#   label   — the noun phrase shown to a reader
#   fmt     — "pct" renders 0.12 as 12%, "num" as a plain number, "in" inches
#   high    — what a HIGH value of this feature means, as a clause
#   low     — what a LOW value means
#
# Features absent from this table still appear in the raw contribution list but
# are never rendered as prose, so an unlabelled new feature degrades to silence
# rather than to a confusing machine name in the UI.
FEATURE_COPY: dict[str, dict[str, str]] = {
    "height":       {"label": "Height", "fmt": "in",
                     "high": "tall enough to work inside", "low": "undersized inside"},
    "weight":       {"label": "Weight", "fmt": "num",
                     "high": "heavy enough to hold position", "low": "light for interior work"},
    "wingspan":     {"label": "Wingspan", "fmt": "in",
                     "high": "long reach", "low": "shorter reach"},

    "avg_drib_per_touch": {"label": "Dribbles per touch", "fmt": "num",
                           "high": "pounds the ball before shooting",
                           "low": "moves it quickly"},
    "avg_sec_per_touch":  {"label": "Seconds per touch", "fmt": "num",
                           "high": "holds the ball", "low": "quick decisions"},
    "touches_per_min":    {"label": "Touches per minute", "fmt": "num",
                           "high": "heavily involved", "low": "rarely touches it"},
    "poss_time_per_min":  {"label": "Possession time", "fmt": "num",
                           "high": "dominates possessions", "low": "off-ball"},
    "drives_per_min":     {"label": "Drives per minute", "fmt": "num",
                           "high": "constant rim pressure", "low": "rarely drives"},
    "drive_fg_pct":       {"label": "Drive FG%", "fmt": "pct",
                           "high": "finishes drives", "low": "struggles finishing drives"},
    "drive_tov_pct":      {"label": "Drive turnover rate", "fmt": "pct",
                           "high": "loses it driving", "low": "secure driving"},
    "drive_pf_pct":       {"label": "Drive foul rate", "fmt": "pct",
                           "high": "draws contact", "low": "rarely draws fouls"},

    "avg_def_dist":   {"label": "Defender distance", "fmt": "num",
                       "high": "usually left open", "low": "closely guarded"},
    "open_share":     {"label": "Open-shot share", "fmt": "pct",
                       "high": "gets clean looks", "low": "rarely open"},
    "tight_share":    {"label": "Contested-shot share", "fmt": "pct",
                       "high": "shoots through contact", "low": "avoids contested looks"},
    "pullup_share":   {"label": "Pull-up share", "fmt": "pct",
                       "high": "creates his own jumper", "low": "seldom pulls up"},
    "catch_shoot_share": {"label": "Catch-and-shoot share", "fmt": "pct",
                          "high": "spot-up shooter", "low": "rarely spots up"},
    "self_created_dribble_share": {"label": "Self-created share", "fmt": "pct",
                                   "high": "makes his own shots",
                                   "low": "depends on others to create"},
    "zero_dribble_share": {"label": "Zero-dribble share", "fmt": "pct",
                           "high": "shoots off the catch", "low": "needs a dribble first"},
    "creation_retention": {"label": "Creation retention", "fmt": "num",
                           "high": "keeps efficiency when creating",
                           "low": "loses efficiency when creating"},
    "self_creation_index": {"label": "Self-creation", "fmt": "num",
                            "high": "elite self-creator", "low": "limited self-creator"},
    "playmaking_gravity": {"label": "Playmaking gravity", "fmt": "num",
                           "high": "bends defences", "low": "little defensive attention"},
    "rim_pressure":   {"label": "Rim pressure", "fmt": "num",
                       "high": "lives at the rim", "low": "little rim pressure"},

    # Supporting cast — his TEAMMATES, with his own contribution excluded. The
    # copy says "teammates" everywhere for that reason: "his team moves the
    # ball" would be the raw quantity, which for a high-usage star is mostly
    # a description of the star himself.
    "cast_ast_rate": {"label": "Teammates' ball movement", "fmt": "pct",
                      "high": "teammates move the ball",
                      "low": "teammates rarely create for him"},
    "cast_3p_rate":  {"label": "Teammates' 3P%", "fmt": "pct",
                     "high": "shooters around him stretch the floor",
                     "low": "little shooting around him"},
    "cast_efg":      {"label": "Teammates' efficiency", "fmt": "pct",
                     "high": "teammates command attention",
                     "low": "teammates draw little defensive attention"},
    "diet_to_date":    {"label": "His rate here this season", "fmt": "pct",
                      "high": "already going there this season",
                      "low": "has not gone there this season"},
    "diet_att_to_date": {"label": "Shots so far this season", "fmt": "num",
                      "high": "plenty of evidence this season",
                      "low": "little evidence this season yet"},
    "season_progress": {"label": "Season elapsed", "fmt": "pct",
                      "high": "well into the season", "low": "early in the season"},
    "career_diet":     {"label": "His career rate here", "fmt": "pct",
                      "high": "a career-long habit", "low": "never been part of his game"},
    "career_diet_att": {"label": "Career shots behind it", "fmt": "num",
                      "high": "long track record", "low": "short track record"},
    "prior_zone_share": {"label": "His rate here last season", "fmt": "pct",
                      "high": "a spot he already lives in",
                      "low": "not a spot he has gone to before"},

    "cast_att":      {"label": "Teammate attempts", "fmt": "num",
                     "high": "well-established supporting cast",
                     "low": "thin evidence on his supporting cast"},
}

# Feature-name prefixes that describe the SHOT rather than the player. These
# form the structural half of the decomposition.
ZONE_PREFIXES = ("zone_is_", "is_three")
POSITION_PREFIX = "pos_is_"

# The player's own prior-season share, held out of the factor list on purpose.
#
# It is not a trait. Every other entry answers "what about his game moves this
# number" — handle, spacing, the cast around him — and its own history answers
# a different question entirely: where he starts from. Left in the list it also
# swamps everything else, because shot diet is stable at r=0.94 season over
# season, so the top reason for every shot on the floor would read "he took
# this shot last year" and nothing else would ever surface.
#
# It gets its own line in the decomposition instead: history, then what moves
# him off it.
HISTORY_FEATURE = "diet_to_date"

# Falls back to last season when the current one has not started.
HISTORY_FALLBACK = "prior_zone_share"

# Reported alongside, never as a "trait".
HISTORY_SUPPORT = ("diet_att_to_date", "season_progress", "career_diet",
                   "career_diet_att", "prior_zone_share")

POSITION_LABEL = {"G": "guard", "F": "forward", "C": "center"}


def creation_note(creation: dict) -> str | None:
    """
    One line on WHO generates a player's shots from a spot.

    Attainability is a frequency — what share of a player's shots come from
    here — and a frequency cannot distinguish a look he manufactures at will
    from one that only exists when somebody finds him. League-wide the two
    separate sharply: 44% of restricted-area makes are unassisted against 4%
    of corner threes, so "you can go get this" and "you need this created for
    you" are genuinely different advice at the same attainability number.

    Returns None when the sample is too thin to say anything, rather than
    reporting a shrunk estimate as though it were observed.
    """
    share = creation.get("self_created_share")
    league = creation.get("league_self_created_share")
    makes = creation.get("makes") or 0

    if share is None or league is None:
        return None
    if makes < MIN_CREATION_MAKES:
        return (f"Only {makes} made shots here — not enough to say how he "
                f"generates them.")

    pct, league_pct = round(share * 100), round(league * 100)
    if share < 0.25:
        return (f"He creates {pct}% of these himself against a league {league_pct}% "
                f"— this shot mostly has to be created for him.")
    if share > league + 0.10:
        return (f"He creates {pct}% of these himself against a league {league_pct}% "
                f"— he can generate this look on his own.")
    return (f"He creates {pct}% of these himself, about the league {league_pct}% "
            f"for this spot.")


# Below this many made shots in a zone, the shrunk self-created share is
# almost entirely the league prior showing through, and reporting it as a
# fact about the player would be reporting the prior back as evidence.
MIN_CREATION_MAKES = 20


def _percentile(value: float | None, reference: dict) -> float | None:
    """
    Approximate league percentile for `value`, interpolated from the stored
    quantile grid. Returns None when the feature has no reference (a model
    trained before the grid existed, or a feature that was always null).
    """
    quantiles = (reference or {}).get("quantiles")
    if not quantiles or value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    points = sorted((float(q), v) for q, v in quantiles.items())
    qs = [q for q, _ in points]
    vs = [v for _, v in points]

    if value <= vs[0]:
        return round(qs[0] * 100, 1)
    if value >= vs[-1]:
        return round(qs[-1] * 100, 1)
    # np.interp needs an increasing x; the stored values are monotone in q for
    # every real distribution, but guard against a degenerate flat segment.
    return round(float(np.interp(value, vs, qs)) * 100, 1)


def _format_value(value, fmt: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    if fmt == "pct":
        return f"{value * 100:.0f}%"
    if fmt == "in":
        feet, inches = divmod(int(round(value)), 12)
        return f"{feet}'{inches}\""
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.2f}"


def explain_attainability(
    model: xgb.XGBRegressor,
    metadata: dict,
    feature_row: pd.DataFrame,
    zone: str,
    top_n: int = 4,
) -> dict:
    """
    Decompose one attainability prediction into readable reasons.

    `feature_row` is a single-row frame already carrying every column in
    `metadata["feature_cols"]` — the caller builds it the same way it builds
    the row it predicts from, so the explanation describes the prediction that
    was actually served rather than a reconstruction of it.
    """
    cols: list[str] = metadata["feature_cols"]
    reference: dict = metadata.get("feature_reference", {})
    league_zone_shares: dict = metadata.get("league_zone_shares", {})

    X = feature_row[cols].astype(float)
    booster = model.get_booster()
    contribs = booster.predict(
        xgb.DMatrix(X, missing=np.nan), pred_contribs=True
    )[0]

    bias = float(contribs[-1])
    per_feature = {col: float(contribs[i]) for i, col in enumerate(cols)}
    predicted = float(np.clip(bias + sum(per_feature.values()), 0.0, 1.0))

    zone_effect = sum(
        v for c, v in per_feature.items() if c.startswith(ZONE_PREFIXES)
    )
    position_effect = sum(
        v for c, v in per_feature.items() if c.startswith(POSITION_PREFIX)
    )
    # What a league-average player would get in this zone: the model's own
    # bias plus everything it knows about the SHOT, with the player's own
    # traits held out.
    baseline = float(np.clip(bias + zone_effect, 0.0, 1.0))

    # The whole history block — this season so far, the career behind it, and
    # last season — moves together and is reported as one quantity.
    history_effect = per_feature.get(HISTORY_FEATURE, 0.0) + sum(
        per_feature.get(c, 0.0) for c in HISTORY_SUPPORT
    )
    # Which window the reported figure comes from decides how the sentence
    # names it. Quoting a season-to-date rate as "last season" would be a
    # confident mislabel of the number the reader is looking at.
    prior_share = feature_row.iloc[0].get(HISTORY_FEATURE)
    history_window = "this season"
    if pd.isna(prior_share):
        prior_share = feature_row.iloc[0].get(HISTORY_FALLBACK)
        history_window = "last season"
    prior_share = None if pd.isna(prior_share) else float(prior_share)

    factors = []
    for col, contribution in per_feature.items():
        if (col.startswith(ZONE_PREFIXES) or col.startswith(POSITION_PREFIX)
                or col == HISTORY_FEATURE or col in HISTORY_SUPPORT):
            continue
        copy = FEATURE_COPY.get(col)
        if copy is None or abs(contribution) < 1e-6:
            continue

        value = feature_row.iloc[0].get(col)
        value = None if pd.isna(value) else float(value)
        pct = _percentile(value, reference.get(col, {}))
        # Which clause applies is a statement about the player's VALUE, not
        # about the direction the model moved — a low pull-up share can raise
        # attainability at the rim, and describing that as "creates his own
        # jumper" because the contribution was positive would be wrong.
        trait = copy["high"] if (pct is not None and pct >= 50) else copy["low"]

        factors.append({
            "feature": col,
            "label": copy["label"],
            "value": value,
            "display_value": _format_value(value, copy["fmt"]),
            "percentile": pct,
            "impact": round(contribution, 5),
            "direction": "raises" if contribution > 0 else "lowers",
            "detail": trait,
        })

    factors.sort(key=lambda f: -abs(f["impact"]))
    factors = factors[:top_n]

    position = None
    for bucket in metadata.get("position_buckets", []):
        col = f"{POSITION_PREFIX}{bucket}"
        if col in feature_row.columns and float(feature_row.iloc[0][col]) == 1.0:
            position = POSITION_LABEL.get(bucket, bucket)
            break

    return {
        "attainability": round(predicted, 4),
        "zone": zone,
        "league_zone_share": league_zone_shares.get(zone),
        "baseline": round(baseline, 4),
        "player_effect": round(predicted - baseline, 4),
        "position": position,
        "position_effect": round(position_effect, 5),
        # Where he starts from, kept distinct from what moves him off it.
        "history": {
            "prior_share": prior_share,
            "window": history_window,
            "effect": round(float(history_effect), 5),
        },
        "factors": factors,
        "summary": _summarize(predicted, baseline, zone, factors, position,
                              prior_share, float(history_effect),
                              history_window),
    }


# Below this share of a player's diet, a zone is rare enough that its scarcity
# is worth stating outright — the reader's first question about a 2% estimate
# is whether anybody shoots from there.
RARE_ZONE_SHARE = 0.10

# Player effects smaller than this are not worth narrating: they are inside the
# model's own error (test MAE is around 0.05), so naming a "cause" would be
# reading noise aloud.
NEGLIGIBLE_EFFECT = 0.01


def _share_text(share: float) -> str:
    """
    A shot share as a percentage, with a decimal only where rounding to whole
    numbers would destroy the figure.

    Giannis took 0.2% of his shots from the left corner. Printed as "0%" that
    reads as a rendering failure, and it sits in the same sentence as a
    non-zero projection, which makes the sentence look self-contradictory.
    """
    pct = share * 100
    if 0 < pct < 1:
        return f"{pct:.1f}%"
    return f"{pct:.0f}%"


def _driver_clause(factors: list[dict], limit: int = 2) -> str:
    """
    ", mostly X (val) and Y (val)" — or "" when nothing is worth naming.

    Neutral LABELS, not the trait clauses the factor list uses. A trait clause
    asserts a direction ("tall enough to work inside"), which reads as a
    contradiction when the feature's contribution runs the other way — and it
    legitimately can, since a high value need not push the estimate up. The
    factor list can afford the vivid phrasing because it prints the signed
    contribution beside it; a sentence cannot.
    """
    # A factor with no value is not evidence. Naming one produced sentences
    # like "mostly teammates' ball movement (unknown)" — citing a quantity in
    # the same breath as admitting it is not known, which reads as a rendering
    # fault and undercuts the reasons that ARE backed by data. The factor still
    # appears in the ranked list below with its contribution; it just does not
    # get quoted in prose.
    named = [
        f for f in factors
        if abs(f["impact"]) >= NEGLIGIBLE_EFFECT / 4
        and f.get("value") is not None
    ]
    if not named:
        return ""
    joined = " and ".join(
        f"{d['label'].lower()} ({d['display_value']})" for d in named[:limit]
    )
    return f", mostly {joined}"


def _summarize(predicted: float, baseline: float, zone: str,
               factors: list[dict], position: str | None,
               prior_share: float | None = None,
               history_effect: float = 0.0,
               history_window: str = "last season") -> str:
    """
    One sentence a reader can take away without parsing the factor list.

    Leads with the player's own prior season when there is one, because that is
    genuinely where the estimate comes from — shot diet repeats at r=0.94 year
    over year, and an explanation that opened with "pull-up share 50%" while
    silently resting on "he took 27% of his shots here last season" would be
    naming a secondary cause and hiding the primary one.

    Quotes `baseline` rather than `league_share` because that is the number
    shown beside it in the UI — they are close but not identical (one is the
    model's own bias-plus-zone decomposition, the other a raw league mean), and
    printing both invites the reader to reconcile two different "typical"
    figures that answer slightly different questions.

    Deliberately hedged where the model is: when the player effect is inside
    the model's error, the honest statement is that nothing about his game
    moves this much.
    """
    zone_name = zone.lower()

    # History first, when it exists and is doing real work.
    #
    # `history_effect` is the contribution of the player's OWN prior share, so
    # the sentence must attribute it to his history and not to his game — an
    # earlier draft said "his game pulls him off it" while quoting the history
    # term, which named the wrong cause. Traits get their own clause after.
    if prior_share is not None and abs(history_effect) >= NEGLIGIBLE_EFFECT:
        prior_txt = _share_text(prior_share)
        pred_txt = _share_text(predicted)
        base_txt = _share_text(baseline)
        drivers = _driver_clause(factors)

        taken = ("has taken" if history_window == "this season" else "took")
        if history_effect > 0:
            lead = (f"He {taken} {prior_txt} of his shots here {history_window}, and "
                    f"that is most of why this projects at {pred_txt} against "
                    f"{base_txt} for a typical player")
        else:
            lead = (f"He {taken} {prior_txt} of his shots here {history_window}, well "
                    f"under the {base_txt} a typical player takes, which is most of "
                    f"why this projects at {pred_txt}")
        return f"{lead}{drivers}."

    baseline_txt = f"{baseline * 100:.0f}%" if baseline >= 0.1 else f"{baseline * 100:.1f}%"
    delta = predicted - baseline
    rare = baseline < RARE_ZONE_SHARE

    if abs(delta) < NEGLIGIBLE_EFFECT:
        if rare:
            return (f"Rare for anyone — {zone_name} shots are about {baseline_txt} "
                    f"of a typical player's diet, and nothing about his game "
                    f"moves that much.")
        return (f"About what you'd expect: {baseline_txt} for a typical player "
                f"here, and his game neither adds nor costs much.")

    who = f" {position}s" if position else ""
    if rare:
        lead = (f"{zone_name.capitalize()} shots are a small slice of anyone's diet "
                f"({baseline_txt}), and he gets there "
                f"{'more' if delta > 0 else 'less'} than most{who}")
    else:
        lead = (f"A typical player takes {baseline_txt} of his shots here; "
                f"he gets there {'more' if delta > 0 else 'less'} than most{who}")

    # Same rule as `_driver_clause`: only quote factors whose value is known.
    drivers = [f for f in factors
               if f["direction"] == ("raises" if delta > 0 else "lowers")
               and f.get("value") is not None]
    if not drivers:
        return lead + "."

    # Neutral LABELS here, not the trait clauses the factor list uses. A trait
    # clause asserts a direction ("tall enough to work inside"), which reads as
    # a contradiction when the feature's contribution runs the other way — and
    # it legitimately can, since a high value need not push the estimate up.
    # The factor list can afford the vivid phrasing because it prints the
    # signed contribution beside it; a sentence cannot.
    named = " and ".join(
        f"{d['label'].lower()} ({d['display_value']})" for d in drivers[:2]
    )
    return f"{lead} — mostly {named}."


# ═══════════════════════════════════════════════════════════════════════════
# Shot-quality (make-probability) explanation
#
# Everything above explains ATTAINABILITY — whether this player would take a
# shot like this. It never mentions a defender, because attainability doesn't
# have one: it is a frequency question about the shooter's own diet.
#
# This section explains the OTHER model: given the shot is being taken, will
# it go in. That model's feature vector already carries the defender (his
# zone-level FG%-allowed, his matchup share, the offense/defense interaction
# terms) — it was simply never decomposed into prose. The result was an
# explanation UI that only ever talked about league averages and the
# attacker's own history, with no defender in the story at all, for a
# MATCHUP engine.
# ═══════════════════════════════════════════════════════════════════════════

# Column families, by which side of the matchup they describe. Built from
# FEATURE_GROUPS rather than hand-listed, so a feature added to a group later
# is picked up automatically instead of silently landing nowhere.
#
# NOTE on a mistake worth recording: `FEATURE_GROUPS["creation"]` is the
# shooter's HANDLE/skill numbers (self_creation_index, drives_per_min, ...),
# not the mech_* creation-mechanic one-hots (driving/spot_up/post_up/...).
# The two are easy to conflate — they were conflated once already, describing
# an ablation result — because both are called "creation" in different parts
# of this codebase for different reasons. mech_* and finish_* are not members
# of ANY FEATURE_GROUPS entry at all; derive_features adds them by a prefix
# scan (`all_feature_columns`), so they are picked up here the same way.
_OFFENSE_GROUPS = ("shooter_physical", "shooter_skill", "creation", "possession_origin")
_DEFENSE_GROUPS = ("defender", "defender_physical", "opponent_defence")
_MATCHUP_GROUPS = ("interaction",)
_CONTEXT_GROUPS = ("spatial", "context", "shot_context")


def _group_columns(*group_names: str) -> set[str]:
    cols: set[str] = set()
    for name in group_names:
        cols.update(FEATURE_GROUPS.get(name, []))
    return cols


def _side_columns(cols: list[str]) -> dict[str, str]:
    """{column: side} for every shot-quality feature column, where side is
    one of "offense", "defense", "matchup", "context". mech_*/finish_* are
    offense (they describe how THIS shooter created/finished the look);
    anything matched by none of the above (e.g. an ablated-off feature that
    still has a column) falls back to "context" rather than being dropped."""
    offense = _group_columns(*_OFFENSE_GROUPS)
    defense = _group_columns(*_DEFENSE_GROUPS)
    matchup = _group_columns(*_MATCHUP_GROUPS)
    context = _group_columns(*_CONTEXT_GROUPS)

    sides = {}
    for col in cols:
        if col.startswith("mech_") or col.startswith("finish_"):
            sides[col] = "offense"
        elif col in offense:
            sides[col] = "offense"
        elif col in defense:
            sides[col] = "defense"
        elif col in matchup:
            sides[col] = "matchup"
        elif col in context:
            sides[col] = "context"
        else:
            sides[col] = "context"
    return sides


# Copy for shot-quality features not already covered by FEATURE_COPY above
# (which was built for attainability's feature set). Kept as a SEPARATE dict
# rather than merged into FEATURE_COPY: several column names mean different
# things in the two models' feature sets are actually disjoint here, but
# keeping the tables apart avoids ever having to reconcile a future collision
# by accident.
SQ_FEATURE_COPY: dict[str, dict[str, str]] = {
    "zone_rate": {"label": "His rate from this zone", "fmt": "pct",
                 "high": "shoots this zone well", "low": "struggles from this zone"},
    "overall_rate": {"label": "His overall FG%", "fmt": "pct",
                     "high": "efficient scorer overall", "low": "below-average efficiency"},
    "three_rate": {"label": "His 3P%", "fmt": "pct",
                  "high": "knocks down threes", "low": "not a three-point threat"},
    "recent_10_fg": {"label": "Last 10 games FG%", "fmt": "pct",
                     "high": "shooting well lately", "low": "cold lately"},
    "def_fg_pct_zone": {"label": "Defender's FG% allowed here", "fmt": "pct",
                        "high": "gets shot on in this zone", "low": "locks this zone down"},
    "def_pct_plusminus_zone": {"label": "Defender's zone plus-minus", "fmt": "pct",
                               "high": "worse than league average here",
                               "low": "better than league average here"},
    "def_pct_plusminus": {"label": "Defender's overall plus-minus", "fmt": "pct",
                          "high": "worse than league average overall",
                          "low": "better than league average overall"},
    "def_matchup_share": {"label": "Primary-defender share", "fmt": "pct",
                          "high": "closely tracked by this defender",
                          "low": "loosely switched onto this defender"},
    "def_freq_zone": {"label": "Defender's time in this zone", "fmt": "pct",
                      "high": "spends a lot of time defending this zone",
                      "low": "rarely defends this zone"},
    "matchup_advantage": {"label": "Matchup edge", "fmt": "pct",
                          "high": "clear edge over this defender",
                          "low": "defender has the edge here"},
    "expected_contest": {"label": "Expected space", "fmt": "num",
                         "high": "projects open", "low": "projects contested"},
    "openness_vs_defender": {"label": "Openness vs. this defender", "fmt": "pct",
                             "high": "gets more space than this defender usually allows",
                             "low": "gets less space than this defender usually allows"},
    "opp_zone_def_rate": {"label": "Opponent's zone defense", "fmt": "pct",
                          "high": "team defends this zone well",
                          "low": "team is exploitable in this zone"},
}


def _sq_percentile(value, reference: dict) -> float | None:
    return _percentile(value, reference)


def explain_shot_quality(
    model: xgb.XGBClassifier,
    feature_cols: list[str],
    feature_row: pd.DataFrame,
    reference: dict | None = None,
    top_n: int = 3,
) -> dict:
    """
    Decompose one shot-quality (make-probability) prediction by side of the
    matchup: offense, defense, matchup interaction, and shot/game context.

    `feature_row` must already be the exact single row that was scored
    (post `apply_hierarchy`/`derive_features`, one specific mechanic — not a
    marginalised mixture, which has no single feature vector to attribute a
    contribution to). Built via `ShotRecommender._build_feature_frame`, the
    same assembly logic `recommend()` uses, so the explanation always
    describes the number that was actually served.

    Returns a decomposition additive by construction (TreeSHAP): bias +
    offense_effect + defense_effect + matchup_effect + context_effect sums
    to the predicted probability.
    """
    X = feature_row[feature_cols].astype(float)
    booster = model.get_booster()
    contribs = booster.predict(
        xgb.DMatrix(X, missing=np.nan), pred_contribs=True
    )[0]

    bias = float(contribs[-1])
    per_feature = {col: float(contribs[i]) for i, col in enumerate(feature_cols)}
    sides = _side_columns(feature_cols)

    effects = {"offense": 0.0, "defense": 0.0, "matchup": 0.0, "context": 0.0}
    for col, contribution in per_feature.items():
        effects[sides[col]] += contribution

    predicted = float(np.clip(bias + sum(effects.values()), 0.0, 1.0))
    # What this shooter would be projected at from here against a LEAGUE
    # AVERAGE defender: bias + context + offense, with the defense/matchup
    # terms (which describe the NAMED defender specifically) held out. This
    # is the number the defender's presence is measured against.
    offense_only = float(np.clip(bias + effects["context"] + effects["offense"], 0.0, 1.0))

    reference = reference or {}

    def _factors_for(side: str) -> list[dict]:
        items = []
        for col, contribution in per_feature.items():
            if sides[col] != side or abs(contribution) < 1e-6:
                continue
            copy = SQ_FEATURE_COPY.get(col) or FEATURE_COPY.get(col)
            if copy is None:
                continue
            value = feature_row.iloc[0].get(col)
            value = None if pd.isna(value) else float(value)
            pct = _sq_percentile(value, reference.get(col, {}))
            # Unlike attainability's model, shot-quality's metadata carries no
            # feature_reference quantile grid — every percentile lookup here
            # returns None. Defaulting to `copy["low"]` in that case (the
            # naive `pct is not None and pct >= 50` reads as False when pct is
            # None) would assert "this is a low value" for every single
            # factor regardless of what the value actually is — a comparison
            # this function cannot support without real reference data.
            # Honest fallback: no percentile, no high/low claim.
            trait = None
            if pct is not None:
                trait = copy["high"] if pct >= 50 else copy["low"]
            items.append({
                "feature": col,
                "label": copy["label"],
                "value": value,
                "display_value": _format_value(value, copy["fmt"]),
                "percentile": pct,
                "impact": round(contribution, 5),
                "direction": "raises" if contribution > 0 else "lowers",
                "detail": trait,
            })
        items.sort(key=lambda f: -abs(f["impact"]))
        return items[:top_n]

    return {
        "make_probability": round(predicted, 4),
        "bias": round(bias, 5),
        "offense_effect": round(effects["offense"], 5),
        "defense_effect": round(effects["defense"], 5),
        "matchup_effect": round(effects["matchup"], 5),
        "context_effect": round(effects["context"], 5),
        "offense_only_probability": round(offense_only, 4),
        "defender_swing": round(predicted - offense_only, 4),
        "offense_factors": _factors_for("offense"),
        "defense_factors": _factors_for("defense") + _factors_for("matchup"),
    }


def _pct_points(p: float) -> str:
    return f"{p * 100:.0f}%"


def build_matchup_narrative(
    sq: dict,
    attainability: dict | None,
    attacker_name: str,
    zone: str,
    points: int,
    defender_name: str | None = None,
    secondary_defender_name: str | None = None,
    is_league_average_defender: bool = False,
) -> str:
    """
    Compose the full matchup narrative: make probability (offense vs
    defense), expected points as a consequence of it, attainability woven in
    rather than standing alone, and a double-team clause when a second
    defender is named.

    This is the function that turns four separately-computed numbers
    (make_probability, expected_points, attainability, and — when present —
    the double-team heuristic) into one paragraph a reader can follow, rather
    than four facts they have to reconcile themselves.
    """
    zone_lower = zone.lower()
    make_pct = _pct_points(sq["make_probability"])
    ep = sq["make_probability"] * points

    sentences = []

    # ── 1. Offense vs defense on make probability ─────────────────────────
    if is_league_average_defender or defender_name is None:
        off_pct = _pct_points(sq["offense_only_probability"])
        sentences.append(
            f"{attacker_name} projects at {make_pct} from the {zone_lower} "
            f"against a league-average defender ({off_pct} is his own baseline "
            f"here before any specific matchup is applied)."
        )
    else:
        off_pct = _pct_points(sq["offense_only_probability"])
        swing = sq["defender_swing"]
        swing_pp = abs(round(swing * 100))
        direction = "up" if swing > 0 else "down"
        if swing_pp < 1:
            sentences.append(
                f"{attacker_name} projects at {make_pct} from the {zone_lower} "
                f"against {defender_name} — essentially his own baseline here "
                f"({off_pct}); this particular matchup barely moves it either way."
            )
        else:
            named = [f for f in sq["defense_factors"] if f.get("value") is not None][:2]
            drivers = ""
            if named:
                drivers = ", mostly " + " and ".join(
                    f"{d['label'].lower()} ({d['display_value']})" for d in named
                )
            sentences.append(
                f"{attacker_name} shoots {off_pct} from the {zone_lower} against a "
                f"league-average defender; against {defender_name} specifically that "
                f"moves {direction} {swing_pp} points to {make_pct}{drivers}."
            )

    # ── 2. Double-team clause ───────────────────────────────────────────
    if secondary_defender_name:
        sentences.append(
            f"With {secondary_defender_name} also involved, this uses the tougher "
            f"of the two defenders' numbers per category as a conservative floor — "
            f"there is no genuinely double-teamed shot in the training data to learn "
            f"a real suppression effect from, so this is a documented heuristic, not "
            f"a learned one."
        )

    # ── 3. Expected points as a consequence, not a separate fact ───────────
    sentences.append(
        f"At {points} points for a make, that make probability is worth "
        f"{ep:.2f} expected points from this spot."
    )

    # ── 4. Attainability woven in ──────────────────────────────────────────
    if attainability is not None:
        att_pct = _pct_points(attainability["attainability"])
        att_summary = attainability.get("summary", "")
        sentences.append(
            f"How often {attacker_name} actually gets a look like this: {att_pct} "
            f"of his shot diet. {att_summary}"
        )

    return " ".join(sentences)
