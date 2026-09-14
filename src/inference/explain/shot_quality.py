"""
Shot-quality (make-probability) explanation, and the matchup narrative that
weaves it together with attainability.

Everything in attainability.py explains ATTAINABILITY — whether this player
would take a shot like this. It never mentions a defender, because
attainability doesn't have one: it is a frequency question about the
shooter's own diet.

This module explains the OTHER model: given the shot is being taken, will
it go in. That model's feature vector already carries the defender (his
zone-level FG%-allowed, his matchup share, the offense/defense interaction
terms) — it was simply never decomposed into prose. The result was an
explanation UI that only ever talked about league averages and the
attacker's own history, with no defender in the story at all, for a
MATCHUP engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.spec import FEATURE_GROUPS
from .attainability import FEATURE_COPY, _driver_clause, _format_value, _percentile

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
_OFFENSE_GROUPS = ("shooter_physical", "shooter_skill", "creation",
                   "possession_origin", "team_creation")
_DEFENSE_GROUPS = ("defender", "defender_physical", "opponent_defence", "help_defense")
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

    # On-court lineup context (src/features/point_in_time.build_lineup_
    # context / lookup_lineup_context) — who else is on the floor, not the
    # shooter or the primary defender. "_max" entries are the single peak
    # threat among the other four on that side, not their average — see
    # that module's docstring for why averaging dilutes a lone star.
    "oncourt_off_creation": {"label": "Teammates' self-creation (avg)", "fmt": "num",
                             "high": "surrounded by shot-creators",
                             "low": "surrounded by non-creators"},
    "oncourt_off_creation_max": {"label": "Best teammate creator on the floor", "fmt": "num",
                                 "high": "has an elite self-creator alongside him",
                                 "low": "no real shot-creator alongside him"},
    "oncourt_off_gravity": {"label": "Teammates' playmaking gravity (avg)", "fmt": "num",
                            "high": "teammates draw real defensive attention",
                            "low": "teammates don't draw much attention"},
    "oncourt_off_gravity_max": {"label": "Best playmaker on the floor", "fmt": "num",
                                "high": "plays alongside an elite table-setter",
                                "low": "no real playmaking threat alongside him"},
    "oncourt_off_rim_pressure": {"label": "Teammates' rim pressure (avg)", "fmt": "num",
                                 "high": "teammates collapse the defense off drives",
                                 "low": "teammates don't pressure the rim"},
    "oncourt_off_rim_pressure_max": {"label": "Best downhill threat on the floor", "fmt": "num",
                                     "high": "has a real driving threat alongside him",
                                     "low": "no downhill pressure alongside him"},
    "oncourt_off_foul_rate_max": {"label": "Best foul-drawer on the floor", "fmt": "pct",
                                  "high": "a teammate draws heavy contact, forcing early help",
                                  "low": "no real foul-drawing threat alongside him"},
    "oncourt_def_fg_pct": {"label": "Help defenders' FG% allowed here (avg)", "fmt": "pct",
                           "high": "help defense is soft here",
                           "low": "help defense is stout here"},
    "oncourt_def_fg_pct_min": {"label": "Toughest help defender here", "fmt": "pct",
                               "high": "even the best help defender is beatable here",
                               "low": "a lockdown help defender patrols this shot"},
    "oncourt_def_blk_max": {"label": "Best shot-blocker on the floor", "fmt": "num",
                            "high": "an elite rim protector is patrolling the paint",
                            "low": "no real rim protection on the floor"},
    "oncourt_def_stl_max": {"label": "Most disruptive defender (steals)", "fmt": "num",
                            "high": "a high-steal defender is on the floor",
                            "low": "no real ball-pressure on the floor"},
    "oncourt_def_deflections_max": {"label": "Most disruptive defender (deflections)",
                                    "fmt": "num",
                                    "high": "a hand-in-every-lane defender is on the floor",
                                    "low": "no real disruption on the floor"},
    "oncourt_def_gravity_max": {"label": "Best defensive disruptor on the floor", "fmt": "num",
                                "high": "an elite disruptor changes what's attempted",
                                "low": "no real defensive deterrent on the floor"},

    # Game-state context. These were always live model inputs (the "context"
    # FEATURE_GROUPS bucket) with real TreeSHAP contributions, but nothing in
    # this module gave them a label or a narrative clause — the explanation
    # only ever talked about offense/defense, so a shot whose make probability
    # moved because of the score or the clock told a story that silently left
    # that part out. Confirmed real gap the caller flagged directly.
    "score_diff": {"label": "Score margin", "fmt": "signed_int",
                  "high": "team comfortably ahead", "low": "team behind or tied"},
    "clutch_flag": {"label": "Clutch situation (4th, under 2min, within 5)", "fmt": "bool",
                    "high": "a clutch-time possession", "low": "not a clutch situation"},
    # The game clock. `seconds_remaining_in_game` is the single derived
    # feature the model leans on most for this (one number instead of
    # forcing the trees to reconstruct it via repeated quarter/time_remaining
    # splits — see spec/__init__.py's own comment), but quarter and
    # time_remaining are separate live columns too, so each gets its own
    # label rather than only the derived one showing up.
    "seconds_remaining_in_game": {
        "label": "Time left in the game", "fmt": "sec",
        "high": "early — plenty of time left", "low": "late in the game, time running out",
    },
    "quarter": {"label": "Quarter", "fmt": "int",
               "high": "later in the game", "low": "earlier in the game"},
    "time_remaining": {"label": "Time left in the period", "fmt": "sec",
                       "high": "early in the period", "low": "period winding down"},
    # Whether THIS PLAYER specifically raises or lowers his game in the
    # clutch, relative to his own normal baseline — not just whether the
    # moment itself is clutch (clutch_flag, above). Exactly zero on every
    # non-clutch shot by construction (see point_in_time.clutch_performance),
    # so this only ever appears in a clutch-situation explanation, and its
    # displayed value is the player's actual clutch-vs-normal FG% gap.
    "clutch_edge": {
        "label": "Clutch performer (vs. his own baseline)", "fmt": "pct",
        "high": "historically raises his game in the clutch",
        "low": "historically shoots worse in the clutch than usual",
    },
    # NOT literal shot-clock-remaining — the NBA's public stats API (the only
    # source this project ingests from) never publishes that as a per-shot
    # field; it only exists as a season-aggregate FG% split by clock BUCKET
    # (playerdashptshots' ShotClockShooting dataset), which cannot say what
    # THIS shot's actual clock read. `seconds_since_prev_event` is the
    # closest per-shot proxy this project actually has for "was this shot
    # rushed or a set-defense possession" — already a live model feature
    # (see shot_context.py's own docstring), just never surfaced in prose
    # until now. Labeled honestly as what it is, not as a real shot clock.
    "seconds_since_prev_event": {
        "label": "Time since last live-ball event (shot-clock proxy)", "fmt": "sec",
        "high": "came off a long, set-defense possession",
        "low": "came quickly — early clock or transition",
    },
}


def _sigmoid(margin: float) -> float:
    """
    Log-odds -> probability. XGBoost's `binary:logistic` TreeSHAP
    contributions are additive in log-odds, so every sum of them has to come
    through here before it can be called a probability.
    """
    return float(1.0 / (1.0 + np.exp(-margin)))


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

    TreeSHAP contributions for a `binary:logistic` model are additive in
    LOG-ODDS, not in probability: they sum to the margin, which has to go
    through a sigmoid to become the number the model actually predicts.
    Treating that sum as a probability directly — which this function did
    until it was caught — silently reports the margin as if it were a rate.
    It looks merely wrong when the margin lands inside [0, 1] (a rim shot
    read 0.48 when the model said 0.62) and absurd when it doesn't (Stephen
    Curry above the break clipped to 0%, because his margin was -0.30).

    So: group effects are summed in log-odds, and every number reported to a
    caller is converted to probability. Per-feature `impact` is a MARGINAL
    effect — how far the probability moves when that one feature's
    contribution is taken out — which is interpretable in percentage points
    but, unlike the log-odds contributions underneath it, does NOT sum
    exactly to the total. That is a property of a logistic model, not a bug
    to reconcile away.
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

    margin = bias + sum(effects.values())
    predicted = _sigmoid(margin)
    # What this shooter would be projected at from here against a LEAGUE
    # AVERAGE defender: bias + context + offense, with the defense/matchup
    # terms (which describe the NAMED defender specifically) held out. This
    # is the number the defender's presence is measured against.
    offense_only = _sigmoid(bias + effects["context"] + effects["offense"])

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
            # Marginal effect in PROBABILITY, not the raw log-odds
            # contribution: how far the estimate moves when this one
            # feature's contribution is removed. The UI prints these as
            # percentage points, which the log-odds number is not.
            impact = predicted - _sigmoid(margin - contribution)
            items.append({
                "feature": col,
                "label": copy["label"],
                "value": value,
                "display_value": _format_value(value, copy["fmt"]),
                "percentile": pct,
                "impact": round(impact, 5),
                "log_odds_contribution": round(contribution, 5),
                "direction": "raises" if contribution > 0 else "lowers",
                "detail": trait,
            })
        items.sort(key=lambda f: -abs(f["impact"]))
        return items[:top_n]

    return {
        "make_probability": round(predicted, 4),
        # Log-odds, explicitly named as such. These are the quantities that
        # really are additive (they sum with `bias` to the margin); the
        # probabilities above are what that margin becomes after a sigmoid.
        "bias": round(bias, 5),
        "offense_effect_log_odds": round(effects["offense"], 5),
        "defense_effect_log_odds": round(effects["defense"], 5),
        "matchup_effect_log_odds": round(effects["matchup"], 5),
        "context_effect_log_odds": round(effects["context"], 5),
        "offense_only_probability": round(offense_only, 4),
        "defender_swing": round(predicted - offense_only, 4),
        "offense_factors": _factors_for("offense"),
        "defense_factors": _factors_for("defense") + _factors_for("matchup"),
        "context_factors": _factors_for("context"),
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

    # ── 1b. Why the baseline is what it is — the shooter's own profile plus
    # who else is on the floor with him (team_creation), the offense-side
    # half of the matchup the narrative previously never spoke about at
    # all: only the defender's swing (1, above) was ever quoted in prose,
    # leaving a real playmaking teammate's gravity invisible even when it
    # was a top-ranked SHAP contributor to the exact same prediction.
    offense_drivers = _driver_clause(sq["offense_factors"])
    if offense_drivers:
        sentences.append(f"That baseline reflects his own scoring profile{offense_drivers}.")

    # ── 1c. Game-state context (score margin, clutch, shot-clock proxy) —
    # these are real model inputs with real SHAP contributions to the same
    # prediction above, but until now nothing here ever named them, so a
    # shot whose probability moved because of the score or the clock told a
    # story that silently left that part out.
    context_drivers = _driver_clause(sq["context_factors"])
    if context_drivers:
        sentences.append(f"Game situation matters here too{context_drivers}.")

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
        # The comparison attainability's own model can't make: not just how
        # often the attacker shoots here, but whether THIS defender's
        # opponents attack this zone more or less than a typical defender's
        # do. Only present when a real (non-league-average) defender was
        # named — see ShotRecommender.explain_attainability.
        defender_tendency = (attainability.get("defender") or {}).get("note")
        if defender_tendency:
            sentences.append(defender_tendency)

    return " ".join(sentences)
