"""
Point-in-time features — the leakage fix.

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

What this package does
-----------------------
For every quantity a shot's features need that could otherwise leak the
future into the past — a player's own shooting rate, a defender's quality,
how contested a player's shots have been, what his supporting cast gives
him, who else is on the floor — it computes that quantity from games
strictly BEFORE the game in question, at both a career-to-date and (where
meaningful) a season-to-date horizon. Two constructions of each quantity
exist throughout: a `build_*` function computing it in bulk for the whole
league (used at training time) and a `lookup_*` function computing it for
one player/request (used at serving time) — the parity between the two is
what tests/test_train_serve_parity.py exists to check.

This used to be one 1,701-line point_in_time.py. It is now:
    zones.py             — the six court zones and the defense-category map
    shooting_rates.py     — the three-level shrinkage hierarchy for a
                            player's own shooting rate
    defender_quality.py   — point-in-time defender FG%-allowed by category
    contest.py             — how contested a player's shots have been
    supporting_cast.py    — leave-one-out teammate shooting/passing
    creation_priors.py    — per-zone "did he create this himself" priors
    lineup_context.py     — the on-court lineup context (see its own
                            docstring for why it calls back into this
                            package via `point_in_time.X` rather than a
                            direct submodule import)
    __init__.py            — this file: re-exports everything

This __init__.py re-exports everything (including several underscore-
prefixed "private" helpers that tests and spark_jobs/export_shot_zone_counts.py
reach into directly) so every existing `from src.features.point_in_time
import X` — and every `import src.features.point_in_time as pit; pit.X` —
keeps working unchanged.
"""
from .zones import (
    DEFENSE_CATEGORIES,
    THREE_POINT_ZONES,
    ZONE_SUFFIX,
    ZONE_TO_DEF_CATEGORY,
    ZONES,
)
from .shooting_rates import (
    SEASON_TO_CAREER_STRENGTH,
    _load_player_game_zone_counts,
    apply_hierarchy,
    build_prior_counts,
    build_rolling_form,
    fit_league_zone_priors,
    lookup_prior_counts,
    lookup_recent_form,
)
from .defender_quality import (
    _load_defender_exposure,
    apply_opponent_defence,
    build_defender_category_rates,
    build_opponent_zone_defence,
    fit_league_category_priors,
    league_average_defender,
    lookup_defender_category_rates,
)
from .contest import (
    CONTEST_BAND_SUFFIX,
    CONTEST_BANDS,
    MIN_CONTEST_ATTEMPTS,
    build_contest_history,
)
from .supporting_cast import (
    MIN_CAST_ATTEMPTS,
    build_supporting_cast,
    lookup_supporting_cast,
)
from .creation_priors import fit_league_creation_priors, lookup_zone_creation
from .clutch_performance import (
    CLUTCH_MARGIN,
    CLUTCH_TIME_REMAINING,
    build_clutch_performance,
    fit_league_clutch_prior,
    lookup_clutch_performance,
)
from .lineup_context import (
    LINEUP_FEATURE_COLS,
    _resolve_recent_lineup_ids,
    _resolve_roster_fallback_ids,
    build_lineup_context,
    lookup_lineup_context,
)

__all__ = [
    "ZONES", "THREE_POINT_ZONES", "ZONE_SUFFIX", "ZONE_TO_DEF_CATEGORY",
    "DEFENSE_CATEGORIES",
    "SEASON_TO_CAREER_STRENGTH",
    "build_prior_counts", "build_rolling_form", "fit_league_zone_priors",
    "apply_hierarchy", "lookup_prior_counts", "lookup_recent_form",
    "league_average_defender", "build_opponent_zone_defence",
    "apply_opponent_defence", "fit_league_category_priors",
    "build_defender_category_rates", "lookup_defender_category_rates",
    "CONTEST_BANDS", "CONTEST_BAND_SUFFIX", "MIN_CONTEST_ATTEMPTS",
    "build_contest_history",
    "MIN_CAST_ATTEMPTS", "build_supporting_cast", "lookup_supporting_cast",
    "fit_league_creation_priors", "lookup_zone_creation",
    "LINEUP_FEATURE_COLS", "build_lineup_context", "lookup_lineup_context",
    "CLUTCH_TIME_REMAINING", "CLUTCH_MARGIN", "fit_league_clutch_prior",
    "build_clutch_performance", "lookup_clutch_performance",
]
