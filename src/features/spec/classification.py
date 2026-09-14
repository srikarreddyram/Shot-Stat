"""
Rule-based classifiers over raw play-by-play text: what CREATED a shot
(mechanic), what it physically WAS (finish), and how the possession began
(origin). All three are first-match-wins ordered scans over substring
rules, and all three round-trip through their own classifier — a bucket
name maps to itself — because the serving path assigns a name to a
hypothetical grid row and re-classifies it rather than reimplementing the
encoding.
"""
from __future__ import annotations

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
