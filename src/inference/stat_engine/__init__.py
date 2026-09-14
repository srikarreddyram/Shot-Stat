"""
Stat Engine — every stat this project tracks, browsable per player and per
team, basic through advanced. Distinct from the `/stats/features` route
(src/inference/api/model_features.py): that page answers "what does the
MODEL use and how much"; this one answers "what does this PLAYER (or team)
actually do", independent of whether any of it ends up in a model at all.

Design
------
One wide table per player (`build_player_stat_table`) merging every source
this project has already built for other reasons — box-score/physical
identity (`players`), zone-level shooting (`player_zone_stats`), handle/
passing/rim-pressure tracking (`creation.load_creation_profiles`), defensive
activity (`defensive_activity.load_defensive_activity_profiles`), FG%-allowed
by category (`defender_stats`), our own computed ratings
(`player_ratings.rating_lookup`), and NBA 2K's (`player_two_k_ratings`).
None of these are recomputed here — this module is a read-only join over
data other modules already own, on purpose: two sources of truth for the
same number is exactly the class of bug this project has hit and fixed
several times already (def_freq_zone, the bare-zone bugs).

Unlike the creation/defensive-activity profiles' USE elsewhere in this
project (always lagged one season for leak-free training features — see
creation.py's module docstring), this reads them for the season they
describe with NO lag: a stats browser is reporting what happened, not
predicting a future shot, so there is nothing to leak.

metadata.STAT_GROUPS documents every column's section and label once, so
the API and a client can agree on how to organize the page without either
one hand-maintaining a duplicate list.

This used to be one 758-line stat_engine.py. It is now:
    metadata.py    — the STAT_GROUPS/QUALIFIERS catalogue and the two
                     *_column_metadata functions
    formatting.py  — the JSON-safety `clean()` helper
    players.py     — build_player_stat_table, player_full_profile
    teams.py       — build_team_stat_table, team_full_profile

This __init__.py re-exports everything so every existing
`from src.inference.stat_engine import X` keeps working unchanged.
"""
from .metadata import (
    DEF_CATEGORY_LABEL,
    DEF_CATEGORY_SLUG,
    GROUP_LABELS,
    GROUP_ORDER,
    GROUP_SIDE,
    PLAY_TYPE_LABEL,
    PLAY_TYPE_SLUG,
    QUALIFIERS,
    SHOT_CONTEXT_LABEL,
    SHOT_CONTEXT_SLUG,
    STAT_GROUPS,
    TEAM_STAT_LABELS,
    ZONE_SLUG,
    player_column_metadata,
    team_column_metadata,
)
from .formatting import _clean
from .players import build_player_stat_table, player_full_profile
from .teams import build_team_stat_table, team_full_profile

__all__ = [
    "ZONE_SLUG", "SHOT_CONTEXT_SLUG", "PLAY_TYPE_SLUG", "PLAY_TYPE_LABEL",
    "SHOT_CONTEXT_LABEL", "QUALIFIERS", "STAT_GROUPS",
    "DEF_CATEGORY_SLUG", "DEF_CATEGORY_LABEL",
    "GROUP_ORDER", "GROUP_LABELS", "GROUP_SIDE", "TEAM_STAT_LABELS",
    "player_column_metadata", "team_column_metadata",
    "build_player_stat_table", "player_full_profile",
    "build_team_stat_table", "team_full_profile",
]
