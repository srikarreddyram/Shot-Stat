"""
Attainability — can this player actually GET this shot? See train.py's
module docstring for the full explanation of what this model estimates and
why.

This used to be one 960-line attainability.py. It is now:
    features.py  — encoding shared by training and serving (sub-zones, the
                   one-hot columns, the fixed feature-column lists)
    serving.py   — point-in-time lookups for ONE player, called at request
                   time by ShotRecommender
    build.py     — training-matrix construction for the WHOLE league
    train.py     — the train() entrypoint
    __main__.py  — `python -m src.training.attainability --name ...`

This __init__.py re-exports everything so every existing
`from src.training.attainability import X` keeps working unchanged.
"""
from .features import (
    ANGLE_SPLIT_ZONES,
    ATTAINABILITY_FEATURE_COLS,
    CAST_FEATURE_COLS,
    DIET_SNAPSHOTS,
    POSITION_BUCKETS,
    PRIOR_DIET_COLS,
    SUB_ZONE_ANGLE_BOUNDARY,
    SUB_ZONE_SUFFIX,
    SUB_ZONES,
    attach_sub_zone,
    encode_zone_and_position,
    with_bare_zone_entries,
)
from .serving import lookup_diet_history, lookup_prior_diet
from .build import (
    attach_prior_diet,
    build_attainability_matrix,
    build_season_cast,
    build_zone_frequency_targets,
    build_diet_snapshots,
)
from .train import MODEL_DIR, train

__all__ = [
    "ANGLE_SPLIT_ZONES", "ATTAINABILITY_FEATURE_COLS", "CAST_FEATURE_COLS",
    "DIET_SNAPSHOTS", "POSITION_BUCKETS", "PRIOR_DIET_COLS",
    "SUB_ZONE_ANGLE_BOUNDARY", "SUB_ZONE_SUFFIX", "SUB_ZONES",
    "attach_sub_zone", "encode_zone_and_position", "with_bare_zone_entries",
    "lookup_diet_history", "lookup_prior_diet",
    "attach_prior_diet", "build_attainability_matrix", "build_season_cast",
    "build_zone_frequency_targets", "build_diet_snapshots",
    "MODEL_DIR", "train",
]
