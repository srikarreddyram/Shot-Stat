"""
TreeSHAP-backed plain-English explanations for the two models this project
serves: attainability (would this player take a shot like this) and shot
quality (would it go in).

This used to be one 961-line explain.py. It is now two files that mirror
the module's own docstring split — attainability.py and shot_quality.py —
plus this __init__.py re-exporting everything so every existing
`from src.inference.explain import X` keeps working unchanged.
"""
from .attainability import (
    FEATURE_COPY,
    ZONE_PREFIXES,
    POSITION_PREFIX,
    HISTORY_FEATURE,
    HISTORY_FALLBACK,
    HISTORY_SUPPORT,
    POSITION_LABEL,
    MIN_CREATION_MAKES,
    NEGLIGIBLE_FREQ_GAP,
    RARE_ZONE_SHARE,
    NEGLIGIBLE_EFFECT,
    creation_note,
    defender_zone_tendency_note,
    explain_attainability,
)
# Re-exported for tests reaching them as `explain._percentile` etc. (see
# tests/test_explain.py) — redundant-alias imports spell that out as
# intentional rather than an unused-import.
from .attainability import _percentile as _percentile
from .attainability import _format_value as _format_value
from .attainability import _share_text as _share_text
from .attainability import _driver_clause as _driver_clause
from .attainability import _summarize as _summarize
from .shot_quality import (
    SQ_FEATURE_COPY,
    explain_shot_quality,
    build_matchup_narrative,
)
from .shot_quality import _sigmoid as _sigmoid
from .shot_quality import _sq_percentile as _sq_percentile
from .shot_quality import _pct_points as _pct_points
from .shot_quality import _side_columns as _side_columns
from .shot_quality import _group_columns as _group_columns
from .comparable_shots import ComparableShotsIndex, build_comparable_shots_index

__all__ = [
    "FEATURE_COPY", "SQ_FEATURE_COPY",
    "ZONE_PREFIXES", "POSITION_PREFIX",
    "HISTORY_FEATURE", "HISTORY_FALLBACK", "HISTORY_SUPPORT", "POSITION_LABEL",
    "MIN_CREATION_MAKES", "NEGLIGIBLE_FREQ_GAP",
    "RARE_ZONE_SHARE", "NEGLIGIBLE_EFFECT",
    "creation_note", "defender_zone_tendency_note",
    "explain_attainability", "explain_shot_quality", "build_matchup_narrative",
    "ComparableShotsIndex", "build_comparable_shots_index",
]
