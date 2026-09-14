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
    _percentile,
    _format_value,
    _share_text,
    _driver_clause,
    _summarize,
)
from .shot_quality import (
    SQ_FEATURE_COPY,
    explain_shot_quality,
    build_matchup_narrative,
    _sigmoid,
    _sq_percentile,
    _pct_points,
    _side_columns,
    _group_columns,
)

__all__ = [
    "FEATURE_COPY", "SQ_FEATURE_COPY",
    "ZONE_PREFIXES", "POSITION_PREFIX",
    "HISTORY_FEATURE", "HISTORY_FALLBACK", "HISTORY_SUPPORT", "POSITION_LABEL",
    "MIN_CREATION_MAKES", "NEGLIGIBLE_FREQ_GAP",
    "RARE_ZONE_SHARE", "NEGLIGIBLE_EFFECT",
    "creation_note", "defender_zone_tendency_note",
    "explain_attainability", "explain_shot_quality", "build_matchup_narrative",
]
