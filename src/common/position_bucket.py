"""
Shared position-bucketing logic.

Used by both src/training/position_priors.py (to compute rookie-season
averages per bucket) and the recommender's stats resolver (to look up the
right bucket for a player with zero NBA history). Both sides MUST use this
same function — a mismatch would silently make priors unreachable.

Buckets are deliberately broad (G / F / C) rather than the 5 traditional
positions: splitting further shrinks the historical rookie sample per
bucket enough to make the averages noisy, which is the exact problem this
project's no-imputation policy exists to avoid.
"""

_PRIMARY_TO_BUCKET = {
    "PG": "G", "SG": "G", "G": "G",
    "SF": "F", "PF": "F", "F": "F",
    "C": "C",
}


def position_bucket(position: str | None) -> str | None:
    """Map a position string (e.g. 'F-C', 'PG') to a broad bucket: G, F, or C."""
    if not position or not position.strip():
        return None
    primary = position.strip().upper().split("-")[0]
    return _PRIMARY_TO_BUCKET.get(primary)


# Median height (inches), computed once from historical rookie-season data
# grouped by position_bucket (see src/training/position_priors.py, which
# printed these when the height-band split was added: G=76.0, F=80.0,
# C=83.0). Fixed rather than recomputed on the fly so a rookie looked up
# today lands in the same band a prior computed yesterday would predict —
# lookup time and prior-computation time MUST agree on the same thresholds.
HEIGHT_BAND_MEDIAN = {"G": 76.0, "F": 80.0, "C": 83.0}


def height_band(height: float | None, bucket: str | None) -> str | None:
    """
    Split a position bucket into "short" / "tall" halves by height, so a
    7'4" rookie center isn't averaged in with a 6'8" one. Returns None if
    height or bucket is unavailable/unrecognized — callers should treat
    that as "no finer bucket available, use the plain position bucket."
    """
    if height is None or bucket not in HEIGHT_BAND_MEDIAN:
        return None
    return "tall" if height > HEIGHT_BAND_MEDIAN[bucket] else "short"


def fine_bucket(height: float | None, position: str | None) -> str | None:
    """Combined position+height bucket key, e.g. 'C-tall'. None if either
    position or height can't be resolved (caller should fall back to the
    plain position_bucket, which always has broader historical coverage)."""
    bucket = position_bucket(position)
    band = height_band(height, bucket)
    if bucket is None or band is None:
        return None
    return f"{bucket}-{band}"
