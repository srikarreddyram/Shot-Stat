"""JSON-safety helper shared by players.py and teams.py."""
import numpy as np


def _clean(value):
    """
    NaN -> None (bare NaN is not valid JSON — some clients tolerate it,
    browsers' own JSON.parse does not, so a NaN reaching the wire is a
    silent frontend crash waiting for whichever row hits it first) and
    numpy scalar -> native Python (FastAPI's default encoder chokes on
    numpy int64/float64 the same way it would on a Decimal).
    """
    if value is None:
        return None
    if isinstance(value, (np.floating, float)) and np.isnan(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
