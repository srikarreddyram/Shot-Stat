"""
Spatial basis: radial basis functions over the half court, in FEET from the
basket.

Trees split on one axis at a time, so given only `loc_x` and `loc_y` the
fitted surface is a union of axis-aligned rectangles. That is why the court
render needed interpolation to look continuous — the underlying estimate
genuinely is blocky, and no amount of smoothing in the canvas changes what
the model believes.

Each basis function is a smooth bump centred somewhere on the floor, so a
single split on one of them carves out a circular region instead of a
rectangle, and sums of them approximate a smooth surface directly. The
centres are laid out in polar coordinates because shot difficulty varies far
more with distance from the rim than with left-right position, and because
the court is symmetric about the centre line.
"""
import numpy as np

_BASIS_RADII = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 23.5, 27.0]
_BASIS_ANGLES = [20.0, 60.0, 90.0, 120.0, 160.0]
SPATIAL_SIGMA_FT = 5.5


def _basis_centers() -> list[tuple[float, float]]:
    """Centres in feet, (x, y), basket at the origin."""
    centers = [(0.0, 0.0)]
    for radius in _BASIS_RADII[1:]:
        for angle in _BASIS_ANGLES:
            rad = np.radians(angle)
            centers.append((radius * np.cos(rad), radius * np.sin(rad)))
    return centers


SPATIAL_CENTERS = _basis_centers()
SPATIAL_FEATURES = [f"rbf_{i}" for i in range(len(SPATIAL_CENTERS))]
