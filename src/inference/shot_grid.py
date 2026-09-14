"""The candidate shot-location grid `ShotRecommender` scores every request
against — pure geometry, no database, no model, no `self`. Split out of
recommender.py because it's a genuinely standalone piece: nothing else in
that file depends on how the grid is built, only on the fact that
`SHOT_GRID` exists.
"""
import numpy as np

ZONE_POINTS = {
    "Restricted Area": 2, "In The Paint (Non-RA)": 2, "Mid-Range": 2,
    "Left Corner 3": 3, "Right Corner 3": 3, "Above the Break 3": 3,
}


# Observed shot-distance range (feet) per zone, from every shot 2016-17
# onward. The analytic grid below can otherwise place candidates outside the
# region where shots of that type are actually attempted — restricted-area
# points nearly five feet out, for instance — and the model has no training
# signal there, so it extrapolates. Filtering to the observed envelope keeps
# every scored location one the model has genuinely seen.
ZONE_DISTANCE_RANGE = {
    "Restricted Area": (0.0, 3.0),
    "In The Paint (Non-RA)": (4.0, 15.0),
    "Mid-Range": (8.0, 23.0),
    "Left Corner 3": (22.0, 26.0),
    "Right Corner 3": (22.0, 26.0),
    "Above the Break 3": (23.0, 32.0),
}


def _generate_court_grid() -> list[dict]:
    """
    Candidate shot locations across all six zones.

    Coordinates are NBA shot-chart units: the basket is (0, 0), ten units to
    the foot, y increasing toward half court. Zone membership is assigned by
    construction here rather than recomputed from coordinates, so a grid point
    can never disagree with the zone whose statistics it is scored against.
    """
    grid: list[dict] = []

    def add(loc_x, loc_y, zone, shot_type):
        distance = round(float(np.hypot(loc_x, loc_y)) / 10.0, 1)
        low, high = ZONE_DISTANCE_RANGE[zone]
        if not (low <= distance <= high):
            return
        grid.append({
            "loc_x": float(loc_x),
            "loc_y": float(loc_y),
            "shot_distance": distance,
            "zone": zone,
            "shot_type": shot_type,
        })

    for x in np.linspace(-35, 35, 8):
        for y in np.linspace(2, 35, 4):
            if np.hypot(x, y) <= 42:
                add(x, y, "Restricted Area", "2PT Field Goal")

    for x in np.linspace(-75, 75, 7):
        for y in np.linspace(40, 90, 5):
            if 42 < np.hypot(x, y) <= 100 and abs(x) <= 80:
                add(x, y, "In The Paint (Non-RA)", "2PT Field Goal")

    for angle in np.linspace(10, 170, 12):
        for r in (100, 130, 160, 190):
            x, y = r * np.cos(np.radians(angle)), r * np.sin(np.radians(angle))
            d = np.hypot(x, y) / 10.0
            if 9.0 < d < 22.0 and y > 0:
                add(x, y, "Mid-Range", "2PT Field Goal")

    for x in np.linspace(-235, -220, 4):
        for y in np.linspace(5, 85, 5):
            if np.hypot(x, y) / 10.0 >= 22.0 and y <= 93:
                add(x, y, "Left Corner 3", "3PT Field Goal")

    for x in np.linspace(220, 235, 4):
        for y in np.linspace(5, 85, 5):
            if np.hypot(x, y) / 10.0 >= 22.0 and y <= 93:
                add(x, y, "Right Corner 3", "3PT Field Goal")

    for angle in np.linspace(15, 165, 14):
        for r in (237, 250, 270):
            x, y = r * np.cos(np.radians(angle)), r * np.sin(np.radians(angle))
            if np.hypot(x, y) / 10.0 >= 23.0 and y > 93:
                add(x, y, "Above the Break 3", "3PT Field Goal")

    return grid


SHOT_GRID = _generate_court_grid()
