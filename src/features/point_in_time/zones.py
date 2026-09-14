"""The six court zones and the defense-category mapping every other module
in this package (and spec.py, recommender.py, api.py) reads as the single
source of truth."""

# The six shot areas, matching PlayerZoneStats.zone and Shot.zone exactly.
ZONES = [
    "Restricted Area",
    "In The Paint (Non-RA)",
    "Mid-Range",
    "Left Corner 3",
    "Right Corner 3",
    "Above the Break 3",
]

THREE_POINT_ZONES = ["Left Corner 3", "Right Corner 3", "Above the Break 3"]

# Zone name → the suffix used in feature column names. Kept explicit rather
# than slugified on the fly so a zone rename upstream breaks loudly here
# instead of silently producing a new, unmatched feature column.
ZONE_SUFFIX = {
    "Restricted Area": "restricted_area",
    "In The Paint (Non-RA)": "paint",
    "Mid-Range": "midrange",
    "Left Corner 3": "left_corner_3",
    "Right Corner 3": "right_corner_3",
    "Above the Break 3": "above_break_3",
}

# Zone → the LeagueDashPtDefend category whose FG%-allowed best describes
# defending that zone. Single source of truth: the recommender, the API's
# /matchup endpoint, and the training join all read this. Defined here rather
# than in `spec.py` (which imports it back) so `build_defender_category_rates`
# can use it without spec.py importing point_in_time importing spec.
ZONE_TO_DEF_CATEGORY = {
    "Restricted Area": "Less Than 6Ft",
    "In The Paint (Non-RA)": "Less Than 10Ft",
    "Mid-Range": "Greater Than 15Ft",
    "Left Corner 3": "3 Pointers",
    "Right Corner 3": "3 Pointers",
    "Above the Break 3": "3 Pointers",
}

DEFENSE_CATEGORIES = sorted(set(ZONE_TO_DEF_CATEGORY.values()))
