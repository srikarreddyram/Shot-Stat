"""
The Stat Engine's catalogue: every stat column this project can emit, its
section, its reader-facing label, its format, and (where relevant) the
volume qualifier a leaderboard needs before ranking on it.

This one file is what lets the API and a client agree on how to organize
the stats page without either one hand-maintaining a duplicate list — see
`player_column_metadata`/`team_column_metadata`, and players.py/teams.py's
`*_full_profile` functions, all of which read STAT_GROUPS rather than a
column list of their own.
"""
from __future__ import annotations

ZONE_SLUG = {
    "Restricted Area": "rim",
    "In The Paint (Non-RA)": "paint",
    "Mid-Range": "midrange",
    "Left Corner 3": "left_corner3",
    "Right Corner 3": "right_corner3",
    "Above the Break 3": "above_break3",
}

# player_shot_profile carries the same four (player_id, season, split_type,
# split_value) shape as player_zone_stats, but the split is shot DIFFICULTY
# CONTEXT rather than court location: how contested the shot was, how many
# dribbles preceded it, how long the ball was held, and whether it was off
# the catch or off the dribble. creation.py already reads this table for
# open_share/tight_share/pullup_share/catch_shoot_share — the SHARE of a
# player's diet in each bucket. What's added here is the other half: how
# well they SHOOT in each bucket, which is a different question from how
# often they end up there, and neither is redundant with the other.
SHOT_CONTEXT_SLUG: dict[tuple[str, str], str] = {
    ("def_dist", "0-2 Feet - Very Tight"): "very_tight",
    ("def_dist", "2-4 Feet - Tight"): "tight",
    ("def_dist", "4-6 Feet - Open"): "open",
    ("def_dist", "6+ Feet - Wide Open"): "wide_open",
    ("dribbles", "0 Dribbles"): "catch_and_fire",
    ("dribbles", "1 Dribble"): "one_dribble",
    ("dribbles", "2 Dribbles"): "two_dribbles",
    ("dribbles", "3-6 Dribbles"): "three_to_six_dribbles",
    ("dribbles", "7+ Dribbles"): "seven_plus_dribbles",
    ("general", "Catch and Shoot"): "catch_shoot",
    ("general", "Pull Ups"): "pullup",
    ("touch_time", "Touch < 2 Seconds"): "quick_touch",
    ("touch_time", "Touch 2-6 Seconds"): "medium_touch",
    ("touch_time", "Touch 6+ Seconds"): "long_touch",
}
PLAY_TYPE_SLUG: dict[str, str] = {
    "Isolation": "isolation",
    "Transition": "transition",
    "PRBallHandler": "pnr_ball_handler",
    "PRRollman": "pnr_roll_man",
    "Postup": "postup",
    "Spotup": "spotup",
    "Handoff": "handoff",
    "Cut": "cut",
    "OffScreen": "off_screen",
    "OffRebound": "putback",
}
PLAY_TYPE_LABEL: dict[str, str] = {
    "isolation": "Isolation",
    "transition": "Transition",
    "pnr_ball_handler": "Pick-and-roll (ball-handler)",
    "pnr_roll_man": "Pick-and-roll (roll man)",
    "postup": "Post-up",
    "spotup": "Spot-up",
    "handoff": "Hand-off",
    "cut": "Cut",
    "off_screen": "Off-screen",
    "putback": "Putback (off. rebound)",
}

SHOT_CONTEXT_LABEL: dict[str, str] = {
    "very_tight": "Very tightly contested (0-2 ft)",
    "tight": "Tightly contested (2-4 ft)",
    "open": "Open (4-6 ft)",
    "wide_open": "Wide open (6+ ft)",
    "catch_and_fire": "Off the catch, no dribble",
    "one_dribble": "Off one dribble",
    "two_dribbles": "Off two dribbles",
    "three_to_six_dribbles": "Off 3-6 dribbles",
    "seven_plus_dribbles": "Off 7+ dribbles",
    "catch_shoot": "Catch-and-shoot",
    "pullup": "Pull-up",
    "quick_touch": "Quick release (< 2s holding)",
    "medium_touch": "Held 2-6 seconds",
    "long_touch": "Held 6+ seconds",
}

# Rate stats are paired with the column counting the attempts behind them,
# and a minimum volume to be RANKED. Without this a "best rim defender"
# leaderboard is topped by whoever defended one shot and got a stop — the
# same failure the zone FG% leaderboard had before attempts were shown. An
# unqualified player still shows their number; they just do not get a rank,
# which is how every real stat leaderboard handles a thin sample.
QUALIFIERS: dict[str, tuple[str, int]] = {
    "def_fg_pct_allowed": ("def_fga_overall", 150),
    "def_plus_minus": ("def_fga_overall", 150),
    **{f"def_{slug}_{suffix}": (f"def_{slug}_fga", minimum)
       for slug, minimum in [("rim", 60), ("paint", 60), ("two_pt", 100),
                             ("mid_long", 60), ("perimeter", 60)]
       for suffix in ("fg_pct", "pm")},
    **{f"zone_fg_pct_{slug}": (f"zone_fga_{slug}", minimum)
       for slug, minimum in [("rim", 50), ("paint", 30), ("midrange", 30),
                             ("left_corner3", 15), ("right_corner3", 15),
                             ("above_break3", 50)]},
    "season_fg_pct": ("games_played", 15),
    "career_fg_pct": ("games_played", 15),
    "career_3p_pct": ("games_played", 15),
    "ft_pct": ("games_played", 15),
    **{f"shotctx_{slug}_fg_pct": (f"shotctx_{slug}_fga", 20)
       for slug in SHOT_CONTEXT_SLUG.values()},
    # PPP is only meaningful once a play type is a real part of a player's
    # offense — a possession or two of garbage-time isolation is not a
    # measurement of anyone's isolation game. Synergy's own site applies a
    # similar minimum before it will even display a play type on a card.
    **{f"playtype_{slug}_ppp": (f"playtype_{slug}_poss", 25)
       for slug in PLAY_TYPE_SLUG.values()},
    **{f"playtype_{slug}_fg_pct": (f"playtype_{slug}_poss", 25)
       for slug in PLAY_TYPE_SLUG.values()},
}

# One entry per stat column this module can emit, keyed by the column name
# in build_player_stat_table's output. "group" orders the profile page
# basic-to-advanced; "label" is the reader-facing name; "fmt" matches
# explain.py's _format_value vocabulary, plus "count" for whole-number
# tallies (attempts, roster size) so a table column does not mix "14" and
# "14.0" down its length.
STAT_GROUPS: dict[str, dict] = {
    # ── Overview ────────────────────────────────────────────────────────
    "height": {"group": "overview", "label": "Height", "fmt": "in"},
    "weight": {"group": "overview", "label": "Weight", "fmt": "count"},
    "wingspan": {"group": "overview", "label": "Wingspan", "fmt": "in"},
    "position": {"group": "overview", "label": "Position", "fmt": "text"},
    "games_played": {"group": "overview", "label": "Games played", "fmt": "count"},
    "min_per_game": {"group": "overview", "label": "Minutes per game", "fmt": "num"},
    "off_rating": {"group": "overview", "label": "Offensive rating", "fmt": "num"},
    "def_rating_ours": {"group": "overview", "label": "Defensive rating", "fmt": "num"},
    # ── Shooting (basic) ────────────────────────────────────────────────
    "career_fg_pct": {"group": "shooting", "label": "Career FG%", "fmt": "pct"},
    "season_fg_pct": {"group": "shooting", "label": "Season FG%", "fmt": "pct"},
    "career_3p_pct": {"group": "shooting", "label": "Career 3P%", "fmt": "pct"},
    "ft_pct": {"group": "shooting", "label": "FT%", "fmt": "pct"},
    # Each zone's percentage is paired with its attempt count on purpose: a
    # bare 60% is unreadable without knowing whether it is 3-for-5 or
    # 300-for-500, and a zone leaderboard sorted on percentage alone is
    # topped entirely by players with a handful of attempts.
    "zone_fg_pct_rim": {"group": "shooting", "label": "Restricted area FG%", "fmt": "pct"},
    "zone_fga_rim": {"group": "shooting", "label": "Restricted area attempts", "fmt": "count"},
    "zone_fg_pct_paint": {"group": "shooting", "label": "Paint (non-RA) FG%", "fmt": "pct"},
    "zone_fga_paint": {"group": "shooting", "label": "Paint (non-RA) attempts", "fmt": "count"},
    "zone_fg_pct_midrange": {"group": "shooting", "label": "Mid-range FG%", "fmt": "pct"},
    "zone_fga_midrange": {"group": "shooting", "label": "Mid-range attempts", "fmt": "count"},
    "zone_fg_pct_left_corner3": {"group": "shooting", "label": "Left corner 3 FG%", "fmt": "pct"},
    "zone_fga_left_corner3": {"group": "shooting", "label": "Left corner 3 attempts", "fmt": "count"},
    "zone_fg_pct_right_corner3": {"group": "shooting", "label": "Right corner 3 FG%", "fmt": "pct"},
    "zone_fga_right_corner3": {"group": "shooting", "label": "Right corner 3 attempts", "fmt": "count"},
    "zone_fg_pct_above_break3": {"group": "shooting", "label": "Above the break 3 FG%", "fmt": "pct"},
    "zone_fga_above_break3": {"group": "shooting", "label": "Above the break 3 attempts", "fmt": "count"},
    # ── Shot difficulty (advanced) ──────────────────────────────────────
    # These are FG% and volume WITHIN each shot-difficulty bucket — a
    # different question from creation.py's open_share/pullup_share/etc.,
    # which are the SHARE of a player's diet that falls in each bucket. Diet
    # says what kind of shots a player takes; this says how well they make
    # them. Populated by the loop below, keyed off SHOT_CONTEXT_SLUG.
    # ── Creation & playmaking (advanced) ───────────────────────────────
    "self_creation_index": {"group": "creation", "label": "Self-creation index", "fmt": "num"},
    "playmaking_gravity": {"group": "creation", "label": "Playmaking gravity", "fmt": "num"},
    "rim_pressure": {"group": "creation", "label": "Rim pressure", "fmt": "num"},
    "avg_drib_per_touch": {"group": "creation", "label": "Dribbles per touch", "fmt": "num"},
    "drives_per_min": {"group": "creation", "label": "Drives per minute", "fmt": "num"},
    "pullup_share": {"group": "creation", "label": "Pull-up share", "fmt": "pct"},
    "catch_shoot_share": {"group": "creation", "label": "Catch-and-shoot share", "fmt": "pct"},
    "open_share": {"group": "creation", "label": "Open-shot share", "fmt": "pct"},
    "tight_share": {"group": "creation", "label": "Tightly-contested share", "fmt": "pct"},
    "drive_pf_pct": {"group": "creation", "label": "Foul-drawn rate on drives", "fmt": "pct"},
    "ast": {"group": "creation", "label": "Assists per game", "fmt": "num"},
    "tov": {"group": "creation", "label": "Turnovers per game", "fmt": "num"},
    # NBA SportVU tracking's own passing-quality numbers — a potential assist
    # is a pass to a shot within one dribble of the catch; a secondary assist
    # is the pass one step before the assist. Neither requires the ensuing
    # shot to actually go in, so together with `ast` they separate "created a
    # good look" from "the look went in", the same distinction shot quality
    # draws for shooting.
    "potential_ast": {"group": "creation", "label": "Potential assists per game", "fmt": "num"},
    "secondary_ast": {"group": "creation", "label": "Secondary (hockey) assists per game", "fmt": "num"},
    "ast_points_created": {"group": "creation", "label": "Points created by assist, per game", "fmt": "num"},
    "ast_to_pass_pct": {"group": "creation", "label": "Assist-to-pass rate", "fmt": "pct"},
    # ── Box score (basic, from LeagueDashPlayerStats) ───────────────────
    # Ordinary counting stats — the numbers on a broadcast graphic — added
    # last of all despite being the most basic, because everything above
    # already answered a more specific question and points/rebounds sat in
    # an already-fetched API response unused the whole time (see
    # defensive_activity_ingestor.py's own note on this). fg3m/fg3a and
    # ftm/fta are volume behind season_fg_pct/career_3p_pct/ft_pct, the same
    # pairing every rate elsewhere in this module gets.
    "pts": {"group": "shooting", "label": "Points per game", "fmt": "num"},
    "fgm": {"group": "shooting", "label": "Field goals made, per game", "fmt": "num"},
    "fga": {"group": "shooting", "label": "Field goals attempted, per game", "fmt": "num"},
    "fg3m": {"group": "shooting", "label": "Three-pointers made, per game", "fmt": "num"},
    "fg3a": {"group": "shooting", "label": "Three-pointers attempted, per game", "fmt": "num"},
    "ftm": {"group": "shooting", "label": "Free throws made, per game", "fmt": "num"},
    "fta": {"group": "shooting", "label": "Free throws attempted, per game", "fmt": "num"},
    "oreb": {"group": "creation", "label": "Offensive rebounds per game", "fmt": "num"},
    "dreb": {"group": "defense", "label": "Defensive rebounds per game", "fmt": "num"},
    "reb": {"group": "defense", "label": "Total rebounds per game", "fmt": "num"},
    "pfd": {"group": "creation", "label": "Fouls drawn per game", "fmt": "num"},
    "pf": {"group": "defense", "label": "Personal fouls per game", "fmt": "num"},
    "dd2": {"group": "overview", "label": "Double-doubles this season", "fmt": "count"},
    "td3": {"group": "overview", "label": "Triple-doubles this season", "fmt": "count"},
    # ── Defense (advanced) ─────────────────────────────────────────────
    "blk_per_game": {"group": "defense", "label": "Blocks per game", "fmt": "num"},
    "stl_per_game": {"group": "defense", "label": "Steals per game", "fmt": "num"},
    "deflections_per_game": {"group": "defense", "label": "Deflections per game", "fmt": "num"},
    "def_fg_pct_allowed": {"group": "defense", "label": "Overall — FG% allowed", "fmt": "pct"},
    "def_plus_minus": {"group": "defense", "label": "Overall — vs league normal", "fmt": "pct"},
    "def_fga_overall": {"group": "defense", "label": "Overall — shots defended", "fmt": "count"},
    "def_rim_fg_pct": {"group": "defense", "label": "Rim (< 6 ft) — FG% allowed", "fmt": "pct"},
    "def_rim_pm": {"group": "defense", "label": "Rim (< 6 ft) — vs league normal", "fmt": "pct"},
    "def_rim_fga": {"group": "defense", "label": "Rim (< 6 ft) — shots defended", "fmt": "count"},
    "def_paint_fg_pct": {"group": "defense", "label": "Paint (< 10 ft) — FG% allowed", "fmt": "pct"},
    "def_paint_pm": {"group": "defense", "label": "Paint (< 10 ft) — vs league normal", "fmt": "pct"},
    "def_paint_fga": {"group": "defense", "label": "Paint (< 10 ft) — shots defended", "fmt": "count"},
    "def_two_pt_fg_pct": {"group": "defense", "label": "Two-point — FG% allowed", "fmt": "pct"},
    "def_two_pt_pm": {"group": "defense", "label": "Two-point — vs league normal", "fmt": "pct"},
    "def_two_pt_fga": {"group": "defense", "label": "Two-point — shots defended", "fmt": "count"},
    "def_mid_long_fg_pct": {"group": "defense", "label": "Long two (15+ ft) — FG% allowed", "fmt": "pct"},
    "def_mid_long_pm": {"group": "defense", "label": "Long two (15+ ft) — vs league normal", "fmt": "pct"},
    "def_mid_long_fga": {"group": "defense", "label": "Long two (15+ ft) — shots defended", "fmt": "count"},
    "def_perimeter_fg_pct": {"group": "defense", "label": "Perimeter (3PT) — FG% allowed", "fmt": "pct"},
    "def_perimeter_pm": {"group": "defense", "label": "Perimeter (3PT) — vs league normal", "fmt": "pct"},
    "def_perimeter_fga": {"group": "defense", "label": "Perimeter (3PT) — shots defended", "fmt": "count"},
    "blk_per_min": {"group": "defense", "label": "Blocks per minute", "fmt": "num"},
    "stl_per_min": {"group": "defense", "label": "Steals per minute", "fmt": "num"},
    "deflections_per_min": {"group": "defense", "label": "Deflections per minute", "fmt": "num"},
    "defensive_gravity": {"group": "defense", "label": "Defensive gravity", "fmt": "num"},
    # ── Hustle (NBA's own "Hustle Stats" category), split by side ───────
    # Effort/activity plays that box-score counting stats don't capture at
    # all: setting a screen that leads to a bucket, boxing out, diving for a
    # loose ball, taking a charge. All per game, from LeagueHustleStatsPlayer
    # — the same endpoint that already supplied `deflections_per_game` above;
    # these columns simply weren't read from that response until now.
    #
    # This one group genuinely mixes both ends of the floor, unlike every
    # other group here — so it is the one place a STAT's side differs from
    # its neighbours', and it is split into two groups (hustle_offense /
    # hustle_defense) rather than one, so GROUP_SIDE below can stay a
    # per-GROUP lookup instead of needing a per-stat override.
    "screen_ast": {"group": "hustle_offense", "label": "Screen assists per game", "fmt": "num"},
    "screen_ast_pts": {"group": "hustle_offense", "label": "Points off screen assists, per game", "fmt": "num"},
    "off_boxouts": {"group": "hustle_offense", "label": "Offensive box-outs per game", "fmt": "num"},
    "off_loose_balls_recovered": {"group": "hustle_offense", "label": "Offensive loose balls recovered per game", "fmt": "num"},
    # The "total" box-out/loose-ball figures (off + def combined) default to
    # the defense side: broadcast convention frames both as fundamentally
    # defensive/rebounding hustle plays, and putting them here rather than
    # inventing a third "mixed" bucket keeps the offense/defense split clean.
    "box_outs": {"group": "hustle_defense", "label": "Box-outs per game", "fmt": "num"},
    "def_boxouts": {"group": "hustle_defense", "label": "Defensive box-outs per game", "fmt": "num"},
    "loose_balls_recovered": {"group": "hustle_defense", "label": "Loose balls recovered per game", "fmt": "num"},
    "def_loose_balls_recovered": {"group": "hustle_defense", "label": "Defensive loose balls recovered per game", "fmt": "num"},
    "charges_drawn": {"group": "hustle_defense", "label": "Charges drawn per game", "fmt": "num"},
    "contested_shots": {"group": "hustle_defense", "label": "Shots contested per game", "fmt": "num"},
    "contested_shots_2pt": {"group": "hustle_defense", "label": "2PT shots contested per game", "fmt": "num"},
    "contested_shots_3pt": {"group": "hustle_defense", "label": "3PT shots contested per game", "fmt": "num"},
    # ── Ratings & the outside opinion ───────────────────────────────────
    "rating_source": {"group": "ratings", "label": "Rating source", "fmt": "text"},
    "two_k_overall": {"group": "ratings", "label": "NBA 2K overall", "fmt": "num"},
    "two_k_offense_avg": {"group": "ratings", "label": "NBA 2K offense (our rollup)", "fmt": "num"},
    "two_k_defense_avg": {"group": "ratings", "label": "NBA 2K defense (our rollup)", "fmt": "num"},
}

# The five non-Overall defence categories, keyed by LeagueDashPtDefend's own
# category name. Overall is deliberately absent: it keeps its original column
# names for the consumers that already read them.
DEF_CATEGORY_SLUG = {
    "Less Than 6Ft": "rim",
    "Less Than 10Ft": "paint",
    "2 Pointers": "two_pt",
    "Greater Than 15Ft": "mid_long",
    "3 Pointers": "perimeter",
}
DEF_CATEGORY_LABEL = {
    "rim": "Rim (< 6 ft)",
    "paint": "Paint (< 10 ft)",
    "two_pt": "Two-point",
    "mid_long": "Long two (15+ ft)",
    "perimeter": "Perimeter (3PT)",
}

GROUP_ORDER = [
    "overview", "shooting", "shot_difficulty", "play_type", "creation",
    "hustle_offense", "defense", "hustle_defense", "ratings",
]
GROUP_LABELS = {
    "overview": "Overview",
    "shooting": "Shooting",
    "shot_difficulty": "Shot Difficulty & Diet",
    "play_type": "Play Type (Synergy)",
    "creation": "Creation & Playmaking",
    "hustle_offense": "Hustle (Offense)",
    "defense": "Defense",
    "hustle_defense": "Hustle (Defense)",
    "ratings": "Ratings",
}

# Which side of the ball a group belongs on, for splitting a player's page
# into OFFENSE / DEFENSE views rather than one long undifferentiated list.
# "info" groups (bio, ratings) don't belong to either side — a client shows
# them in a persistent header instead of inside either tab.
GROUP_SIDE: dict[str, str] = {
    "overview": "info",
    "ratings": "info",
    "shooting": "offense",
    "shot_difficulty": "offense",
    "play_type": "offense",
    "creation": "offense",
    "hustle_offense": "offense",
    "defense": "defense",
    "hustle_defense": "defense",
}

# Registered once from PLAY_TYPE_SLUG for the same reason the shot-context
# buckets are: 10 play types x 4 columns retyped by hand is exactly the kind
# of list that drifts out of sync with its source map.
for _slug, _label in PLAY_TYPE_LABEL.items():
    STAT_GROUPS[f"playtype_{_slug}_poss_pct"] = {
        "group": "play_type", "label": f"{_label} — share of offense", "fmt": "pct",
    }
    STAT_GROUPS[f"playtype_{_slug}_poss"] = {
        "group": "play_type", "label": f"{_label} — possessions", "fmt": "count",
    }
    STAT_GROUPS[f"playtype_{_slug}_ppp"] = {
        "group": "play_type", "label": f"{_label} — points per possession", "fmt": "num",
    }
    STAT_GROUPS[f"playtype_{_slug}_fg_pct"] = {
        "group": "play_type", "label": f"{_label} — FG%", "fmt": "pct",
    }
del _slug, _label

# Registered here, once, from SHOT_CONTEXT_SLUG rather than hand-typed like
# the rest of STAT_GROUPS — 14 buckets x 2 columns is exactly the kind of
# list that silently drifts out of sync with its source map if retyped.
for _slug in SHOT_CONTEXT_SLUG.values():
    STAT_GROUPS[f"shotctx_{_slug}_fg_pct"] = {
        "group": "shot_difficulty", "label": f"{SHOT_CONTEXT_LABEL[_slug]} — FG%", "fmt": "pct",
    }
    STAT_GROUPS[f"shotctx_{_slug}_fga"] = {
        "group": "shot_difficulty", "label": f"{SHOT_CONTEXT_LABEL[_slug]} — attempts", "fmt": "count",
    }
del _slug

# Team columns are few enough and flat enough not to need sections. The
# roster_avg_ prefix is load-bearing, not cosmetic — see
# build_team_stat_table's docstring on why these must read as derived.
TEAM_STAT_LABELS: dict[str, dict] = {
    "def_rating": {"label": "Defensive rating", "fmt": "num", "derived": False, "side": "defense"},
    "roster_size": {"label": "Roster size", "fmt": "count", "derived": False, "side": "info"},
    "roster_avg_season_fg_pct": {"label": "Avg FG%", "fmt": "pct", "derived": True, "side": "offense"},
    "roster_avg_off_rating": {"label": "Avg offensive rating", "fmt": "num", "derived": True, "side": "offense"},
    "roster_avg_def_rating_ours": {"label": "Avg defensive rating", "fmt": "num", "derived": True, "side": "defense"},
    "roster_avg_self_creation_index": {"label": "Avg self-creation", "fmt": "num", "derived": True, "side": "offense"},
    "roster_avg_playmaking_gravity": {"label": "Avg playmaking gravity", "fmt": "num", "derived": True, "side": "offense"},
    "roster_avg_rim_pressure": {"label": "Avg rim pressure", "fmt": "num", "derived": True, "side": "offense"},
    "roster_avg_def_fg_pct_allowed": {"label": "Avg FG% allowed", "fmt": "pct", "derived": True, "side": "defense"},
    "roster_avg_blk_per_min": {"label": "Avg blocks/min", "fmt": "num", "derived": True, "side": "defense"},
    "roster_avg_stl_per_min": {"label": "Avg steals/min", "fmt": "num", "derived": True, "side": "defense"},
    "roster_avg_deflections_per_min": {"label": "Avg deflections/min", "fmt": "num", "derived": True, "side": "defense"},
}


def player_column_metadata(columns) -> list[dict]:
    """Describe the stat columns actually present in a built table, in
    basic-to-advanced order. The leaderboard's column list is derived from
    STAT_GROUPS here rather than retyped client-side — a duplicated column
    list is how a renamed stat quietly becomes an empty column."""
    present = set(columns)
    out = []
    for group in GROUP_ORDER:
        for col, meta in STAT_GROUPS.items():
            if meta["group"] == group and col in present:
                entry = {"key": col, "label": meta["label"], "fmt": meta["fmt"],
                         "group": group, "group_label": GROUP_LABELS[group],
                         "side": GROUP_SIDE[group]}
                qualifier = QUALIFIERS.get(col)
                if qualifier and qualifier[0] in present:
                    entry["qualify_key"], entry["qualify_min"] = qualifier
                out.append(entry)
    return out


def team_column_metadata(columns) -> list[dict]:
    """Same idea as player_column_metadata, flat (teams have ~10 columns)."""
    present = set(columns)
    return [
        {"key": col, "label": meta["label"], "fmt": meta["fmt"],
         "derived": meta["derived"], "side": meta["side"]}
        for col, meta in TEAM_STAT_LABELS.items() if col in present
    ]
