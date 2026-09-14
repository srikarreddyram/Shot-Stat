"""The flat catalogue of every stat a career panel can report — (key,
label, group, fmt, kind) tuples — plus the section metadata (labels, order,
offense/defense/info side) a client uses to lay the page out."""
from __future__ import annotations

from .aggregate import COUNT, DIFF, PER_GAME, RATE
from .sources import (
    BOX_SCORE_COLUMNS,
    DEF_CATEGORIES,
    DEF_CATEGORY_ORDER,
    HUSTLE_STAT_COLUMNS,
    SEASON_TOTAL_COLUMNS,
)

GROUP_LABELS = {
    "availability": "Availability",
    "defense_activity": "Defensive activity",
    "defense_matchup": "Defending the shot",
    "shooting": "Shooting",
    "shot_difficulty": "Shot Difficulty & Diet",
    "play_type": "Play Type (Synergy)",
    "passing": "Passing Quality",
    "hustle_offense": "Hustle (Offense)",
    "hustle_defense": "Hustle (Defense)",
}
GROUP_ORDER = [
    "defense_activity", "defense_matchup", "hustle_defense", "shooting",
    "shot_difficulty", "play_type", "hustle_offense", "passing", "availability",
]

# Same offense/defense/info split as stat_engine.GROUP_SIDE, mirrored here so
# a career page can be divided the same way the season page is. "availability"
# (games played, minutes) is neither — shown outside either tab, same as
# "overview"/"ratings" on the season side.
GROUP_SIDE: dict[str, str] = {
    "availability": "info",
    "defense_activity": "defense",
    "defense_matchup": "defense",
    "hustle_defense": "defense",
    "shooting": "offense",
    "shot_difficulty": "offense",
    "play_type": "offense",
    "passing": "offense",
    "hustle_offense": "offense",
}


def catalogue() -> list[tuple[str, str, str, str, str]]:
    """(key, label, group, fmt, kind) for everything a panel can report."""
    out: list[tuple[str, str, str, str, str]] = [
        ("games_played", "Games played", "availability", "count", COUNT),
        ("minutes", "Minutes", "availability", "num", PER_GAME),
        ("blocks", "Blocks", "defense_activity", "num", PER_GAME),
        ("steals", "Steals", "defense_activity", "num", PER_GAME),
        ("deflections", "Deflections", "defense_activity", "num", PER_GAME),
    ]
    for col, label, side in HUSTLE_STAT_COLUMNS.values():
        out.append((col, label, f"hustle_{side}", "num", PER_GAME))
    for col, label, side in BOX_SCORE_COLUMNS.values():
        out.append((col, label, "shooting" if side == "offense" else "defense_activity", "num", PER_GAME))
    for col, label, side in SEASON_TOTAL_COLUMNS.values():
        out.append((col, label, "shooting", "count", COUNT))
    for slug in DEF_CATEGORY_ORDER:
        label = next(l for s, l in DEF_CATEGORIES.values() if s == slug)
        out += [
            (f"def_{slug}_fg_pct", f"{label} — FG% allowed", "defense_matchup", "pct", RATE),
            (f"def_{slug}_pm", f"{label} — vs league normal", "defense_matchup", "pct", DIFF),
            (f"def_{slug}_fga", f"{label} — shots defended", "defense_matchup", "count", COUNT),
            (f"def_{slug}_fgm", f"{label} — makes allowed", "defense_matchup", "count", COUNT),
        ]
    from src.inference.stat_engine import (
        PLAY_TYPE_LABEL, SHOT_CONTEXT_SLUG, STAT_GROUPS, ZONE_SLUG,
    )
    for slug in ZONE_SLUG.values():
        pct_key, fga_key = f"zone_fg_pct_{slug}", f"zone_fga_{slug}"
        out.append((pct_key, STAT_GROUPS[pct_key]["label"], "shooting", "pct", RATE))
        out.append((fga_key, STAT_GROUPS[fga_key]["label"], "shooting", "count", COUNT))
    for slug in dict.fromkeys(SHOT_CONTEXT_SLUG.values()):
        pct_key, fga_key = f"shotctx_{slug}_fg_pct", f"shotctx_{slug}_fga"
        out.append((pct_key, STAT_GROUPS[pct_key]["label"], "shot_difficulty", "pct", RATE))
        out.append((fga_key, STAT_GROUPS[fga_key]["label"], "shot_difficulty", "count", COUNT))
    out += [
        ("potential_ast", "Potential assists", "passing", "num", PER_GAME),
        ("secondary_ast", "Secondary (hockey) assists", "passing", "num", PER_GAME),
        ("ast_points_created", "Points created by assist", "passing", "num", PER_GAME),
        ("ast_to_pass_pct", "Assist-to-pass rate", "passing", "pct", RATE),
    ]
    for slug, label in PLAY_TYPE_LABEL.items():
        out.append((f"playtype_{slug}_ppp", f"{label} — points per possession", "play_type", "num", RATE))
        out.append((f"playtype_{slug}_fg_pct", f"{label} — FG%", "play_type", "pct", RATE))
        out.append((f"playtype_{slug}_poss", f"{label} — possessions", "play_type", "count", COUNT))
    return out
