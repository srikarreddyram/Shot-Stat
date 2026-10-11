"""
Tests for scripts/refresh_season.py: the step order respects each step's
inputs, and --from / --only / --skip select what they say.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from refresh_season import plan, preseason, select  # noqa: E402


def _order():
    return [st.name for st in plan("2026-27")]


def test_steps_run_after_what_they_read():
    order = _order()
    before = lambda a, b: order.index(a) < order.index(b)  # noqa: E731
    assert order[0] == "schema"
    assert before("games", "shots") and before("player_stats", "shots") and before("rosters", "shots")
    for per_game in ("home_away", "score_diff", "pbp", "lineups", "matchups", "career_stats"):
        assert before("shots", per_game)
    assert before("matchups", "defenders")
    assert before("games", "schedule")
    assert "advanced_stats" in order
    assert order[-1] == "comparable"  # indexes the finished shot data


def test_in_progress_season_steps_ignore_their_done_flags():
    steps = {st.name: st for st in plan("2026-27")}
    assert "--refresh" in steps["shots"].args and "--refresh" in steps["zone_stats"].args
    assert steps["comparable"].args[-2:] == ("--add-season", "2026-27")


def test_from_only_and_skip():
    steps = plan("2026-27")
    resumed = [st.name for st in select(steps, "pbp", None, [])]
    assert resumed[0] == "pbp" and "shots" not in resumed
    assert [st.name for st in select(steps, None, ["game_logs", "games"], [])] == ["games", "game_logs"]
    assert "contest" not in [st.name for st in select(steps, None, None, ["contest"])]
    with pytest.raises(SystemExit):
        select(steps, "nope", None, [])


def test_before_opening_night_only_rosters_refresh():
    kept, skipped = preseason(plan("2026-27"))
    assert [st.name for st in kept] == ["schema", "games", "rosters", "physicals"]
    # Everything that reads games or shots waits for the first game.
    names = {st.name for st in skipped}
    assert {"shots", "pbp", "zone_stats", "game_logs", "comparable"} <= names
    assert len(kept) + len(skipped) == len(plan("2026-27"))
