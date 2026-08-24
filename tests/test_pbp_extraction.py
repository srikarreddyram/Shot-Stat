"""
Tests for play-by-play context extraction.

`extract_context` is a pure function over the endpoint's frame, so all of this
runs without a network call. That matters: the ingest takes hours, and a logic
bug found afterwards means re-fetching twelve thousand games.

One of these encodes a bug that actually shipped during development: the assist
regex that never matched, which quietly marked every shot in the database as
unassisted.
"""
import pandas as pd
import pytest

from src.ingestion.pbp_ingestor import _AST_RE, _clock_seconds, extract_context


def _event(action_number, action_type, clock, period=1, subtype="Jump Shot",
           description="", is_fg=1, team=100):
    return {
        "actionNumber": action_number,
        "actionType": action_type,
        "subType": subtype,
        "clock": clock,
        "period": period,
        "description": description,
        "isFieldGoal": is_fg,
        "teamId": team,
    }


def test_clock_parsing():
    assert _clock_seconds("PT11M42.00S") == pytest.approx(702.0)
    assert _clock_seconds("PT00M04.50S") == pytest.approx(4.5)
    assert _clock_seconds(None) is None
    assert _clock_seconds("garbage") is None


def test_assist_regex_matches_the_real_format():
    """
    Regression. The original pattern anchored on "\\(\\d+" — but the clause is
    "(Anthony 1 AST)", with the assister's NAME between the paren and the
    count, so it never matched anything and every shot in the database came
    back unassisted.
    """
    assert _AST_RE.search("James 20' Pullup Jump Shot (2 PTS) (Anthony 1 AST)")
    assert _AST_RE.search("Davis 1' Dunk (2 PTS) (Russell 11 AST)")
    assert not _AST_RE.search("MISS Rondo 20' Jump Shot")
    assert not _AST_RE.search("Jokic 7' Driving Floating Jump Shot (2 PTS)")


def test_extracts_assist_flag():
    df = pd.DataFrame([
        _event(2, "Made Shot", "PT11M42.00S",
               description="Davis 1' Dunk (2 PTS) (Russell 1 AST)"),
        _event(4, "Missed Shot", "PT11M20.00S", description="MISS Davis 18' Jump Shot"),
    ])
    rows = extract_context(df, "0022300061")
    assert [r["is_assisted"] for r in rows] == [1, 0]
    assert rows[0]["shot_id"] == "0022300061_2"


def test_elapsed_time_is_positive():
    """
    The game clock counts DOWN, so elapsed time is previous minus current.
    Getting the sign backwards would yield negative durations and silently
    disable every putback check, which is the kind of failure that produces a
    column of zeros rather than an error.
    """
    df = pd.DataFrame([
        _event(1, "Rebound", "PT10M00.00S", is_fg=0, description="Reaves REBOUND (Off:1 Def:0)"),
        _event(2, "Made Shot", "PT09M57.00S"),
    ])
    rows = extract_context(df, "G")
    assert rows[0]["seconds_since_prev_event"] == pytest.approx(3.0)


def test_putback_requires_offensive_rebound_same_team_and_quick():
    same_team_quick = pd.DataFrame([
        _event(1, "Rebound", "PT10M00.00S", is_fg=0,
               description="Reaves REBOUND (Off:1 Def:0)", team=100),
        _event(2, "Made Shot", "PT09M58.00S", team=100),
    ])
    assert extract_context(same_team_quick, "G")[0]["is_putback"] == 1

    # A defensive rebound followed by a quick shot is a fast break, not a putback.
    other_team = pd.DataFrame([
        _event(1, "Rebound", "PT10M00.00S", is_fg=0,
               description="Reaves REBOUND (Off:0 Def:1)", team=200),
        _event(2, "Made Shot", "PT09M58.00S", team=100),
    ])
    assert extract_context(other_team, "G")[0]["is_putback"] == 0

    # Too slow to be a putback — the possession reset.
    slow = pd.DataFrame([
        _event(1, "Rebound", "PT10M00.00S", is_fg=0,
               description="Reaves REBOUND (Off:1 Def:0)", team=100),
        _event(2, "Made Shot", "PT09M50.00S", team=100),
    ])
    assert extract_context(slow, "G")[0]["is_putback"] == 0


def test_previous_event_does_not_cross_periods():
    """
    The gap across a period break is not elapsed game time. Looking back into
    the previous period would report a shot early in Q2 as following the last
    Q1 event by minutes.
    """
    df = pd.DataFrame([
        _event(1, "Made Shot", "PT00M20.00S", period=1),
        _event(2, "Made Shot", "PT11M50.00S", period=2),
    ])
    rows = extract_context(df, "G")
    assert rows[1]["seconds_since_prev_event"] is None
    assert rows[1]["prev_event_type"] is None


def test_non_field_goal_events_are_skipped():
    df = pd.DataFrame([
        _event(1, "Substitution", "PT10M00.00S", is_fg=0),
        _event(2, "Made Shot", "PT09M40.00S"),
    ])
    rows = extract_context(df, "G")
    assert len(rows) == 1
    assert rows[0]["shot_id"] == "G_2"
