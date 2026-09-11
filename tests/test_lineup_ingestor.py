"""
Tests for src/ingestion/lineup_ingestor.py's on-court reconstruction.

Pure-function tests over a synthetic PlayByPlayV3-shaped DataFrame — no
network call, matching the pattern src/ingestion/pbp_ingestor.py's own
tests use for the same reason (this logic should be checkable without an
NBA API round trip).
"""
import pandas as pd

from src.ingestion.lineup_ingestor import extract_lineups

TEAM_A, TEAM_B = 1, 2


def _row(action, period, team, person, name, action_type,
        is_fg=0, description=""):
    return {
        "actionNumber": action, "period": period, "teamId": team,
        "personId": person, "playerName": name, "actionType": action_type,
        "isFieldGoal": is_fg, "description": description,
    }


def test_simple_game_five_starters_one_substitution():
    """
    A's starters (1-5) vs B's starters (6-10). A shot before any
    substitution sees the starting five on both sides; a shot after player
    6 subs out for player 11 sees the updated defense.
    """
    rows = [
        _row(1, 1, TEAM_A, 1, "P1", "Rebound"),
        _row(2, 1, TEAM_B, 6, "P6", "Rebound"),
        _row(3, 1, TEAM_A, 2, "P2", "Made Shot", is_fg=1),
        _row(4, 1, TEAM_B, 7, "P7", "Foul"),
        _row(5, 1, TEAM_B, 8, "P8", "Foul"),
        _row(6, 1, TEAM_B, 9, "P9", "Foul"),
        _row(7, 1, TEAM_B, 10, "P10", "Foul"),
        _row(8, 1, TEAM_A, 3, "P3", "Foul"),
        _row(9, 1, TEAM_A, 4, "P4", "Foul"),
        _row(10, 1, TEAM_A, 5, "P5", "Foul"),
        # B subs player 6 out for player 11.
        _row(11, 1, TEAM_B, 6, "P6", "Substitution", description="SUB: P11 FOR P6"),
        _row(12, 1, TEAM_B, 11, "P11", "Foul"),
        _row(13, 1, TEAM_A, 1, "P1", "Made Shot", is_fg=1),
    ]
    df = pd.DataFrame(rows)

    out, unresolved = extract_lineups(df, "GAME1")
    assert unresolved == 0
    out_df = pd.DataFrame(out)

    first_shot = out_df[out_df["shot_id"] == "GAME1_3"]
    assert set(first_shot[first_shot["role"] == "offense"]["player_id"]) == \
        {"1", "2", "3", "4", "5"}
    assert set(first_shot[first_shot["role"] == "defense"]["player_id"]) == \
        {"6", "7", "8", "9", "10"}

    second_shot = out_df[out_df["shot_id"] == "GAME1_13"]
    assert set(second_shot[second_shot["role"] == "offense"]["player_id"]) == \
        {"1", "2", "3", "4", "5"}
    assert set(second_shot[second_shot["role"] == "defense"]["player_id"]) == \
        {"7", "8", "9", "10", "11"}, "player 6 should be replaced by player 11"


def test_starter_who_rests_and_returns_is_not_mistaken_for_a_bench_player():
    """
    Regression test for the real bug found building this: a starter who is
    subbed out and later subbed back IN is `first_incoming`-tagged just like
    a genuine bench player, but that must not disqualify him from having
    started — what matters is whether his FIRST appearance in the period
    precedes his first incoming assignment, not whether he is EVER incoming
    at all. The original version used "ever incoming" as disqualifying and
    left only ~1 real starter per team.
    """
    rows = [
        _row(1, 1, TEAM_A, 1, "P1", "Rebound"),
        _row(2, 1, TEAM_A, 2, "P2", "Rebound"),
        _row(3, 1, TEAM_A, 3, "P3", "Rebound"),
        _row(4, 1, TEAM_A, 4, "P4", "Rebound"),
        _row(5, 1, TEAM_A, 5, "P5", "Made Shot", is_fg=1),
        _row(6, 1, TEAM_B, 6, "P6", "Rebound"),
        _row(7, 1, TEAM_B, 7, "P7", "Rebound"),
        _row(8, 1, TEAM_B, 8, "P8", "Rebound"),
        _row(9, 1, TEAM_B, 9, "P9", "Rebound"),
        _row(10, 1, TEAM_B, 10, "P10", "Rebound"),
        # Starter P1 rests...
        _row(11, 1, TEAM_A, 1, "P1", "Substitution", description="SUB: P12 FOR P1"),
        _row(12, 1, TEAM_A, 12, "P12", "Foul"),
        # ...and returns.
        _row(13, 1, TEAM_A, 12, "P12", "Substitution", description="SUB: P1 FOR P12"),
        _row(14, 1, TEAM_A, 1, "P1", "Made Shot", is_fg=1),
    ]
    df = pd.DataFrame(rows)

    out, unresolved = extract_lineups(df, "GAME2")
    assert unresolved == 0
    out_df = pd.DataFrame(out)

    returned_shot = out_df[out_df["shot_id"] == "GAME2_14"]
    offense = set(returned_shot[returned_shot["role"] == "offense"]["player_id"])
    assert offense == {"1", "2", "3", "4", "5"}, (
        f"P1 should be back on the floor after returning from the bench, got {offense}"
    )


def test_missing_incoming_substitution_excludes_only_the_affected_window():
    """
    A real data gap (confirmed against the actual NBA feed, not a
    hypothetical): a player has real box-score actions and is later validly
    subbed OUT, but no matching "SUB: X FOR <him>" ever brought him in. This
    must not corrupt the rest of the game for that team — shots during his
    unaccounted-for stretch get no lineup row (correctly excluded by the
    5-and-5 invariant), but shots after he is validly subbed back out must
    still resolve normally.
    """
    rows = [
        _row(1, 1, TEAM_A, 1, "P1", "Rebound"),
        _row(2, 1, TEAM_A, 2, "P2", "Rebound"),
        _row(3, 1, TEAM_A, 3, "P3", "Rebound"),
        _row(4, 1, TEAM_A, 4, "P4", "Rebound"),
        _row(5, 1, TEAM_A, 5, "P5", "Rebound"),
        _row(6, 1, TEAM_B, 6, "P6", "Rebound"),
        _row(7, 1, TEAM_B, 7, "P7", "Rebound"),
        _row(8, 1, TEAM_B, 8, "P8", "Rebound"),
        _row(9, 1, TEAM_B, 9, "P9", "Rebound"),
        _row(10, 1, TEAM_B, 10, "P10", "Rebound"),
        # Shot while state is fully known.
        _row(11, 1, TEAM_A, 1, "P1", "Made Shot", is_fg=1),
        # Player 99 appears from nowhere — no incoming sub for him anywhere.
        _row(12, 1, TEAM_A, 99, "P99", "Foul"),
        # A shot while the reconstruction can't know who 99 replaced.
        _row(13, 1, TEAM_A, 2, "P2", "Missed Shot", is_fg=1),
        # 99 is validly subbed back out for a real, well-formed player.
        _row(14, 1, TEAM_A, 99, "P99", "Substitution", description="SUB: P13 FOR P99"),
        _row(15, 1, TEAM_A, 13, "P13", "Foul"),
        # State should be back to normal (minus whoever 99 actually
        # replaced, who is now permanently missing from this team for the
        # rest of the period — an accepted, bounded cost).
        _row(16, 1, TEAM_A, 3, "P3", "Made Shot", is_fg=1),
    ]
    df = pd.DataFrame(rows)

    out, unresolved = extract_lineups(df, "GAME3")
    out_df = pd.DataFrame(out)

    # The shot before the gap resolves normally.
    assert "GAME3_11" in set(out_df["shot_id"])
    # The shot during the unresolvable window is excluded, not guessed.
    assert "GAME3_13" not in set(out_df["shot_id"])
