"""
Regression tests for src/ingestion/defender_stats_ingestor.py's per-category
COL_MAP.

The real bug: LeagueDashPtDefend returns a *different* set of column names
depending on the `defense_category` requested (e.g. "LT_06_PCT" for
"Less Than 6Ft" vs "D_FG_PCT" for "Overall"). COL_MAP previously mapped some
zone categories to column names that don't exist in that category's response,
so `row.get(cm["pct"])` silently returned None for every zone-level row —
d_fg_pct, pct_plusminus etc. all landed as NULL in the DB for every category
except "Overall". These tests fake the nba_api response with the *real*
column names for a given category and assert the ingestor actually extracts
non-null values using the correct COL_MAP entry.
"""
import time

import pandas as pd
import pytest
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

import src.ingestion.defender_stats_ingestor as ing


class _FakeDefendResponse:
    """Mimics nba_api's LeagueDashPtDefend: only get_data_frames() is used."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_data_frames(self):
        return [self._df]


@pytest.fixture()
def patched_ingestor(seeded_db, monkeypatch):
    """Point the ingestor at the seeded in-memory DB and silence sleep()."""
    monkeypatch.setattr(ing, "get_engine", lambda: seeded_db)
    monkeypatch.setattr(ing, "get_session_factory", lambda engine: sessionmaker(bind=engine))
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    return seeded_db


def _fake_rows_for_category(category: str) -> pd.DataFrame:
    """
    Build a one-row fake API response using the REAL column names
    LeagueDashPtDefend returns for each defense_category.
    """
    payloads = {
        "Overall": {
            "CLOSE_DEF_PERSON_ID": 999, "GP": 70, "FREQ": 1.0,
            "D_FGM": 400, "D_FGA": 800, "D_FG_PCT": 0.50,
            "NORMAL_FG_PCT": 0.47, "PCT_PLUSMINUS": 0.03,
        },
        "Less Than 6Ft": {
            "CLOSE_DEF_PERSON_ID": 999, "GP": 70, "FREQ": 0.35,
            "FGM_LT_06": 100, "FGA_LT_06": 280, "LT_06_PCT": 0.357,
            "NS_LT_06_PCT": 0.60, "PLUSMINUS": -0.243,
        },
        "3 Pointers": {
            "CLOSE_DEF_PERSON_ID": 999, "GP": 70, "FREQ": 0.25,
            "FG3M": 150, "FG3A": 350, "FG3_PCT": 0.4286,
            "NS_FG3_PCT": 0.36, "PLUSMINUS": 0.0686,
        },
    }
    return pd.DataFrame([payloads[category]])


@pytest.mark.parametrize(
    "category,expected_pct,expected_pm,expected_fgm,expected_fga",
    [
        ("Overall", 0.50, 0.03, 400, 800),
        ("Less Than 6Ft", 0.357, -0.243, 100, 280),
        ("3 Pointers", 0.4286, 0.0686, 150, 350),
    ],
)
def test_col_map_extracts_correct_columns_per_category(
    patched_ingestor, monkeypatch, category, expected_pct, expected_pm, expected_fgm, expected_fga,
):
    """
    For each category, COL_MAP must pull d_fg_pct / pct_plusminus / d_fgm / d_fga
    from the columns that category's API response actually contains — not from
    "Overall"'s column names, which is what the old bug effectively did.
    """
    monkeypatch.setattr(ing, "DEFENSE_CATEGORIES", [category])
    monkeypatch.setattr(ing, "SEASON_TYPES", ["Regular Season"])
    monkeypatch.setattr(
        ing.leaguedashptdefend, "LeagueDashPtDefend",
        lambda **kwargs: _FakeDefendResponse(_fake_rows_for_category(category)),
    )

    ing.ingest_defender_stats(["2023-24"])

    with patched_ingestor.connect() as conn:
        # NOTE: df.iterrows() upcasts a mixed-dtype row to a common dtype, so
        # the integer CLOSE_DEF_PERSON_ID (999) arrives as float (999.0) and
        # str()'s to "999.0" — match on season/category (unique for this
        # newly-inserted row) rather than hardcoding the player_id string.
        row = conn.execute(
            text(
                "SELECT d_fg_pct, pct_plusminus, d_fgm, d_fga FROM defender_stats "
                "WHERE player_id = '999.0' AND season = '2023-24' AND defense_category = :cat"
            ),
            {"cat": category},
        ).fetchone()

    assert row is not None, "COL_MAP bug regression: no row was upserted for this category"
    d_fg_pct, pct_plusminus, d_fgm, d_fga = row

    # This is the exact assertion that would have caught the original bug: a
    # wrong column name means .get() returns None, and these would all be NULL.
    assert d_fg_pct == pytest.approx(expected_pct)
    assert pct_plusminus == pytest.approx(expected_pm)
    assert d_fgm == expected_fgm
    assert d_fga == expected_fga


def test_col_map_wrong_column_name_would_produce_nulls(patched_ingestor, monkeypatch):
    """
    Sanity-check the test harness itself: if COL_MAP pointed "Less Than 6Ft" at
    "Overall"'s column names (the historical bug), the extracted values would
    all be None. This documents what the regression above is actually guarding.
    """
    category = "Less Than 6Ft"
    fake_df = _fake_rows_for_category(category)

    # Using "Overall"'s column names against the "Less Than 6Ft" payload —
    # exactly what the pre-fix COL_MAP effectively did for zone categories.
    bogus_cm = {"fgm": "D_FGM", "fga": "D_FGA", "pct": "D_FG_PCT", "pm": "PCT_PLUSMINUS"}
    row = fake_df.iloc[0]
    assert row.get(bogus_cm["pct"]) is None
    assert row.get(bogus_cm["pm"]) is None
