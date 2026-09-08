"""
Tests for src/ingestion/physical_ingestor_2k.py's parsing, no network involved.

Two real bugs motivate these cases, both found by running the scraper against
players still missing wingspan in the DB rather than against the one player
(Luka Doncic) it had originally been checked against:

1. 2kratings.com prunes a player's page once he drops out of the current
   roster set — Wesley Matthews, Lou Williams and Evan Fournier all 404 live
   despite having real pages as of an April 2024 Wayback snapshot. That is
   attrition in the source, not a parsing bug, so the fix is a Wayback
   fallback rather than touching the live-page selectors.
2. Some archived snapshots glue adjacent fields into one <p> with no
   separator but the next label itself, e.g.
   "Height:6'2\" (188cm)|Weight:175lbs (79kg)" — the original per-<p>
   exact-prefix match silently dropped every field after the first in a tag
   like that.
"""
from src.ingestion.physical_ingestor_2k import (
    _extract_field,
    _name_to_slug,
    _parse_feet_inches,
    _parse_weight,
)


def test_name_to_slug_strips_diacritics_and_punctuation():
    assert _name_to_slug("Luka Dončić") == "luka-doncic"
    assert _name_to_slug("P.J. Tucker") == "pj-tucker"


def test_parse_feet_inches():
    assert _parse_feet_inches("6'8\" (203cm)") == 80.0
    assert _parse_feet_inches("garbage") is None


def test_parse_weight():
    assert _parse_weight("230lbs (104kg)") == 230.0
    assert _parse_weight("garbage") is None


def test_extract_field_handles_one_field_per_tag():
    """The common case: each field already sits in its own <p>'s text."""
    text = "Height:6'8\" (203cm)"
    assert _extract_field(text, "Height") == "6'8\" (203cm)"


def test_extract_field_handles_fields_glued_together():
    """
    The bug: an older Wayback snapshot glued Height and Weight into one <p>
    with no delimiter, e.g. get_text() on
    <p>Height:<span>6'2" (188cm)</span>|Weight:<span>175lbs (79kg)</span></p>
    yields this single string. Both fields must still resolve.
    """
    text = 'Height:6\'2" (188cm)|Weight:175lbs (79kg)'
    assert _extract_field(text, "Height") == '6\'2" (188cm)|'
    assert _extract_field(text, "Weight") == "175lbs (79kg)"
    # And the downstream numeric parsers tolerate the trailing "|" fine.
    assert _parse_feet_inches(_extract_field(text, "Height")) == 74.0
    assert _parse_weight(_extract_field(text, "Weight")) == 175.0


def test_extract_field_missing_label_returns_none():
    """Lou Williams' 2K23 snapshot never lists a wingspan at all — real
    missing data, and the extractor must say so rather than matching garbage."""
    text = "Height:6'2\" (188cm)Weight:175lbs (79kg)Position:PG/SG"
    assert _extract_field(text, "Wingspan") is None
