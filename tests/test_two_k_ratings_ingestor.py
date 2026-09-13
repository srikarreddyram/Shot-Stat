"""
Tests for src/ingestion/two_k_ratings_ingestor.py's parsing — the schema.org
JSON-LD block on a 2kratings.com page, confirmed live for Anthony Davis and
Alex Caruso, is real JSON, so this exercises that shape directly rather than
depending on a live fetch.
"""
import json

from src.ingestion.two_k_ratings_ingestor import (
    DEFENSE_ATTRIBUTES, OFFENSE_ATTRIBUTES, _extract_attributes,
)


def _fake_page(overall: int, attrs: dict) -> str:
    """A minimal page with the same JSON-LD @graph shape as the real site."""
    props = [{"@type": "PropertyValue", "name": "NBA 2K27 Rating", "value": overall}]
    props += [{"@type": "PropertyValue", "name": k, "value": v} for k, v in attrs.items()]
    graph = {
        "@context": "https://schema.org",
        "@graph": [
            {"@type": "Organization", "name": "2K Ratings"},
            {
                "@type": "Person",
                "name": "Test Player",
                "additionalProperty": props,
            },
        ],
    }
    return f"""<html><head>
        <script type="application/ld+json">{json.dumps(graph)}</script>
    </head><body></body></html>"""


def test_extract_attributes_reads_overall_and_individual_attributes():
    html = _fake_page(90, {"Three-Point Shot Attribute": 70, "Block Attribute": 82})
    attrs = _extract_attributes(html)
    assert attrs["NBA 2K27 Rating"] == 90
    assert attrs["Three-Point Shot Attribute"] == 70
    assert attrs["Block Attribute"] == 82


def test_extract_attributes_returns_none_without_a_json_ld_block():
    assert _extract_attributes("<html><body>no ratings here</body></html>") is None


def test_offense_and_defense_attribute_lists_do_not_overlap():
    # Each 2K attribute should describe exactly one side of the ball in our
    # rollup, or neither (physicals/hustle/potential) — never both, which
    # would double-count it in two different composites.
    assert set(OFFENSE_ATTRIBUTES).isdisjoint(DEFENSE_ATTRIBUTES)


def test_scrape_2k_rating_rollup_matches_manual_average(monkeypatch):
    from src.ingestion import two_k_ratings_ingestor as mod

    offense_subset = {"Three-Point Shot Attribute": 70, "Layup Attribute": 80}
    defense_subset = {"Block Attribute": 60, "Steal Attribute": 40}
    html = _fake_page(85, {**offense_subset, **defense_subset})

    class _FakeResp:
        status_code = 200
        text = html

    monkeypatch.setattr(mod, "_get_with_retry", lambda url, **kw: _FakeResp())

    result = mod.scrape_2k_rating("Test Player")
    assert result["overall"] == 85
    assert result["offense_avg"] == sum(offense_subset.values()) / len(offense_subset)
    assert result["defense_avg"] == sum(defense_subset.values()) / len(defense_subset)
