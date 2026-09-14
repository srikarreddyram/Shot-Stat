"""
Tests for src/inference/api.py using FastAPI's TestClient.

The model artifacts under models/ are small (a few MB), so the app's startup
event loads the real ShotRecommender rather than a mock — this gives genuine
end-to-end coverage of request validation -> DB lookup -> inference. The DB
engine is monkeypatched (in both api.py and recommender.py, since each imported
get_engine into its own module namespace via `from ... import`) to point at the
seeded in-memory DB so no test touches data/nba_shots.db.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.inference.api as api_mod
import src.inference.recommender as rec_mod
from src.features.point_in_time import DEFENSE_CATEGORIES
from src.features.shrinkage import BetaPrior

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_NAME = "shot-quality-v20"

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / f"metadata_{MODEL_NAME}.json").exists(),
    reason=f"{MODEL_NAME} artifacts not present in models/",
)


@pytest.fixture()
def client(seeded_db, monkeypatch):
    monkeypatch.setattr(api_mod, "get_engine", lambda: seeded_db)
    monkeypatch.setattr(rec_mod, "get_engine", lambda: seeded_db)
    with TestClient(api_mod.app) as c:
        # `models/metadata_shot-quality-v9.json` predates point-in-time
        # defender quality, so it carries no `category_priors` and every
        # lookup would short-circuit to empty (see the same override in
        # tests/test_recommender.py's `recommender` fixture).
        api_mod.recommender.category_priors = {
            cat: BetaPrior(mean=0.0, strength=0.0) for cat in DEFENSE_CATEGORIES
        }
        yield c


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_name"] == MODEL_NAME
    assert body["features"] > 0


def test_health_when_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr(api_mod, "recommender", None)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is False
    assert body["model_name"] is None
    assert body["features"] == 0


def test_recommend_success(client):
    resp = client.post("/recommend", json={
        "player_id": "P_TALL", "season": "2023-24", "top_n": 5,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["player_name"] == "Tall Center"
    assert len(body["recommendations"]) <= 5


def test_recommend_unknown_player_returns_404(client):
    resp = client.post("/recommend", json={
        "player_id": "NOT_A_REAL_ID", "season": "2023-24",
    })
    assert resp.status_code == 404


def test_recommend_validation_error_on_bad_quarter(client):
    # quarter is constrained to 1..8 by RecommendRequest
    resp = client.post("/recommend", json={
        "player_id": "P_TALL", "season": "2023-24", "quarter": 99,
    })
    assert resp.status_code == 422


def test_recommend_validation_error_missing_required_field(client):
    # player_id is required
    resp = client.post("/recommend", json={"season": "2023-24"})
    assert resp.status_code == 422


def test_recommend_returns_503_when_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr(api_mod, "recommender", None)
    resp = client.post("/recommend", json={
        "player_id": "P_TALL", "season": "2023-24",
    })
    assert resp.status_code == 503


def test_recommend_summary_success(client):
    resp = client.post("/recommend/summary", json={
        "player_id": "P_TALL", "season": "2023-24",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["player_name"] == "Tall Center"
    assert len(body["zone_summary"]) > 0


def test_get_player_success(client):
    resp = client.get("/player/P_SHORT", params={"season": "2023-24"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Short Guard"


def test_get_player_not_found(client):
    resp = client.get("/player/NOT_A_REAL_ID", params={"season": "2023-24"})
    assert resp.status_code == 404


def test_players_search_requires_min_length_query(client):
    resp = client.get("/players/search", params={"q": "a", "season": "2023-24"})
    assert resp.status_code == 422


def test_players_search_finds_seeded_player(client):
    resp = client.get("/players/search", params={"q": "Tall", "season": "2023-24"})
    assert resp.status_code == 200
    body = resp.json()
    assert any(p["name"] == "Tall Center" for p in body)


def test_matchup_uses_zone_level_defender_fg_pct(client):
    """
    /matchup should use zone-level (not overall) defender FG% for the
    Restricted Area exploit analysis — the same fix ZONE_TO_DEF_CATEGORY
    documents in recommender.py.

    P_DEF is every seeded shooter's only matchup in G1: 2 makes allowed at
    the rim on 2 attempts (S1 off P_TALL, S4 off P_SHORT), against an
    Overall of 2/5 — see test_defender_row_returns_zone_level_stats_not_just_overall
    for the full breakdown.
    """
    resp = client.get("/matchup/P_TALL/P_DEF", params={"season": "2023-24"})
    assert resp.status_code == 200
    body = resp.json()

    rim_zone = next(z for z in body["exploit_zones"] if z["zone"] == "Restricted Area")
    assert rim_zone["defender_fg_pct_allowed"] == pytest.approx(1.0)
    assert rim_zone["defender_fg_pct_allowed"] != body["defender_quality"]["fg_pct_allowed"]


def test_matchup_unknown_defender_returns_404(client):
    resp = client.get("/matchup/P_TALL/NOT_A_REAL_ID", params={"season": "2023-24"})
    assert resp.status_code == 404


def test_explain_matchup_returns_narrative_and_factor_lists(client):
    """
    /explain/matchup is the endpoint the frontend uses for the full
    matchup narrative (offense vs defense make-probability, expected
    points, attainability woven in). The seeded DB has no shot_on_court
    data, so the on-court lineup context resolves to nothing — this also
    checks that absence degrades to missing features rather than an error.
    """
    resp = client.get("/explain/matchup/P_TALL", params={
        "zone": "Restricted Area", "loc_x": 0.0, "loc_y": 40.0,
        "defender_id": "P_DEF", "season": "2023-24",
    })
    assert resp.status_code == 200
    body = resp.json()

    assert body["player_id"] == "P_TALL"
    assert body["defender_id"] == "P_DEF"
    assert isinstance(body["narrative"], str) and body["narrative"]
    assert "offense_factors" in body["shot_quality"]
    assert "defense_factors" in body["shot_quality"]


# ── Team logos ──────────────────────────────────────────────────────────────
# team_id reaches both an outbound URL and a filesystem path, so the shape
# check on it is a security control, not tidiness: without it a crafted id is
# a path-traversal (writing/reading outside the cache) and an SSRF (pointing
# the fetch at an arbitrary host) at the same time.

@pytest.mark.parametrize("bad_id", [
    "../../../etc/passwd",
    "..%2F..%2Fsecret",
    "1610612737/../../evil",
    "abcdefghij",        # right length, not digits
    "161061273",         # digits, wrong length
    "16106127370",       # digits, too long
    "",
    "1610612737 ",
])
def test_team_logo_rejects_anything_that_is_not_a_team_id(client, bad_id, monkeypatch):
    """A malformed id must be refused BEFORE any fetch happens."""
    def explode(*args, **kwargs):
        raise AssertionError("a request was made for a rejected team id")
    monkeypatch.setattr(api_mod.requests, "get", explode)

    response = client.get(f"/team/{bad_id}/logo")
    assert response.status_code == 404


def test_team_logo_fetches_once_then_serves_from_cache(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api_mod.media, "MEDIA_CACHE", tmp_path)
    calls = []

    class FakeResponse:
        content = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        headers = {"content-type": "image/svg+xml"}
        def raise_for_status(self): pass

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse()
    monkeypatch.setattr(api_mod.requests, "get", fake_get)

    first = client.get("/team/1610612737/logo")
    assert first.status_code == 200
    assert first.headers["content-type"] == "image/svg+xml"
    assert len(calls) == 1
    assert "1610612737" in calls[0]

    second = client.get("/team/1610612737/logo")
    assert second.status_code == 200
    assert len(calls) == 1  # served from disk, not re-fetched


def test_team_logo_reports_upstream_failure_rather_than_caching_garbage(client, tmp_path, monkeypatch):
    """An HTML error page from the CDN must not be written to the cache and
    then served forever as if it were a logo."""
    monkeypatch.setattr(api_mod.media, "MEDIA_CACHE", tmp_path)

    class HtmlResponse:
        content = b"<html>404 not found</html>"
        headers = {"content-type": "text/html"}
        def raise_for_status(self): pass

    monkeypatch.setattr(api_mod.requests, "get", lambda url, **kw: HtmlResponse())
    response = client.get("/team/1610612737/logo")
    assert response.status_code == 502
    assert not (tmp_path / "team_logos" / "1610612737.svg").exists()


@pytest.mark.parametrize("bad_id", ["../../etc/passwd", "abc", "", "12345678901", "2544 "])
def test_player_headshot_rejects_anything_that_is_not_a_player_id(client, bad_id, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("a request was made for a rejected player id")
    monkeypatch.setattr(api_mod.requests, "get", explode)
    assert client.get(f"/player/{bad_id}/headshot").status_code == 404


def test_player_headshot_rejects_an_unknown_size(client, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("a request was made for a rejected size")
    monkeypatch.setattr(api_mod.requests, "get", explode)
    assert client.get("/player/2544/headshot?size=enormous").status_code == 400


def test_player_headshot_caches_each_size_separately(client, tmp_path, monkeypatch):
    """Small and large must not share a cache path, or whichever is requested
    first is served for both and a 200 KB hero image ends up in a table row."""
    monkeypatch.setattr(api_mod.media, "MEDIA_CACHE", tmp_path)
    urls = []

    class FakePng:
        content = b"\x89PNG\r\n\x1a\n"
        headers = {"content-type": "image/png"}
        def raise_for_status(self): pass

    monkeypatch.setattr(api_mod.requests, "get", lambda url, **kw: (urls.append(url), FakePng())[1])

    assert client.get("/player/2544/headshot?size=small").status_code == 200
    assert client.get("/player/2544/headshot?size=large").status_code == 200
    assert len(urls) == 2
    assert "260x190" in urls[0] and "1040x760" in urls[1]
    assert (tmp_path / "player_headshots" / "small" / "2544.png").exists()
    assert (tmp_path / "player_headshots" / "large" / "2544.png").exists()

    client.get("/player/2544/headshot?size=small")
    assert len(urls) == 2  # served from disk
