"""NBA media (team logos, the league wordmark, player headshots) —
proxied and cached rather than hot-linked from the page. cdn.nba.com serves
these happily to a plain HTTP client but rejects at least some browser
requests with ERR_HTTP2_PROTOCOL_ERROR, so an <img> pointed straight at it
is not dependable. Going through the API also means the media keeps working
with no network once cached, and NBA's assets stay out of the repo.
"""
from pathlib import Path

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

MEDIA_CACHE = Path(__file__).resolve().parents[3] / "data" / "cache"
TEAM_LOGO_URL = "https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"
# The league wordmark itself, same CDN as the team logos above — used by the
# frontend to brand its own chrome (header, splash page), not to represent
# any single team.
LEAGUE_LOGO_URL = "https://cdn.nba.com/logos/leagues/logo-nba.svg"
# Two sizes: the small one is for list rows and avatars (~15 KB), the large one
# for a profile hero (~200 KB). Serving the large one into a 582-row table
# would pull ~120 MB.
HEADSHOT_URLS = {
    "small": "https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png",
    "large": "https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png",
}


def _nba_id(value: str, *, exact_length: int | None = None) -> str:
    """Validate an NBA id before it reaches an outbound URL or a file path.

    Both uses are injection sinks: unchecked, a crafted id is a path traversal
    (reading or writing outside the cache) and an SSRF (pointing the fetch at
    an arbitrary host) at the same time. Digits only, so neither is reachable.
    """
    if not value.isdigit() or not (1 <= len(value) <= 10):
        raise HTTPException(status_code=404, detail="Unknown id.")
    if exact_length is not None and len(value) != exact_length:
        raise HTTPException(status_code=404, detail="Unknown id.")
    return value


def _cached_media(url: str, cache_path: Path, media_type: str, label: str) -> FileResponse:
    """Fetch once, serve from disk thereafter.

    media_type is passed explicitly rather than derived from a file extension:
    SVG's is "image/svg+xml", and serving the plausible-looking "image/svg"
    means browsers refuse to render it.
    """
    expect = media_type.split("/", 1)[1].split("+", 1)[0]
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Could not fetch {label}: {exc}")
        # Without this the CDN's HTML error page would be cached and then
        # served forever as if it were the asset.
        if expect not in response.headers.get("content-type", ""):
            raise HTTPException(status_code=502, detail=f"{label} endpoint did not return {expect}.")
        cache_path.write_bytes(response.content)

    return FileResponse(
        cache_path,
        media_type=media_type,
        # Neither a logo nor a headshot changes often; let the browser keep it.
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/team/{team_id}/logo")
def team_logo(team_id: str):
    """The team's official logo as SVG, cached on first request."""
    team_id = _nba_id(team_id, exact_length=10)
    return _cached_media(
        TEAM_LOGO_URL.format(team_id=team_id),
        MEDIA_CACHE / "team_logos" / f"{team_id}.svg",
        media_type="image/svg+xml",
        label="logo",
    )


@router.get("/league/logo")
def league_logo():
    """The NBA's own league wordmark as SVG, cached on first request. No id
    to validate here — there's exactly one of these, unlike team logos."""
    return _cached_media(
        LEAGUE_LOGO_URL,
        MEDIA_CACHE / "league_logo.svg",
        media_type="image/svg+xml",
        label="league logo",
    )


@router.get("/player/{player_id}/headshot")
def player_headshot(player_id: str, size: str = "small"):
    """The player's official headshot, cached on first request.

    A player id the NBA has no photo for is NOT an error here: the CDN answers
    with a generic silhouette, which is a perfectly good fallback and is what
    gets cached. There is therefore no way to distinguish "no photo" from
    "photo" by status code, and callers should not try.
    """
    player_id = _nba_id(player_id)
    if size not in HEADSHOT_URLS:
        raise HTTPException(status_code=400, detail="size must be 'small' or 'large'.")
    return _cached_media(
        HEADSHOT_URLS[size].format(player_id=player_id),
        MEDIA_CACHE / "player_headshots" / size / f"{player_id}.png",
        media_type="image/png",
        label="headshot",
    )
