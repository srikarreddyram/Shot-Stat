"""
Phase 2C: NBA 2K Ratings Physical Data Collection (Final Fallback)

Iterates through unique players STILL missing wingspan data after the
NBA API (Phase 2A) and Basketball Reference (Phase 2B) checks,
and scrapes their exact measurements from 2kratings.com.

2kratings.com sources their data directly from the NBA 2K game files,
which contain accurate wingspan measurements for every player ever
modelled in the game (covering every NBA player from 2K's history).

URL pattern:  https://www.2kratings.com/{first-last-name-slug}
  e.g. https://www.2kratings.com/luka-doncic

Rate limit: ~4 seconds between requests to avoid 403 bans. Requests retry
through transient failure with exponential backoff (see `_get_with_retry`);
a genuine 404 falls back to the Wayback Machine rather than retrying, since
2kratings.com prunes pages for players who have left the current roster set.
"""
import sys
import time
import re
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from sqlalchemy import select, update
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db.database import get_engine, get_session_factory
from src.db.models import Player

# ── Configuration ──────────────────────────────────────────────────────────
TWO_K_BASE_URL = "https://www.2kratings.com"
TWO_K_DELAY = 4.0  # seconds between requests — site bans at ~3s
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}


def _name_to_slug(name: str) -> str:
    """
    Convert a player name to a 2kratings.com URL slug.

    Steps:
      1. Strip diacritics:  Dončić  →  Doncic
      2. Lowercase:         Doncic  →  doncic
      3. Remove dots/apos:  P.J. Tucker  →  pj tucker
      4. Replace non-alphanum with dashes:  pj tucker  →  pj-tucker
      5. Collapse multiple dashes and strip leading/trailing
    """
    # Decompose unicode chars, strip combining marks (accents)
    nfkd = unicodedata.normalize("NFD", name)
    ascii_name = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")

    slug = ascii_name.lower().strip()
    slug = re.sub(r"[.']", "", slug)            # Remove . and '
    slug = re.sub(r"[^a-z0-9]+", "-", slug)     # Non-alphanum → dash
    slug = slug.strip("-")
    return slug


def _extract_field(text: str, label: str) -> str | None:
    """
    Pull the value following `label:` out of the info block's full text,
    stopping at the next capitalised label or end of string.

    Extraction runs on the whole block rather than per-<p> tag: on some
    archived (Wayback) snapshots the site glued adjacent fields into one <p>
    with no separator recognisable except the next label itself, e.g.
    "Height:6'2\" (188cm)|Weight:175lbs (79kg)" — a per-tag exact-prefix match
    silently drops every field after the first in a case like that.
    """
    match = re.search(rf"{label}:\s*(.+?)(?=[A-Z][a-zA-Z ]*:|$)", text)
    return match.group(1).strip() if match else None


def _parse_feet_inches(text: str) -> float | None:
    """
    Parse a string like  6'8" (203cm)  into total inches (80.0).
    Returns None if the pattern isn't found.
    """
    match = re.search(r"(\d+)'(\d+)", text)
    if match:
        return int(match.group(1)) * 12.0 + int(match.group(2))
    return None


def _parse_weight(text: str) -> float | None:
    """
    Parse a string like  230lbs (104kg)  into a float (230.0).
    """
    match = re.search(r"(\d+)\s*lbs?", text)
    if match:
        return float(match.group(1))
    return None


def _get_with_retry(url: str, attempts: int = 4, timeout: float = 15,
                    params: dict | None = None) -> requests.Response | None:
    """
    GET with exponential backoff through transient network failure.

    Without this, a single read-timeout or connection blip drops a player for
    the whole run — confirmed directly: a 427-player backfill reported "not
    found" for several players (Dennis Smith Jr., Juan Toscano-Anderson, Ryan
    Arcidiacono among them) whose pages scraped successfully moments later on
    a bare retry, with no code change in between. A real 404 is returned as-is
    rather than retried — that is a genuine "page not here" signal the caller
    uses to fall back to the Wayback Machine, not a transient failure.
    """
    delay = 2.0
    for attempt in range(attempts):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout, params=params)
            if resp.status_code == 200 or resp.status_code == 404:
                return resp
        except requests.exceptions.RequestException:
            pass
        if attempt < attempts - 1:
            time.sleep(delay)
            delay *= 2
    return None


def _wayback_snapshot_url(url: str) -> str | None:
    """
    Look up the closest archived snapshot of `url` via the Wayback Machine's
    availability API. Returns None if archive.org has never captured it (or
    every retry against the availability API itself failed).

    2kratings.com prunes a player's page once he drops out of the current
    roster set (confirmed via direct testing: a batch of retired role players
    — Wesley Matthews, Lou Williams, Evan Fournier — 404 live despite having
    been scraped successfully as of an April 2024 snapshot). That is link rot
    in the source, not a parsing bug, and the archive is the correct fix: the
    page used to exist and its content did not change after the player retired.

    archive.org's own latency is a second, independent thing this has to
    survive: a full backfill run reported Juan Toscano-Anderson as unresolved,
    but a bare retry of this same lookup minutes later took nearly a minute
    and then succeeded — no code change in between. `_get_with_retry`'s default
    timeout (15s) is tuned for 2kratings.com; archive.org needs more slack.
    """
    resp = _get_with_retry(
        "https://archive.org/wayback/available", attempts=4, timeout=25,
        params={"url": url},
    )
    if resp is None or resp.status_code != 200:
        return None
    try:
        snapshot = resp.json().get("archived_snapshots", {}).get("closest")
    except ValueError:
        return None
    if snapshot and snapshot.get("available"):
        return snapshot["url"]
    return None


def scrape_2k_physicals(player_name: str) -> dict | None:
    """
    Scrape height, weight, and wingspan from a player's 2kratings.com page.

    Falls back to the Wayback Machine when the live page 404s — see
    `_wayback_snapshot_url` for why that happens routinely for retired
    players and is not a bug to "fix" by touching the live scrape.

    Returns:
        {"height": float|None, "weight": float|None, "wingspan": float|None}
        or None if the page can't be reached / parsed anywhere.
    """
    slug = _name_to_slug(player_name)
    url = f"{TWO_K_BASE_URL}/{slug}"

    try:
        resp = _get_with_retry(url)
        if resp is None:
            return None
        if resp.status_code != 200:
            archived_url = _wayback_snapshot_url(url)
            if archived_url is None:
                return None
            # web.archive.org itself is the slow host here too (see the
            # latency note on `_wayback_snapshot_url`), not 2kratings.com.
            resp = _get_with_retry(archived_url, attempts=4, timeout=25)
            if resp is None or resp.status_code != 200:
                return None

        soup = BeautifulSoup(resp.content, "lxml")
        info_div = soup.find("div", class_="player-info")
        if not info_div:
            return None

        block_text = info_div.get_text(separator="", strip=True)
        height_val = _extract_field(block_text, "Height")
        weight_val = _extract_field(block_text, "Weight")
        wingspan_val = _extract_field(block_text, "Wingspan")

        return {
            "height": _parse_feet_inches(height_val) if height_val else None,
            "weight": _parse_weight(weight_val) if weight_val else None,
            "wingspan": _parse_feet_inches(wingspan_val) if wingspan_val else None,
        }

    except Exception as e:
        print(f"\n  ✗ 2K scrape error for {player_name} ({slug}): {e}")
        return None


def ingest_physicals_2k():
    """
    Phase 2C: For every player still missing wingspan in the DB,
    attempt to scrape it from 2kratings.com.

    Only updates columns that are still NULL — never overwrites
    existing data from the NBA API or Basketball Reference.
    """
    engine = get_engine()
    Session = get_session_factory(engine)

    with Session() as session:
        # Find unique players still missing wingspan
        query = (
            select(Player.player_id, Player.name)
            .where(Player.wingspan.is_(None))
            .distinct()
        )
        missing_players = session.execute(query).all()

        if not missing_players:
            print("\n  ✓ All players already have wingspan data. Nothing to do.")
            return

        print(f"\n  Found {len(missing_players)} unique players missing wingspan.")
        print(f"  Estimated time: ~{len(missing_players) * TWO_K_DELAY / 60:.0f} minutes\n")

        updated_count = 0
        not_found_count = 0
        not_found_names = []

        for pid, name in tqdm(missing_players, desc="  2K Physicals"):
            phys = scrape_2k_physicals(name)

            if phys and phys["wingspan"]:
                # Build update dict — only fill in what's still missing
                update_vals = {}

                # Check if height is also missing for this player
                existing = session.execute(
                    select(Player.height, Player.weight)
                    .where(Player.player_id == pid)
                    .limit(1)
                ).first()

                if existing and existing.height is None and phys["height"]:
                    update_vals["height"] = phys["height"]
                if existing and existing.weight is None and phys["weight"]:
                    update_vals["weight"] = phys["weight"]

                # Always set wingspan (that's why we're here)
                update_vals["wingspan"] = phys["wingspan"]
                update_vals["wingspan_source"] = "2K"

                stmt = (
                    update(Player)
                    .where(Player.player_id == pid)
                    .values(**update_vals)
                )
                session.execute(stmt)
                updated_count += 1
            else:
                not_found_count += 1
                if len(not_found_names) < 20:  # Keep a sample
                    not_found_names.append(name)

            time.sleep(TWO_K_DELAY)

        session.commit()

        print(f"\n  ✓ Updated wingspan for {updated_count} players (source: 2K).")
        if not_found_count:
            print(f"  ⚠ {not_found_count} players not found on 2kratings.com.")
            if not_found_names:
                print(f"    Sample: {', '.join(not_found_names[:10])}")


if __name__ == "__main__":
    ingest_physicals_2k()
