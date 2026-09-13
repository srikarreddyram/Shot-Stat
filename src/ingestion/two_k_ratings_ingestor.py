"""
NBA 2K Ratings Ingestor.

Populates `player_two_k_ratings` from 2kratings.com — a second, independent
opinion of player quality, meant to check src/inference/player_ratings.py's
own computed ratings against, and eventually to backstop them for a player
our own data has little fresh evidence for (just traded, a rookie, an
injury-shortened recent season).

Unlike src/ingestion/physical_ingestor_2k.py (which scrapes one `player-info`
div with regex), this reads a much richer, already-structured source on the
same pages: a schema.org JSON-LD block embedding every individual 2K
attribute as `{"name": "X Attribute", "value": N}`, confirmed live for
Anthony Davis (NBA 2K27 Rating: 90, 37 individual attributes) and Alex Caruso
(Perimeter Defense: 90, Steal: 97). Parsing real JSON instead of regexing
flattened HTML text is both simpler and less fragile to page changes.

Reuses physical_ingestor_2k.py's slug generation, retry/backoff, and Wayback
Machine fallback rather than re-implementing them — those are generic to
"reach a 2kratings.com player page," not specific to what's scraped off it.

Usage:
    python -m src.ingestion.two_k_ratings_ingestor              # all rostered players
    python -m src.ingestion.two_k_ratings_ingestor --limit 20   # smoke test
"""
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db.database import get_engine, get_session_factory, init_db
from src.db.models import Player, PlayerTwoKRating
from src.ingestion.physical_ingestor_2k import (
    TWO_K_BASE_URL, TWO_K_DELAY, _get_with_retry, _name_to_slug,
    _wayback_snapshot_url,
)

# Attribute-name substrings (2K's own labels, e.g. "Steal Attribute") rolled
# up into offense_avg/defense_avg — a judgement call about which of 2K's ~37
# attributes describe which side of the ball, made explicit here rather than
# implied by which ones happened to get queried. Physical/hustle/potential
# attributes (Speed, Strength, Hustle, Stamina, Durability, Intangibles,
# Potential) are deliberately excluded from both — they're real 2K inputs
# but don't belong to either offense or defense specifically.
OFFENSE_ATTRIBUTES = [
    "Three-Point Shot Attribute", "Mid-Range Shot Attribute", "Close Shot Attribute",
    "Free Throw Attribute", "Shot IQ Attribute", "Offensive Consistency Attribute",
    "Layup Attribute", "Driving Dunk Attribute", "Standing Dunk Attribute",
    "Post Hook Attribute", "Post Fade Attribute", "Post Control Attribute",
    "Draw Foul Attribute", "Hands Attribute", "Ball Handle Attribute",
    "Speed with Ball Attribute", "Pass Accuracy Attribute", "Pass Vision Attribute",
    "Pass IQ Attribute", "Offensive Rebound Attribute",
]
DEFENSE_ATTRIBUTES = [
    "Interior Defense Attribute", "Perimeter Defense Attribute",
    "Block Attribute", "Steal Attribute", "Pass Perception Attribute",
    "Help Defense IQ Attribute", "Defensive Consistency Attribute",
    "Defensive Rebound Attribute",
]

_JSON_LD_RE = re.compile(r"<script[^>]*type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>", re.S)


def _extract_attributes(html: str) -> dict | None:
    """
    Pull `{attribute_name: value}` (plus the "NBA 2K__ Rating" overall entry)
    out of the page's schema.org JSON-LD block. Returns None if the block
    isn't present or doesn't parse — a real possibility on an old Wayback
    snapshot predating this markup, which the caller treats as "not found"
    rather than raising.
    """
    for block in _JSON_LD_RE.findall(html):
        if "additionalProperty" not in block:
            continue
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        graph = data.get("@graph", [data]) if isinstance(data, dict) else data
        for node in graph:
            if isinstance(node, dict) and node.get("@type") == "Person":
                props = node.get("additionalProperty", [])
                return {p["name"]: p["value"] for p in props if "name" in p and "value" in p}
    return None


def scrape_2k_rating(player_name: str) -> dict | None:
    """
    Scrape one player's 2K attribute set. Returns None if the page can't be
    reached/parsed anywhere (live or Wayback), same contract as
    physical_ingestor_2k.scrape_2k_physicals.
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
            resp = _get_with_retry(archived_url, attempts=4, timeout=25)
            if resp is None or resp.status_code != 200:
                return None

        attrs = _extract_attributes(resp.text)
        if not attrs:
            return None

        edition_key = next((k for k in attrs if k.startswith("NBA 2K") and k.endswith("Rating")), None)
        if edition_key is None:
            return None

        offense_vals = [attrs[k] for k in OFFENSE_ATTRIBUTES if k in attrs]
        defense_vals = [attrs[k] for k in DEFENSE_ATTRIBUTES if k in attrs]

        return {
            # "NBA 2K27 Rating" -> "NBA 2K27" — the overall VALUE already
            # says "Rating"; keeping it in the edition string too just
            # duplicates that in every row.
            "edition": edition_key.removesuffix(" Rating"),
            "overall": int(attrs[edition_key]),
            "offense_avg": sum(offense_vals) / len(offense_vals) if offense_vals else None,
            "defense_avg": sum(defense_vals) / len(defense_vals) if defense_vals else None,
            "raw_attributes": json.dumps(attrs),
        }

    except Exception as e:
        print(f"\n  ✗ 2K rating scrape error for {player_name} ({slug}): {e}")
        return None


def ingest_two_k_ratings(limit: int | None = None):
    """
    Scrape a 2K rating for every currently-rostered player (the latest
    season in `players`) who doesn't have one yet. Re-run to refresh —
    upserts, so an existing row is overwritten rather than duplicated.
    """
    engine = get_engine()
    init_db(engine)
    Session = get_session_factory(engine)

    with Session() as session:
        latest_season = session.execute(select(func.max(Player.season))).scalar()

        existing = {
            row[0] for row in session.execute(select(PlayerTwoKRating.player_id)).all()
        }

        rows = session.execute(
            select(Player.player_id, Player.name)
            .where(Player.season == latest_season)
            .distinct()
        ).all()

    already_rated = [(pid, name) for pid, name in rows if pid in existing]
    targets = [(pid, name) for pid, name in rows if pid not in existing]
    truncated = limit is not None and len(targets) > limit
    if limit:
        targets = targets[:limit]

    print(f"\n{'='*60}")
    print("  NBA 2K RATINGS INGESTOR")
    print(f"  {len(targets)} players to fetch"
          + (f" ({len(already_rated)} already have a rating)" if already_rated else "")
          + (f" — truncated by --limit {limit}" if truncated else ""))
    print(f"  Estimated time: ~{len(targets) * TWO_K_DELAY / 60:.0f} minutes")
    print(f"{'='*60}\n")

    found = missing = 0
    missing_names = []

    for pid, name in tqdm(targets, desc="  2K Ratings"):
        rating = scrape_2k_rating(name)
        time.sleep(TWO_K_DELAY)

        if rating is None:
            missing += 1
            if len(missing_names) < 20:
                missing_names.append(name)
            continue

        found += 1
        with engine.begin() as conn:
            stmt = sqlite_insert(PlayerTwoKRating).values(
                player_id=pid, fetched_at=date.today(), **rating
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["player_id"],
                set_={
                    "edition": stmt.excluded.edition,
                    "overall": stmt.excluded.overall,
                    "offense_avg": stmt.excluded.offense_avg,
                    "defense_avg": stmt.excluded.defense_avg,
                    "raw_attributes": stmt.excluded.raw_attributes,
                    "fetched_at": stmt.excluded.fetched_at,
                },
            )
            conn.execute(stmt)

    print(f"\n{'='*60}")
    print(f"  ✓ {found} ratings fetched, {missing} not found")
    if missing_names:
        print(f"  Sample not found: {', '.join(missing_names)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest NBA 2K player ratings.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only fetch this many players (smoke test).")
    args = parser.parse_args()

    ingest_two_k_ratings(limit=args.limit)
