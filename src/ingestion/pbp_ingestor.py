"""
Play-by-play ingestor — per-shot context.

Populates `shot_context` from PlayByPlayV3, one API call per game. For every
field-goal event it records whether the shot was assisted, the league's own
shot-type label, whether it followed an offensive rebound, and how long since
the previous event.

Why this is worth three hours of API calls
------------------------------------------
Everything the model knew about how a shot came about was a season average —
a player's typical dribble count, his typical openness. That describes a
player, not a shot. Play-by-play is the first source that distinguishes two
above-the-break threes by the same shooter: one catch-and-shoot off a kick-out,
one step-back over a set defender.

Resumability
------------
Games already present in `shot_context` are skipped, so the run can be
interrupted and restarted freely — which matters when it takes hours and the
NBA's endpoint intermittently drops requests.

Usage:
    python -m src.ingestion.pbp_ingestor --seasons 2016-17+   # the training window
    python -m src.ingestion.pbp_ingestor --season 2023-24
    python -m src.ingestion.pbp_ingestor --limit 50           # smoke test
"""
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playbyplayv3
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db.database import get_engine, init_db
from src.db.models import ShotContext

# Fetching is network-bound, so threads cost almost no CPU. The constraint is
# not this machine, it is the endpoint.
#
# Measured the hard way: six workers at 0.65s spacing (~9 requests/second) ran
# beautifully for about 1,600 games and then the endpoint throttled the client
# completely — every subsequent request, including single ones from a fresh
# process, timed out at 20 seconds. Recovery took minutes of total silence.
#
# Two workers at 1.0s spacing is ~2 requests/second, close to the serial rate
# that ran for hours without trouble. The full backfill is a multi-hour job at
# this rate and that is simply the cost; going faster does not finish sooner,
# it finishes never.
MAX_WORKERS = 2

REQUEST_DELAY = 1.0
RETRY_DELAY = 8.0
MAX_RETRIES = 3

# SQLite takes a single writer, so all database work happens on the main thread
# behind this lock while the workers only fetch.
_WRITE_LOCK = threading.Lock()

# "Davis 1' Dunk (2 PTS) (Russell 1 AST)".
#
# ⚠ `is_assisted` IS TARGET LEAKAGE AND MUST NEVER BE A MODEL FEATURE.
#
# An assist is only credited on a MADE basket, so a missed shot can never carry
# the clause. The column is a perfect predictor of the label — measured FG% is
# exactly 1.000 for is_assisted=1 and 0.230 for is_assisted=0. A model given
# this feature would score near-perfectly offline and be worthless in
# production, because at prediction time nobody knows whether the shot that
# has not been taken yet will be assisted.
#
# It is still worth storing: it supports descriptive work (a player's assisted
# rate is a real and useful quantity) and it is the correct denominator for
# self-creation analysis. It is excluded from FEATURE_GROUPS by construction,
# and tests/test_no_leaky_features.py asserts it stays excluded.
#
# The clause is "(Anthony 1 AST)" — the assister's NAME sits between the open
# paren and the count, so an anchor on "\(\d+" never matches. Anchoring on the
# trailing "N AST)" is both correct and robust to name formatting.
_AST_RE = re.compile(r"\d+\s+AST\)")

# ISO-8601 duration as the endpoint reports the game clock: "PT11M42.00S".
_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")

# An offensive rebound this recently before a shot makes it a putback. Four
# seconds is deliberately tight: a longer window sweeps in ordinary reset
# possessions that merely began with an offensive board.
PUTBACK_WINDOW_SECONDS = 4.0


def _clock_seconds(clock: str) -> float | None:
    """'PT11M42.00S' → seconds remaining in the period."""
    if not isinstance(clock, str):
        return None
    m = _CLOCK_RE.match(clock)
    if not m:
        return None
    return int(m.group(1)) * 60 + float(m.group(2))


def _fetch_game(game_id: str) -> pd.DataFrame | None:
    for attempt in range(MAX_RETRIES):
        try:
            frames = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=45).get_data_frames()
            if not frames or frames[0].empty:
                return None
            return frames[0]
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return None
    return None


def extract_context(df: pd.DataFrame, game_id: str) -> list[dict]:
    """
    Turn one game's play-by-play into per-shot context rows.

    Pure function over the endpoint's frame, so it can be tested without a
    network call.
    """
    if df.empty:
        return []

    df = df.sort_values("actionNumber").reset_index(drop=True)
    df["_secs"] = df["clock"].map(_clock_seconds)

    rows = []
    for i, ev in df.iterrows():
        if not ev.get("isFieldGoal"):
            continue

        # Previous event, for the transition and putback signals. Only within
        # the same period — the gap across a period break is not game time.
        prev = None
        for j in range(i - 1, -1, -1):
            if df.at[j, "period"] == ev["period"]:
                prev = df.loc[j]
                break

        elapsed = None
        prev_type = None
        if prev is not None:
            prev_type = prev.get("actionType")
            a, b = ev["_secs"], prev["_secs"]
            if a is not None and b is not None:
                # The clock counts DOWN, so time elapsed is previous minus current.
                elapsed = max(0.0, float(b) - float(a))

        is_putback = 0
        if (
            prev is not None
            and prev.get("actionType") == "Rebound"
            and "Off:" in str(prev.get("description", ""))
            and elapsed is not None
            and elapsed <= PUTBACK_WINDOW_SECONDS
            # Same team — a defensive rebound by the other side then a quick
            # shot is a fast break, not a putback.
            and prev.get("teamId") == ev.get("teamId")
        ):
            is_putback = 1

        description = str(ev.get("description", ""))
        rows.append({
            "shot_id": f"{game_id}_{int(ev['actionNumber'])}",
            "game_id": game_id,
            "is_assisted": 1 if _AST_RE.search(description) else 0,
            "shot_subtype": (ev.get("subType") or None),
            "action_type": (ev.get("actionType") or None),
            "is_putback": is_putback,
            "seconds_since_prev_event": elapsed,
            "prev_event_type": prev_type,
            "period": int(ev["period"]) if pd.notna(ev.get("period")) else None,
        })

    return rows


def _pending_games(engine, seasons: list[str] | None, limit: int | None) -> list[str]:
    """
    Games with shots but no context yet, NEWEST first.

    Order matters more than it looks. `build_matrix` refuses to use these
    features until they cover 90% of the training window, because a partial
    backfill in game-id order would cover only the oldest seasons — making
    "context is missing" a near-perfect proxy for "recent season", which the
    model would happily learn instead of the mechanics. Working backwards from
    the present means the test and validation seasons complete first, so the
    features become usable for a real experiment long before the full history
    is done.
    """
    clause = ""
    params = {}
    if seasons:
        placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
        clause = f"AND s.season IN ({placeholders})"
        params = {f"s{i}": s for i, s in enumerate(seasons)}

    sql = f"""
        SELECT DISTINCT s.game_id
        FROM shots s
        WHERE NOT EXISTS (
            SELECT 1 FROM shot_context c WHERE c.game_id = s.game_id
        )
        {clause}
        ORDER BY s.game_id DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"

    with engine.connect() as conn:
        return [r[0] for r in conn.execute(text(sql), params).fetchall()]


def ingest(seasons: list[str] | None = None, limit: int | None = None) -> int:
    engine = get_engine()
    init_db(engine)

    games = _pending_games(engine, seasons, limit)

    print(f"\n{'='*62}")
    print("  PLAY-BY-PLAY INGESTOR (per-shot context)")
    print(f"  {len(games):,} games pending")
    print(f"  ~{len(games) * (REQUEST_DELAY + 0.4) / MAX_WORKERS / 3600:.1f} hours "
          f"at {MAX_WORKERS} workers")
    print(f"{'='*62}")

    total = 0
    failed = 0

    def fetch(game_id):
        """Runs on a worker thread: network only, no database access."""
        df = _fetch_game(game_id)
        time.sleep(REQUEST_DELAY)
        return game_id, df

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch, g): g for g in games}
        with tqdm(total=len(games), desc="  Games") as bar:
            for future in as_completed(futures):
                bar.update(1)
                try:
                    game_id, df = future.result()
                except Exception:
                    failed += 1
                    continue
                if df is None:
                    failed += 1
                    continue

                rows = extract_context(df, game_id)
                if not rows:
                    continue

                with _WRITE_LOCK, engine.begin() as conn:
                    for i in range(0, len(rows), 400):
                        chunk = rows[i:i + 400]
                        stmt = sqlite_insert(ShotContext).values(chunk)
                        conn.execute(stmt.on_conflict_do_update(
                            index_elements=["shot_id"],
                            set_={
                                c.name: stmt.excluded[c.name]
                                for c in ShotContext.__table__.columns
                                if c.name != "shot_id"
                            },
                        ))
                total += len(rows)

    print(f"\n{'='*62}")
    print(f"  DONE — {total:,} shot-context rows, {failed} games failed")
    print(f"{'='*62}\n")
    return total


if __name__ == "__main__":
    import argparse

    import config

    parser = argparse.ArgumentParser(description="Ingest per-shot play-by-play context.")
    parser.add_argument("--season", type=str, default=None)
    parser.add_argument("--seasons", type=str, default=None,
                        help="e.g. '2016-17+' for that season onward")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.season:
        target = [args.season]
    elif args.seasons and args.seasons.endswith("+"):
        floor = args.seasons[:-1]
        target = [s for s in config.ALL_SEASONS if s >= floor]
    else:
        target = None

    ingest(seasons=target, limit=args.limit)
