"""
On-court lineup reconstruction — who was actually on the floor for each shot.

Populates `shot_on_court` from the SAME PlayByPlayV3 endpoint pbp_ingestor.py
already calls per game — not a new data source, a deeper read of one already
in use. Every event row it returns carries a `personId`/`teamId`, and
substitution rows ("SUB: X FOR Y") are enough to replay the on-court state
across a whole game without any extra API call.

The one wrinkle: a substitution row's own `personId` is the OUTGOING player
(confirmed by inspection — description "SUB: Reid FOR DiVincenzo" pairs with
personId/playerName == DiVincenzo). The incoming player's name is free text
only. Resolved by matching that name against every other player who appears
on the SAME team, in the SAME game, via a row that DOES carry a real
personId (a shot, rebound, foul, or the outgoing side of another sub) — a
tight, natural search space (a box score has ~8-12 players per team), rather
than matching against a full season roster.

Starters are derived from the substitutions, not fetched separately: a
player who is ever the OUTGOING side of a substitution, but never the
INCOMING side of an earlier one, must have started (you cannot be subbed out
of a game you were not already in). Anyone else seen in the game's events who
never appears as an incoming substitution target also started (played
uninterrupted). This needs no extra API call either.

Known limitation, accepted rather than solved
-----------------------------------------------
A player subbed in who never takes a shot, rebound, foul, turnover, or any
other recordable action for the rest of the game has no OTHER row to resolve
their name against, so that one incoming assignment is skipped — the
previous occupant of that slot is left in the on-court set one substitution
too long. Rare (garbage-time cameo subs), and the resulting features are
already being treated as an approximate reconstruction to test a hypothesis,
not a guaranteed-exact box score.

Usage:
    python -m src.ingestion.lineup_ingestor --seasons 2016-17+
    python -m src.ingestion.lineup_ingestor --limit 50   # smoke test
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
from src.db.models import ShotOnCourt

# Same rate-limit-safe pattern pbp_ingestor.py already proved out over hours
# of real runs against this endpoint — see that file for why 2 workers /
# 1.0s spacing, not a faster guess.
MAX_WORKERS = 2
REQUEST_DELAY = 1.0
RETRY_DELAY = 8.0
MAX_RETRIES = 3

_WRITE_LOCK = threading.Lock()

_SUB_RE = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+)", re.IGNORECASE)


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


def _build_name_index(df: pd.DataFrame) -> dict[tuple, str]:
    """{(team_id, player_name): person_id} from every row that carries both —
    the pool a substitution's incoming free-text name is resolved against."""
    real = df[(df["personId"].notna()) & (df["personId"] != 0) & (df["playerName"] != "")]
    index = {}
    for _, row in real.iterrows():
        index[(row["teamId"], str(row["playerName"]).strip())] = str(int(row["personId"]))
    return index


def extract_lineups(df: pd.DataFrame, game_id: str) -> tuple[list[dict], int]:
    """
    Pure function over one game's play-by-play frame -> shot_on_court rows.

    Returns (rows, unresolved_sub_count) so the caller can track how often
    an incoming player's name could not be matched, without that count
    silently vanishing into a skipped row.
    """
    if df.empty:
        return [], 0

    df = df.sort_values("actionNumber").reset_index(drop=True)
    team_ids = [t for t in df["teamId"].dropna().unique() if t]
    if len(team_ids) != 2:
        return [], 0
    name_index = _build_name_index(df)

    rows = []
    unresolved = 0

    # Rederive each team's on-court five FRESH at the start of every period,
    # rather than replaying substitutions across the whole game from one
    # initial guess. This bounds how far a single bad data point can spread.
    #
    # The failure mode that made this necessary: the NBA's own feed
    # occasionally drops one substitution event (a player has real
    # box-score actions and is later correctly subbed OUT, but no matching
    # "SUB: <him> FOR <someone>" ever brought him IN — confirmed by
    # inspecting the raw play-by-play directly, not a parsing bug). A
    # missing INCOMING event means the corresponding OUTGOING half of some
    # LATER, perfectly well-formed substitution discards a player who was
    # never actually tracked as present — silently growing or shrinking the
    # on-court set for the rest of the GAME, not just for the one player's
    # actual stint. A whole-game replay lets one dropped event corrupt
    # everything after it; a per-period rederivation limits the damage to
    # at most one quarter, since each period gets an independent, from-
    # scratch starting five computed the same principled way.
    for period in sorted(df["period"].dropna().unique()):
        period_df = df[df["period"] == period]
        subs = period_df[period_df["actionType"] == "Substitution"]

        first_incoming: dict[tuple, float] = {}
        resolved_subs = []  # (action_number, team_id, incoming_id, outgoing_id)

        for _, row in subs.iterrows():
            m = _SUB_RE.search(str(row.get("description", "")))
            if not m:
                continue
            incoming_name = m.group(1).strip()
            team_id = row["teamId"]
            outgoing_id = str(int(row["personId"])) if pd.notna(row["personId"]) else None
            incoming_id = name_index.get((team_id, incoming_name))
            if incoming_id is None or outgoing_id is None:
                unresolved += 1
                continue
            key = (team_id, incoming_id)
            if key not in first_incoming or row["actionNumber"] < first_incoming[key]:
                first_incoming[key] = row["actionNumber"]
            resolved_subs.append((row["actionNumber"], team_id, incoming_id, outgoing_id))

        # This period's starting five: whoever's first appearance THIS
        # PERIOD (a shot, rebound, foul, or being the outgoing side of a
        # sub) precedes their first incoming assignment THIS PERIOD — same
        # principle as a game's real starters, just rescoped per period so
        # each one gets its own resync point. A player who rests the whole
        # period never enters this candidate pool at all, which is correct:
        # he did not start it.
        on_court: dict[float, set] = {}
        for team_id in team_ids:
            team_rows = period_df[period_df["teamId"] == team_id]
            first_seen: dict[str, float] = {}
            for _, row in team_rows.iterrows():
                if pd.isna(row["personId"]) or not row["personId"]:
                    continue
                pid = str(int(row["personId"]))
                an = row["actionNumber"]
                if pid not in first_seen or an < first_seen[pid]:
                    first_seen[pid] = an

            candidates = [
                pid for pid, seen_at in first_seen.items()
                if first_incoming.get((team_id, pid)) is None
                or seen_at < first_incoming[(team_id, pid)]
            ]
            candidates.sort(key=lambda pid: first_seen[pid])
            on_court[team_id] = set(candidates[:5])

        sub_idx = 0
        resolved_subs.sort(key=lambda s: s[0])
        # Once a team's tracked five is caught disagreeing with reality —
        # some OTHER event names a player who isn't in the tracked set at
        # all — the set can still happen to read as exactly 5 members (an
        # untracked player standing in for whoever he silently replaced,
        # net size unchanged) and pass the size check while being flatly
        # wrong. A dropped incoming event does not always announce itself
        # as a size mismatch; sometimes it only shows up later, when a
        # perfectly well-formed substitution tries to remove a player who
        # was never tracked as present, discards nothing, and the set grows
        # by one deferred step behind the actual drift. Marking the team
        # broken the instant ANY event exposes the gap — not waiting for a
        # substitution to eventually prove it — is what closes that window.
        broken: dict[float, bool] = {t: False for t in team_ids}

        for _, ev in period_df.iterrows():
            # Apply every substitution up to (not including) this event
            # first, so a shot taken exactly as a sub is being logged sees
            # the state BEFORE that sub — the PBP orders subs immediately
            # after the shot that prompted a coach's decision, not before.
            while (sub_idx < len(resolved_subs)
                   and resolved_subs[sub_idx][0] < ev["actionNumber"]):
                _, team_id, incoming_id, outgoing_id = resolved_subs[sub_idx]
                on_court[team_id].discard(outgoing_id)
                on_court[team_id].add(incoming_id)
                sub_idx += 1

            # Drift check: any event OTHER than a substitution names a real
            # player on a tracked team. If he isn't in that team's current
            # on-court set, the set is wrong right now, regardless of its
            # size — mark it broken for the rest of the period rather than
            # waiting to see whether a later substitution happens to also
            # produce a detectable size mismatch.
            ev_team = ev.get("teamId")
            ev_pid = ev.get("personId")
            if (ev.get("actionType") != "Substitution" and ev_team in team_ids
                    and pd.notna(ev_pid) and ev_pid):
                if str(int(ev_pid)) not in on_court[ev_team]:
                    broken[ev_team] = True

            if not ev.get("isFieldGoal"):
                continue

            offense_team = ev["teamId"]
            defense_team = next((t for t in team_ids if t != offense_team), None)
            if defense_team is None:
                continue
            if broken[offense_team] or broken[defense_team]:
                continue
            offense_five = on_court.get(offense_team, set())
            defense_five = on_court.get(defense_team, set())
            # Both invariants below are the correctness check this
            # reconstruction lives or dies on. A shot whose lineup isn't
            # exactly 5-and-5 at this instant gets NO row at all, rather
            # than a wrong 4-or-6-man "lineup" shipped as if it were real —
            # the training pipeline already treats missing lineup data as
            # an ordinary missing feature.
            if len(offense_five) != 5 or len(defense_five) != 5:
                continue

            shot_id = f"{game_id}_{int(ev['actionNumber'])}"
            for pid in offense_five:
                rows.append({"shot_id": shot_id, "player_id": pid,
                            "team_id": str(int(offense_team)), "role": "offense"})
            for pid in defense_five:
                rows.append({"shot_id": shot_id, "player_id": pid,
                            "team_id": str(int(defense_team)), "role": "defense"})

    return rows, unresolved


def _pending_games(engine, seasons: list[str] | None, limit: int | None) -> list[str]:
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
            SELECT 1 FROM shot_on_court soc
            JOIN shots s2 ON s2.shot_id = soc.shot_id
            WHERE s2.game_id = s.game_id
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
    print("  LINEUP INGESTOR (on-court reconstruction from substitutions)")
    print(f"  {len(games):,} games pending")
    print(f"  ~{len(games) * (REQUEST_DELAY + 0.4) / MAX_WORKERS / 3600:.1f} hours "
          f"at {MAX_WORKERS} workers")
    print(f"{'='*62}")

    total_rows = 0
    total_shots_covered = 0
    failed = 0
    incomplete_games = 0
    unresolved_subs = 0

    def fetch(game_id):
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

                rows, unresolved = extract_lineups(df, game_id)
                unresolved_subs += unresolved
                if not rows:
                    incomplete_games += 1
                    continue

                with _WRITE_LOCK, engine.begin() as conn:
                    for i in range(0, len(rows), 500):
                        chunk = rows[i:i + 500]
                        stmt = sqlite_insert(ShotOnCourt).values(chunk)
                        conn.execute(stmt.on_conflict_do_nothing(
                            index_elements=["shot_id", "player_id"]
                        ))
                total_rows += len(rows)
                total_shots_covered += len(rows) // 10

    print(f"\n{'='*62}")
    print(f"  DONE — {total_rows:,} on-court rows ({total_shots_covered:,} shots "
          f"covered), {failed} games failed to fetch, "
          f"{incomplete_games} games produced no usable lineup data, "
          f"{unresolved_subs} substitutions had an unresolvable incoming name")
    print(f"{'='*62}\n")
    return total_rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct on-court lineups per shot.")
    parser.add_argument("--season", type=str, default=None)
    parser.add_argument("--seasons", type=str, default=None,
                        help="e.g. '2016-17+' for that season onward")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    seasons = None
    if args.season:
        seasons = [args.season]
    elif args.seasons:
        if args.seasons.endswith("+"):
            import config
            floor = args.seasons[:-1]
            seasons = [s for s in config.ALL_SEASONS if s >= floor]
        else:
            seasons = [args.seasons]

    ingest(seasons=seasons, limit=args.limit)
