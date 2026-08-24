"""
Player/defender stats resolver.

Decouples "which season is the caller asking about" from "which season's
real data do we actually have." A requested season (e.g. the current
offseason's roster year) may have zero games played and therefore zero
real shot/defender data for every player in the league — that's expected,
not an error condition.

For a requested (player_id, season), resolution falls through three tiers:
  1. Real data for the exact requested season → use it.
  2. Real data for a prior season → use that player's own most recent
     season with real zone/defender data (an active veteran during the
     offseason, or a player who sat out the requested season).
  3. No real data in ANY season (the player has never recorded a single
     real NBA shot/defensive possession — e.g. this year's draft class)
     → fall back to the position-bucket prior computed by
     src/training/position_priors.py. Where the player's own measured
     height lets us compute a finer bucket (e.g. "C-tall" instead of just
     "C") AND that finer bucket had enough historical sample to be written
     (see position_priors.py's MIN_* thresholds), the finer value is used
     per-row; otherwise that row falls back to the coarse position bucket.

Tier 3 is the ONLY sanctioned exception to this project's "exact data or
NULL, never impute" policy — see PositionPrior's docstring in
src/db/models.py for why it exists. Every dict returned from here carries
`stats_source` ("measured" | "prior") and `resolved_season` (the season
whose real data was used, or None for a prior-sourced result) so callers
can always tell which tier a value came from and label it accordingly.
"""
from sqlalchemy import text

from src.common.position_bucket import position_bucket, fine_bucket

ZONES = [
    "Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
    "Left Corner 3", "Right Corner 3", "Above the Break 3",
]
DEFENSE_CATEGORIES = [
    "Overall", "3 Pointers", "2 Pointers",
    "Less Than 6Ft", "Less Than 10Ft", "Greater Than 15Ft",
]


def _identity_row(conn, player_id: str, season: str) -> dict | None:
    """Name/position/physicals for the exact season, falling back to the
    player's most recent row of any season if the exact one is missing."""
    row = conn.execute(text("""
        SELECT name, height, weight, wingspan, position
        FROM players WHERE player_id = :pid AND season = :season
    """), {"pid": player_id, "season": season}).fetchone()
    if row is not None:
        return dict(row._mapping)

    row = conn.execute(text("""
        SELECT name, height, weight, wingspan, position
        FROM players WHERE player_id = :pid
        ORDER BY season DESC LIMIT 1
    """), {"pid": player_id}).fetchone()
    return dict(row._mapping) if row is not None else None


def _most_recent_measured_season(conn, table: str, player_id: str, season: str) -> str | None:
    row = conn.execute(text(f"""
        SELECT MAX(season) FROM {table}
        WHERE player_id = :pid AND season <= :season
    """), {"pid": player_id, "season": season}).fetchone()
    return row[0] if row and row[0] else None


def _position_prior_zone_stats(conn, bucket: str, fine: str | None) -> dict:
    """Zone fg_pct by zone, preferring the fine (position+height) bucket
    per-zone where it exists and falling back to the coarse bucket
    otherwise (a rare zone/height combo may not have cleared the sample
    threshold in position_priors.py even when the coarse bucket did)."""
    rows = conn.execute(text("""
        SELECT stat_key, fg_pct FROM position_priors
        WHERE position_bucket = :b AND stat_type = 'zone'
    """), {"b": bucket}).fetchall()
    result = {r[0]: r[1] for r in rows}
    if fine:
        fine_rows = conn.execute(text("""
            SELECT stat_key, fg_pct FROM position_priors
            WHERE position_bucket = :b AND stat_type = 'zone'
        """), {"b": fine}).fetchall()
        result.update({r[0]: r[1] for r in fine_rows})
    return result


def _position_prior_overall(conn, bucket: str, fine: str | None) -> dict | None:
    row = conn.execute(text("""
        SELECT fg_pct, fg3_pct, ast, tov, ft_pct FROM position_priors
        WHERE position_bucket = :b AND stat_type = 'overall' AND stat_key = 'career'
    """), {"b": bucket}).fetchone()
    result = dict(row._mapping) if row is not None else None
    if fine:
        fine_row = conn.execute(text("""
            SELECT fg_pct, fg3_pct, ast, tov, ft_pct FROM position_priors
            WHERE position_bucket = :b AND stat_type = 'overall' AND stat_key = 'career'
        """), {"b": fine}).fetchone()
        if fine_row is not None:
            result = dict(fine_row._mapping)
    return result


def _position_prior_defense(conn, bucket: str, fine: str | None) -> dict:
    rows = conn.execute(text("""
        SELECT stat_key, d_fg_pct, pct_plusminus FROM position_priors
        WHERE position_bucket = :b AND stat_type = 'defense'
    """), {"b": bucket}).fetchall()
    result = {r[0]: {"d_fg_pct": r[1], "pct_plusminus": r[2], "freq": None} for r in rows}
    if fine:
        fine_rows = conn.execute(text("""
            SELECT stat_key, d_fg_pct, pct_plusminus FROM position_priors
            WHERE position_bucket = :b AND stat_type = 'defense'
        """), {"b": fine}).fetchall()
        result.update({r[0]: {"d_fg_pct": r[1], "pct_plusminus": r[2], "freq": None} for r in fine_rows})
    return result


def _recent_form(conn, player_id: str) -> dict:
    """
    Shot-weighted FG% over this player's most recent 10 / 20 real games
    (across their full history, any season) — the inference-time
    counterpart of feature_engineering.py's leakage-safe rolling window.
    At inference time "as of right now" naturally means their true most
    recent games, so there's no leakage concern to guard against here the
    way there is at training time.
    """
    rows = conn.execute(text("""
        SELECT s.game_id, SUM(s.shot_made) AS makes, COUNT(*) AS attempts
        FROM shots s
        JOIN games g ON s.game_id = g.game_id
        WHERE s.player_id = :pid
        GROUP BY s.game_id, g.date
        ORDER BY g.date DESC
        LIMIT 20
    """), {"pid": player_id}).fetchall()

    result = {}
    for window, label in ((10, "recent_10_fg"), (20, "recent_20_fg")):
        games = rows[:window]
        attempts = sum(r[2] for r in games)
        result[label] = (sum(r[1] for r in games) / attempts) if attempts else None
    return result


def resolve_player_stats(conn, player_id: str, season: str) -> dict | None:
    """Attacker-side lookup: identity + zone_stats + career/season shooting."""
    identity = _identity_row(conn, player_id, season)
    if identity is None:
        return None

    measured_season = _most_recent_measured_season(conn, "player_zone_stats", player_id, season)

    if measured_season is not None:
        zone_rows = conn.execute(text("""
            SELECT zone, fg_pct FROM player_zone_stats
            WHERE player_id = :pid AND season = :season
        """), {"pid": player_id, "season": measured_season}).fetchall()
        zone_stats = {r[0]: r[1] for r in zone_rows}

        stat_row = conn.execute(text("""
            SELECT career_fg_pct, career_3p_pct, season_fg_pct, ast, tov, ft_pct
            FROM players WHERE player_id = :pid AND season = :season
        """), {"pid": player_id, "season": measured_season}).fetchone()
        stats = dict(stat_row._mapping) if stat_row is not None else {}

        return {
            **identity,
            **stats,
            **_recent_form(conn, player_id),
            "zone_stats": zone_stats,
            "stats_source": "measured",
            "resolved_season": measured_season,
            "position_bucket": None,
        }

    # Tier 3: no real shot data in any season — labeled position prior,
    # refined by height where we have enough sample (see fine_bucket()).
    bucket = position_bucket(identity.get("position"))
    fine = fine_bucket(identity.get("height"), identity.get("position"))
    zone_stats = _position_prior_zone_stats(conn, bucket, fine) if bucket else {}
    overall = _position_prior_overall(conn, bucket, fine) if bucket else None

    return {
        **identity,
        "career_fg_pct": overall["fg_pct"] if overall else None,
        "career_3p_pct": overall["fg3_pct"] if overall else None,
        "season_fg_pct": overall["fg_pct"] if overall else None,
        "ast": overall["ast"] if overall else None,
        "tov": overall["tov"] if overall else None,
        "ft_pct": overall["ft_pct"] if overall else None,
        # No prior-based substitute for recent form — "recent form" is
        # inherently about THIS player's actual last games, which a true
        # zero-history rookie doesn't have. Left as None (XGBoost treats
        # as missing), not filled from the position prior.
        "recent_10_fg": None,
        "recent_20_fg": None,
        "zone_stats": zone_stats,
        "stats_source": "prior",
        "resolved_season": None,
        "position_bucket": fine or bucket,
    }


def resolve_defender_stats(conn, defender_id: str, season: str) -> dict | None:
    """Defender-side lookup: identity + per-category d_fg_pct/pct_plusminus."""
    identity = _identity_row(conn, defender_id, season)
    if identity is None:
        return None

    measured_season = _most_recent_measured_season(conn, "defender_stats", defender_id, season)

    if measured_season is not None:
        def_rows = conn.execute(text("""
            SELECT defense_category, d_fg_pct, pct_plusminus, freq
            FROM defender_stats WHERE player_id = :pid AND season = :season
        """), {"pid": defender_id, "season": measured_season}).fetchall()
        def_stats = {r[0]: {"d_fg_pct": r[1], "pct_plusminus": r[2], "freq": r[3]} for r in def_rows}

        return {
            **identity,
            "def_stats": def_stats,
            "stats_source": "measured",
            "resolved_season": measured_season,
            "position_bucket": None,
        }

    bucket = position_bucket(identity.get("position"))
    fine = fine_bucket(identity.get("height"), identity.get("position"))
    def_stats = _position_prior_defense(conn, bucket, fine) if bucket else {}

    return {
        **identity,
        "def_stats": def_stats,
        "stats_source": "prior",
        "resolved_season": None,
        "position_bucket": fine or bucket,
    }
