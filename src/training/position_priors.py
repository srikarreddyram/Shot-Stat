"""
DEPRECATED — superseded by empirical-Bayes shrinkage.

Kept so the `position_priors` table can still be rebuilt for the old pipeline.
Do not extend it.

This module existed to give a player with zero NBA history a labelled starting
estimate, which then had to be special-cased through the entire stack via
`stats_source="prior"`. `src/features/shrinkage.py` makes that a continuum
rather than a special case: a rookie's first shot resolves to the league prior,
his four-hundredth barely uses it, and no consumer has to branch on which case
it is or surface a different provenance string.

Use:
    from src.features.shrinkage import fit_beta_prior, shrink
"""

# Original module docstring follows.
"""
Position Priors — historical rookie-season averages bucketed by position
group (G / F / C), refined by height within each bucket where there's
enough sample to trust it (e.g. "C-tall" vs "C-short").

This is the ONLY sanctioned exception to the project's "exact data or
NULL, never impute" policy (docs/prd_checklist.md §5): it exists solely to
give a labeled, clearly-flagged fallback estimate for players with zero
real NBA shot history (e.g. this year's draft class before their first
game). See the PositionPrior docstring in src/db/models.py for the full
rationale — any consumer of these rows must be able to tell they came
from here rather than from measured data.

Two tiers of rows are written, both keyed by `position_bucket`:
  - Coarse: "G", "F", "C" — always written, broadest historical sample.
  - Fine:   "G-short", "G-tall", "F-short", "F-tall", "C-short", "C-tall"
            — written only when MIN_SAMPLE thresholds are met, so a rare
            combination (e.g. very few historical rookie centers shooting
            corner 3s) doesn't produce a noisy estimate. A resolver should
            try the fine key first (using src.common.position_bucket.
            fine_bucket on the rookie's own measured height) and fall back
            to the coarse key when the fine row doesn't exist.

"Rookie season" for a player = the earliest season for which they have at
least one row in the `shots` table. Only that specific season's shots /
defender_stats / players row are used for that player — not their whole
career. Players with no shots history at all contribute nothing (there's
no rookie season to anchor on).

Aggregation is done in SQL (GROUP BY) against the real 3.5M-row `shots`
table rather than in a Python loop, so this runs in seconds. The one
per-player step that has to happen in Python is position/height bucketing
(src.common.position_bucket is not expressible in SQL) — that small
(player_id -> rookie_season, bucket, fine_bucket) mapping is written to a
temp table and joined back against shots/defender_stats/players in SQL.

Usage:
    python -m src.training.position_priors
"""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db.database import get_engine, get_session_factory
from src.db.models import PositionPrior
from src.common.position_bucket import position_bucket, fine_bucket

ZONES = [
    "Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
    "Left Corner 3", "Right Corner 3", "Above the Break 3",
]
THREE_PT_ZONES = ("Left Corner 3", "Right Corner 3", "Above the Break 3")
DEFENSE_CATEGORIES = [
    "Overall", "3 Pointers", "2 Pointers",
    "Less Than 6Ft", "Less Than 10Ft", "Greater Than 15Ft",
]
BUCKETS = ["G", "F", "C"]
FINE_BUCKETS = [f"{b}-{band}" for b in BUCKETS for band in ("short", "tall")]

# A fine (position+height) row is only written if it clears these — below
# this, the noise from a small sample is worse than just using the coarser
# position-only estimate. Zone attempts are shot-level (plentiful); defense
# and overall rows are player-season-level (much scarcer), hence the lower bar.
MIN_ZONE_ATTEMPTS = 500
MIN_DEFENSE_ROWS = 15
MIN_OVERALL_ROWS = 15


def _build_rookie_bucket_map(conn) -> pd.DataFrame:
    """
    One row per player_id who has ever attempted a shot: their rookie
    season (earliest season with a `shots` row), the position/height
    as-of that season, and the resulting bucket ("C") plus fine bucket
    ("C-tall", or None if height is unavailable).

    Position/height fallback: if the players row for the rookie season
    itself has a NULL position or height, fall back to the earliest other
    season for that player that does have a non-null value — physical
    attributes and position are near-constant across a career, so this is
    a reasonable substitute, not an imputation of the on-court stats
    themselves.
    """
    rookie_seasons = pd.read_sql(
        text("SELECT player_id, MIN(season) AS rookie_season FROM shots GROUP BY player_id"),
        conn,
    )

    players = pd.read_sql(text("SELECT player_id, season, position, height FROM players"), conn)

    merged = rookie_seasons.merge(
        players,
        left_on=["player_id", "rookie_season"],
        right_on=["player_id", "season"],
        how="left",
    )[["player_id", "rookie_season", "position", "height"]]

    for col in ("position", "height"):
        fallback = (
            players.dropna(subset=[col])
            .sort_values("season")
            .drop_duplicates(subset="player_id", keep="first")
            .rename(columns={col: f"fallback_{col}"})[["player_id", f"fallback_{col}"]]
        )
        merged = merged.merge(fallback, on="player_id", how="left")
        merged[col] = merged[col].fillna(merged[f"fallback_{col}"])
        merged = merged.drop(columns=[f"fallback_{col}"])

    # position_bucket()/fine_bucket() expect a str/float or None — pandas
    # gives us NaN for missing values, so normalize before calling them.
    merged["bucket"] = merged["position"].apply(
        lambda p: position_bucket(p) if isinstance(p, str) else None
    )
    merged["fine"] = merged.apply(
        lambda r: fine_bucket(r["height"] if pd.notna(r["height"]) else None,
                               r["position"] if isinstance(r["position"], str) else None),
        axis=1,
    )
    merged = merged[merged["bucket"].notna()][["player_id", "rookie_season", "bucket", "fine"]]
    return merged.reset_index(drop=True)


def _clean(val):
    """Convert pandas/numpy NaN to None for SQLite insertion."""
    if val is None or pd.isna(val):
        return None
    return val


def _zone_rows(conn, group_col: str, min_attempts: int) -> list[dict]:
    """Offense priors (stat_type='zone'), grouped by the given temp-table
    column ('bucket' for coarse, 'fine' for position+height). Rows below
    min_attempts are dropped — see MIN_ZONE_ATTEMPTS for why."""
    zone_placeholders = ", ".join(f"'{z}'" for z in ZONES)
    zone_df = pd.read_sql(text(f"""
        SELECT t.{group_col} AS position_bucket, s.zone AS stat_key,
               SUM(s.shot_made) AS makes, COUNT(*) AS attempts,
               MAX(t.rookie_season) AS max_season
        FROM shots s
        JOIN _tmp_rookie_bucket t
          ON t.player_id = s.player_id AND t.rookie_season = s.season
        WHERE s.zone IN ({zone_placeholders}) AND t.{group_col} IS NOT NULL
        GROUP BY t.{group_col}, s.zone
    """), conn)

    fg3_df = pd.read_sql(text(f"""
        SELECT t.{group_col} AS position_bucket, s.zone AS stat_key,
               SUM(s.shot_made) AS fg3_makes, COUNT(*) AS fg3_attempts
        FROM shots s
        JOIN _tmp_rookie_bucket t
          ON t.player_id = s.player_id AND t.rookie_season = s.season
        WHERE s.shot_type = '3PT Field Goal'
          AND s.zone IN ('Left Corner 3', 'Right Corner 3', 'Above the Break 3')
          AND t.{group_col} IS NOT NULL
        GROUP BY t.{group_col}, s.zone
    """), conn)

    zone_df = zone_df.merge(fg3_df, on=["position_bucket", "stat_key"], how="left")

    out = []
    for _, r in zone_df.iterrows():
        attempts = r["attempts"]
        if attempts < min_attempts:
            continue
        fg_pct = (r["makes"] / attempts) if attempts else None
        fg3_attempts = r.get("fg3_attempts")
        fg3_pct = None
        if pd.notna(fg3_attempts) and fg3_attempts:
            fg3_pct = r["fg3_makes"] / fg3_attempts
        out.append({
            "position_bucket": r["position_bucket"], "stat_type": "zone", "stat_key": r["stat_key"],
            "fg_pct": _clean(fg_pct), "fg3_pct": _clean(fg3_pct),
            "d_fg_pct": None, "pct_plusminus": None, "ast": None, "tov": None, "ft_pct": None,
            "sample_size": int(attempts), "computed_through_season": r["max_season"],
        })
    return out


def _defense_rows(conn, group_col: str, min_rows: int) -> list[dict]:
    """Defense priors (stat_type='defense'). Shot/attempt-weighted average,
    using `gp` (games played) as the weight — defender_stats doesn't expose
    a direct attempt count against that player, and gp is the best available
    proxy for how much data underlies a given player's d_fg_pct/pct_plusminus.
    Rows with gp NULL or 0 are excluded (no meaningful weight)."""
    defense_df = pd.read_sql(text(f"""
        SELECT t.{group_col} AS position_bucket, d.defense_category AS stat_key,
               SUM(CASE WHEN d.d_fg_pct IS NOT NULL THEN d.d_fg_pct * d.gp ELSE 0 END) AS wsum_dfg,
               SUM(CASE WHEN d.d_fg_pct IS NOT NULL THEN d.gp ELSE 0 END) AS wgt_dfg,
               SUM(CASE WHEN d.pct_plusminus IS NOT NULL THEN d.pct_plusminus * d.gp ELSE 0 END) AS wsum_pm,
               SUM(CASE WHEN d.pct_plusminus IS NOT NULL THEN d.gp ELSE 0 END) AS wgt_pm,
               COUNT(*) AS n_rows,
               MAX(t.rookie_season) AS max_season
        FROM defender_stats d
        JOIN _tmp_rookie_bucket t
          ON t.player_id = d.player_id AND t.rookie_season = d.season
        WHERE d.gp IS NOT NULL AND d.gp > 0 AND t.{group_col} IS NOT NULL
        GROUP BY t.{group_col}, d.defense_category
    """), conn)

    out = []
    for _, r in defense_df.iterrows():
        if r["n_rows"] < min_rows:
            continue
        d_fg_pct = (r["wsum_dfg"] / r["wgt_dfg"]) if r["wgt_dfg"] else None
        pct_pm = (r["wsum_pm"] / r["wgt_pm"]) if r["wgt_pm"] else None
        out.append({
            "position_bucket": r["position_bucket"], "stat_type": "defense", "stat_key": r["stat_key"],
            "fg_pct": None, "fg3_pct": None,
            "d_fg_pct": _clean(d_fg_pct), "pct_plusminus": _clean(pct_pm),
            "ast": None, "tov": None, "ft_pct": None,
            "sample_size": int(r["n_rows"]), "computed_through_season": r["max_season"],
        })
    return out


def _overall_rows(conn, group_col: str, min_rows: int) -> list[dict]:
    """Overall priors (stat_type='overall', stat_key='career'). Simple
    unweighted mean of non-null values — the players table has no natural
    per-row weight (e.g. minutes/games) at this granularity. AVG() ignores
    NULLs in SQLite, matching "mean of non-null values" directly."""
    overall_df = pd.read_sql(text(f"""
        SELECT t.{group_col} AS position_bucket,
               AVG(p.season_fg_pct) AS fg_pct,
               AVG(p.career_3p_pct) AS fg3_pct,
               AVG(p.ft_pct) AS ft_pct,
               AVG(p.ast) AS ast,
               AVG(p.tov) AS tov,
               COUNT(*) AS n_rows,
               MAX(t.rookie_season) AS max_season
        FROM players p
        JOIN _tmp_rookie_bucket t
          ON t.player_id = p.player_id AND t.rookie_season = p.season
        WHERE t.{group_col} IS NOT NULL
        GROUP BY t.{group_col}
    """), conn)

    out = []
    for _, r in overall_df.iterrows():
        if r["n_rows"] < min_rows:
            continue
        out.append({
            "position_bucket": r["position_bucket"], "stat_type": "overall", "stat_key": "career",
            "fg_pct": _clean(r["fg_pct"]), "fg3_pct": _clean(r["fg3_pct"]),
            "d_fg_pct": None, "pct_plusminus": None,
            "ast": _clean(r["ast"]), "tov": _clean(r["tov"]), "ft_pct": _clean(r["ft_pct"]),
            "sample_size": int(r["n_rows"]), "computed_through_season": r["max_season"],
        })
    return out


def compute_position_priors() -> list[dict]:
    """
    Compute all position-prior rows (offense/zone, defense, overall) at
    both the coarse (G/F/C) and fine (e.g. "C-tall") granularity, and
    upsert them into the position_priors table. Fine rows are only written
    where MIN_ZONE_ATTEMPTS/MIN_DEFENSE_ROWS/MIN_OVERALL_ROWS are met — a
    resolver should look up the fine key first and fall back to the coarse
    one when it's missing. Returns the list of row dicts that were written.
    """
    engine = get_engine()
    rows = []

    with engine.connect() as conn:
        bucket_map = _build_rookie_bucket_map(conn)
        n_fine = bucket_map["fine"].notna().sum()
        print(f"  {len(bucket_map)} players with a rookie season and resolvable position bucket "
              f"({n_fine} of those also have a resolvable height band) "
              f"(of {len(pd.read_sql(text('SELECT DISTINCT player_id FROM players'), conn))} total players)")

        bucket_map.to_sql("_tmp_rookie_bucket", conn, if_exists="replace", index=False)
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_tmp_rookie_bucket ON _tmp_rookie_bucket(player_id, rookie_season)"
        ))
        conn.commit()

        rows += _zone_rows(conn, "bucket", min_attempts=0)
        rows += _defense_rows(conn, "bucket", min_rows=0)
        rows += _overall_rows(conn, "bucket", min_rows=0)

        rows += _zone_rows(conn, "fine", min_attempts=MIN_ZONE_ATTEMPTS)
        rows += _defense_rows(conn, "fine", min_rows=MIN_DEFENSE_ROWS)
        rows += _overall_rows(conn, "fine", min_rows=MIN_OVERALL_ROWS)

        conn.execute(text("DROP TABLE IF EXISTS _tmp_rookie_bucket"))
        conn.commit()

    # ── Upsert into position_priors ─────────────────────────────────────────
    Session = get_session_factory(engine)
    with Session() as session:
        for row in rows:
            stmt = sqlite_upsert(PositionPrior.__table__).values(**row)
            stmt = stmt.on_conflict_do_update(
                index_elements=["position_bucket", "stat_type", "stat_key"],
                set_={
                    "fg_pct": stmt.excluded.fg_pct,
                    "fg3_pct": stmt.excluded.fg3_pct,
                    "d_fg_pct": stmt.excluded.d_fg_pct,
                    "pct_plusminus": stmt.excluded.pct_plusminus,
                    "ast": stmt.excluded.ast,
                    "tov": stmt.excluded.tov,
                    "ft_pct": stmt.excluded.ft_pct,
                    "sample_size": stmt.excluded.sample_size,
                    "computed_through_season": stmt.excluded.computed_through_season,
                },
            )
            session.execute(stmt)
        session.commit()

    print(f"  ✓ Upserted {len(rows)} position_priors rows")
    return rows


def _fmt_pct(v):
    return f"{v:.3f}" if v is not None else "  N/A"


def print_report(rows: list[dict]):
    """Print a human-readable sanity-check report grouped by position bucket."""
    by_key = {(r["position_bucket"], r["stat_type"], r["stat_key"]): r for r in rows}

    print(f"\n{'='*72}")
    print("  POSITION PRIORS — SUMMARY REPORT")
    print(f"{'='*72}")

    for bucket in BUCKETS:
        print(f"\n── Position bucket: {bucket} " + "─" * (50 - len(bucket)))

        print("  Zone FG% (offense, rookie-season shots):")
        for zone in ZONES:
            r = by_key.get((bucket, "zone", zone))
            if r is None:
                print(f"    {zone:<26} no data")
                continue
            fg3_part = f"  fg3%={_fmt_pct(r['fg3_pct'])}" if zone in THREE_PT_ZONES else ""
            print(f"    {zone:<26} fg%={_fmt_pct(r['fg_pct'])}{fg3_part}  n={r['sample_size']}")

        print("  Defense d_FG% allowed (rookie-season defender_stats):")
        for cat in DEFENSE_CATEGORIES:
            r = by_key.get((bucket, "defense", cat))
            if r is None:
                print(f"    {cat:<20} no data")
                continue
            print(f"    {cat:<20} d_fg%={_fmt_pct(r['d_fg_pct'])}  +/-={_fmt_pct(r['pct_plusminus'])}  n={r['sample_size']}")

        r = by_key.get((bucket, "overall", "career"))
        if r is not None:
            print("  Overall (rookie-season players-table averages):")
            print(f"    season_fg%={_fmt_pct(r['fg_pct'])}  3p%={_fmt_pct(r['fg3_pct'])}  "
                  f"ft%={_fmt_pct(r['ft_pct'])}  ast={_fmt_pct(r['ast'])}  tov={_fmt_pct(r['tov'])}  "
                  f"n={r['sample_size']}  through={r['computed_through_season']}")

    print(f"\n{'='*72}")
    print("  FINE (POSITION + HEIGHT) BUCKETS — only where sample size clears the bar")
    print(f"  (min {MIN_ZONE_ATTEMPTS} shot attempts / {MIN_DEFENSE_ROWS} defender-seasons / {MIN_OVERALL_ROWS} rookie-seasons)")
    print(f"{'='*72}")
    for fb in FINE_BUCKETS:
        zone_rows = {z: by_key[(fb, "zone", z)] for z in ZONES if (fb, "zone", z) in by_key}
        if not zone_rows and (fb, "overall", "career") not in by_key:
            print(f"\n── {fb}: no rows cleared the sample-size threshold (falls back to coarse bucket)")
            continue
        print(f"\n── Fine bucket: {fb} " + "─" * (46 - len(fb)))
        for zone in ZONES:
            r = zone_rows.get(zone)
            if r is None:
                print(f"    {zone:<26} below threshold — falls back to coarse")
                continue
            print(f"    {zone:<26} fg%={_fmt_pct(r['fg_pct'])}  n={r['sample_size']}")

    # Quick built-in sanity checks
    ra_g = by_key.get(("G", "zone", "Restricted Area"))
    ra_c = by_key.get(("C", "zone", "Restricted Area"))
    abt3_g = by_key.get(("G", "zone", "Above the Break 3"))
    abt3_c = by_key.get(("C", "zone", "Above the Break 3"))
    ra_c_short = by_key.get(("C-short", "zone", "Restricted Area"))
    ra_c_tall = by_key.get(("C-tall", "zone", "Restricted Area"))
    print(f"\n{'='*72}")
    print("  SANITY CHECKS")
    print(f"{'='*72}")
    if ra_g and ra_c:
        ok = ra_c["fg_pct"] > ra_g["fg_pct"]
        print(f"  Centers > Guards at rim (Restricted Area):  "
              f"C={_fmt_pct(ra_c['fg_pct'])} vs G={_fmt_pct(ra_g['fg_pct'])}  -> {'PASS' if ok else 'FAIL'}")
    if abt3_g and abt3_c:
        ok = abt3_g["fg_pct"] > abt3_c["fg_pct"]
        print(f"  Guards > Centers from 3 (Above the Break 3): "
              f"G={_fmt_pct(abt3_g['fg_pct'])} vs C={_fmt_pct(abt3_c['fg_pct'])}  -> {'PASS' if ok else 'FAIL'}")
    if ra_c_short and ra_c_tall:
        ok = ra_c_tall["fg_pct"] >= ra_c_short["fg_pct"]
        print(f"  Tall centers >= short centers at rim:       "
              f"tall={_fmt_pct(ra_c_tall['fg_pct'])} vs short={_fmt_pct(ra_c_short['fg_pct'])}  -> {'PASS' if ok else 'FAIL (not necessarily a bug — check n)'}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    print("Computing position priors from rookie-season data...")
    written_rows = compute_position_priors()
    print_report(written_rows)
