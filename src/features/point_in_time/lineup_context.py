"""
On-court lineup context — what the REST of the lineup looks like at the
moment of a shot, not the shooter (already fully described elsewhere) and
not the primary defender (`defender_id`, already its own feature group).

This module calls back into `build_defender_category_rates` and
`lookup_defender_category_rates` (defender_quality.py) via
`from src.features import point_in_time as _pit` rather than importing them
directly — deliberately. Tests monkeypatch both by name on the `point_in_time`
package (e.g. `monkeypatch.setattr(point_in_time, "build_defender_category_rates", fake)`),
and a plain `from .defender_quality import build_defender_category_rates`
would bind this module's OWN copy of the name at import time, which a patch
applied to the package's re-exported copy would never reach — the same
"attribute access sees the patch, a frozen name import does not" distinction
documented at length in src/inference/api/__init__.py. Going through `_pit.`
means both this module and any test see the exact same, patchable value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .zones import DEFENSE_CATEGORIES, ZONE_TO_DEF_CATEGORY

LINEUP_FEATURE_COLS = [
    "oncourt_off_creation", "oncourt_off_gravity", "oncourt_off_rim_pressure",
    "oncourt_off_n", "oncourt_def_fg_pct", "oncourt_def_n",
    # Peak-threat versions. A defense honors the single best playmaker on the
    # floor, not the average of him and three role players — averaging four
    # z-scores dilutes an elite passer to a quarter of his actual gravity.
    # These sit alongside (not instead of) the means, since depth of shooting
    #/passing threats is arguably its own real thing; the tree decides which
    # carries signal.
    "oncourt_off_creation_max", "oncourt_off_gravity_max",
    "oncourt_off_rim_pressure_max",
    # Same peak-threat logic applied to foul-drawing: a teammate who draws
    # heavy contact on drives forces the same early-help/rotation stress a
    # gravity teammate does, and until now that pressure only existed as the
    # SHOOTER's own drive_pf_pct — never as a fact about who else is on the
    # floor with him.
    "oncourt_off_foul_rate_max",
    # Toughest single help defender present, mirroring the offense-side
    # peak-threat logic: one elite rim-protector rotating over matters more
    # than the average of him and three replacement-level defenders.
    "oncourt_def_fg_pct_min",
    # Defensive activity peak-threat: does the floor have a defender whose
    # sheer disruptiveness changes what the offense is willing to attempt at
    # all, not just how well a shot that WAS taken gets contested. See
    # src/features/defensive_activity.py for why blocks/deflections/steals
    # matter here specifically (the Wembanyama case: opponents stop
    # attacking the rim when he's patrolling it, regardless of who their
    # primary defender is).
    "oncourt_def_blk_max", "oncourt_def_stl_max",
    "oncourt_def_deflections_max", "oncourt_def_gravity_max",
]


def build_lineup_context(engine, through_season: str) -> pd.DataFrame:
    """
    Per shot, what the REST of the on-court lineup looks like — not the
    shooter (already fully described elsewhere) and not the primary
    defender (`defender_id`, already its own feature group), but the other
    four offensive teammates and the other four defenders who were also on
    the floor for that specific shot.

    This is the experiment the "gravity" and "team/roster" questions this
    session kept circling back to actually need: `playmaking_gravity`
    already exists but only as a SELF-effect (a player's own passing making
    HIS OWN shot marginally easier), and the supporting-cast features
    (cast_ast_rate etc.) are a whole-SEASON roster aggregate, not who was
    literally on the floor for this possession. `shot_on_court` (see
    src/ingestion/lineup_ingestor.py) is what makes the real, per-shot
    version of both questions answerable at all.

    Offense side: both the MEAN and the MAX of the other four teammates'
    self_creation_index / playmaking_gravity / rim_pressure / drive_pf_pct,
    from the same lagged-one-season creation profile the shooter's own
    features already use (so a shot in 2023-24 sees teammates' 2022-23
    profiles — same availability and leakage reasoning as
    attach_creation_features). The max matters as much as the mean here: a
    defense honors the single best playmaker on the floor, not the average
    of him and three role players, so averaging a Jokić-caliber teammate in
    with three replacement-level ones dilutes his gravity to a quarter
    strength and reads as "unremarkable". drive_pf_pct's max captures the
    same peak-threat logic for foul-drawing pressure — a teammate who draws
    heavy contact forces early defensive help the same way an elite passer
    does, and until this was added that pressure only existed as a fact
    about the SHOOTER himself, never about who else is on the floor.

    Defense side: the MEAN and the MIN (i.e. the toughest single defender
    present) point-in-time FG%-allowed of the other four defenders, matched
    to THIS SHOT'S zone via the same ZONE_TO_DEF_CATEGORY mapping the
    primary defender's own def_fg_pct_zone already uses — not the pooled
    "Overall" category. A defender's FG%-allowed pooled across every shot
    type he's ever guarded says little about whether this specific zone is
    covered well: a rim-protecting shot-blocker with slow closeouts reads as
    merely average on the pooled number, which is backwards for a shot at
    the rim and backwards again for a shot from three. Same
    build_defender_category_rates the primary defender's own features
    already use — not a second, differently-built defender-quality number,
    just read at the category that actually matches the shot.

    Defense side, activity: the MAX (peak-threat again, not the mean — see
    src/features/defensive_activity.py) blk_per_min / stl_per_min /
    deflections_per_min / defensive_gravity among the other four defenders,
    lagged one season the same way the offense-side creation profile is.
    Zone-agnostic by design, unlike d_fg_pct above: a rim protector's real
    effect is suppressing whether the offense attacks the paint AT ALL, which
    is a fact about the lineup, not about the zone the shot eventually
    happened to be taken from.

    Coverage is bounded by shot_on_court's own coverage (see
    lineup_ingestor.py — real substitution data gaps mean some shots have
    no reconstructed lineup at all), so these columns are NaN for a real
    share of rows. XGBoost treats that as an ordinary missing feature, the
    same way it already does for contest and defender coverage gaps.
    """
    from src.features import point_in_time as _pit
    from src.features.creation import _previous_season, load_creation_profiles
    from src.features.defensive_activity import load_defensive_activity_profiles

    onc = pd.read_sql("""
        SELECT soc.shot_id, soc.player_id, soc.role, soc.team_id,
               s.season, s.game_id, s.player_id AS shooter_id,
               s.defender_id AS primary_defender_id, s.zone
        FROM shot_on_court soc
        JOIN shots s ON s.shot_id = soc.shot_id
    """, engine)
    if onc.empty:
        return pd.DataFrame(columns=["shot_id"] + LINEUP_FEATURE_COLS)
    onc["category"] = onc["zone"].map(ZONE_TO_DEF_CATEGORY)

    # ── Offense: the other four teammates' creation profile ────────────────
    offense = onc[(onc["role"] == "offense") & (onc["player_id"] != onc["shooter_id"])].copy()
    profiles = load_creation_profiles(engine)
    offense["_lag_season"] = offense["season"].map(_previous_season)
    lagged = profiles.rename(columns={"season": "_lag_season"})
    offense = offense.merge(
        lagged[["player_id", "_lag_season", "self_creation_index",
               "playmaking_gravity", "rim_pressure", "drive_pf_pct"]],
        on=["player_id", "_lag_season"], how="left",
    )
    off_agg = offense.groupby("shot_id").agg(
        oncourt_off_creation=("self_creation_index", "mean"),
        oncourt_off_gravity=("playmaking_gravity", "mean"),
        oncourt_off_rim_pressure=("rim_pressure", "mean"),
        oncourt_off_creation_max=("self_creation_index", "max"),
        oncourt_off_gravity_max=("playmaking_gravity", "max"),
        oncourt_off_rim_pressure_max=("rim_pressure", "max"),
        oncourt_off_foul_rate_max=("drive_pf_pct", "max"),
        oncourt_off_n=("player_id", "count"),
    ).reset_index()

    # ── Defense: the other four defenders' point-in-time quality ───────────
    defense = onc[
        (onc["role"] == "defense")
        & (onc["primary_defender_id"].notna())
        & (onc["player_id"] != onc["primary_defender_id"])
    ].copy()
    rates = _pit.build_defender_category_rates(engine, through_season=through_season)
    category_rates = rates.rename(
        columns={"defense_player_id": "player_id", "defense_category": "category"}
    )[["player_id", "game_id", "category", "d_fg_pct"]]
    # Zone-matched, not "Overall": a defender's FG%-allowed pooled across every
    # shot type he's ever guarded says little about whether THIS shot, at
    # THIS zone, is contested well — a good rim protector with mediocre
    # closeout speed reads as merely average on the pooled number, exactly
    # backwards from what matters for a shot at the rim. Same category
    # mapping (ZONE_TO_DEF_CATEGORY) the primary defender's own def_fg_pct_zone
    # already uses, so a help defender is judged by the same yardstick.
    defense = defense.merge(
        category_rates, on=["player_id", "game_id", "category"], how="left",
    )

    # Defensive activity (blocks/steals/deflections/gravity): a floor-presence
    # effect, so it uses the same peak-threat MAX logic as the offense-side
    # teammate features, not the zone-matched FG% logic above — a rim
    # protector deters attempts regardless of the shot's actual zone, unlike
    # d_fg_pct which only describes a shot that was already taken.
    activity = load_defensive_activity_profiles(engine)
    defense["_lag_season"] = defense["season"].map(_previous_season)
    lagged_activity = activity.rename(columns={"season": "_lag_season"})
    defense = defense.merge(
        lagged_activity[["player_id", "_lag_season", "stl_per_min",
                         "blk_per_min", "deflections_per_min",
                         "defensive_gravity"]],
        on=["player_id", "_lag_season"], how="left",
    )

    def_agg = defense.groupby("shot_id").agg(
        oncourt_def_fg_pct=("d_fg_pct", "mean"),
        oncourt_def_fg_pct_min=("d_fg_pct", "min"),
        oncourt_def_n=("player_id", "count"),
        oncourt_def_blk_max=("blk_per_min", "max"),
        oncourt_def_stl_max=("stl_per_min", "max"),
        oncourt_def_deflections_max=("deflections_per_min", "max"),
        oncourt_def_gravity_max=("defensive_gravity", "max"),
    ).reset_index()

    return off_agg.merge(def_agg, on="shot_id", how="outer")


def _resolve_recent_lineup_ids(conn, anchor_id: str, role: str, as_of_date=None) -> list[str]:
    """
    The (up to) four other players who most recently shared the floor with
    `anchor_id` on the given side ("offense" or "defense"), read from the
    real substitution-reconstructed shot_on_court table — the true recent
    lineup, not a guess.

    Returns an empty list (never raises) when `anchor_id` has no
    shot_on_court presence at all — no games covered by lineup_ingestor.py
    yet, or a brand-new player with no NBA minutes. The caller is
    responsible for the roster-based fallback in that case.
    """
    from sqlalchemy import text

    date_clause = "AND g.date < :as_of" if as_of_date is not None else ""
    params = {"anchor": str(anchor_id), "role": role}
    if as_of_date is not None:
        params["as_of"] = str(as_of_date)

    row = conn.execute(text(f"""
        SELECT soc.shot_id
        FROM shot_on_court soc
        JOIN shots s ON s.shot_id = soc.shot_id
        JOIN games g ON g.game_id = s.game_id
        WHERE soc.player_id = :anchor AND soc.role = :role
          {date_clause}
        ORDER BY g.date DESC, soc.shot_id DESC
        LIMIT 1
    """), params).fetchone()
    if row is None:
        return []

    others = conn.execute(text("""
        SELECT player_id FROM shot_on_court
        WHERE shot_id = :shot_id AND role = :role AND player_id != :anchor
    """), {"shot_id": row[0], "role": role, "anchor": str(anchor_id)}).fetchall()
    return [str(r[0]) for r in others][:4]


def _resolve_roster_fallback_ids(conn, anchor_id: str, season: str) -> list[str]:
    """
    Tier-2 fallback when `_resolve_recent_lineup_ids` finds nothing: the
    anchor's CURRENT team roster (players.team_id, this season — kept fresh
    by src/ingestion/{roster,current_roster}_ingestor.py), ranked by minutes
    played LAST season (src/ingestion/tracking_ingestor.py's min_per_game) —
    this season's minutes are unreliable this early in a season, and last
    season's is a solid proxy for "who's actually in the rotation".
    """
    from sqlalchemy import text

    from src.features.creation import _previous_season

    team_row = conn.execute(text("""
        SELECT team_id FROM players WHERE player_id = :pid AND season = :season
    """), {"pid": str(anchor_id), "season": season}).fetchone()
    if team_row is None or team_row[0] is None:
        return []

    prev_season = _previous_season(season)
    rows = conn.execute(text("""
        SELECT p.player_id
        FROM players p
        JOIN player_tracking_stats pts
          ON pts.player_id = p.player_id AND pts.season = :prev_season
        WHERE p.season = :season AND p.team_id = :team_id AND p.player_id != :anchor
        ORDER BY pts.min_per_game DESC
        LIMIT 4
    """), {"prev_season": prev_season, "season": season,
           "team_id": team_row[0], "anchor": str(anchor_id)}).fetchall()
    return [str(r[0]) for r in rows]


def lookup_lineup_context(
    conn, creation_profiles: pd.DataFrame, defensive_profiles: pd.DataFrame,
    category_priors: dict, player_id: str, defender_id: str | None,
    season: str, as_of_date=None,
) -> dict:
    """
    Serving-path equivalent of `build_lineup_context` for one hypothetical
    matchup. There is no real on-court group for a "what if X shoots against
    Y" query the way there is for a shot that already happened, so this
    resolves a STAND-IN lineup instead of replaying substitutions:

      1. The (up to) four players who most recently ACTUALLY shared the
         floor with the shooter/defender — real shot_on_court data via
         `_resolve_recent_lineup_ids` — as true to a real, recent lineup as
         the data gets.
      2. If that's empty (no games covered by lineup_ingestor.py yet, or a
         player with no shot_on_court presence at all), the player's
         CURRENT roster teammates, ranked by minutes played last season
         (`_resolve_roster_fallback_ids`).

    Deliberately NOT parameterized by zone, unlike build_lineup_context:
    a caller here is usually scoring a whole grid of candidate locations at
    once (ShotRecommender.recommend), and the help defenders' zone-matched
    FG%-allowed depends on each row's OWN zone. Recomputing the stand-in
    lineup per zone would be wasteful (four DB round-trips per zone instead
    of once total) since the four resolved names don't change. Instead this
    returns `oncourt_def_fg_pct_by_category`/`oncourt_def_fg_pct_min_by_
    category` — one value per DEFENSE_CATEGORY — so the caller does one
    cheap zone → category → dict-lookup per grid row afterward, the same
    "resolve once, select per row" shape `_build_feature_frame` already uses
    for the primary defender's own def_fg_pct_zone.

    Every other returned key is zone-independent and constant across a
    whole grid, matching LINEUP_FEATURE_COLS minus the two dict-valued keys
    above. `creation_profiles`/`defensive_profiles` are the full,
    already-loaded tables (load_creation_profiles/load_defensive_activity_
    profiles) — the caller is expected to load and cache these once, not
    per request.
    """
    from src.features import point_in_time as _pit
    from src.features.creation import _previous_season

    result: dict = {c: None for c in LINEUP_FEATURE_COLS
                    if c not in ("oncourt_def_fg_pct", "oncourt_def_fg_pct_min")}
    result["oncourt_def_fg_pct_by_category"] = {}
    result["oncourt_def_fg_pct_min_by_category"] = {}
    lag_season = _previous_season(season)

    off_ids = _resolve_recent_lineup_ids(conn, player_id, "offense", as_of_date)
    if not off_ids:
        off_ids = _resolve_roster_fallback_ids(conn, player_id, season)

    if off_ids:
        prof = creation_profiles[
            (creation_profiles["player_id"].isin(off_ids))
            & (creation_profiles["season"] == lag_season)
        ]
        for col, mean_key, max_key in (
            ("self_creation_index", "oncourt_off_creation", "oncourt_off_creation_max"),
            ("playmaking_gravity", "oncourt_off_gravity", "oncourt_off_gravity_max"),
            ("rim_pressure", "oncourt_off_rim_pressure", "oncourt_off_rim_pressure_max"),
        ):
            if col in prof.columns and prof[col].notna().any():
                result[mean_key] = float(prof[col].mean())
                result[max_key] = float(prof[col].max())
        if "drive_pf_pct" in prof.columns and prof["drive_pf_pct"].notna().any():
            result["oncourt_off_foul_rate_max"] = float(prof["drive_pf_pct"].max())
        result["oncourt_off_n"] = len(off_ids)

    if defender_id is not None:
        def_ids = _resolve_recent_lineup_ids(conn, defender_id, "defense", as_of_date)
        if not def_ids:
            def_ids = _resolve_roster_fallback_ids(conn, defender_id, season)

        if def_ids:
            per_category: dict[str, list[float]] = {cat: [] for cat in DEFENSE_CATEGORIES}
            for did in def_ids:
                rates = _pit.lookup_defender_category_rates(
                    conn, did, category_priors, as_of_date=as_of_date
                )
                for cat, stats in rates.get("by_category", {}).items():
                    if cat in per_category and stats.get("d_fg_pct") is not None:
                        per_category[cat].append(stats["d_fg_pct"])
            for cat, pcts in per_category.items():
                if pcts:
                    result["oncourt_def_fg_pct_by_category"][cat] = float(np.mean(pcts))
                    result["oncourt_def_fg_pct_min_by_category"][cat] = float(np.min(pcts))

            dprof = defensive_profiles[
                (defensive_profiles["player_id"].isin(def_ids))
                & (defensive_profiles["season"] == lag_season)
            ]
            for col, max_key in (
                ("blk_per_min", "oncourt_def_blk_max"),
                ("stl_per_min", "oncourt_def_stl_max"),
                ("deflections_per_min", "oncourt_def_deflections_max"),
                ("defensive_gravity", "oncourt_def_gravity_max"),
            ):
                if col in dprof.columns and dprof[col].notna().any():
                    result[max_key] = float(dprof[col].max())
            result["oncourt_def_n"] = len(def_ids)

    return result
