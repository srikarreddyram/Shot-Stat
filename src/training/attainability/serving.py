"""Point-in-time serving-path lookups — the request-time equivalents of the
training-time builders in build.py, computed for one player rather than
the whole league."""
from __future__ import annotations

import pandas as pd

from .features import ANGLE_SPLIT_ZONES, SUB_ZONES, attach_sub_zone


def lookup_diet_history(conn, player_id: str, season: str,
                        sub_zone_priors: dict, as_of_date=None,
                        max_progress: float = 0.7) -> dict:
    """
    Serving-path equivalent of `build.build_diet_snapshots` for one player:
    what is known about his shot diet right now.

    Returns `{sub_zone: {diet_to_date, career_diet, prior_zone_share, ...}}`
    plus a shared `season_progress`, matching the feature names the training
    frame carries.

    `as_of_date` makes this genuinely point-in-time: the diet-to-date counts
    only games before that date, so a request in December sees December's
    evidence and no more. Without one it uses everything recorded for `season`,
    which is the right reading of "as of now" for a season still in progress.

    `season_progress` is capped at the largest snapshot the model was trained
    on. Serving at 1.0 would ask it to extrapolate past every training row, and
    the honest answer at a completed season is the same one it learned at the
    furthest point it actually saw.
    """
    from sqlalchemy import text

    params = {"pid": str(player_id), "season": season}
    date_clause = ""
    if as_of_date is not None:
        date_clause = "AND g.date < :as_of"
        params["as_of"] = str(as_of_date)

    current = conn.execute(text(f"""
        SELECT s.zone, s.loc_x, s.loc_y
        FROM shots s JOIN games g ON g.game_id = s.game_id
        WHERE s.player_id = :pid AND s.season = :season
          AND s.zone IS NOT NULL AND s.zone != 'Backcourt' {date_clause}
    """), params).fetchall()

    season_total = conn.execute(text("""
        SELECT COUNT(*) FROM shots
        WHERE player_id = :pid AND season = :season
          AND zone IS NOT NULL AND zone != 'Backcourt'
    """), {"pid": str(player_id), "season": season}).scalar() or 0

    history = conn.execute(text("""
        SELECT s.season, s.zone, s.loc_x, s.loc_y
        FROM shots s
        WHERE s.player_id = :pid AND s.season < :season
          AND s.zone IS NOT NULL AND s.zone != 'Backcourt'
    """), {"pid": str(player_id), "season": season}).fetchall()

    def _shares(rows, cols):
        if not rows:
            return {}, 0.0
        frame = attach_sub_zone(pd.DataFrame(rows, columns=cols))
        counts = frame["sub_zone"].value_counts()
        return counts.to_dict(), float(counts.sum())

    seen, seen_total = _shares(current, ["zone", "loc_x", "loc_y"])
    hist_frame = (
        attach_sub_zone(pd.DataFrame(history, columns=["season", "zone", "loc_x", "loc_y"]))
        if history else pd.DataFrame(columns=["season", "sub_zone"])
    )
    career, career_total = ({}, 0.0)
    if not hist_frame.empty:
        c = hist_frame["sub_zone"].value_counts()
        career, career_total = c.to_dict(), float(c.sum())

    prior_shares = {}
    if not hist_frame.empty:
        last = hist_frame["season"].max()
        lf = hist_frame[hist_frame["season"] == last]["sub_zone"].value_counts()
        if lf.sum() > 0:
            prior_shares = (lf / lf.sum()).to_dict()

    progress = min(
        (seen_total / season_total) if season_total else 0.0, max_progress
    )

    out = {"season_progress": progress, "zones": {}}
    for zone in SUB_ZONES:
        prior = sub_zone_priors.get(zone)
        alpha = prior.alpha if prior is not None else 0.0
        strength = prior.strength if prior is not None else 0.0
        out["zones"][zone] = {
            "diet_to_date": (
                (seen.get(zone, 0) + alpha) / (seen_total + strength)
                if seen_total > 0 else None
            ),
            "diet_att_to_date": seen_total,
            "career_diet": (
                (career.get(zone, 0) + alpha) / (career_total + strength)
                if career_total > 0 else None
            ),
            "career_diet_att": career_total,
            "prior_zone_share": prior_shares.get(zone),
        }

    # Bare-zone aggregates for the two angle-split zones, keyed by the
    # UNSPLIT name (e.g. "Above the Break 3", not "... (centre)"/"(wing)").
    #
    # `attach_sub_zone` — the same function this file uses to build the keys
    # above — resolves `sub_zone` to the bare zone name whenever loc_x/loc_y
    # are not supplied (see its own docstring: "rows without coordinates
    # keep their unsplit zone name"). A caller of explain_attainability
    # without exact coordinates therefore looks up `out["zones"]["Above the
    # Break 3"]`, a key that never existed here — every diet-history figure
    # silently came back None regardless of how much real data the player
    # had, and the model's missing-value handling then applied whatever
    # default it learned for that gap, which is not "no effect": querying
    # Stephen Curry's Above-the-Break-3 attainability without coordinates
    # returned 2%, driven almost entirely by this None history, while the
    # visible explanation never mentioned history at all (the summary logic
    # only narrates it when a share IS known) — so the reason shown to the
    # user did not match the reason for the number.
    #
    # These aggregates are plain ratios, not the shrunk sub-zone estimates
    # above: the fitted Beta priors are per SUB-ZONE, and there is no prior
    # fit for the unsplit zone to shrink toward. A real, high-volume rate is
    # far more informative than another None, even unshrunk.
    for zone in ANGLE_SPLIT_ZONES:
        centre, wing = f"{zone} (centre)", f"{zone} (wing)"
        zone_seen = seen.get(centre, 0) + seen.get(wing, 0)
        zone_career = career.get(centre, 0) + career.get(wing, 0)
        # None (not 0.0) when there is no prior season at all — the same
        # "no prior season" vs "a real, measured zero" distinction the
        # per-sub-zone entries above make via `prior_shares.get(zone)`
        # returning None on an empty dict.
        zone_prior_share = (
            None if not prior_shares
            else prior_shares.get(centre, 0.0) + prior_shares.get(wing, 0.0)
        )
        out["zones"][zone] = {
            "diet_to_date": (zone_seen / seen_total) if seen_total > 0 else None,
            "diet_att_to_date": seen_total,
            "career_diet": (zone_career / career_total) if career_total > 0 else None,
            "career_diet_att": career_total,
            "prior_zone_share": zone_prior_share,
        }
    return out


def lookup_prior_diet(conn, player_id: str, season: str) -> dict:
    """
    Serving-path equivalent of `build.attach_prior_diet` for one player: his
    shot diet in the season BEFORE `season`, keyed by sub-zone.

    Raw shares here rather than the shrunk ones the training frame carries.
    The shrinkage only matters below roughly 50 attempts, and a served player
    with fewer than that in a whole season is not someone the recommender is
    being asked about; matching the estimator exactly would mean refitting the
    per-sub-zone Beta priors at request time for a difference in the third
    decimal.

    Returns {} when there is no prior season, which leaves the feature NaN —
    the same encoding a rookie gets in training.
    """
    from sqlalchemy import text

    row = conn.execute(text("""
        SELECT MAX(season) FROM shots WHERE player_id = :pid AND season < :season
    """), {"pid": str(player_id), "season": season}).fetchone()
    if row is None or row[0] is None:
        return {}

    rows = conn.execute(text("""
        SELECT zone, loc_x, loc_y FROM shots
        WHERE player_id = :pid AND season = :prior
          AND zone IS NOT NULL AND zone != 'Backcourt'
    """), {"pid": str(player_id), "prior": row[0]}).fetchall()
    if not rows:
        return {}

    frame = attach_sub_zone(
        pd.DataFrame(rows, columns=["zone", "loc_x", "loc_y"])
    )
    counts = frame["sub_zone"].value_counts()
    total = float(counts.sum())
    return {z: float(counts.get(z, 0)) / total for z in SUB_ZONES}
