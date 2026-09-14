"""Per-zone priors over "did the player create this shot himself" — the
question attainability's own frequency number cannot answer on its own."""
from __future__ import annotations

import re

import pandas as pd

from src.features.shrinkage import BetaPrior, fit_priors, shrink


def fit_league_creation_priors(engine, through_season: str) -> dict[str, BetaPrior]:
    """
    Per zone, a Beta prior over "what share of a player's makes here did he
    create himself".

    The quantity the recommender's attainability number cannot express on its
    own. Attainability is a frequency — what share of a player's shots come
    from a spot — and frequency conflates two very different situations: a
    shot he can manufacture whenever he wants, and a shot that only exists
    when a teammate finds him. League-wide those separate sharply: 44% of
    restricted-area makes are unassisted against 4% of corner threes. A corner
    three is not a shot anybody goes and gets.

    Keyed by SUB-zone, so a dead-centre three and a wing three get separate
    priors — the split is sharpest exactly here (27% self-created within ten
    degrees of dead centre against 9.5% beyond sixty), and reporting one
    blended above-the-break figure for both would hide the distinction the
    number exists to make.

    Fit through the last TRAINING season only, same rule as every other prior
    in this package.
    """
    from src.training.attainability import ANGLE_SPLIT_ZONES, attach_sub_zone

    df = pd.read_sql(f"""
        SELECT s.zone, s.loc_x, s.loc_y, s.player_id, s.season, s.shot_made,
               CASE WHEN sc.is_assisted = 1 THEN 0 ELSE 1 END AS self_make
        FROM shots s
        JOIN shot_context sc ON sc.shot_id = s.shot_id
        WHERE s.shot_made = 1
          AND s.zone IS NOT NULL AND s.zone != 'Backcourt'
          AND s.season <= '{through_season}'
    """, engine)
    if df.empty:
        return {}

    df = attach_sub_zone(df)
    grouped = df.groupby(["sub_zone", "player_id", "season"], as_index=False).agg(
        self_makes=("self_make", "sum"), makes=("shot_made", "size")
    )
    priors = fit_priors(
        grouped, ["sub_zone"], makes_col="self_makes", attempts_col="makes"
    )
    out = {z: priors[(z,)] for z in grouped["sub_zone"].unique() if (z,) in priors}

    # Bare-zone aggregate priors for the two angle-split zones, keyed by the
    # UNSPLIT name — same bug, same fix as lookup_diet_history's missing
    # bare-zone entries. `lookup_zone_creation` resolves to the bare zone
    # name whenever it isn't given exact coordinates (attach_sub_zone's
    # documented behaviour), and `creation_priors` had no such key for
    # either angle-split zone, only its "(centre)"/"(wing)" splits. The
    # lookup's own fallback (`.get(zone) or .get(parent)`) cannot rescue
    # this, because zone == parent in exactly the case that needs rescuing.
    #
    # The visible symptom: asking for Jamal Murray's self-created share at
    # Above the Break 3 came back with no prior at all, so `share` and
    # `league_self_created_share` were both None and `creation_note`
    # returned nothing — despite `self_creation_index` correctly rating him
    # an elite self-creator and his supporting cast (Jokić) showing a 71%
    # teammate assist rate. The gap read as "we don't know," not as the
    # (false) "nobody creates this for him" it was mistaken for, but the
    # missing prior is the same class of silent gap either way.
    parent_grouped = df.groupby(["zone", "player_id", "season"], as_index=False).agg(
        self_makes=("self_make", "sum"), makes=("shot_made", "size")
    )
    parent_priors = fit_priors(
        parent_grouped, ["zone"], makes_col="self_makes", attempts_col="makes"
    )
    for zone in ANGLE_SPLIT_ZONES:
        if (zone,) in parent_priors:
            out[zone] = parent_priors[(zone,)]
    return out


def lookup_zone_creation(conn, player_id: str, zone: str,
                         creation_priors: dict[str, BetaPrior],
                         season: str | None = None, as_of_date=None) -> dict:
    """
    How this player's makes in one zone were generated: by himself, or by a
    teammate finding him.

    Shrunk toward the zone's league rate, so a player with four makes in a
    corner does not read as a 100% self-creator. Career-to-date rather than
    season-to-date: shot creation is a stable trait and the season-only sample
    per zone is thin for everyone but high-volume starters.

    Returns the player's shrunk self-created share, the league rate for the
    same zone, and the raw counts behind it so a caller can say how much
    evidence there is.
    """
    from sqlalchemy import text

    from src.training.attainability import ANGLE_SPLIT_ZONES, attach_sub_zone

    # `zone` may arrive as either a plain zone or an already-split sub-zone.
    # The parent zone is what the shots table stores, so query on that and
    # narrow to the sub-zone in pandas afterwards.
    parent = re.sub(r" \((centre|wing)\)$", "", zone)

    clauses = []
    params = {"pid": str(player_id), "zone": parent}
    if season is not None:
        clauses.append("AND s.season <= :season")
        params["season"] = season
    if as_of_date is not None:
        clauses.append("AND g.date < :as_of")
        params["as_of"] = str(as_of_date)

    rows = conn.execute(text(f"""
        SELECT s.zone, s.loc_x, s.loc_y,
               CASE WHEN sc.is_assisted = 1 THEN 0 ELSE 1 END AS self_make
        FROM shots s
        JOIN shot_context sc ON sc.shot_id = s.shot_id
        JOIN games g ON g.game_id = s.game_id
        WHERE s.player_id = :pid
          AND s.zone = :zone
          AND s.shot_made = 1
          {' '.join(clauses)}
    """), params).fetchall()

    frame = pd.DataFrame(rows, columns=["zone", "loc_x", "loc_y", "self_make"])
    if parent in ANGLE_SPLIT_ZONES and not frame.empty:
        frame = attach_sub_zone(frame)
        # An unsplit parent name means "either half"; a split name narrows.
        if zone != parent:
            frame = frame[frame["sub_zone"] == zone]

    prior = creation_priors.get(zone) or creation_priors.get(parent)
    league = float(prior.mean) if prior is not None else None

    self_makes = float(frame["self_make"].sum()) if not frame.empty else 0.0
    makes = float(len(frame))

    if prior is None:
        share = None
    else:
        share = float(shrink(self_makes, makes, prior))

    return {
        "self_created_share": share,
        "league_self_created_share": league,
        "makes": int(makes),
        "self_makes": int(self_makes),
    }
