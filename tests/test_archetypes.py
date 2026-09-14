"""
Tests for src/inference/archetypes.py.

Every bug this file guards against was found by running the module against
REAL players before trusting it, not by reasoning about the code in the
abstract — the pattern this whole project follows. In order:

  1. Curry, Draymond and Luka were all labeled "Passing Big" because the
     archetype had no size requirement at all — just "elite playmaking".
  2. An automated regex patch meant to add position gates silently landed
     several of them on the WRONG archetype (a lazy regex backtracked past
     a line with a trailing `min_stat=None` and matched the next
     archetype's closing paren instead) — "Point-of-Attack Defender" and
     "Defensive Anchor" ended up with each other's gates swapped.
  3. The position gates themselves were built against a PG/SG/SF/PF
     taxonomy the data does not contain — the real `position` column only
     ever holds {G, F, C, G-F, F-G, F-C, C-F} — so a bare "G" or "F" (the
     two single most common values in the table) could never match a Wing
     archetype at all, regardless of their actual trait scores.
  4. "Traffic Cone" required being below the LEAGUE MEDIAN on three
     measures at once, one of which (rim_protection) is structurally near-
     zero for almost every guard/wing who has simply never blocked a shot —
     a merely unremarkable wing defender qualified as "bad" purely for not
     being a shot-blocking center.

Every one of these is a real defect a plausible-looking system had; none of
them would show up from reading the scoring formula in isolation.
"""
import numpy as np
import pandas as pd
import pytest

from src.inference import archetypes as arc


def _row(**kwargs) -> pd.Series:
    """A minimal player row: every trait NaN unless overridden, so a test
    only has to state the handful of values it actually cares about."""
    base = {f"trait_{t.key}": np.nan for t in arc.TRAITS}
    base.update({"position": None, "games_played": 82})
    base.update(kwargs)
    return pd.Series(base)


# ── Position gating ──────────────────────────────────────────────────────────

def test_position_gate_rejects_a_guard_from_a_big_only_archetype():
    """The exact bug that let Curry/Draymond/Luka get called "Passing Big":
    elite playmaking alone must not be sufficient without the size gate."""
    passing_big = arc.ARCHETYPE_BY_KEY["passing_big"]
    guard_row = _row(position="G", trait_passing_hub=0.99)
    assert not arc._qualifies(guard_row, passing_big)

    big_row = _row(position="C", trait_passing_hub=0.99)
    assert arc._qualifies(big_row, passing_big)


def test_bare_g_and_f_qualify_for_wing_archetypes():
    """The real position column only ever holds {G, F, C, G-F, F-G, F-C,
    C-F} — never SG/PG/SF/PF. A Wing gate built against the finer taxonomy
    silently excluded the two most common values in the whole table.

    A bare "G" additionally needs real wing height (see min_height) — a
    combo "G-F"/"F-G" already self-declares wing eligibility regardless of
    measured height, but a bare "G" of unknown or clearly guard-only height
    must not get the same free pass a "G-F" gets."""
    three_and_d = arc.ARCHETYPE_BY_KEY["three_and_d_wing"]
    bare_g = _row(position="G", height=80.0, trait_catch_and_shoot=0.90, trait_defensive_quality=0.90)
    bare_f = _row(position="F", trait_catch_and_shoot=0.90, trait_defensive_quality=0.90)
    assert arc._qualifies(bare_g, three_and_d)
    assert arc._qualifies(bare_f, three_and_d)


def test_bare_g_needs_real_wing_height_for_a_wing_archetype():
    """Confirmed real bug: Aaron Holiday, a 6'0" (72in) pure point guard,
    was qualifying for "3&D Wing" purely because a bare "G" was treated as
    wing-eligible with no size check at all. A "G-F" combo is unaffected —
    the combo notation is itself the size signal."""
    three_and_d = arc.ARCHETYPE_BY_KEY["three_and_d_wing"]
    short_guard = _row(position="G", height=72.0, trait_catch_and_shoot=0.90, trait_defensive_quality=0.90)
    assert not arc._qualifies(short_guard, three_and_d)

    unknown_height = _row(position="G", trait_catch_and_shoot=0.90, trait_defensive_quality=0.90)
    assert not arc._qualifies(unknown_height, three_and_d)

    combo_guard_forward = _row(position="G-F", height=72.0, trait_catch_and_shoot=0.90, trait_defensive_quality=0.90)
    assert arc._qualifies(combo_guard_forward, three_and_d)


def test_unknown_position_never_qualifies_a_gated_archetype():
    rim_protector = arc.ARCHETYPE_BY_KEY["rim_protector"]
    row = _row(position=None, trait_rim_protection=0.99)
    assert not arc._qualifies(row, rim_protector)


def test_position_agnostic_archetype_has_no_gate_at_all():
    shot_creator = arc.ARCHETYPE_BY_KEY["shot_creator"]
    assert shot_creator.positions is None
    row = _row(position="C", trait_shot_creation=0.9, trait_self_creation=0.9)
    assert arc._qualifies(row, shot_creator)


def test_poa_defender_and_defensive_anchor_gates_are_not_swapped():
    """The regression this test exists for: an automated patch put
    Point-of-Attack Defender's gate on Defensive Anchor and vice versa."""
    poa = arc.ARCHETYPE_BY_KEY["poa_defender"]
    anchor = arc.ARCHETYPE_BY_KEY["defensive_anchor"]
    assert poa.positions == arc.GUARD_POS | arc.WING_POS
    assert anchor.positions == arc.BIG_POS


# ── Threshold semantics ──────────────────────────────────────────────────────

def test_positive_threshold_is_a_floor():
    arch = arc.ArchetypeDef("t", "T", "cat", "", requires=[("playmaking", 0.80)], rank_traits=[])
    assert arc._qualifies(_row(trait_playmaking=0.85), arch)
    assert not arc._qualifies(_row(trait_playmaking=0.79), arch)


def test_negative_threshold_is_a_ceiling_at_its_absolute_value():
    """-0.25 means "must be in the bottom quarter of the league" — NOT
    "must be below the median", which is what an earlier version of this
    module used every negative threshold to mean."""
    arch = arc.ArchetypeDef("t", "T", "cat", "", requires=[("defensive_activity", -0.25)], rank_traits=[])
    assert arc._qualifies(_row(trait_defensive_activity=0.10), arch)   # bottom 25%: qualifies
    assert not arc._qualifies(_row(trait_defensive_activity=0.40), arch)  # below median but not bottom 25%: must not


def test_traffic_cone_does_not_fire_on_a_merely_unremarkable_wing_defender():
    """The exact bug: a wing below league median on three measures at once
    (defensive_activity, rim_protection, perimeter_defense), elite at none
    of them, used to qualify as "bad" — because rim_protection is near-zero
    for almost any non-shot-blocking wing regardless of real defensive
    quality. Traffic Cone must require a real bottom-quartile bar and must
    not use rim_protection as a gate at all.

    Gated on defensive_quality (FG% allowed vs. league normal), not
    defensive_activity (steal/block/deflection rate) — see the module's own
    comment on defensive_quality: a low-event defender who is fine by
    outcome (real case: Aaron Nesmith) must not read as "bad", only a
    low-event defender who is ALSO bad by outcome should."""
    traffic_cone = arc.ARCHETYPE_BY_KEY["traffic_cone"]
    assert not any(t == "rim_protection" for t, _ in traffic_cone.requires)

    unremarkable_wing = _row(trait_defensive_quality=0.44, trait_rim_protection=0.43, trait_perimeter_defense=0.46)
    assert not arc._qualifies(unremarkable_wing, traffic_cone)

    genuinely_bad = _row(trait_defensive_quality=0.10, trait_rim_protection=0.05, trait_perimeter_defense=0.12)
    assert arc._qualifies(genuinely_bad, traffic_cone)


def test_stretch_big_rejects_a_bare_forward_wing():
    """Confirmed complaint: BIG_POS admits a bare "F" (this dataset cannot
    split small forward from power forward there), which let a shooting WING
    qualify for "Stretch Big" off floor-spacing volume alone. Stretch Big's
    own name is a claim about traditional back-to-the-basket size, so it
    gates on TRUE_BIG_POS instead — a true big in this dataset is always a
    combo with "C" or plain "C", never a bare "F"."""
    stretch_big = arc.ARCHETYPE_BY_KEY["stretch_big"]
    assert stretch_big.positions == arc.TRUE_BIG_POS

    wing = _row(position="F", trait_floor_spacing=0.95, trait_shooting_efficiency=0.90)
    assert not arc._qualifies(wing, stretch_big)

    real_big = _row(position="C-F", trait_floor_spacing=0.95, trait_shooting_efficiency=0.90)
    assert arc._qualifies(real_big, stretch_big)


def test_three_level_scorer_requires_every_zone_individually():
    """Real bug: the old version averaged two blended traits (season FG%/3P%/
    FT% and a rim-finishing trait mixing in PnR-roll/cut playtypes), so an
    elite-at-two-of-three-zones player with an average blend could pass while
    a genuinely elite-at-all-three player with a hard shot diet (Dončić:
    excellent at the rim, mid-range and three individually, but a middling
    blended season FG% because of shot difficulty) could fail. Each zone must
    now clear its own bar."""
    three_level = arc.ARCHETYPE_BY_KEY["three_level_scorer"]
    all_three_good = _row(trait_rim_efficiency=0.60, trait_midrange_efficiency=0.60,
                          trait_three_point_efficiency=0.60)
    assert arc._qualifies(all_three_good, three_level)

    weak_at_rim = _row(trait_rim_efficiency=0.10, trait_midrange_efficiency=0.90,
                       trait_three_point_efficiency=0.90)
    assert not arc._qualifies(weak_at_rim, three_level)


def test_missing_trait_never_silently_qualifies():
    """A NaN trait must fail every requirement that names it — never read
    as 0.0 (which would satisfy every negative/ceiling threshold for free)
    and never read as passing a floor by some other default."""
    arch = arc.ArchetypeDef("t", "T", "cat", "", requires=[("rim_protection", -0.30)], rank_traits=[])
    row = _row()  # trait_rim_protection is NaN
    assert not arc._qualifies(row, arch)


# ── compute_traits ───────────────────────────────────────────────────────────

def test_trait_is_nan_when_none_of_its_inputs_are_measured():
    table = pd.DataFrame([
        {"player_id": "P1", "self_creation_index": np.nan, "avg_drib_per_touch": np.nan, "pullup_share": np.nan},
        {"player_id": "P2", "self_creation_index": 1.5, "avg_drib_per_touch": 3.0, "pullup_share": 0.4},
    ])
    out = arc.compute_traits(table)
    assert pd.isna(out.loc[0, "trait_self_creation"])
    assert not pd.isna(out.loc[1, "trait_self_creation"])


def test_trait_uses_whichever_inputs_it_has_not_all_or_nothing():
    # P1 only has one of the three self_creation inputs measured — the
    # trait must still compute from that one, not go NaN just because the
    # other two are missing (the same partial-axis reasoning stat-engine.ts
    # already uses for the frontend radar).
    table = pd.DataFrame([
        {"player_id": "P1", "self_creation_index": 2.0, "avg_drib_per_touch": np.nan, "pullup_share": np.nan},
        {"player_id": "P2", "self_creation_index": -2.0, "avg_drib_per_touch": np.nan, "pullup_share": np.nan},
    ])
    out = arc.compute_traits(table)
    assert not pd.isna(out.loc[0, "trait_self_creation"])
    assert out.loc[0, "trait_self_creation"] > out.loc[1, "trait_self_creation"]


def test_lower_is_better_stats_are_inverted_before_averaging():
    # Two players differing only in turnovers (lower is better): the one
    # with fewer turnovers must score HIGHER on ball_security, not lower.
    table = pd.DataFrame([
        {"player_id": "LOW_TOV", "tov": 1.0, "ast_to_pass_pct": 0.1},
        {"player_id": "HIGH_TOV", "tov": 4.0, "ast_to_pass_pct": 0.1},
    ])
    out = arc.compute_traits(table)
    low = out[out["player_id"] == "LOW_TOV"]["trait_ball_security"].iloc[0]
    high = out[out["player_id"] == "HIGH_TOV"]["trait_ball_security"].iloc[0]
    assert low > high


# ── End-to-end against a real (in-memory) database ──────────────────────────
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.db.database import register_unaccent
from src.db.models import (
    Base, DefenderStats, Player, PlayerDefensiveActivity, PlayerTwoKRating,
    PlayerZoneStats, REGULAR_SEASON, Shot, TeamStats,
)


@pytest.fixture()
def engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    register_unaccent(engine)
    Base.metadata.create_all(engine)
    return engine


def _seed_rim_protector(session, player_id: str, games_played: int):
    """A center whose rim-defense numbers should clear Rim Protector's bar
    outright, at a controllable games-played count for the MIN_GAMES test."""
    session.add(Player(
        player_id=player_id, season="2024-25", name=f"Big {player_id}",
        position="C", team_id="T1", height=84.0, weight=250.0,
    ))
    session.add(DefenderStats(
        player_id=player_id, season="2024-25", defense_category="Overall",
        season_type=REGULAR_SEASON, gp=games_played, freq=1.0,
        d_fgm=100, d_fga=250, d_fg_pct=0.40, normal_fg_pct=0.50, pct_plusminus=-0.10,
    ))
    session.add(DefenderStats(
        player_id=player_id, season="2024-25", defense_category="Less Than 6Ft",
        season_type=REGULAR_SEASON, gp=games_played, freq=0.4,
        d_fgm=40, d_fga=110, d_fg_pct=0.36, normal_fg_pct=0.65, pct_plusminus=-0.29,
    ))
    session.add(PlayerDefensiveActivity(
        player_id=player_id, season="2024-25", gp=games_played, min_per_game=30.0,
        blk=3.0, stl=0.8, deflections=1.5,
        # Elite rebounding too, deliberately, so this fixture also covers the
        # dominant-two-way-big case: qualifying for Rim Protector, Paint
        # Anchor AND Rebounding Specialist at once is the exact real-world
        # shape (confirmed live: Victor Wembanyama) the secondary-archetype
        # redundancy check exists to de-duplicate.
        oreb=4.0, dreb=10.0,
    ))
    session.add(Shot(
        shot_id=f"s-{player_id}", game_id="g1", player_id=player_id, season="2024-25",
        shot_made=1, zone="Restricted Area",
    ))


def _seed_role_players(session, count=6):
    """Percentile ranking against a league of one player is mathematically
    meaningless — every stat trivially ranks at the 100th percentile, and a
    trait mixing a LOWER_IS_BETTER input with a normal one then averages a
    percentile with its own inversion into a content-free 0.5. Every test
    that checks a specific archetype outcome needs a real comparison
    population, the same way the league-wide functions are actually used."""
    for i in range(count):
        session.add(Player(player_id=f"ROLE{i}", season="2024-25",
                           name=f"Role {i}", position="G", team_id="T1"))
        session.add(DefenderStats(
            player_id=f"ROLE{i}", season="2024-25", defense_category="Overall",
            season_type=REGULAR_SEASON, gp=70, freq=1.0,
            d_fgm=100, d_fga=200, d_fg_pct=0.50, normal_fg_pct=0.50, pct_plusminus=0.0,
        ))
        session.add(DefenderStats(
            player_id=f"ROLE{i}", season="2024-25", defense_category="Less Than 6Ft",
            season_type=REGULAR_SEASON, gp=70, freq=0.2,
            d_fgm=50, d_fga=80, d_fg_pct=0.625, normal_fg_pct=0.65, pct_plusminus=-0.025,
        ))
        session.add(PlayerDefensiveActivity(
            player_id=f"ROLE{i}", season="2024-25", gp=70, min_per_game=20.0,
            blk=0.3, stl=0.5, deflections=1.0,
            # Modest, not zero/missing — BIG1's oreb/dreb must rank against a
            # real distribution. Ranking against all-NaN role players would
            # make BIG1 the only non-NaN value and trivially "100th
            # percentile" regardless of how elite the raw number actually is
            # (the exact trap this fixture's own docstring warns about).
            oreb=1.0, dreb=3.0,
        ))
    # One player who out-rebounds BIG1 but has no rim/shot-blocking presence
    # at all — otherwise BIG1's rebounding, like his rim protection, would
    # rank a trivial #1 of N and TIE rim_protection's score exactly (both
    # 1.0), and a tie is resolved by ARCHETYPES list order rather than by
    # which archetype the player actually fits better. A real league always
    # has someone who boards better than even an elite rim protector.
    session.add(Player(player_id="BOARDS_ONLY", season="2024-25",
                       name="Boards Only", position="C", team_id="T1"))
    session.add(DefenderStats(
        player_id="BOARDS_ONLY", season="2024-25", defense_category="Overall",
        season_type=REGULAR_SEASON, gp=70, freq=1.0,
        d_fgm=100, d_fga=200, d_fg_pct=0.50, normal_fg_pct=0.50, pct_plusminus=0.0,
    ))
    session.add(DefenderStats(
        player_id="BOARDS_ONLY", season="2024-25", defense_category="Less Than 6Ft",
        season_type=REGULAR_SEASON, gp=70, freq=0.2,
        d_fgm=50, d_fga=80, d_fg_pct=0.625, normal_fg_pct=0.65, pct_plusminus=-0.025,
    ))
    session.add(PlayerDefensiveActivity(
        player_id="BOARDS_ONLY", season="2024-25", gp=70, min_per_game=20.0,
        blk=0.3, stl=0.5, deflections=1.0, oreb=6.0, dreb=14.0,
    ))

def test_compute_archetypes_end_to_end_assigns_a_real_rim_protector(engine):
    Session = sessionmaker(bind=engine)
    with Session() as session:
        _seed_rim_protector(session, "BIG1", games_played=70)
        # A thin bench of role players so BIG1's shot-blocking is a real
        # league-relative outlier rather than the only data point.
        _seed_role_players(session)
        session.commit()

    table = arc.compute_archetypes(engine, "2024-25")
    row = table[table["player_id"] == "BIG1"].iloc[0]
    assert row["archetype_key"] == "rim_protector"
    assert row["archetype_score"] is not None and row["archetype_score"] > 0


def test_secondary_archetypes_drop_a_fully_redundant_one(engine):
    """Confirmed real bug: a dominant two-way big (Victor Wembanyama)
    qualified for Rim Protector, Paint Anchor, Rebounding Specialist AND
    Defensive Anchor simultaneously, but Rebounding Specialist's entire gate
    (rebounding alone) was already covered by Paint Anchor's (rim_protection
    + rebounding) — showing it added a badge with zero new information.
    Paint Anchor, which adds the new "rebounding" fact on top of Rim
    Protector's "rim_protection", must still show."""
    Session = sessionmaker(bind=engine)
    with Session() as session:
        _seed_rim_protector(session, "BIG1", games_played=70)
        _seed_role_players(session)
        session.commit()

    detail = arc.player_archetype_detail(engine, "BIG1", "2024-25")
    secondary_keys = {s["key"] for s in detail["secondary"]}
    assert "rebounding_specialist" not in secondary_keys
    assert "paint_anchor" in secondary_keys


def test_compute_archetypes_excludes_players_under_min_games(engine):
    Session = sessionmaker(bind=engine)
    with Session() as session:
        _seed_rim_protector(session, "CALLUP", games_played=arc.MIN_GAMES - 1)
        session.commit()

    table = arc.compute_archetypes(engine, "2024-25")
    row = table[table["player_id"] == "CALLUP"].iloc[0]
    assert row["archetype_key"] is None
    assert row["archetype_label"] is None


def test_player_archetype_detail_is_json_safe_and_transparent(engine):
    Session = sessionmaker(bind=engine)
    with Session() as session:
        _seed_rim_protector(session, "BIG1", games_played=70)
        _seed_role_players(session)
        session.commit()

    detail = arc.player_archetype_detail(engine, "BIG1", "2024-25")
    assert detail is not None
    encoded = json.dumps(detail)  # must round-trip with no NaN reaching JSON
    json.loads(encoded)

    assert detail["eligible"] is True
    assert detail["primary"]["key"] == "rim_protector"
    # Every trait must appear, even ones this player has no data for, so a
    # client can render a full breakdown rather than a partial one.
    assert len(detail["traits"]) == len(arc.TRAITS)
    assert any(t["key"] == "rim_protection" and t["percentile"] is not None for t in detail["traits"])
    # The archetypes this player qualified for but did NOT win must also be
    # visible, so "why isn't he X" is answerable from the same payload.
    assert isinstance(detail["qualified"], list)
    assert any(q["key"] == "rim_protector" for q in detail["qualified"])


def test_player_archetype_detail_returns_none_for_unknown_player(engine):
    Session = sessionmaker(bind=engine)
    with Session() as session:
        _seed_rim_protector(session, "BIG1", games_played=70)
        session.commit()
    assert arc.player_archetype_detail(engine, "NOBODY", "2024-25") is None


def test_archetype_catalogue_covers_every_defined_archetype():
    catalogue = arc.archetype_catalogue()
    assert len(catalogue) == len(arc.ARCHETYPES)
    assert all(c["label"] and c["category"] and c["blurb"] for c in catalogue)
    assert set(c["category"] for c in catalogue) <= set(arc.CATEGORY_ORDER)
