"""/matchup/{attacker_id}/{defender_id} — the physical and zone-by-zone
exploit-zone comparison behind the Shot Engine's "Matchup Edge" panel.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException

from src.inference import api as _state
from src.inference.recommender import ZONE_TO_DEF_CATEGORY

router = APIRouter()


@router.get("/matchup/{attacker_id}/{defender_id}")
def get_matchup(attacker_id: str, defender_id: str, season: Optional[str] = None):
    """Get the full physical and statistical mismatch breakdown (defaults to the latest ingested season)."""
    if _state.recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    season = season or _state.latest_season

    try:
        attacker = _state.recommender._player_row(attacker_id, season)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Attacker {attacker_id} not found in {season}")

    defender = _state.recommender._defender_row(defender_id, season)
    if not defender:
        raise HTTPException(status_code=404, detail=f"Defender {defender_id} not found in {season}")

    # Physical comparison
    physical = {}
    for attr in ["height", "weight", "wingspan"]:
        a_val = attacker.get(attr)
        d_val = defender.get(attr)
        physical[attr] = {
            "attacker": a_val,
            "defender": d_val,
            "diff": round(a_val - d_val, 1) if (a_val and d_val) else None,
        }

    height_diff = physical["height"]["diff"]
    size_mismatch = abs(height_diff) >= 4 if height_diff else False

    # Defender quality
    def_stats = defender.get("def_stats", {})
    def_overall = def_stats.get("Overall", {})

    # Zone-to-defense-category mapping (shared with recommender.py's grid scoring)
    zone_to_def_cat = ZONE_TO_DEF_CATEGORY

    # Zone-by-zone exploit analysis — uses the defender's zone-specific FG% allowed
    # (falling back to overall when the zone category is missing), so an elite rim
    # protector with mediocre overall numbers (e.g. blended with weak perimeter D)
    # is still correctly flagged as tough at the rim specifically.
    zones = ["Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
             "Left Corner 3", "Right Corner 3", "Above the Break 3"]
    # The attacker's shrunk, point-in-time rate per zone — the same quantity
    # the model consumes, rather than the raw season split the old endpoint
    # showed. Small samples are regressed to the league prior here too, so the
    # matchup screen and the recommendation cannot tell different stories.
    import pandas as _pd
    from src.features.point_in_time import ZONE_SUFFIX as _SUFFIX, apply_hierarchy as _apply
    _rates = _apply(_pd.DataFrame([attacker]), _state.recommender.zone_priors).iloc[0]
    attacker_zone_rates = {
        z: float(_rates[f"zone_rate_{_SUFFIX[z]}"]) for z in zones
    }

    exploit_zones = []
    for zone in zones:
        atk_eff = attacker_zone_rates.get(zone)
        def_cat = zone_to_def_cat.get(zone)
        zone_def = def_stats.get(def_cat, {})
        def_fg = zone_def.get("d_fg_pct")
        if def_fg is None:
            def_fg = def_overall.get("d_fg_pct")
        advantage = None
        if atk_eff is not None and def_fg is not None:
            advantage = round(atk_eff - def_fg, 3)
        exploit_zones.append({
            "zone": zone,
            "attacker_fg_pct": atk_eff,
            "defender_fg_pct_allowed": def_fg,
            "matchup_advantage": advantage,
            "exploit": advantage is not None and advantage > 0,
        })

    # Sort by advantage (biggest exploit first)
    exploit_zones.sort(key=lambda z: z["matchup_advantage"] or -999, reverse=True)

    return {
        "attacker": {
            "player_id": attacker_id,
            "name": attacker.get("name"),
            "position": attacker.get("position"),
            "headshot_url": f"https://cdn.nba.com/headshots/nba/latest/1040x760/{attacker_id}.png",
            "stats_source": "measured",
            "resolved_season": attacker.get("_latest_season"),
        },
        "defender": {
            "player_id": defender_id,
            "name": defender.get("name"),
            "position": defender.get("position"),
            "headshot_url": f"https://cdn.nba.com/headshots/nba/latest/1040x760/{defender_id}.png",
            "stats_source": "measured",
            "resolved_season": season,
        },
        "season": season,
        "physical_comparison": physical,
        "size_mismatch": size_mismatch,
        "defender_quality": {
            "fg_pct_allowed": def_overall.get("d_fg_pct"),
            "plus_minus": def_overall.get("pct_plusminus"),
        },
        "exploit_zones": exploit_zones,
    }
