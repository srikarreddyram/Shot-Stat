"""/stats/features — every feature the live shot-quality model reasons
with, published as data rather than buried in src/features/spec.py. This
answers "what does the MODEL use and how much" — model transparency, not
player/team stats browsing (that's stat_engine_routes.py).

The number that matters here is `gain_share` — XGBoost's gain importance
normalised across the whole model. A feature can exist in the matrix and
do nothing; this is what separates the two.
"""
from fastapi import APIRouter, HTTPException

from src.inference import api as _state

router = APIRouter()

# Prefix-matched families. derive_features emits these as one-hot dummies via
# a prefix scan rather than listing them in FEATURE_GROUPS (see
# spec.all_feature_columns), so they need their own grouping here or they'd
# all fall through as "ungrouped".
_PREFIX_GROUPS = {
    "mech_": "shot_mechanic",
    "finish_": "finish",
    "zone_is_": "zone_indicator",
    "pos_": "position_indicator",
    "origin_": "possession_origin",
}

_GROUP_BLURBS = {
    "shooter_skill": "What this player shoots, from strictly-prior games only — never including the shot being predicted.",
    "shooter_physical": "Measured physical attributes. Exact data or NULL; never estimated from position.",
    "creation": "Handle and passing, lagged one full season. Separates shooting skill from shot difficulty.",
    "defender": "The named defender, blended as a possession-weighted mixture — nobody guards one player for a whole game.",
    "defender_physical": "Defender size. Off by default: no measurable signal, and a large spurious effect at serving time.",
    "opponent_defence": "The defending TEAM's zone-level tendencies, distinct from the individual matchup.",
    "interaction": "Explicit offense-versus-defense terms — the matchup itself rather than either side alone.",
    "spatial": "Where on the floor the shot is taken from.",
    "context": "Game state: clock, score, rest, home/away.",
    "shot_context": "Play-by-play context — what happened immediately before the shot.",
    "contest": "Per-game defender-distance bands. Off by default: measured the most harmful group in ablation.",
    "team_creation": "Who else is on the FLOOR with the shooter — teammates' creation, gravity, rim pressure and foul-drawing, mean and peak-threat.",
    "help_defense": "The other four defenders — zone-matched FG% allowed, plus blocks/steals/deflections and the defensive-gravity composite.",
    "shot_mechanic": "How the shot was created — the move itself.",
    "finish": "How the shot was finished: dunk, layup, floater, jumper.",
    "zone_indicator": "Which of the six court zones the shot is in.",
    "position_indicator": "The shooter's position bucket.",
    "possession_origin": "How the possession began — after a rebound, off a turnover, out of a dead ball.",
    "spatial_basis": "Radial spatial basis. Off by default: measured slightly harmful.",
}


def _humanize(column: str) -> str:
    """Fallback label for a feature with no hand-written copy."""
    return column.replace("_", " ").strip().capitalize()


@router.get("/stats/features")
def list_engine_features():
    """
    Every feature in the live shot-quality model, grouped, labelled, and
    ranked by how much the trained model actually leans on it.

    `gain_share` is XGBoost's gain importance normalised to a share of the
    whole model, so the groups are directly comparable to each other —
    that is the number that says a feature is doing real work rather than
    merely being present. `direction` reports a monotone constraint where
    one is bound: basketball facts the model is not permitted to
    contradict however the training data happens to wiggle (see
    src/training/train.py's MONOTONE_CONSTRAINTS).
    """
    if _state.recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    from src.features.spec import FEATURE_GROUPS
    from src.inference.explain import FEATURE_COPY, SQ_FEATURE_COPY
    from src.training.train import MONOTONE_CONSTRAINTS

    feature_cols = _state.recommender.feature_cols
    booster = _state.recommender.model.get_booster()
    booster.feature_names = feature_cols
    gain = booster.get_score(importance_type="gain")
    total_gain = sum(gain.values()) or 1.0

    group_of: dict[str, str] = {}
    for group, columns in FEATURE_GROUPS.items():
        for column in columns:
            group_of[column] = group

    grouped: dict[str, list[dict]] = {}
    for column in feature_cols:
        group = group_of.get(column)
        if group is None:
            for prefix, prefix_group in _PREFIX_GROUPS.items():
                if column.startswith(prefix):
                    group = prefix_group
                    break
        group = group or "other"

        copy = SQ_FEATURE_COPY.get(column) or FEATURE_COPY.get(column) or {}
        grouped.setdefault(group, []).append({
            "name": column,
            "label": copy.get("label") or _humanize(column),
            "high": copy.get("high"),
            "low": copy.get("low"),
            "gain_share": gain.get(column, 0.0) / total_gain,
            "direction": (
                "raises make probability" if MONOTONE_CONSTRAINTS.get(column) == 1
                else "lowers make probability" if MONOTONE_CONSTRAINTS.get(column) == -1
                else None
            ),
        })

    groups = []
    for group, features in grouped.items():
        features.sort(key=lambda f: -f["gain_share"])
        groups.append({
            "group": group,
            "blurb": _GROUP_BLURBS.get(group),
            "n_features": len(features),
            "gain_share": sum(f["gain_share"] for f in features),
            "features": features,
        })
    groups.sort(key=lambda g: -g["gain_share"])

    return {
        "model_name": _state.recommender.metadata.get("model_name", "shot-quality"),
        "n_features": len(feature_cols),
        "metrics": _state.recommender.metadata.get("test_metrics", {}),
        "groups": groups,
    }
