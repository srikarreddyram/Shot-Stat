"""
Shot archetypes: PCA + K-Means over the existing engineered feature set.

What gets clustered, and why
-----------------------------
This clusters SHOTS, not players, into groups like "above-the-break
spot-up" or "contested rim finish" — the archetype belongs to a shot
attempt, and a player's archetype MIX is derived afterward from that.

The features are deliberately restricted to ones that describe the shot
ITSELF, not the shooter's season-long tendencies:
  - spatial: loc_x, loc_y, shot_distance, shot_angle, is_three   (WHERE)
  - mech_*: driving, spot_up, pullup, post_up, stepback, ...     (HOW CREATED)
  - finish_*: dunk, layup, hook, floater, jumper, other          (HOW FINISHED)
  - expected_contest, openness_vs_defender                       (HOW CONTESTED)

`shooter_skill` and `creation` (avg_drib_per_touch, self_creation_index, ...)
are season-level aggregates about the PLAYER, not the shot, and are
excluded on purpose — including them would cluster by "type of player"
more than "type of shot," which is a different (also reasonable, but
different) question than the one archetypes are meant to answer here.

Reuse, not rewrite
------------------
The feature values come from `src.features.build.build_matrix` — the same
function training uses — not from a second, hand-rolled feature pass. This
module's own code is the clustering (scaling, PCA, K-Means, and turning
cluster centroids into human-readable labels), not feature engineering.

Usage:
    python -m src.analysis.shot_archetypes --fit
    python -m src.analysis.shot_archetypes --fit --k 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, init_db
from src.db.models import ShotArchetype
from src.features.build import build_matrix

MECH_COLS = [
    "mech_alley_oop", "mech_cutting", "mech_driving", "mech_post_up",
    "mech_pullup", "mech_putback", "mech_spot_up", "mech_stepback",
    "mech_transition",
]
FINISH_COLS = [
    "finish_dunk", "finish_layup", "finish_hook", "finish_floater",
    "finish_jumper", "finish_other",
]
SPATIAL_COLS = ["loc_x", "loc_y", "shot_distance", "shot_angle", "is_three"]
CONTEST_COLS = ["expected_contest", "openness_vs_defender"]

ARCHETYPE_FEATURE_COLS = SPATIAL_COLS + MECH_COLS + FINISH_COLS + CONTEST_COLS

MODEL_DIR = config.PROJECT_ROOT / "models"
ARCHETYPE_ARTIFACT = MODEL_DIR / "shot_archetypes.joblib"
ARCHETYPE_METADATA = MODEL_DIR / "shot_archetypes_metadata.json"

ZONE_LABELS = {
    "Restricted Area": "rim", "In The Paint (Non-RA)": "short mid-range",
    "Mid-Range": "mid-range", "Left Corner 3": "corner three",
    "Right Corner 3": "corner three", "Above the Break 3": "above-the-break three",
}
MECH_LABELS = {
    "mech_alley_oop": "alley-oop", "mech_cutting": "cut", "mech_driving": "driving",
    "mech_post_up": "post-up", "mech_pullup": "pull-up", "mech_putback": "putback",
    "mech_spot_up": "spot-up", "mech_stepback": "step-back", "mech_transition": "transition",
}
FINISH_LABELS = {
    "finish_dunk": "dunk", "finish_layup": "layup", "finish_hook": "hook",
    "finish_floater": "floater", "finish_jumper": "jumper", "finish_other": "shot",
}


def _label_cluster(centroid: pd.Series, zone_mode: str, population_mean_contest: float) -> str:
    """
    Turn a cluster's centroid back into a short human label, e.g.
    "contested rim putback" or "open above-the-break-three spot-up" — the
    dominant mechanic/finish by centroid weight, plus the dominant real zone
    for that cluster (passed in separately since zone isn't a clustering
    feature — loc_x/loc_y/shot_distance are used instead, to keep the space
    continuous; the zone name is only for the label).
    """
    top_mech = max(MECH_COLS, key=lambda c: centroid[c])
    top_finish = max(FINISH_COLS, key=lambda c: centroid[c])
    # "open" vs "contested" is relative to THIS population's average, not a
    # fixed number of feet. Two absolute cutoffs were tried and both failed
    # the same way, just in opposite directions: the technical clip floor
    # (0.5 ft) called every cluster "contested"; the NBA's own Tight/Open
    # boundary (4 ft, CONTEST_BANDS in point_in_time.py) then called every
    # cluster "open", because this dataset's average `expected_contest`
    # already sits above 4 ft — a jump-shot-heavy league does not average
    # "tight" contests. A label that never varies across clusters carries no
    # information regardless of which absolute number produced it; comparing
    # each cluster to the population mean at least guarantees the label
    # reflects real separation between archetypes rather than a population-
    # wide constant.
    contest = "open" if centroid["expected_contest"] >= population_mean_contest else "contested"
    zone_word = ZONE_LABELS.get(zone_mode, zone_mode)
    return f"{contest} {zone_word} {MECH_LABELS[top_mech]} {FINISH_LABELS[top_finish]}"


def _write_assignments_to_db(shot_ids: list[str], cluster_ids: list[int],
                             chunk_size: int = 50_000) -> None:
    """
    Bulk-write (shot_id -> cluster_id) for the current MODEL_VERSION.

    Clears any existing rows for this version first, so re-running --fit is
    idempotent rather than accumulating stale duplicate assignments; a
    different MODEL_VERSION's rows (from a differently-configured fit) are
    left untouched.
    """
    engine = get_engine()
    init_db(engine)

    with engine.begin() as conn:
        conn.execute(
            ShotArchetype.__table__.delete()
            .where(ShotArchetype.model_version == MODEL_VERSION)
        )
        rows = [
            {"shot_id": sid, "cluster_id": int(cid), "model_version": MODEL_VERSION}
            for sid, cid in zip(shot_ids, cluster_ids)
        ]
        for i in range(0, len(rows), chunk_size):
            conn.execute(ShotArchetype.__table__.insert(), rows[i:i + chunk_size])

    print(f"✓ wrote {len(shot_ids):,} archetype assignments "
          f"(model_version={MODEL_VERSION!r}) to shot_archetypes")


def build_archetype_dataset(seasons: list[str] | None = None) -> pd.DataFrame:
    seasons = seasons or [s for s in config.ALL_SEASONS if s >= "2016-17"]
    matrix, _feature_cols, _artifacts = build_matrix(
        seasons, prior_through_season=seasons[-2], verbose=True,
    )
    missing = [c for c in ARCHETYPE_FEATURE_COLS if c not in matrix.columns]
    if missing:
        raise ValueError(f"build_matrix output is missing archetype columns: {missing}")

    keep = ["shot_id", "player_id", "season", "zone", "shot_made"] + ARCHETYPE_FEATURE_COLS
    df = matrix[keep].dropna(subset=ARCHETYPE_FEATURE_COLS)
    return df


MODEL_VERSION = "pca_kmeans_v1"

# k=7 was chosen by silhouette score over a random 20,000-shot sample,
# comparing k in {4,5,6,7,8,10,12}: k=7 scored highest (0.408), ahead of the
# default-guess k=8 (0.365) and even k=12 (0.404) — a larger k does not
# reliably buy a cleaner partition once "above-the-break spot-up" starts
# splitting into near-duplicate clusters that differ only in exact contest
# level. See the ad-hoc sweep this was picked from; not re-run automatically
# on every fit because it roughly doubles the fitting time for a value that
# is stable across re-fits of the same feature set.
DEFAULT_K = 7


def fit(k: int = DEFAULT_K, seasons: list[str] | None = None,
       write_db: bool = True) -> dict:
    df = build_archetype_dataset(seasons)
    print(f"Clustering {len(df):,} shots into {k} archetypes "
          f"over {len(ARCHETYPE_FEATURE_COLS)} shot-descriptive features")

    X = df[ARCHETYPE_FEATURE_COLS].to_numpy(dtype="float64")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Enough components to explain most of the variance without collapsing
    # the one-hot mechanic/finish families into noise; printed so the choice
    # is auditable rather than a magic number.
    pca = PCA(n_components=8, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  PCA: {pca.n_components_} components explain "
          f"{pca.explained_variance_ratio_.sum():.1%} of variance")

    # MiniBatchKMeans rather than full KMeans: at 2M+ shots, full Lloyd's
    # algorithm re-examines every point every iteration, while MiniBatch
    # converges on random subsamples per step — the standard scalability
    # trade for K-Means at this row count, at a small, well-documented cost
    # in cluster-boundary precision.
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=10_000,
                             n_init=10)
    labels = kmeans.fit_predict(X_pca)
    df = df.assign(cluster=labels)

    print(f"  inertia: {kmeans.inertia_:.1f}")

    population_mean_contest = float(df["expected_contest"].mean())
    print(f"  population mean expected_contest: {population_mean_contest:.2f} ft "
          f"(labels below call a cluster \"open\" relative to this, not to a "
          f"fixed number of feet)")

    clusters = []
    for cluster_id in range(k):
        sub = df[df["cluster"] == cluster_id]
        centroid = sub[ARCHETYPE_FEATURE_COLS].mean()
        zone_mode = sub["zone"].mode().iat[0]
        label = _label_cluster(centroid, zone_mode, population_mean_contest)
        clusters.append({
            "cluster_id": int(cluster_id),
            "label": label,
            "n_shots": int(len(sub)),
            "share_of_shots": float(len(sub) / len(df)),
            "fg_pct": float(sub["shot_made"].mean()),
            "dominant_zone": zone_mode,
            "avg_shot_distance": float(sub["shot_distance"].mean()),
            "avg_expected_contest": float(centroid["expected_contest"]),
            "mechanic_mix": {
                MECH_LABELS[c]: float(sub[c].mean()) for c in MECH_COLS
            },
            "finish_mix": {
                FINISH_LABELS[c]: float(sub[c].mean()) for c in FINISH_COLS
            },
        })
    clusters.sort(key=lambda c: -c["n_shots"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "scaler": scaler, "pca": pca, "kmeans": kmeans,
        "feature_cols": ARCHETYPE_FEATURE_COLS,
    }, ARCHETYPE_ARTIFACT)

    metadata = {
        "k": k,
        "model_version": MODEL_VERSION,
        "n_shots": len(df),
        "feature_cols": ARCHETYPE_FEATURE_COLS,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "clusters": clusters,
    }
    ARCHETYPE_METADATA.write_text(json.dumps(metadata, indent=2))

    print(f"\n✓ saved {ARCHETYPE_ARTIFACT.name}, {ARCHETYPE_METADATA.name}")

    if write_db:
        _write_assignments_to_db(df["shot_id"].tolist(), labels.tolist())
    print("\nArchetypes found:")
    for c in clusters:
        print(f"  [{c['cluster_id']}] {c['label']:<45} "
              f"{c['share_of_shots']:>6.1%} of shots, {c['fg_pct']:.1%} FG")

    return {"clusters": clusters, "labels": labels, "shot_ids": df["shot_id"].tolist()}


def load_archetype_model() -> dict:
    if not ARCHETYPE_ARTIFACT.exists():
        raise FileNotFoundError(
            f"No archetype model at {ARCHETYPE_ARTIFACT}. "
            f"Run: python -m src.analysis.shot_archetypes --fit"
        )
    return joblib.load(ARCHETYPE_ARTIFACT)


def assign_archetype(feature_row: pd.DataFrame, model: dict | None = None) -> int:
    """Assign a single already-derived feature row (must contain
    ARCHETYPE_FEATURE_COLS) to its nearest archetype cluster."""
    model = model or load_archetype_model()
    X = feature_row[model["feature_cols"]].to_numpy(dtype="float64")
    X_scaled = model["scaler"].transform(X)
    X_pca = model["pca"].transform(X_scaled)
    return int(model["kmeans"].predict(X_pca)[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="store_true", help="Fit the archetype model")
    parser.add_argument("--k", type=int, default=8, help="Number of archetypes")
    args = parser.parse_args()
    if args.fit:
        fit(k=args.k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
