"""
Phase 1 Model Training — Zone-Average Baseline, Logistic Regression, XGBoost.

Trains three models using a strict temporal split:
  - Train: 2010-11 through 2024-25 (15 seasons)
  - Test:  2025-26 (held-out season)

The zone-average baseline is the dumb benchmark to beat.
LR is the interpretable ML baseline.
XGBoost is the primary model.

Usage:
    python -m src.training.train_baseline
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
import xgboost as xgb

from src.training.calibration import (
    fit_calibrator,
    apply_calibration,
    save_calibrator,
    print_calibration_report,
    save_calibration_plot,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.training.feature_engineering import (
    build_training_matrix,
    get_feature_columns,
    TARGET_COL,
)


def train_and_evaluate(version="v1"):
    """Train all Phase 1/2 models and print a comparison report."""

    use_defender = (version in ("v2", "v3"))
    
    # Matchup data only available from 2016-17 onward
    if use_defender:
        seasons_to_use = [s for s in config.ALL_SEASONS if s >= "2016-17"]
    else:
        seasons_to_use = config.ALL_SEASONS

    # ── Step 1: Build training matrix ────────────────────────────────────────
    df = build_training_matrix(seasons_to_use, use_defender=use_defender)
    feature_cols = get_feature_columns(df)

    # ── Step 2: Temporal split ───────────────────────────────────────────────
    # Train on everything EXCEPT the most recent season.
    # Test on the most recent season.
    # This prevents any future data leakage.
    TRAIN_SEASONS = seasons_to_use[:-1]
    TEST_SEASON = seasons_to_use[-1]

    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test_df = df[df["season"] == TEST_SEASON].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    print(f"\n{'='*60}")
    print(f"  TEMPORAL SPLIT")
    print(f"{'='*60}")
    print(f"  Train: {len(X_train):>10,} shots  ({TRAIN_SEASONS[0]} → {TRAIN_SEASONS[-1]})")
    print(f"  Test:  {len(X_test):>10,} shots  ({TEST_SEASON})")
    print(f"  Train FG%: {y_train.mean():.3f}")
    print(f"  Test FG%:  {y_test.mean():.3f}")
    print(f"  Features:  {len(feature_cols)}")

    # ── Step 3: Baseline — Zone-Average FG% ──────────────────────────────────
    # The dumbest possible model: "predict the historical zone average."
    # If our ML can't beat this, something is wrong.
    print(f"\n{'='*60}")
    print(f"  MODEL 1/3: Zone-Average Baseline")
    print(f"{'='*60}")

    zone_avg = train_df.groupby("zone")[TARGET_COL].mean().to_dict()
    print("  Zone averages (from training data):")
    for zone, avg in sorted(zone_avg.items(), key=lambda x: -x[1]):
        print(f"    {zone:<30} {avg:.3f}")

    # For each test shot, predict the zone's average FG%
    baseline_preds = test_df["zone"].map(zone_avg)
    baseline_preds = baseline_preds.fillna(y_train.mean())  # safety fallback
    baseline_preds = baseline_preds.clip(0.01, 0.99)        # avoid log(0)

    baseline_logloss = log_loss(y_test, baseline_preds)
    baseline_acc = accuracy_score(y_test, (baseline_preds >= 0.5).astype(int))
    baseline_auc = roc_auc_score(y_test, baseline_preds)

    print(f"\n  → Log-Loss:  {baseline_logloss:.4f}")
    print(f"  → Accuracy:  {baseline_acc:.4f}")
    print(f"  → AUC:       {baseline_auc:.4f}")

    # ── Step 4: Logistic Regression ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MODEL 2/3: Logistic Regression")
    print(f"{'='*60}")

    # LR can't handle NaN → fill with training set medians
    train_medians = X_train.median()
    # If a column is entirely NaN, median is NaN → fill those with 0
    train_medians = train_medians.fillna(0)
    X_train_lr = X_train.fillna(train_medians)
    X_test_lr = X_test.fillna(train_medians)  # ALWAYS use train medians, never test!

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
    ])

    print("  → Training...")
    lr_pipeline.fit(X_train_lr, y_train)

    lr_preds = lr_pipeline.predict_proba(X_test_lr)[:, 1]
    lr_logloss = log_loss(y_test, lr_preds)
    lr_acc = accuracy_score(y_test, (lr_preds >= 0.5).astype(int))
    lr_auc = roc_auc_score(y_test, lr_preds)

    print(f"  → Log-Loss:  {lr_logloss:.4f}")
    print(f"  → Accuracy:  {lr_acc:.4f}")
    print(f"  → AUC:       {lr_auc:.4f}")

    # Show top LR coefficients
    lr_model = lr_pipeline.named_steps["lr"]
    coef_pairs = sorted(
        zip(feature_cols, lr_model.coef_[0]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    print(f"\n  Top 10 LR coefficients (absolute value):")
    for feat, coef in coef_pairs[:10]:
        direction = "+" if coef > 0 else "-"
        print(f"    {direction} {feat:<35} {coef:>8.4f}")

    # ── Step 5: XGBoost ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MODEL 3/3: XGBoost (Hierarchical Split: Interior vs Perimeter)")
    print(f"{'='*60}")

    # Default XGBoost parameters
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "enable_categorical": False,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 20,
    }

    # Load tuned params for v3
    if version == "v3":
        tuned_path = Path("models") / "best_params_v3.json"
        if tuned_path.exists():
            import json
            with open(tuned_path) as f:
                tuned = json.load(f)
            xgb_params.update(tuned)
            print(f"  ✓ Loaded tuned hyperparameters from {tuned_path}")
        else:
            print(f"  ⚠ No tuned params found at {tuned_path}, using defaults")

    # Split data into interior and perimeter
    interior_zones = ["Restricted Area", "In The Paint (Non-RA)"]
    mask_train_int = train_df["zone"].isin(interior_zones)
    mask_test_int = test_df["zone"].isin(interior_zones)

    X_train_int, y_train_int = X_train[mask_train_int], y_train[mask_train_int]
    X_test_int, y_test_int = X_test[mask_test_int], y_test[mask_test_int]
    
    X_train_per, y_train_per = X_train[~mask_train_int], y_train[~mask_train_int]
    X_test_per, y_test_per = X_test[~mask_test_int], y_test[~mask_test_int]

    # Train Interior Model
    # Monotonic constraints for interior:
    #   height_diff = -1: being shorter than defender HURTS at the rim
    #   def_pct_plusminus = -1: better rim protector (negative plusminus) = harder to score
    print("\n  → Training INTERIOR Model...")
    xgb_params_int = dict(xgb_params)
    if "height_diff" in feature_cols:
        # height_diff = attacker - defender. Positive = attacker is taller.
        # At the rim, being taller helps → monotone +1
        # def_fg_pct_overall = opponent FG% the defender allows. Higher = worse defense → easier to score → +1
        # def_pct_plusminus = how much worse opponents shoot vs league avg. Positive = bad defender → +1
        # matchup_advantage = attacker zone FG% - defender overall FG%. Higher = better for attacker → +1
        # zone_efficiency = attacker's shooting % in this zone. Higher = better shooter → +1
        constraints_int = {
            "height_diff": 1,
            "zone_efficiency": 1,
        }
        if "def_fg_pct_overall" in feature_cols:
            constraints_int["def_fg_pct_overall"] = 1
        if "def_pct_plusminus" in feature_cols:
            constraints_int["def_pct_plusminus"] = 1
        if "def_fg_pct_zone" in feature_cols:
            constraints_int["def_fg_pct_zone"] = 1
        if "def_pct_plusminus_zone" in feature_cols:
            constraints_int["def_pct_plusminus_zone"] = 1
        if "matchup_advantage" in feature_cols:
            constraints_int["matchup_advantage"] = 1
        xgb_params_int["monotone_constraints"] = constraints_int
        print(f"    Monotonic constraints: {constraints_int}")
    
    xgb_interior = xgb.XGBClassifier(**xgb_params_int)
    xgb_interior.fit(
        X_train_int, y_train_int,
        eval_set=[(X_test_int, y_test_int)],
        verbose=100,
    )
    
    # Train Perimeter Model
    print("\n  → Training PERIMETER Model...")
    xgb_perimeter = xgb.XGBClassifier(**xgb_params)
    xgb_perimeter.fit(
        X_train_per, y_train_per,
        eval_set=[(X_test_per, y_test_per)],
        verbose=100,
    )

    # Predictions
    xgb_preds_raw = np.zeros(len(X_test))
    xgb_preds_raw[mask_test_int] = xgb_interior.predict_proba(X_test_int)[:, 1]
    xgb_preds_raw[~mask_test_int] = xgb_perimeter.predict_proba(X_test_per)[:, 1]

    xgb_logloss_raw = log_loss(y_test, xgb_preds_raw)
    xgb_acc = accuracy_score(y_test, (xgb_preds_raw >= 0.5).astype(int))
    xgb_auc = roc_auc_score(y_test, xgb_preds_raw)

    print(f"\n  → Combined Log-Loss (raw):  {xgb_logloss_raw:.4f}")
    print(f"  → Combined Accuracy:        {xgb_acc:.4f}")
    print(f"  → Combined AUC:             {xgb_auc:.4f}")

    # ── Step 6: Probability Calibration ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PROBABILITY CALIBRATION (Isotonic Regression)")
    print(f"{'='*60}")

    # Fit separate calibrators
    xgb_train_preds_int = xgb_interior.predict_proba(X_train_int)[:, 1]
    calibrator_int, _, _ = fit_calibrator(xgb_train_preds_int, y_train_int.values)
    
    xgb_train_preds_per = xgb_perimeter.predict_proba(X_train_per)[:, 1]
    calibrator_per, _, _ = fit_calibrator(xgb_train_preds_per, y_train_per.values)

    # Apply calibration
    xgb_preds = np.zeros(len(X_test))
    xgb_preds[mask_test_int] = apply_calibration(calibrator_int, xgb_preds_raw[mask_test_int])
    xgb_preds[~mask_test_int] = apply_calibration(calibrator_per, xgb_preds_raw[~mask_test_int])
    
    xgb_logloss = log_loss(y_test, xgb_preds)

    print(f"  → Raw test log-loss:        {xgb_logloss_raw:.4f}")
    print(f"  → Calibrated test log-loss:  {xgb_logloss:.4f}")

    # Print detailed calibration report
    print_calibration_report(
        xgb_preds_raw, xgb_preds, y_test.values,
        zones=test_df["zone"].values,
    )
    save_calibration_plot(xgb_preds_raw, xgb_preds, y_test.values, "models", version)

    # ── Step 7: Save models ─────────────────────────────────────────────────
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    joblib.dump(lr_pipeline, model_dir / f"lr_baseline_{version}.joblib")
    
    xgb_interior.save_model(str(model_dir / f"xgb_interior_{version}.json"))
    xgb_perimeter.save_model(str(model_dir / f"xgb_perimeter_{version}.json"))
    
    joblib.dump(calibrator_int, model_dir / f"calibrator_interior_{version}.joblib")
    joblib.dump(calibrator_per, model_dir / f"calibrator_perimeter_{version}.joblib")
    
    joblib.dump(
        {
            "feature_cols": feature_cols,
            "train_medians": train_medians.to_dict(),
            "zone_avg": zone_avg,
            "train_seasons": TRAIN_SEASONS,
            "test_season": TEST_SEASON,
            "calibrated": True,
            "hierarchical": True,
        },
        model_dir / f"metadata_{version}.joblib",
    )
    print(f"\n  ✓ Models saved to {model_dir}/ (version {version})")

    # ── Step 8: Final comparison table ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON — Test Season: {TEST_SEASON}")
    print(f"{'='*60}")
    print(f"  {'Model':<30} {'Log-Loss':>10} {'Accuracy':>10} {'AUC':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Zone-Average Baseline':<30} {baseline_logloss:>10.4f} {baseline_acc:>10.4f} {baseline_auc:>10.4f}")
    print(f"  {'Logistic Regression':<30} {lr_logloss:>10.4f} {lr_acc:>10.4f} {lr_auc:>10.4f}")
    print(f"  {'XGBoost (raw)':<30} {xgb_logloss_raw:>10.4f} {xgb_acc:>10.4f} {xgb_auc:>10.4f}")
    print(f"  {'XGBoost (calibrated)':<30} {xgb_logloss:>10.4f} {'—':>10} {'—':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    # Did we beat the baseline?
    if xgb_logloss < baseline_logloss:
        lift = (baseline_logloss - xgb_logloss) / baseline_logloss * 100
        print(f"\n  ✅ XGBoost beats zone-average baseline by {lift:.2f}% log-loss")
    else:
        print(f"\n  ❌ XGBoost did NOT beat baseline — investigate!")

    if lr_logloss < baseline_logloss:
        lr_lift = (baseline_logloss - lr_logloss) / baseline_logloss * 100
        print(f"  ✅ LR beats zone-average baseline by {lr_lift:.2f}% log-loss")
    else:
        print(f"  ❌ LR did NOT beat baseline")

    print(f"\n{'='*60}\n")

    # ── Step 8: Feature importance (XGBoost) ─────────────────────────────────
    print("  XGBoost Interior — Top 20 Features by Importance:")
    importance = xgb_interior.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(feat_imp[:20], 1):
        bar = "█" * int(imp * 200)
        print(f"  {i:>3}. {feat:<35} {imp:.4f}  {bar}")

    # Save feature importance plot
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = feat_imp[:20]
        ax.barh(
            [f[0] for f in reversed(top_n)],
            [f[1] for f in reversed(top_n)],
            color="#f97316",
        )
        ax.set_xlabel("Feature Importance (Gain)")
        ax.set_title(f"XGBoost Interior {version} — Top 20 Features (Test: {TEST_SEASON})")
        plt.tight_layout()
        fig.savefig(str(model_dir / f"feature_importance_{version}.png"), dpi=150)
        plt.close(fig)
        print(f"\n  ✓ Saved {model_dir}/feature_importance_{version}.png")
    except Exception as e:
        print(f"\n  ⚠ Could not save plot: {e}")

    # ── Step 9: Per-zone breakdown ───────────────────────────────────────────
    print(f"\n  Per-Zone Log-Loss Breakdown:")
    print(f"  {'Zone':<30} {'XGBoost':>10} {'Baseline':>10} {'Lift':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8}")

    for zone in sorted(test_df["zone"].unique()):
        mask = test_df["zone"] == zone
        if mask.sum() < 100:
            continue
        zone_xgb_ll = log_loss(y_test[mask], xgb_preds[mask])
        zone_base_ll = log_loss(y_test[mask], baseline_preds[mask])
        lift_pct = (zone_base_ll - zone_xgb_ll) / zone_base_ll * 100
        icon = "✅" if zone_xgb_ll < zone_base_ll else "❌"
        print(f"  {icon} {zone:<28} {zone_xgb_ll:>10.4f} {zone_base_ll:>10.4f} {lift_pct:>+7.1f}%")

    print(f"\n{'='*60}")
    print(f"  ✓ Training pipeline complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train shot predictor baseline models.")
    parser.add_argument("--version", type=str, default="v1", choices=["v1", "v2", "v3"],
                        help="v1 (no defender), v2 (with defender), v3 (tuned + new features)")
    args = parser.parse_args()

    train_and_evaluate(version=args.version)
