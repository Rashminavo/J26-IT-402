"""Exploratory participant-grouped regression comparison for wearable stress ratings.

Uses task-level aggregates and continuous self-reported stress ratings. It does not
create Low/Moderate/High labels and does not fit a deployment model.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


INPUT_PATH = Path("data/processed/f_stress_task_level_dataset.csv")
OUTPUT_DIR = Path("data/processed/model_comparison")
TARGET = "self_reported_stress_rating"
GROUP = "participant_id"
NON_FEATURE_COLUMNS = {
    "participant_id",
    "session_id",
    "task",
    TARGET,
    "window_count",
    "task_start_seconds",
    "task_end_seconds",
    "task_duration_seconds",
}
RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3


def make_pipeline(model, scale: bool) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=f_regression, k="all")),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def safe_r2(y_true, y_pred) -> float:
    # Keep sklearn's NaN behavior explicit if a test fold has constant targets.
    return float(r2_score(y_true, y_pred))


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"ERROR: task-level dataset not found: {INPUT_PATH}")
        print("Run scripts\\build_task_dataset.py first.")
        return 1

    data = pd.read_csv(INPUT_PATH)
    needed = {GROUP, "session_id", "task", TARGET}
    missing = needed - set(data.columns)
    if missing:
        print(f"ERROR: missing required columns: {sorted(missing)}")
        return 1

    if data[[GROUP, "task"]].isna().any().any():
        print("ERROR: participant_id/task has missing values.")
        return 1
    if data.duplicated([GROUP, "task"]).any():
        print("ERROR: duplicate participant-task rows; refusing to train.")
        return 1

    feature_columns = [c for c in data.columns if c not in NON_FEATURE_COLUMNS]
    if not feature_columns:
        print("ERROR: no physiological feature columns found.")
        return 1

    # Require all model inputs and target to be numeric and finite.
    try:
        X = data[feature_columns].apply(pd.to_numeric, errors="raise")
        y = pd.to_numeric(data[TARGET], errors="raise")
    except Exception as exc:
        print(f"ERROR: non-numeric target/features: {exc}")
        return 1

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        print("ERROR: model features contain NaN/inf. Fix data before training.")
        return 1
    if y.isna().any() or not np.isfinite(y.to_numpy(dtype=float)).all():
        print("ERROR: target contains missing/non-finite values.")
        return 1

    groups = data[GROUP].astype(str).to_numpy()
    unique_participants = np.unique(groups)
    n_participants = len(unique_participants)
    if n_participants < 6:
        print("ERROR: at least 6 participants are needed for this grouped CV setup.")
        return 1

    # Five grouped outer folds estimate out-of-participant performance.
    outer_splits = min(N_OUTER_SPLITS, n_participants)
    inner_splits = min(N_INNER_SPLITS, n_participants - int(np.ceil(n_participants / outer_splits)))
    if inner_splits < 2:
        print("ERROR: not enough participant groups for nested GroupKFold.")
        return 1

    # All feature selection is embedded in the pipeline and therefore refit only
    # on each inner training fold. Candidate feature counts are tuned within CV.
    model_specs = {
        "RandomForestRegressor": {
            "pipeline": make_pipeline(
                RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=60, n_jobs=1),
                scale=False,
            ),
            "param_grid": {
                "select__k": [10, "all"],
                "model__max_depth": [None, 8],
                "model__min_samples_leaf": [2, 4],
            },
        },
        "SVR_RBF": {
            "pipeline": make_pipeline(SVR(kernel="rbf"), scale=True),
            "param_grid": {
                "select__k": [10, "all"],
                "model__C": [1.0, 10.0],
                "model__epsilon": [0.1, 0.5],
            },
        },
        "RidgeRegression": {
            "pipeline": make_pipeline(Ridge(), scale=True),
            "param_grid": {
                "select__k": [10, "all"],
                "model__alpha": [1.0, 10.0],
            },
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof_records = []
    fold_records = []
    selected_records = []
    correlation_records = []
    fold_metrics_records = []

    outer_cv = GroupKFold(n_splits=outer_splits)
    print("=== PARTICIPANT-GROUPED REGRESSION EXPERIMENT ===")
    print(f"Input: {INPUT_PATH}")
    print(f"Rows: {len(data)} | Participants: {n_participants} | Features: {len(feature_columns)}")
    print(f"Outer GroupKFold: {outer_splits} folds | Inner GroupKFold: {inner_splits} folds")
    print("Candidate models: Random Forest, RBF-SVR, Ridge Regression")
    print("Reference comparator: mean-only baseline (not an ML candidate)")
    print("Target: continuous self-reported rating; no stress classes are created.\n")

    # Each participant's task records appear entirely in either train or test
    # for each outer fold, including both f14_a and f14_b records.
    split_indices = list(outer_cv.split(X, y, groups))

    for fold_number, (train_idx, test_idx) in enumerate(split_indices, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        overlap = set(groups_train) & set(groups_test)
        if overlap:
            raise RuntimeError(f"Participant leakage in fold {fold_number}: {overlap}")

        print(
            f"Fold {fold_number}/{outer_splits}: "
            f"train={len(train_idx)} rows/{len(set(groups_train))} participants; "
            f"test={len(test_idx)} rows/{len(set(groups_test))} participants"
        )

        # Correlation report is computed only from each outer training partition.
        # These are descriptive Pearson correlations; no IID p-values are reported
        # because multiple task records belong to each participant.
        train_correlations = X_train.corrwith(y_train, method="pearson")
        for feature in feature_columns:
            corr = train_correlations.get(feature, np.nan)
            correlation_records.append({
                "outer_fold": fold_number,
                "feature": feature,
                "pearson_r_training_only": float(corr) if pd.notna(corr) else np.nan,
                "absolute_pearson_r": abs(float(corr)) if pd.notna(corr) else np.nan,
                "training_rows": len(train_idx),
                "training_participants": len(set(groups_train)),
            })

        # Mean-only reference, fit on training participants only.
        baseline = DummyRegressor(strategy="mean")
        baseline.fit(np.zeros((len(train_idx), 1)), y_train)
        baseline_pred = baseline.predict(np.zeros((len(test_idx), 1)))
        baseline_model_name = "MeanBaseline_reference"
        for idx, pred in zip(test_idx, baseline_pred):
            oof_records.append({
                "model": baseline_model_name,
                "outer_fold": fold_number,
                "participant_id": data.iloc[idx][GROUP],
                "session_id": data.iloc[idx]["session_id"],
                "task": data.iloc[idx]["task"],
                "actual_rating": float(y.iloc[idx]),
                "predicted_rating": float(pred),
                "best_parameters": "reference_mean_only",
                "selected_features": "none_model_ignores_features",
            })

        b_mae = mean_absolute_error(y_test, baseline_pred)
        b_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
        fold_metrics_records.append({
            "model": baseline_model_name,
            "outer_fold": fold_number,
            "mae": float(b_mae),
            "rmse": b_rmse,
            "r2": safe_r2(y_test, baseline_pred),
            "test_rows": len(test_idx),
            "test_participants": len(set(groups_test)),
        })

        for model_name, spec in model_specs.items():
            search = GridSearchCV(
                estimator=clone(spec["pipeline"]),
                param_grid=spec["param_grid"],
                scoring="neg_mean_absolute_error",
                cv=GroupKFold(n_splits=inner_splits),
                refit=True,
                n_jobs=1,
                error_score="raise",
                return_train_score=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                search.fit(X_train, y_train, groups=groups_train)

            best_estimator = search.best_estimator_
            predictions = best_estimator.predict(X_test)
            selector = best_estimator.named_steps["select"]
            selected_features = np.asarray(feature_columns)[selector.get_support()].tolist()
            best_params_text = json.dumps(search.best_params_, sort_keys=True, default=str)
            selected_text = ";".join(selected_features)

            for idx, pred in zip(test_idx, predictions):
                oof_records.append({
                    "model": model_name,
                    "outer_fold": fold_number,
                    "participant_id": data.iloc[idx][GROUP],
                    "session_id": data.iloc[idx]["session_id"],
                    "task": data.iloc[idx]["task"],
                    "actual_rating": float(y.iloc[idx]),
                    "predicted_rating": float(pred),
                    "best_parameters": best_params_text,
                    "selected_features": selected_text,
                })

            for feature in selected_features:
                selected_records.append({
                    "model": model_name,
                    "outer_fold": fold_number,
                    "feature": feature,
                    "selected_feature_count": len(selected_features),
                    "best_parameters": best_params_text,
                })

            mae = mean_absolute_error(y_test, predictions)
            rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
            r2 = safe_r2(y_test, predictions)
            fold_metrics_records.append({
                "model": model_name,
                "outer_fold": fold_number,
                "mae": float(mae),
                "rmse": rmse,
                "r2": r2,
                "test_rows": len(test_idx),
                "test_participants": len(set(groups_test)),
            })
            fold_records.append({
                "model": model_name,
                "outer_fold": fold_number,
                "n_train_rows": len(train_idx),
                "n_test_rows": len(test_idx),
                "n_train_participants": len(set(groups_train)),
                "n_test_participants": len(set(groups_test)),
                "inner_cv_best_mae": float(-search.best_score_),
                "outer_test_mae": float(mae),
                "outer_test_rmse": float(rmse),
                "outer_test_r2": float(r2),
                "selected_feature_count": len(selected_features),
                "selected_features": selected_text,
                "best_parameters": best_params_text,
            })
            print(
                f"  {model_name}: outer MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}; "
                f"selected={len(selected_features)} features"
            )

    predictions_df = pd.DataFrame(oof_records)
    fold_metrics_df = pd.DataFrame(fold_metrics_records)
    correlation_df = pd.DataFrame(correlation_records)
    selected_df = pd.DataFrame(selected_records)
    fold_search_df = pd.DataFrame(fold_records)

    # Pooled outer-fold predictions: every participant/task record is predicted
    # exactly once by a model that never saw that participant during fitting.
    summary_records = []
    for model_name, group_df in predictions_df.groupby("model", sort=False):
        actual = group_df["actual_rating"].to_numpy(dtype=float)
        predicted = group_df["predicted_rating"].to_numpy(dtype=float)
        fold_subset = fold_metrics_df[fold_metrics_df["model"] == model_name]
        summary_records.append({
            "model": model_name,
            "oof_samples": len(group_df),
            "oof_participants": group_df[GROUP].nunique(),
            "oof_mae_rating_points": float(mean_absolute_error(actual, predicted)),
            "oof_rmse_rating_points": float(np.sqrt(mean_squared_error(actual, predicted))),
            "oof_r2": float(r2_score(actual, predicted)),
            "mean_outer_fold_mae": float(fold_subset["mae"].mean()),
            "std_outer_fold_mae": float(fold_subset["mae"].std(ddof=1)),
            "mean_outer_fold_rmse": float(fold_subset["rmse"].mean()),
            "mean_outer_fold_r2": float(fold_subset["r2"].mean()),
        })

    results = pd.DataFrame(summary_records).sort_values(
        "oof_mae_rating_points", ascending=True
    )

    correlation_summary = (
        correlation_df.groupby("feature", as_index=False)
        .agg(
            mean_training_pearson_r=("pearson_r_training_only", "mean"),
            mean_absolute_training_pearson_r=("absolute_pearson_r", "mean"),
            folds_with_valid_correlation=("pearson_r_training_only", "count"),
        )
        .sort_values("mean_absolute_training_pearson_r", ascending=False)
    )

    predictions_df.to_csv(OUTPUT_DIR / "out_of_fold_predictions.csv", index=False)
    results.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False)
    fold_metrics_df.to_csv(OUTPUT_DIR / "outer_fold_metrics.csv", index=False)
    fold_search_df.to_csv(OUTPUT_DIR / "fold_model_selection_details.csv", index=False)
    correlation_df.to_csv(OUTPUT_DIR / "training_fold_feature_correlations.csv", index=False)
    correlation_summary.to_csv(OUTPUT_DIR / "training_correlation_summary.csv", index=False)
    selected_df.to_csv(OUTPUT_DIR / "selected_features_by_fold.csv", index=False)

    print("\n=== OUT-OF-PARTICIPANT MODEL COMPARISON ===")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\n=== TOP TRAINING-FOLD CORRELATIONS (DESCRIPTIVE ONLY) ===")
    print(correlation_summary.head(10).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\n=== OUTPUT FILES ===")
    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"{path} ({path.stat().st_size:,} bytes)")

    print("\n=== INTERPRETATION NOTES ===")
    print("- Lower MAE/RMSE is better; R2 closer to 1 is better, and can be negative.")
    print("- Compare candidates against MeanBaseline_reference, not just against each other.")
    print("- Correlations are descriptive and computed using outer-training participants only.")
    print("- Hyperparameter selection and feature selection happen inside grouped inner CV.")
    print("- Outer folds hold out whole participants; the OOF estimates are exploratory because there are only 18 participants.")
    print("- No final deployment model was saved; review results and supervisor guidance first.")
    print("- Ratings are self-reported scores, not doctor-confirmed stress diagnoses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
