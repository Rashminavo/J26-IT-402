"""Nested participant-grouped modality comparison for wearable stress ratings.

Exploratory only. Uses continuous self-reported ratings, does not create stress
classes, and never splits task records from a participant across train/test.
"""
from __future__ import annotations

import json
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
    "participant_id", "session_id", "task", TARGET, "window_count",
    "task_start_seconds", "task_end_seconds", "task_duration_seconds",
}
RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 3


def make_pipeline(model, scale: bool) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=f_regression, k="all")),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def get_model_specs(n_features: int):
    # Small feature groups use all their features. For wider groups, selecting
    # either 10 or all features is tuned only inside grouped inner CV.
    k_values = [10, "all"] if n_features > 10 else ["all"]
    return {
        "RandomForestRegressor": {
            "pipeline": make_pipeline(
                RandomForestRegressor(
                    random_state=RANDOM_STATE, n_estimators=60, n_jobs=1
                ), scale=False
            ),
            "param_grid": {
                "select__k": k_values,
                "model__max_depth": [None, 8],
                "model__min_samples_leaf": [2, 4],
            },
        },
        "SVR_RBF": {
            "pipeline": make_pipeline(SVR(kernel="rbf"), scale=True),
            "param_grid": {
                "select__k": k_values,
                "model__C": [1.0, 10.0],
                "model__epsilon": [0.1, 0.5],
            },
        },
        "RidgeRegression": {
            "pipeline": make_pipeline(Ridge(), scale=True),
            "param_grid": {
                "select__k": k_values,
                "model__alpha": [1.0, 10.0],
            },
        },
    }


def metric_values(y_true, y_pred):
    return {
        "mae_rating_points": float(mean_absolute_error(y_true, y_pred)),
        "rmse_rating_points": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"ERROR: input dataset not found: {INPUT_PATH}")
        print("Run scripts\\build_task_dataset.py first.")
        return 1

    data = pd.read_csv(INPUT_PATH)
    needed = {GROUP, "session_id", "task", TARGET}
    missing = needed - set(data.columns)
    if missing:
        print(f"ERROR: missing required columns: {sorted(missing)}")
        return 1
    if data[[GROUP, "task"]].isna().any().any():
        print("ERROR: missing participant/task values.")
        return 1
    if data.duplicated([GROUP, "task"]).any():
        print("ERROR: duplicated participant-task records; stopping.")
        return 1

    all_features = [c for c in data.columns if c not in NON_FEATURE_COLUMNS]
    X_all = data[all_features].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(data[TARGET], errors="raise")
    if not np.isfinite(X_all.to_numpy(dtype=float)).all():
        print("ERROR: non-finite features; stopping.")
        return 1
    if y.isna().any() or not np.isfinite(y.to_numpy(dtype=float)).all():
        print("ERROR: invalid target ratings; stopping.")
        return 1

    groups = data[GROUP].astype(str).to_numpy()
    n_participants = len(np.unique(groups))
    if n_participants < 6:
        print("ERROR: at least six participants are required.")
        return 1

    outer_splits = min(OUTER_SPLITS, n_participants)
    inner_splits = min(
        INNER_SPLITS,
        n_participants - int(np.ceil(n_participants / outer_splits)),
    )
    if inner_splits < 2:
        print("ERROR: insufficient participant groups for nested CV.")
        return 1

    # Group-specific experiments use either all sensors or one sensor family.
    # ACC is represented by its X/Y/Z statistics and vector-magnitude statistics.
    sensor_groups = {
        "HR": [c for c in all_features if c.startswith("heart_rate_")],
        "EDA": [c for c in all_features if c.startswith("eda_")],
        "BVP": [c for c in all_features if c.startswith("bvp_")],
        "TEMP": [c for c in all_features if c.startswith("temp_")],
        "ACC": [c for c in all_features if c.startswith("acc_")],
    }
    groups_to_test = {"ALL_SENSORS": list(all_features)}
    # Single-family runs check each modality's standalone predictive signal.
    for name, columns in sensor_groups.items():
        if columns:
            groups_to_test[f"{name}_ONLY"] = columns
    # Leave-one-family-out runs estimate whether removing a modality changes
    # performance relative to the all-sensors setup.
    for name, columns in sensor_groups.items():
        remaining = [c for c in all_features if c not in columns]
        if remaining:
            groups_to_test[f"ALL_EXCEPT_{name}"] = remaining

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    fold_rows = []
    selection_rows = []

    print("=== SENSOR-GROUP ABLATION: PARTICIPANT-GROUPED REGRESSION ===")
    print(f"Rows: {len(data)} | Participants: {n_participants}")
    print(f"Outer GroupKFold: {outer_splits} | Inner GroupKFold: {inner_splits}")
    print("Groups tested:")
    for name, cols in groups_to_test.items():
        print(f"  {name}: {len(cols)} features")
    print("Target: continuous self-reported rating; no Low/Moderate/High labels.\n")

    outer_cv = GroupKFold(n_splits=outer_splits)
    outer_indices = list(outer_cv.split(X_all, y, groups))

    for group_name, feature_columns in groups_to_test.items():
        print(f"--- {group_name} ({len(feature_columns)} features) ---")
        X = X_all[feature_columns]
        model_specs = get_model_specs(len(feature_columns))

        for fold_number, (train_idx, test_idx) in enumerate(outer_indices, start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            group_train, group_test = groups[train_idx], groups[test_idx]
            overlap = set(group_train) & set(group_test)
            if overlap:
                raise RuntimeError(f"Participant leakage: {sorted(overlap)}")

            # The baseline is refit from training data for every outer fold.
            dummy = DummyRegressor(strategy="mean")
            dummy.fit(np.zeros((len(train_idx), 1)), y_train)
            baseline_pred = dummy.predict(np.zeros((len(test_idx), 1)))
            fold_rows.append({
                "sensor_group": group_name,
                "model": "MeanBaseline_reference",
                "outer_fold": fold_number,
                **metric_values(y_test, baseline_pred),
                "test_rows": len(test_idx),
                "test_participants": len(set(group_test)),
                "feature_count": 0,
            })
            for idx, pred in zip(test_idx, baseline_pred):
                prediction_rows.append({
                    "sensor_group": group_name,
                    "model": "MeanBaseline_reference",
                    "outer_fold": fold_number,
                    "participant_id": data.iloc[idx][GROUP],
                    "session_id": data.iloc[idx]["session_id"],
                    "task": data.iloc[idx]["task"],
                    "actual_rating": float(y.iloc[idx]),
                    "predicted_rating": float(pred),
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
                    search.fit(X_train, y_train, groups=group_train)

                best = search.best_estimator_
                pred = best.predict(X_test)
                fold_rows.append({
                    "sensor_group": group_name,
                    "model": model_name,
                    "outer_fold": fold_number,
                    **metric_values(y_test, pred),
                    "test_rows": len(test_idx),
                    "test_participants": len(set(group_test)),
                    "feature_count": len(feature_columns),
                })
                selected_cols = np.asarray(feature_columns)[
                    best.named_steps["select"].get_support()
                ].tolist()
                selection_rows.append({
                    "sensor_group": group_name,
                    "model": model_name,
                    "outer_fold": fold_number,
                    "best_parameters": json.dumps(
                        search.best_params_, sort_keys=True, default=str
                    ),
                    "selected_features": ";".join(selected_cols),
                    "selected_feature_count": len(selected_cols),
                })
                for idx, value in zip(test_idx, pred):
                    prediction_rows.append({
                        "sensor_group": group_name,
                        "model": model_name,
                        "outer_fold": fold_number,
                        "participant_id": data.iloc[idx][GROUP],
                        "session_id": data.iloc[idx]["session_id"],
                        "task": data.iloc[idx]["task"],
                        "actual_rating": float(y.iloc[idx]),
                        "predicted_rating": float(value),
                    })

            print(f"  outer fold {fold_number}/{outer_splits} complete")

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    selections = pd.DataFrame(selection_rows)

    summary_rows = []
    for (group_name, model_name), part in predictions.groupby(
        ["sensor_group", "model"], sort=False
    ):
        y_true = part["actual_rating"].to_numpy(dtype=float)
        y_pred = part["predicted_rating"].to_numpy(dtype=float)
        fold_part = fold_metrics[
            (fold_metrics["sensor_group"] == group_name)
            & (fold_metrics["model"] == model_name)
        ]
        summary_rows.append({
            "sensor_group": group_name,
            "model": model_name,
            "feature_count": int(
                len(groups_to_test[group_name]) if model_name != "MeanBaseline_reference" else 0
            ),
            "oof_samples": len(part),
            "oof_participants": part["participant_id"].nunique(),
            **metric_values(y_true, y_pred),
            "mean_outer_fold_mae": float(fold_part["mae_rating_points"].mean()),
            "std_outer_fold_mae": float(fold_part["mae_rating_points"].std(ddof=0)),
            "mean_outer_fold_rmse": float(fold_part["rmse_rating_points"].mean()),
            "mean_outer_fold_r2": float(fold_part["r2"].mean()),
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["sensor_group", "mae_rating_points"]
    )
    paths = {
        "summary": OUTPUT_DIR / "sensor_group_ablation_results.csv",
        "predictions": OUTPUT_DIR / "sensor_group_ablation_oof_predictions.csv",
        "fold_metrics": OUTPUT_DIR / "sensor_group_ablation_fold_metrics.csv",
        "selected_features": OUTPUT_DIR / "sensor_group_ablation_selected_features.csv",
    }
    summary.to_csv(paths["summary"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    fold_metrics.to_csv(paths["fold_metrics"], index=False)
    selections.to_csv(paths["selected_features"], index=False)

    print("\n=== OUT-OF-PARTICIPANT SENSOR-GROUP RESULTS ===")
    print(summary.round(3).to_string(index=False))
    print("\n=== OUTPUT FILES ===")
    for label, path in paths.items():
        print(f"{label}: {path} ({path.stat().st_size:,} bytes)")
    print("\nInterpretation: lower MAE/RMSE is better; higher R2 is better.")
    print("This is exploratory evidence, not clinical validation or a final deployed model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
