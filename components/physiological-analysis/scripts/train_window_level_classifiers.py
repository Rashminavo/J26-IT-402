"""Exploratory window-level stress classification with participant-grouped CV.

This experiment predicts relative classes derived from participants' self-reported
ratings. Ratings are task-level and are repeated across windows; therefore:
- quantile cutoffs are derived from UNIQUE participant-task ratings in training folds;
- each participant-task contributes equal total training weight regardless of window count;
- outer CV holds out whole participants;
- task-level metrics (mean window scores per held-out task) are primary;
- window-level metrics are included only as supplementary diagnostics.

These labels are exploratory relative bands, not clinically validated stress levels.
No final deployment model is saved by this script.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

LABELS = {0: "Low", 1: "Moderate", 2: "High"}
CLASS_IDS = [0, 1, 2]
GROUP_KEYS = ["participant_id", "task"]
REQUIRED_COLUMNS = {
    "participant_id",
    "session_id",
    "task",
    "self_reported_stress_rating",
    "window_start_seconds",
    "window_end_seconds",
}
NON_FEATURE_COLUMNS = REQUIRED_COLUMNS | {
    "window_duration_seconds",
    "window_count",
    "task_start_seconds",
    "task_end_seconds",
    "task_duration_seconds",
    "rating",
    "target",
    "label",
    "class_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/processed/f_stress_15s_labeled_windows.csv",
        help="Input window-level dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/model_comparison",
        help="Directory for result CSV/JSON files.",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    return parser.parse_args()


def rating_task_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one unique numeric rating per participant-task (not per window)."""
    table = frame[["participant_id", "task", "self_reported_stress_rating"]].copy()
    conflict = table.groupby(GROUP_KEYS)["self_reported_stress_rating"].nunique()
    if (conflict > 1).any():
        bad = conflict[conflict > 1].index.tolist()[:10]
        raise ValueError(f"Conflicting ratings for participant-task pairs: {bad}")
    return table.drop_duplicates(GROUP_KEYS).reset_index(drop=True)


def learn_cutoffs(train_frame: pd.DataFrame) -> tuple[float, float]:
    """Learn tertile cutoffs from unique training participant-task ratings only."""
    ratings = rating_task_table(train_frame)["self_reported_stress_rating"].astype(float)
    if ratings.empty or not np.isfinite(ratings.to_numpy()).all():
        raise ValueError("Cannot derive cutoffs from empty/non-finite training ratings.")
    lower = float(ratings.quantile(1 / 3))
    upper = float(ratings.quantile(2 / 3))
    if lower >= upper:
        raise ValueError(
            f"Tertile cutoffs are not distinct (lower={lower}, upper={upper}); "
            "these data do not support three exploratory rating bands in this fold."
        )
    return lower, upper


def encode_ratings(values: pd.Series | np.ndarray, lower: float, upper: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.where(arr <= lower, 0, np.where(arr <= upper, 1, 2)).astype(int)


def create_training_weights(frame: pd.DataFrame, lower: float, upper: float) -> np.ndarray:
    """Weight each participant-task equally and balance classes at task level."""
    keys = frame[GROUP_KEYS].astype(str).agg("||".join, axis=1)
    window_counts = keys.value_counts()
    per_window_group_count = keys.map(window_counts).to_numpy(dtype=float)
    base = 1.0 / per_window_group_count

    task_table = rating_task_table(frame)
    task_table["class_id"] = encode_ratings(
        task_table["self_reported_stress_rating"], lower, upper
    )
    task_class_counts = task_table["class_id"].value_counts().to_dict()
    n_tasks = len(task_table)
    present_classes = len(task_class_counts)
    if present_classes < 2:
        raise ValueError("Training fold has fewer than two classes; cannot train classifier.")

    class_weight = {
        cls: n_tasks / (present_classes * count)
        for cls, count in task_class_counts.items()
    }
    class_by_key = {
        str(row.participant_id) + "||" + str(row.task): int(row.class_id)
        for row in task_table.itertuples(index=False)
    }
    per_window_class_weight = np.array([class_weight[class_by_key[key]] for key in keys])
    weights = base * per_window_class_weight
    mean_weight = float(weights.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("Training sample weights are invalid.")
    return weights / mean_weight


def model_candidates() -> dict[str, list[dict[str, Any]]]:
    return {
        "RandomForestClassifier": [
            {"n_estimators": 120, "max_depth": None, "min_samples_leaf": 2},
            {"n_estimators": 120, "max_depth": 8, "min_samples_leaf": 4},
        ],
        "SVC_RBF": [
            {"C": 0.5, "gamma": "scale"},
            {"C": 2.0, "gamma": "scale"},
        ],
        "LogisticRegression": [
            {"C": 0.1, "max_iter": 3000},
            {"C": 1.0, "max_iter": 3000},
        ],
    }


def make_pipeline(model_name: str, params: dict[str, Any]) -> Pipeline:
    if model_name == "RandomForestClassifier":
        estimator = RandomForestClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "SVC_RBF":
        estimator = SVC(
            kernel="rbf",
            probability=False,
            decision_function_shape="ovr",
            random_state=42,
            **params,
        )
    elif model_name == "LogisticRegression":
        estimator = LogisticRegression(
            **params,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", estimator),
    ])


def fit_model(
    model: Pipeline,
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    lower: float,
    upper: float,
) -> Pipeline:
    y = encode_ratings(train_frame["self_reported_stress_rating"], lower, upper)
    if set(np.unique(y)) != set(CLASS_IDS):
        raise ValueError(
            f"Training fold does not contain all 3 classes: {sorted(set(np.unique(y)))}"
        )
    weights = create_training_weights(train_frame, lower, upper)
    model.fit(
        train_frame[feature_columns].to_numpy(dtype=float),
        y,
        classifier__sample_weight=weights,
    )
    return model


def prediction_scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return three columns of model scores in the fixed Low/Moderate/High order."""
    clf = model.named_steps["classifier"]
    trained_classes = np.asarray(clf.classes_, dtype=int)
    if hasattr(clf, "predict_proba"):
        raw = np.asarray(model.predict_proba(X), dtype=float)
    else:
        raw = np.asarray(model.decision_function(X), dtype=float)
        if raw.ndim == 1:
            raw = np.column_stack([-raw, raw])

    full = np.full((len(X), 3), -1e6 if not hasattr(clf, "predict_proba") else 0.0)
    for column_index, class_id in enumerate(trained_classes):
        full[:, class_id] = raw[:, column_index]
    return full


def aggregate_scores_to_tasks(
    frame: pd.DataFrame,
    scores: np.ndarray,
    lower: float,
    upper: float,
    fold: int,
    model_name: str,
) -> pd.DataFrame:
    work = frame[[
        "participant_id", "session_id", "task", "self_reported_stress_rating",
        "window_start_seconds", "window_end_seconds",
    ]].copy().reset_index(drop=True)
    for class_id in CLASS_IDS:
        work[f"score_{class_id}"] = scores[:, class_id]

    group_columns = ["participant_id", "session_id", "task"]
    agg = work.groupby(group_columns, as_index=False).agg(
        self_reported_stress_rating=("self_reported_stress_rating", "first"),
        window_count=("window_start_seconds", "size"),
        mean_window_start_seconds=("window_start_seconds", "mean"),
        mean_window_end_seconds=("window_end_seconds", "mean"),
        score_0=("score_0", "mean"),
        score_1=("score_1", "mean"),
        score_2=("score_2", "mean"),
    )
    agg["actual_class_id"] = encode_ratings(
        agg["self_reported_stress_rating"], lower, upper
    )
    mean_scores = agg[["score_0", "score_1", "score_2"]].to_numpy(dtype=float)
    agg["predicted_class_id"] = mean_scores.argmax(axis=1)
    agg["actual_class"] = agg["actual_class_id"].map(LABELS)
    agg["predicted_class"] = agg["predicted_class_id"].map(LABELS)
    agg["outer_fold"] = fold
    agg["model"] = model_name
    agg["lower_rating_cutoff"] = lower
    agg["upper_rating_cutoff"] = upper
    return agg


def metric_dict(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=CLASS_IDS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, labels=CLASS_IDS, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(actual, predicted, labels=CLASS_IDS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(actual, predicted, labels=CLASS_IDS, average="macro", zero_division=0)),
    }


def choose_params(
    outer_train: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    candidates: list[dict[str, Any]],
    inner_folds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_groups = outer_train["participant_id"].nunique()
    n_splits = min(inner_folds, n_groups)
    if n_splits < 2:
        raise ValueError("Not enough training participants for inner grouped CV.")
    splitter = GroupKFold(n_splits=n_splits)
    scores_by_candidate: list[list[float]] = [[] for _ in candidates]
    detail_rows: list[dict[str, Any]] = []

    for inner_fold, (tr_idx, va_idx) in enumerate(
        splitter.split(outer_train, groups=outer_train["participant_id"]), start=1
    ):
        inner_train = outer_train.iloc[tr_idx].copy().reset_index(drop=True)
        inner_valid = outer_train.iloc[va_idx].copy().reset_index(drop=True)
        lower, upper = learn_cutoffs(inner_train)
        for candidate_index, params in enumerate(candidates):
            model = make_pipeline(model_name, params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_model(model, inner_train, feature_columns, lower, upper)
            scores = prediction_scores(
                model, inner_valid[feature_columns].to_numpy(dtype=float)
            )
            task_predictions = aggregate_scores_to_tasks(
                inner_valid, scores, lower, upper, inner_fold, model_name
            )
            y_true = task_predictions["actual_class_id"].to_numpy(dtype=int)
            y_pred = task_predictions["predicted_class_id"].to_numpy(dtype=int)
            score = float(f1_score(y_true, y_pred, labels=CLASS_IDS, average="macro", zero_division=0))
            scores_by_candidate[candidate_index].append(score)
            detail_rows.append({
                "model": model_name,
                "candidate_index": candidate_index,
                "parameters": json.dumps(params, sort_keys=True),
                "inner_fold": inner_fold,
                "inner_task_macro_f1": score,
                "lower_rating_cutoff": lower,
                "upper_rating_cutoff": upper,
                "train_participants": inner_train["participant_id"].nunique(),
                "validation_participants": inner_valid["participant_id"].nunique(),
            })

    mean_scores = [float(np.mean(scores)) for scores in scores_by_candidate]
    best_index = int(np.argmax(mean_scores))
    for row in detail_rows:
        row["candidate_mean_inner_macro_f1"] = mean_scores[row["candidate_index"]]
    return candidates[best_index], detail_rows


def validate_input(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    frame = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Input dataset is missing columns: {sorted(missing_columns)}")

    feature_columns = [
        col for col in frame.columns
        if col not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if len(feature_columns) != 32:
        raise ValueError(
            f"Expected 32 numeric physiological features, found {len(feature_columns)}: "
            f"{feature_columns}"
        )
    numeric_columns = feature_columns + [
        "self_reported_stress_rating", "window_start_seconds", "window_end_seconds"
    ]
    for col in numeric_columns:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Input numeric columns contain NaN or infinity.")
    if frame.empty:
        raise ValueError("Input dataset is empty.")
    if frame["participant_id"].isna().any() or frame["task"].isna().any():
        raise ValueError("Participant IDs/tasks cannot be missing.")
    if frame.duplicated(["participant_id", "session_id", "window_start_seconds"]).any():
        raise ValueError("Duplicate participant-session-window-start keys detected.")
    if (frame["window_end_seconds"] <= frame["window_start_seconds"]).any():
        raise ValueError("Invalid window interval: end must exceed start.")
    task_table = rating_task_table(frame)
    if len(task_table) < 20:
        raise ValueError(f"Expected more participant-task records; found {len(task_table)}.")
    if frame["participant_id"].nunique() < 6:
        raise ValueError("Not enough participants for grouped cross-validation.")
    return frame.reset_index(drop=True), feature_columns


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    frame, feature_columns = validate_input(input_path)
    n_participants = int(frame["participant_id"].nunique())
    n_outer = min(int(args.outer_folds), n_participants)
    if n_outer < 2:
        raise ValueError("At least 2 outer folds are required.")
    outer = GroupKFold(n_splits=n_outer)
    candidates_by_model = model_candidates()

    print("=== WINDOW-LEVEL CLASSIFICATION WITH PARTICIPANT-GROUPED CV ===")
    print(f"Input: {input_path}")
    print(f"Windows: {len(frame)} | Participants: {n_participants} | Sessions: {frame['session_id'].nunique()}")
    print(f"Unique participant-task ratings: {len(rating_task_table(frame))}")
    print(f"Physiological features: {len(feature_columns)}")
    print(f"Outer GroupKFold: {n_outer} | Inner GroupKFold: {min(args.inner_folds, n_participants - 1)}")
    print("Training sample weights make each participant-task contribute equally.")
    print("Primary evaluation: held-out task predictions aggregated across windows.")
    print("Labels are fold-specific tertiles of unique training participant-task self-reports.")
    print("These are exploratory rating bands, not clinically validated stress levels.")
    print("No final/deployment model will be saved by this experiment.\n")

    all_task_predictions: list[pd.DataFrame] = []
    all_window_predictions: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    cutoff_rows: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(
        outer.split(frame, groups=frame["participant_id"]), start=1
    ):
        outer_train = frame.iloc[train_idx].copy().reset_index(drop=True)
        outer_test = frame.iloc[test_idx].copy().reset_index(drop=True)
        lower, upper = learn_cutoffs(outer_train)
        y_train = encode_ratings(outer_train["self_reported_stress_rating"], lower, upper)
        train_class_counts = rating_task_table(outer_train).assign(
            class_id=lambda d: encode_ratings(d["self_reported_stress_rating"], lower, upper)
        )["class_id"].value_counts().to_dict()
        test_task_table = rating_task_table(outer_test)
        test_class_counts = pd.Series(
            encode_ratings(test_task_table["self_reported_stress_rating"], lower, upper)
        ).value_counts().to_dict()
        cutoff_rows.append({
            "outer_fold": fold,
            "lower_rating_cutoff": lower,
            "upper_rating_cutoff": upper,
            "train_participants": outer_train["participant_id"].nunique(),
            "test_participants": outer_test["participant_id"].nunique(),
            "train_task_count": len(rating_task_table(outer_train)),
            "test_task_count": len(test_task_table),
            "train_class_counts": json.dumps({LABELS[k]: int(v) for k, v in train_class_counts.items()}, sort_keys=True),
            "test_task_class_counts": json.dumps({LABELS[int(k)]: int(v) for k, v in test_class_counts.items()}, sort_keys=True),
        })
        print(
            f"Outer fold {fold}/{n_outer}: train={outer_train['participant_id'].nunique()} participants, "
            f"test={outer_test['participant_id'].nunique()} participants; cutoffs Low<={lower:.3g}, "
            f"Moderate<={upper:.3g}, High>{upper:.3g}"
        )

        train_task_classes = encode_ratings(
            rating_task_table(outer_train)["self_reported_stress_rating"], lower, upper
        )
        values, counts = np.unique(train_task_classes, return_counts=True)
        majority_class = int(values[np.argmax(counts)])

        for model_name, candidates in candidates_by_model.items():
            best_params, tuning_details = choose_params(
                outer_train, feature_columns, model_name, candidates, args.inner_folds
            )
            for detail in tuning_details:
                detail["outer_fold"] = fold
                detail["selected_candidate"] = (detail["candidate_index"] == candidates.index(best_params))
                tuning_rows.append(detail)

            model = make_pipeline(model_name, best_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_model(model, outer_train, feature_columns, lower, upper)
            test_X = outer_test[feature_columns].to_numpy(dtype=float)
            scores = prediction_scores(model, test_X)
            window_pred = scores.argmax(axis=1)
            window_true = encode_ratings(
                outer_test["self_reported_stress_rating"], lower, upper
            )

            window_out = outer_test[[
                "participant_id", "session_id", "task", "window_start_seconds",
                "window_end_seconds", "self_reported_stress_rating",
            ]].copy()
            window_out["outer_fold"] = fold
            window_out["model"] = model_name
            window_out["lower_rating_cutoff"] = lower
            window_out["upper_rating_cutoff"] = upper
            window_out["actual_class_id"] = window_true
            window_out["actual_class"] = window_out["actual_class_id"].map(LABELS)
            window_out["predicted_class_id"] = window_pred
            window_out["predicted_class"] = window_out["predicted_class_id"].map(LABELS)
            all_window_predictions.append(window_out)

            task_out = aggregate_scores_to_tasks(
                outer_test, scores, lower, upper, fold, model_name
            )
            all_task_predictions.append(task_out)
            task_metrics = metric_dict(
                task_out["actual_class_id"].to_numpy(dtype=int),
                task_out["predicted_class_id"].to_numpy(dtype=int),
            )
            window_metrics = metric_dict(window_true, window_pred)
            fold_metric_rows.append({
                "outer_fold": fold,
                "model": model_name,
                "best_parameters": json.dumps(best_params, sort_keys=True),
                "train_participants": outer_train["participant_id"].nunique(),
                "test_participants": outer_test["participant_id"].nunique(),
                "test_windows": len(outer_test),
                "test_tasks": len(task_out),
                "lower_rating_cutoff": lower,
                "upper_rating_cutoff": upper,
                **{f"task_{k}": v for k, v in task_metrics.items()},
                **{f"window_{k}": v for k, v in window_metrics.items()},
            })
            print(
                f"  {model_name}: task Macro-F1={task_metrics['macro_f1']:.3f}, "
                f"balanced accuracy={task_metrics['balanced_accuracy']:.3f}, "
                f"window Macro-F1={window_metrics['macro_f1']:.3f}; params={best_params}"
            )

        # Majority baseline is based on unique outer-training task labels.
        test_task_labels = encode_ratings(
            test_task_table["self_reported_stress_rating"], lower, upper
        )
        baseline_task_out = test_task_table.copy()
        baseline_task_out["actual_class_id"] = test_task_labels
        baseline_task_out["predicted_class_id"] = majority_class
        baseline_task_out["actual_class"] = baseline_task_out["actual_class_id"].map(LABELS)
        baseline_task_out["predicted_class"] = baseline_task_out["predicted_class_id"].map(LABELS)
        baseline_task_out["outer_fold"] = fold
        baseline_task_out["model"] = "MajorityBaseline_reference"
        baseline_task_out["lower_rating_cutoff"] = lower
        baseline_task_out["upper_rating_cutoff"] = upper
        for cid in CLASS_IDS:
            baseline_task_out[f"score_{cid}"] = float(cid == majority_class)
        all_task_predictions.append(baseline_task_out)
        baseline_metrics = metric_dict(test_task_labels, np.full_like(test_task_labels, majority_class))
        fold_metric_rows.append({
            "outer_fold": fold,
            "model": "MajorityBaseline_reference",
            "best_parameters": "reference_majority_class",
            "train_participants": outer_train["participant_id"].nunique(),
            "test_participants": outer_test["participant_id"].nunique(),
            "test_windows": len(outer_test),
            "test_tasks": len(baseline_task_out),
            "lower_rating_cutoff": lower,
            "upper_rating_cutoff": upper,
            **{f"task_{k}": v for k, v in baseline_metrics.items()},
            **{f"window_{k}": np.nan for k in baseline_metrics},
        })

    task_oof = pd.concat(all_task_predictions, ignore_index=True)
    window_oof = pd.concat(all_window_predictions, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    tuning_details = pd.DataFrame(tuning_rows)
    cutoffs_df = pd.DataFrame(cutoff_rows)

    summary_rows = []
    for model_name, group in task_oof.groupby("model", sort=False):
        metrics = metric_dict(
            group["actual_class_id"].to_numpy(dtype=int),
            group["predicted_class_id"].to_numpy(dtype=int),
        )
        fold_group = fold_metrics[fold_metrics["model"] == model_name]
        row = {
            "model": model_name,
            "oof_tasks": len(group),
            "oof_participants": group["participant_id"].nunique(),
            **metrics,
            "mean_outer_fold_macro_f1": float(fold_group["task_macro_f1"].mean()),
            "std_outer_fold_macro_f1": float(fold_group["task_macro_f1"].std(ddof=0)),
            "mean_outer_fold_balanced_accuracy": float(fold_group["task_balanced_accuracy"].mean()),
        }
        if model_name != "MajorityBaseline_reference":
            win = window_oof[window_oof["model"] == model_name]
            row["oof_windows_supplementary"] = len(win)
            row["window_accuracy_supplementary"] = float(
                accuracy_score(win["actual_class_id"], win["predicted_class_id"])
            )
            row["window_macro_f1_supplementary"] = float(
                f1_score(win["actual_class_id"], win["predicted_class_id"], labels=CLASS_IDS, average="macro", zero_division=0)
            )
        else:
            row["oof_windows_supplementary"] = np.nan
            row["window_accuracy_supplementary"] = np.nan
            row["window_macro_f1_supplementary"] = np.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)

    # Confusion matrices use task-level out-of-fold predictions as the primary analysis.
    confusion_rows = []
    for model_name, group in task_oof.groupby("model", sort=False):
        mat = confusion_matrix(
            group["actual_class_id"].to_numpy(dtype=int),
            group["predicted_class_id"].to_numpy(dtype=int),
            labels=CLASS_IDS,
        )
        for actual_id, actual_label in LABELS.items():
            for pred_id, pred_label in LABELS.items():
                confusion_rows.append({
                    "model": model_name,
                    "actual_class": actual_label,
                    "predicted_class": pred_label,
                    "count": int(mat[actual_id, pred_id]),
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "window_level_classification_comparison.csv"
    task_path = output_dir / "window_level_classification_oof_task_predictions.csv"
    window_path = output_dir / "window_level_classification_oof_window_predictions.csv"
    fold_path = output_dir / "window_level_classification_fold_metrics.csv"
    tuning_path = output_dir / "window_level_classification_nested_cv_details.csv"
    cutoff_path = output_dir / "window_level_classification_cutoffs.csv"
    confusion_path = output_dir / "window_level_classification_task_confusion_matrices.csv"
    metadata_path = output_dir / "window_level_classification_metadata.json"

    summary.to_csv(summary_path, index=False)
    task_oof.to_csv(task_path, index=False)
    window_oof.to_csv(window_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    tuning_details.to_csv(tuning_path, index=False)
    cutoffs_df.to_csv(cutoff_path, index=False)
    pd.DataFrame(confusion_rows).to_csv(confusion_path, index=False)
    metadata = {
        "experiment": "window_level_exploratory_stress_classification",
        "input": str(input_path),
        "windows": int(len(frame)),
        "participants": int(n_participants),
        "sessions": int(frame["session_id"].nunique()),
        "unique_participant_task_ratings": int(len(rating_task_table(frame))),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "outer_folds": int(n_outer),
        "inner_folds_requested": int(args.inner_folds),
        "threshold_method": "one-third and two-thirds quantiles of unique training participant-task self-reported ratings",
        "training_weights": "equal total weight per participant-task, with additional task-level class balancing",
        "primary_metric_level": "task-level; mean model scores over all held-out windows in each task",
        "window_metrics": "supplementary only; overlapping windows and repeated task ratings are correlated",
        "clinical_validation": False,
        "deployment_model_saved": False,
        "limitations": [
            "Only 18 participants.",
            "Labels are exploratory relative bands of self-reported task ratings.",
            "Tertile boundaries vary by training fold; they are not fixed clinical severity thresholds.",
            "Window-level ratings are repeated from tasks and are not independent labels.",
            "Overlapping 15-second windows are correlated.",
            "No final/deployment model is saved by this evaluation script.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== OUT-OF-PARTICIPANT TASK-LEVEL COMPARISON (PRIMARY) ===")
    print(summary.round(3).to_string(index=False))
    print("\n=== TASK-LEVEL CONFUSION MATRICES ===")
    confusion_df = pd.DataFrame(confusion_rows)
    for model_name in summary["model"].tolist():
        part = confusion_df[confusion_df["model"] == model_name]
        matrix = part.pivot(index="actual_class", columns="predicted_class", values="count")
        matrix = matrix.reindex(index=["Low", "Moderate", "High"], columns=["Low", "Moderate", "High"]).fillna(0).astype(int)
        print(f"\n{model_name} (rows=actual, columns=predicted)")
        print(matrix.to_string())

    print("\n=== SAVED OUTPUTS ===")
    for path in [summary_path, task_path, window_path, fold_path, tuning_path, cutoff_path, confusion_path, metadata_path]:
        print(path)
    print("\nImportant: primary metrics use aggregated held-out task predictions; window-level metrics are supplementary.")
    print("Exploratory self-report bands only; no clinical validity or deployment readiness is claimed.")
    print("No final/deployment model was saved.")


if __name__ == "__main__":
    main()
