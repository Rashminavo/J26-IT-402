"""Exploratory Low/Moderate/High wearable stress-rating classification.

Labels are data-relative, exploratory tertile bands from numeric self-reported
ratings. They are NOT clinically validated severity cutoffs.

Evaluation uses nested participant-grouped CV. For every inner split, tertile
cutoffs are calculated only from inner-training ratings; for outer evaluation,
cutoffs are calculated only from outer-training ratings and applied unchanged to
the held-out participants. No final deployment model is saved by this script.
"""
from __future__ import annotations

import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

INPUT_PATH = Path("data/processed/f_stress_task_level_dataset.csv")
OUTPUT_DIR = Path("data/processed/model_comparison")
TARGET = "self_reported_stress_rating"
GROUP = "participant_id"
NON_FEATURE_COLUMNS = {
    "participant_id", "session_id", "task", TARGET, "window_count",
    "task_start_seconds", "task_end_seconds", "task_duration_seconds",
}
LABELS = [0, 1, 2]
LABEL_NAMES = {0: "Low", 1: "Moderate", 2: "High"}
RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 3


def rating_cutoffs(y_train: np.ndarray) -> tuple[float, float]:
    """Calculate one-third and two-third quantiles from training ratings only."""
    lower, upper = np.quantile(np.asarray(y_train, dtype=float), [1 / 3, 2 / 3])
    lower, upper = float(lower), float(upper)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError(
            f"Cannot form 3 rating bands from training ratings: Q1/3={lower}, Q2/3={upper}"
        )
    return lower, upper


def label_ratings(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Apply fixed training-derived tertile cutoffs to ratings."""
    values = np.asarray(values, dtype=float)
    return np.where(values <= lower, 0, np.where(values <= upper, 1, 2)).astype(int)


def build_pipeline(model_name: str, params: dict, n_features: int) -> Pipeline:
    k = params["k"]
    if k != "all":
        k = min(int(k), n_features)

    if model_name == "RandomForestClassifier":
        estimator = RandomForestClassifier(
            n_estimators=200,
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    elif model_name == "SVC_RBF":
        estimator = SVC(
            kernel="rbf",
            C=params["C"],
            class_weight=params["class_weight"],
        )
    elif model_name == "LogisticRegression":
        estimator = LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            max_iter=3000,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=f_classif, k=k)),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def parameter_candidates(model_name: str) -> list[dict]:
    """Small, explicit grids to keep this exploratory experiment practical."""
    if model_name == "RandomForestClassifier":
        return [
            {"k": k, "max_depth": depth, "min_samples_leaf": leaf}
            for k, depth, leaf in product([10, "all"], [None, 8], [1, 3])
        ]
    if model_name == "SVC_RBF":
        return [
            {"k": k, "C": c, "class_weight": cw}
            for k, c, cw in product([10, "all"], [0.3, 3.0], [None, "balanced"])
        ]
    if model_name == "LogisticRegression":
        return [
            {"k": k, "C": c, "class_weight": cw}
            for k, c, cw in product([10, "all"], [0.1, 1.0], [None, "balanced"])
        ]
    raise ValueError(f"Unknown model: {model_name}")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1(y_true, y_pred),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
    }


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train_numeric: np.ndarray,
    groups_train: np.ndarray,
    inner_splits: int,
) -> tuple[dict, float, int]:
    """Manually tune hyperparameters with fold-local label cutoffs."""
    inner_cv = GroupKFold(n_splits=inner_splits)
    split_list = list(inner_cv.split(X_train, y_train_numeric, groups_train))
    candidates = parameter_candidates(model_name)
    best_params = None
    best_score = -np.inf

    for params in candidates:
        fold_scores = []
        for inner_train_idx, inner_valid_idx in split_list:
            numeric_inner_train = y_train_numeric[inner_train_idx]
            numeric_inner_valid = y_train_numeric[inner_valid_idx]
            lower, upper = rating_cutoffs(numeric_inner_train)
            labels_inner_train = label_ratings(numeric_inner_train, lower, upper)
            labels_inner_valid = label_ratings(numeric_inner_valid, lower, upper)

            if len(np.unique(labels_inner_train)) < 3:
                raise ValueError(
                    f"Inner training fold for {model_name} did not contain all 3 classes."
                )

            estimator = build_pipeline(model_name, params, X_train.shape[1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                warnings.simplefilter("ignore", category=UserWarning)
                estimator.fit(X_train.iloc[inner_train_idx], labels_inner_train)
                predicted = estimator.predict(X_train.iloc[inner_valid_idx])
            fold_scores.append(macro_f1(labels_inner_valid, predicted))

        candidate_score = float(np.mean(fold_scores))
        if candidate_score > best_score:
            best_score = candidate_score
            best_params = params.copy()

    if best_params is None:
        raise RuntimeError(f"Hyperparameter search failed for {model_name}.")
    return best_params, best_score, len(candidates)


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"ERROR: task-level dataset not found: {INPUT_PATH}")
        print("Run scripts\\build_task_dataset.py first.")
        return 1

    data = pd.read_csv(INPUT_PATH)
    required = {GROUP, "session_id", "task", TARGET}
    missing = required - set(data.columns)
    if missing:
        print(f"ERROR: missing required columns: {sorted(missing)}")
        return 1
    if data[[GROUP, "task"]].isna().any().any():
        print("ERROR: participant_id/task contain missing values.")
        return 1
    if data.duplicated([GROUP, "task"]).any():
        print("ERROR: duplicate participant-task rows; refusing to train.")
        return 1

    feature_columns = [c for c in data.columns if c not in NON_FEATURE_COLUMNS]
    X = data[feature_columns].apply(pd.to_numeric, errors="raise")
    y_numeric = pd.to_numeric(data[TARGET], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        print("ERROR: feature matrix contains NaN or infinity; stopping.")
        return 1
    if not np.isfinite(y_numeric).all():
        print("ERROR: target ratings contain NaN or infinity; stopping.")
        return 1

    groups = data[GROUP].astype(str).to_numpy()
    n_participants = len(np.unique(groups))
    if n_participants < 6:
        print("ERROR: at least 6 participants required for grouped nested CV.")
        return 1

    outer_splits = min(OUTER_SPLITS, n_participants)
    inner_splits = min(
        INNER_SPLITS,
        n_participants - int(np.ceil(n_participants / outer_splits)),
    )
    if inner_splits < 2:
        print("ERROR: insufficient participants for inner grouped CV.")
        return 1

    model_names = ["RandomForestClassifier", "SVC_RBF", "LogisticRegression"]
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_records = []
    fold_metric_records = []
    selected_feature_records = []
    tuning_records = []
    cutoff_records = []

    print("=== EXPLORATORY LOW / MODERATE / HIGH CLASSIFICATION ===")
    print(f"Input: {INPUT_PATH}")
    print(f"Rows: {len(data)} | Participants: {n_participants} | Features: {len(feature_columns)}")
    print(f"Outer GroupKFold: {outer_splits} folds | Inner GroupKFold: {inner_splits} folds")
    print("Models: Random Forest, RBF-SVC, Logistic Regression")
    print("Labels: training-rating tertiles; exploratory, not clinically validated.")
    print("Cutoffs are learned from training participants only in each split.")
    print("No final deployment model will be saved by this experiment.\n")

    outer_cv = GroupKFold(n_splits=outer_splits)
    outer_splits_list = list(outer_cv.split(X, y_numeric, groups))

    for fold_no, (train_idx, test_idx) in enumerate(outer_splits_list, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_numeric, y_test_numeric = y_numeric[train_idx], y_numeric[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]
        overlap = set(groups_train) & set(groups_test)
        if overlap:
            raise RuntimeError(f"Participant leakage in outer fold {fold_no}: {sorted(overlap)}")

        lower, upper = rating_cutoffs(y_train_numeric)
        y_train = label_ratings(y_train_numeric, lower, upper)
        y_test = label_ratings(y_test_numeric, lower, upper)
        if len(np.unique(y_train)) < 3:
            raise ValueError(f"Outer training fold {fold_no} does not contain all 3 classes.")

        train_counts = {LABEL_NAMES[c]: int(np.sum(y_train == c)) for c in LABELS}
        test_counts = {LABEL_NAMES[c]: int(np.sum(y_test == c)) for c in LABELS}
        cutoff_records.append({
            "outer_fold": fold_no,
            "lower_rating_cutoff": lower,
            "upper_rating_cutoff": upper,
            "train_low": train_counts["Low"],
            "train_moderate": train_counts["Moderate"],
            "train_high": train_counts["High"],
            "test_low": test_counts["Low"],
            "test_moderate": test_counts["Moderate"],
            "test_high": test_counts["High"],
            "train_participants": len(set(groups_train)),
            "test_participants": len(set(groups_test)),
        })

        print(
            f"Outer fold {fold_no}/{outer_splits}: "
            f"train={len(train_idx)} rows/{len(set(groups_train))} participants, "
            f"test={len(test_idx)} rows/{len(set(groups_test))} participants; "
            f"cutoffs Low<= {lower:.3g}, Moderate<= {upper:.3g}, High> {upper:.3g}"
        )
        print(f"  Class counts train={train_counts} | test={test_counts}")

        # Majority-class reference using outer-training labels only.
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(np.zeros((len(train_idx), 1)), y_train)
        dummy_pred = dummy.predict(np.zeros((len(test_idx), 1)))
        base_metrics = metrics(y_test, dummy_pred)
        fold_metric_records.append({
            "model": "MajorityBaseline_reference",
            "outer_fold": fold_no,
            **base_metrics,
            "test_rows": len(test_idx),
            "test_participants": len(set(groups_test)),
            "lower_rating_cutoff": lower,
            "upper_rating_cutoff": upper,
            "best_parameters": "reference_majority_class_only",
        })
        for idx, predicted in zip(test_idx, dummy_pred):
            oof_records.append({
                "model": "MajorityBaseline_reference",
                "outer_fold": fold_no,
                "participant_id": data.iloc[idx][GROUP],
                "session_id": data.iloc[idx]["session_id"],
                "task": data.iloc[idx]["task"],
                "actual_rating": float(y_numeric[idx]),
                "lower_rating_cutoff": lower,
                "upper_rating_cutoff": upper,
                "actual_class": LABEL_NAMES[int(y_test[np.where(test_idx == idx)[0][0]])],
                "predicted_class": LABEL_NAMES[int(predicted)],
                "actual_class_id": int(y_test[np.where(test_idx == idx)[0][0]]),
                "predicted_class_id": int(predicted),
                "best_parameters": "reference_majority_class_only",
                "selected_features": "none_model_ignores_features",
            })

        for model_name in model_names:
            best_params, inner_score, candidate_count = tune_model(
                model_name, X_train, y_train_numeric, groups_train, inner_splits
            )
            model = build_pipeline(model_name, best_params, len(feature_columns))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                warnings.simplefilter("ignore", category=UserWarning)
                model.fit(X_train, y_train)
                predicted = model.predict(X_test)

            model_metrics = metrics(y_test, predicted)
            params_text = json.dumps(best_params, sort_keys=True)
            fold_metric_records.append({
                "model": model_name,
                "outer_fold": fold_no,
                **model_metrics,
                "test_rows": len(test_idx),
                "test_participants": len(set(groups_test)),
                "lower_rating_cutoff": lower,
                "upper_rating_cutoff": upper,
                "inner_best_macro_f1": inner_score,
                "candidate_configurations": candidate_count,
                "best_parameters": params_text,
            })
            tuning_records.append({
                "model": model_name,
                "outer_fold": fold_no,
                "inner_best_macro_f1": inner_score,
                "candidate_configurations": candidate_count,
                "best_parameters": params_text,
                "lower_rating_cutoff": lower,
                "upper_rating_cutoff": upper,
            })

            selector = model.named_steps["select"]
            selected = np.asarray(feature_columns)[selector.get_support()].tolist()
            for feature in selected:
                selected_feature_records.append({
                    "model": model_name,
                    "outer_fold": fold_no,
                    "feature": feature,
                    "selected_feature_count": len(selected),
                })

            for local_pos, (idx, predicted_id) in enumerate(zip(test_idx, predicted)):
                actual_id = int(y_test[local_pos])
                oof_records.append({
                    "model": model_name,
                    "outer_fold": fold_no,
                    "participant_id": data.iloc[idx][GROUP],
                    "session_id": data.iloc[idx]["session_id"],
                    "task": data.iloc[idx]["task"],
                    "actual_rating": float(y_numeric[idx]),
                    "lower_rating_cutoff": lower,
                    "upper_rating_cutoff": upper,
                    "actual_class": LABEL_NAMES[actual_id],
                    "predicted_class": LABEL_NAMES[int(predicted_id)],
                    "actual_class_id": actual_id,
                    "predicted_class_id": int(predicted_id),
                    "best_parameters": params_text,
                    "selected_features": ";".join(selected),
                })

            print(
                f"  {model_name}: macro-F1={model_metrics['macro_f1']:.3f}, "
                f"balanced accuracy={model_metrics['balanced_accuracy']:.3f}, "
                f"accuracy={model_metrics['accuracy']:.3f}; selected={len(selected)} features"
            )

    oof = pd.DataFrame(oof_records)
    fold_df = pd.DataFrame(fold_metric_records)
    cutoff_df = pd.DataFrame(cutoff_records)
    tuning_df = pd.DataFrame(tuning_records)
    selected_df = pd.DataFrame(selected_feature_records)

    summary_records = []
    for model_name, group_df in oof.groupby("model", sort=False):
        y_true = group_df["actual_class_id"].to_numpy(dtype=int)
        y_pred = group_df["predicted_class_id"].to_numpy(dtype=int)
        row = {
            "model": model_name,
            "oof_samples": len(group_df),
            "oof_participants": group_df["participant_id"].nunique(),
            **metrics(y_true, y_pred),
        }
        model_fold = fold_df[fold_df["model"] == model_name]
        row.update({
            "mean_outer_fold_macro_f1": float(model_fold["macro_f1"].mean()),
            "std_outer_fold_macro_f1": float(model_fold["macro_f1"].std(ddof=0)),
            "mean_outer_fold_balanced_accuracy": float(model_fold["balanced_accuracy"].mean()),
        })
        summary_records.append(row)
    summary_df = pd.DataFrame(summary_records).sort_values("macro_f1", ascending=False)

    # Overall-data quantiles are shown only as a possible full-data training
    # threshold after evaluation. They were NOT used to define outer test labels.
    full_lower, full_upper = rating_cutoffs(y_numeric)
    full_labels = label_ratings(y_numeric, full_lower, full_upper)
    full_counts = {LABEL_NAMES[c]: int(np.sum(full_labels == c)) for c in LABELS}

    report_records = []
    for model_name, group_df in oof.groupby("model", sort=False):
        report = classification_report(
            group_df["actual_class_id"].to_numpy(dtype=int),
            group_df["predicted_class_id"].to_numpy(dtype=int),
            labels=LABELS,
            target_names=[LABEL_NAMES[i] for i in LABELS],
            output_dict=True,
            zero_division=0,
        )
        for name in ["Low", "Moderate", "High"]:
            report_records.append({
                "model": model_name,
                "class": name,
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1_score": report[name]["f1-score"],
                "support": report[name]["support"],
            })
        cm = confusion_matrix(
            group_df["actual_class_id"].to_numpy(dtype=int),
            group_df["predicted_class_id"].to_numpy(dtype=int),
            labels=LABELS,
        )
        for i, actual_name in enumerate([LABEL_NAMES[c] for c in LABELS]):
            for j, predicted_name in enumerate([LABEL_NAMES[c] for c in LABELS]):
                report_records.append({
                    "model": model_name,
                    "class": f"CONFUSION actual={actual_name}, predicted={predicted_name}",
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1_score": int(cm[i, j]),
                    "support": np.nan,
                })

    # Save reproducible artifacts.
    outputs = {
        "classification_model_comparison.csv": summary_df,
        "classification_oof_predictions.csv": oof,
        "classification_outer_fold_metrics.csv": fold_df,
        "classification_label_cutoffs.csv": cutoff_df,
        "classification_nested_cv_details.csv": tuning_df,
        "classification_selected_features_by_fold.csv": selected_df,
        "classification_per_class_report_and_confusion.csv": pd.DataFrame(report_records),
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    metadata = {
        "experiment_type": "exploratory three-class classification",
        "target_source": "numeric self-reported stress ratings in Stress_Level_v2.csv",
        "label_method": "per-split one-third/two-third quantiles of training ratings",
        "class_rule": {
            "Low": "rating <= training one-third quantile",
            "Moderate": "rating > training one-third quantile and <= training two-third quantile",
            "High": "rating > training two-third quantile",
        },
        "full_dataset_quantile_cutoffs_for_future_full_data_fit_only": {
            "lower": full_lower,
            "upper": full_upper,
            "class_counts": full_counts,
        },
        "outer_cv": "GroupKFold by participant_id; all tasks for each participant are held out together",
        "inner_cv": "GroupKFold by participant_id; label cutoffs re-estimated from each inner-training split",
        "model_candidates": model_names,
        "random_state": RANDOM_STATE,
        "clinical_validation": False,
        "deployment_model_saved": False,
        "caution": "Class cutoffs are exploratory data-relative bands, not doctor- or clinically-validated thresholds. They may differ by fold. Metrics are uncertain with only 18 participants.",
    }
    (output_dir / "classification_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n=== OUT-OF-PARTICIPANT CLASSIFICATION COMPARISON ===")
    print(summary_df.round(3).to_string(index=False))
    print("\n=== OUTER-FOLD EXPLORATORY LABEL CUT-OFFS ===")
    print(cutoff_df.to_string(index=False))
    print("\n=== FULL-DATA TERTILES (FOR A FUTURE FULL-DATA FIT ONLY) ===")
    print(f"Lower cut-off: {full_lower:.3g} | Upper cut-off: {full_upper:.3g}")
    print(f"Label counts: {full_counts}")
    print("\n=== CLASSIFICATION REPORTS / CONFUSION MATRICES ===")
    print(output_dir / "classification_per_class_report_and_confusion.csv")
    print("\n=== OUTPUT FILES ===")
    for filename, frame in outputs.items():
        print(f"{filename}: {len(frame)} rows")
    print("classification_experiment_metadata.json: saved")
    print("\nNo final/deployment model was saved.")
    print("Labels are exploratory relative bands of self-reported ratings, not clinical severity classes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
