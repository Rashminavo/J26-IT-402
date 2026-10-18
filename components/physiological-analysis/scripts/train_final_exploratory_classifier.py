"""Fit and save a provisional exploratory Low/Moderate/High classifier.

This is for research prototyping only. Labels are derived from quantiles of the
full available self-reported rating dataset, not clinically validated stress
severity boundaries. Cross-validated performance must be reported separately;
this script does not estimate generalisation performance.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Running this script from scripts/ makes the neighboring research pipeline
# importable without changing PYTHONPATH.
from train_stress_classifiers import (
    GROUP,
    INNER_SPLITS,
    INPUT_PATH,
    LABEL_NAMES,
    NON_FEATURE_COLUMNS,
    OUTPUT_DIR,
    TARGET,
    build_pipeline,
    label_ratings,
    rating_cutoffs,
    tune_model,
)

MODEL_NAME = "LogisticRegression"
MODEL_PATH = OUTPUT_DIR / "provisional_exploratory_logistic_classifier.joblib"
METADATA_PATH = OUTPUT_DIR / "provisional_exploratory_logistic_classifier_metadata.json"


def main() -> int:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Task-level dataset not found: {INPUT_PATH}")

    data = pd.read_csv(INPUT_PATH)
    required = {GROUP, "session_id", "task", TARGET}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required dataset columns: {sorted(missing)}")
    if data[[GROUP, "task"]].isna().any().any():
        raise ValueError("participant_id/task contain missing values.")
    if data.duplicated([GROUP, "task"]).any():
        raise ValueError("Duplicate participant-task rows found; refusing to train.")

    feature_columns = [c for c in data.columns if c not in NON_FEATURE_COLUMNS]
    X = data[feature_columns].apply(pd.to_numeric, errors="raise")
    y_numeric = pd.to_numeric(data[TARGET], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Feature matrix contains NaN or infinity.")
    if not np.isfinite(y_numeric).all():
        raise ValueError("Target ratings contain NaN or infinity.")

    groups = data[GROUP].astype(str).to_numpy()
    n_participants = len(np.unique(groups))
    if n_participants < 6:
        raise ValueError("At least 6 participants are required for grouped tuning.")
    inner_splits = min(INNER_SPLITS, n_participants - 1)

    # Tune only Logistic Regression because it had the highest OOF Macro-F1 in
    # the exploratory comparison. Tuning remains participant-grouped.
    best_params, inner_macro_f1, candidate_count = tune_model(
        MODEL_NAME,
        X,
        y_numeric,
        groups,
        inner_splits,
    )

    # For the full-data research artifact, derive its explicit label boundaries
    # from the available training ratings, then fit once to all task-level rows.
    lower_cutoff, upper_cutoff = rating_cutoffs(y_numeric)
    y_class = label_ratings(y_numeric, lower_cutoff, upper_cutoff)
    class_counts = {
        LABEL_NAMES[int(label)]: int((y_class == label).sum())
        for label in sorted(np.unique(y_class))
    }
    if set(np.unique(y_class)) != {0, 1, 2}:
        raise ValueError(f"Training labels do not contain all 3 classes: {class_counts}")

    pipeline = build_pipeline(MODEL_NAME, best_params, X.shape[1])
    pipeline.fit(X, y_class)

    selected_mask = pipeline.named_steps["select"].get_support()
    selected_features = [
        name for name, selected in zip(feature_columns, selected_mask) if selected
    ]

    bundle = {
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "class_names": LABEL_NAMES,
        "model_name": MODEL_NAME,
        "selected_parameters": best_params,
        "selected_features": selected_features,
        "rating_cutoffs": {
            "lower_rating_cutoff": float(lower_cutoff),
            "upper_rating_cutoff": float(upper_cutoff),
            "rule": "Low: rating <= lower; Moderate: lower < rating <= upper; High: rating > upper",
        },
        "labeling_method": "full-dataset one-third and two-thirds quantiles of self-reported numeric ratings",
        "clinical_validation": False,
        "research_use_only": True,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    metadata = {
        "model_path": str(MODEL_PATH),
        "input_dataset": str(INPUT_PATH),
        "model_name": MODEL_NAME,
        "n_rows": int(len(data)),
        "n_participants": int(n_participants),
        "n_features_before_selection": int(len(feature_columns)),
        "n_features_after_selection": int(len(selected_features)),
        "selected_features": selected_features,
        "selected_parameters": best_params,
        "inner_group_cv_folds": int(inner_splits),
        "best_inner_cv_macro_f1": float(inner_macro_f1),
        "hyperparameter_candidates_evaluated": int(candidate_count),
        "rating_cutoffs": {
            "lower_rating_cutoff": float(lower_cutoff),
            "upper_rating_cutoff": float(upper_cutoff),
        },
        "class_counts": class_counts,
        "class_mapping": {str(k): v for k, v in LABEL_NAMES.items()},
        "target": TARGET,
        "labeling_caveat": (
            "Exploratory relative bands of self-reported ratings only; these are not "
            "clinically validated stress severity thresholds. The final full-data "
            "quantile cutoffs may differ from the fold-specific cutoffs used during OOF evaluation."
        ),
        "performance_caveat": (
            "The inner CV score is for hyperparameter selection, not an unbiased final estimate. "
            "Use the previously saved participant-held-out OOF results as the exploratory "
            "performance estimate; no new independent test set exists."
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Smoke test the serialized artifact and feature schema.
    loaded = joblib.load(MODEL_PATH)
    X_sample = X.iloc[:5].loc[:, loaded["feature_columns"]]
    predicted = loaded["pipeline"].predict(X_sample)
    if len(predicted) != min(5, len(X)) or not set(np.unique(predicted)).issubset({0, 1, 2}):
        raise RuntimeError("Saved-model smoke test failed.")

    print("=== PROVISIONAL EXPLORATORY CLASSIFIER SAVED ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Rows: {len(data)} | Participants: {n_participants}")
    print(f"Features before selection: {len(feature_columns)}")
    print(f"Features after selection: {len(selected_features)}")
    print(f"Selected parameters: {best_params}")
    print(f"Inner grouped-CV Macro-F1 (tuning only): {inner_macro_f1:.3f}")
    print(f"Full-data rating cutoffs: lower={lower_cutoff:g}, upper={upper_cutoff:g}")
    print(f"Full-data class counts: {class_counts}")
    print(f"Saved-model smoke test predictions: {predicted.tolist()}")
    print("WARNING: research prototype only; class thresholds are exploratory and not clinically validated.")
    print("No new unbiased performance estimate was calculated by this final-fit script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
