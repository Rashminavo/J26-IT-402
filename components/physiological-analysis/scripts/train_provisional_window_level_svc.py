"""Fit and save a provisional window-level SVC stress classifier.

This script builds a deployment-shaped *research prototype* using the same
window-level input format as the exploratory participant-grouped CV experiment.
Hyperparameters are selected with GroupKFold grouped by participant. Quantile
cutoffs are re-estimated from unique training participant-task ratings inside
each tuning fold, then a final fit uses full-data cutoffs.

The saved scores from prior nested CV remain the exploratory performance
estimate. This final full-data fit does not provide a new unbiased metric.
The labels are exploratory bands of self-reported ratings, not clinical stress
severity categories. Do not use for clinical decisions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_window_level_classifiers as experiment  # noqa: E402


def main() -> None:
    input_path = Path("data/processed/f_stress_15s_labeled_windows.csv")
    output_dir = Path("data/processed/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "provisional_window_level_svc.joblib"
    metadata_path = output_dir / "provisional_window_level_svc_metadata.json"
    tuning_path = output_dir / "provisional_window_level_svc_tuning.csv"

    frame, feature_columns = experiment.validate_input(input_path)
    participant_count = int(frame["participant_id"].nunique())
    if participant_count < 6:
        raise ValueError("Need at least 6 participants for grouped tuning.")

    model_name = "SVC_RBF"
    candidates = experiment.model_candidates()[model_name]

    print("=== PROVISIONAL WINDOW-LEVEL SVC FINAL FIT ===")
    print(f"Input: {input_path}")
    print(
        f"Windows: {len(frame)} | Participants: {participant_count} | "
        f"Sessions: {frame['session_id'].nunique()} | "
        f"Participant-task ratings: {len(experiment.rating_task_table(frame))}"
    )
    print(f"Physiological features: {len(feature_columns)}")
    print("Tuning: 3-fold GroupKFold by participant; task-level Macro-F1")
    print("Each tuning fold derives its cutoffs from training participants only.")

    # Tune the final fit using grouped folds. This tuning score is not a new
    # unbiased performance estimate; the previous nested outer-CV is the reportable estimate.
    best_params, tuning_details = experiment.choose_params(
        frame,
        feature_columns,
        model_name,
        candidates,
        inner_folds=3,
    )
    pd.DataFrame(tuning_details).to_csv(tuning_path, index=False)

    lower, upper = experiment.learn_cutoffs(frame)
    task_table = experiment.rating_task_table(frame)
    task_labels = experiment.encode_ratings(
        task_table["self_reported_stress_rating"], lower, upper
    )
    class_counts = {
        experiment.LABELS[int(class_id)]: int(count)
        for class_id, count in zip(*np.unique(task_labels, return_counts=True))
    }

    pipeline = experiment.make_pipeline(model_name, best_params)
    experiment.fit_model(pipeline, frame, feature_columns, lower, upper)

    bundle: dict[str, Any] = {
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "class_names": experiment.LABELS,
        "model_name": model_name,
        "rating_cutoffs": {
            "lower_rating_cutoff": float(lower),
            "upper_rating_cutoff": float(upper),
            "rule": (
                "Low: rating <= lower; Moderate: lower < rating <= upper; "
                "High: rating > upper"
            ),
            "threshold_method": (
                "one-third and two-thirds quantiles of unique participant-task "
                "self-reported ratings from the full development dataset"
            ),
        },
        "clinical_validation": False,
        "research_use_only": True,
        "input_level": "15-second window-level physiological features",
        "window_size_seconds": 15,
        "training_rows": int(len(frame)),
        "training_participants": participant_count,
        "training_sessions": int(frame["session_id"].nunique()),
        "unique_participant_task_ratings": int(len(task_table)),
        "training_class_counts_by_task": class_counts,
        "selected_parameters": best_params,
        "selection_method": (
            "3-fold GroupKFold by participant; within each fold, cutoffs learned "
            "from unique training participant-task self-reports; select by mean "
            "task-level Macro-F1"
        ),
        "performance_note": (
            "This full-data fit does not provide an unbiased performance estimate. "
            "Use window_level_classification_comparison.csv from the nested "
            "participant-grouped CV experiment as the exploratory estimate."
        ),
        "warning": (
            "Exploratory relative bands of self-reported task ratings only; "
            "not clinically validated stress-severity categories."
        ),
    }
    joblib.dump(bundle, model_path)

    metadata = {key: value for key, value in bundle.items() if key != "pipeline"}
    metadata["model_artifact"] = str(model_path)
    metadata["tuning_details_file"] = str(tuning_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Round-trip smoke test: reload the file and predict all windows.
    loaded_bundle = joblib.load(model_path)
    X = frame[feature_columns].to_numpy(dtype=float)
    predictions = np.asarray(loaded_bundle["pipeline"].predict(X))
    allowed = set(experiment.CLASS_IDS)
    if len(predictions) != len(frame):
        raise RuntimeError("Round-trip prediction count does not match input rows.")
    if not set(np.unique(predictions)).issubset(allowed):
        raise RuntimeError(f"Unexpected predicted class IDs: {np.unique(predictions)}")
    predicted_counts = {
        experiment.LABELS[int(class_id)]: int(count)
        for class_id, count in zip(*np.unique(predictions, return_counts=True))
    }

    print("\n=== MODEL ARTIFACT SAVED ===")
    print(f"Model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Tuning details: {tuning_path}")
    print(f"Selected parameters: {best_params}")
    print(f"Full-data cutoffs: Low <= {lower:g}, Moderate <= {upper:g}, High > {upper:g}")
    print(f"Training task-class counts: {class_counts}")
    print(f"Reloaded-model predictions checked: {len(predictions)}")
    print(f"In-sample predicted window counts (smoke test only): {predicted_counts}")
    print("No new unbiased performance estimate was calculated by this fit.")
    print("WARNING: exploratory research prototype; not clinically validated or deployment-ready.")


if __name__ == "__main__":
    main()
