"""Inference wrapper for the provisional exploratory stress classifier.

Research prototype only. The classifier predicts relative categories derived
from self-reported task ratings; it is not a clinically validated stress test.

Input must contain the 32 numeric, task-level aggregate physiological features
used during training. A single raw sensor sample or an isolated 15-second
window is not equivalent to the task-level training inputs.
"""
from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "model_comparison"
    / "provisional_exploratory_logistic_classifier.joblib"
)
EXPECTED_CLASS_NAMES = {"Low", "Moderate", "High"}


class ExploratoryStressClassifier:
    """Load and run the saved exploratory classifier.

    Load an instance once (e.g. when the app starts), then call ``predict``
    with task-level aggregate features for each prediction.
    """

    def __init__(self, bundle: Mapping[str, Any], model_path: Path | None = None):
        if not isinstance(bundle, Mapping):
            raise ValueError("Saved model artifact must contain a mapping/bundle.")

        required_keys = {
            "pipeline",
            "feature_columns",
            "class_names",
            "model_name",
            "rating_cutoffs",
            "clinical_validation",
            "research_use_only",
        }
        missing_keys = required_keys - set(bundle.keys())
        if missing_keys:
            raise ValueError(
                f"Saved model artifact is missing required keys: {sorted(missing_keys)}"
            )

        feature_columns = bundle["feature_columns"]
        if (
            not isinstance(feature_columns, (list, tuple))
            or not feature_columns
            or not all(isinstance(name, str) and name for name in feature_columns)
        ):
            raise ValueError("Artifact feature_columns must be a non-empty list of names.")
        if len(set(feature_columns)) != len(feature_columns):
            raise ValueError("Artifact feature_columns contain duplicates.")
        if not callable(getattr(bundle["pipeline"], "predict", None)):
            raise ValueError("Artifact pipeline does not provide a predict() method.")
        if not isinstance(bundle["class_names"], Mapping):
            raise ValueError("Artifact class_names must be a mapping of class IDs to labels.")

        try:
            class_names = {int(key): str(value) for key, value in bundle["class_names"].items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Artifact class_names contains an invalid class ID.") from exc

        if set(class_names.values()) != EXPECTED_CLASS_NAMES or len(class_names) != 3:
            raise ValueError(
                "Artifact must map exactly three class IDs to Low, Moderate, and High."
            )

        cutoffs = bundle["rating_cutoffs"]
        if not isinstance(cutoffs, Mapping):
            raise ValueError("Artifact rating_cutoffs must be a mapping.")
        for key in ("lower_rating_cutoff", "upper_rating_cutoff"):
            if key not in cutoffs:
                raise ValueError(f"Artifact rating_cutoffs is missing {key}.")

        self._pipeline = bundle["pipeline"]
        self.feature_columns = tuple(feature_columns)
        self.class_names = class_names
        self.model_name = str(bundle["model_name"])
        self.rating_cutoffs = dict(cutoffs)
        self.clinical_validation = bool(bundle["clinical_validation"])
        self.research_use_only = bool(bundle["research_use_only"])
        self.model_path = model_path

    @classmethod
    def load(
        cls, model_path: str | Path | None = None
    ) -> "ExploratoryStressClassifier":
        """Load the persisted model bundle from disk."""
        path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        if not path.is_file():
            raise FileNotFoundError(f"Saved exploratory model not found: {path}")
        try:
            bundle = joblib.load(path)
        except Exception as exc:  # include deserialization/version problems clearly
            raise RuntimeError(f"Could not load saved model artifact: {path}") from exc
        return cls(bundle, model_path=path)

    def predict(self, features: Mapping[str, Any]) -> dict[str, Any]:
        """Predict one exploratory class from task-level aggregated features.

        ``features`` must provide all expected model features. Extra keys are
        ignored. Values must be finite real numbers; booleans and numeric
        strings are rejected to catch upstream schema mistakes.
        """
        if not isinstance(features, Mapping):
            raise TypeError("features must be a mapping of feature names to numeric values.")

        missing = [name for name in self.feature_columns if name not in features]
        if missing:
            preview = ", ".join(missing[:8])
            suffix = "..." if len(missing) > 8 else ""
            raise ValueError(f"Missing required features ({len(missing)}): {preview}{suffix}")

        ordered_values: dict[str, float] = {}
        for name in self.feature_columns:
            value = features[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"Feature '{name}' must be a numeric value (not bool/string).")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"Feature '{name}' must be finite; got {value!r}.")
            ordered_values[name] = numeric_value

        input_frame = pd.DataFrame([ordered_values], columns=self.feature_columns)
        prediction = self._pipeline.predict(input_frame)
        if len(prediction) != 1:
            raise RuntimeError("Model returned an unexpected number of predictions.")

        raw_class_id = prediction[0]
        try:
            class_id = int(raw_class_id)
            if float(raw_class_id) != class_id:
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"Model returned invalid class ID: {raw_class_id!r}") from exc

        if class_id not in self.class_names:
            raise RuntimeError(f"Model returned unknown class ID: {class_id}")

        return {
            "predicted_class": self.class_names[class_id],
            "class_id": class_id,
            "model_name": self.model_name,
            "rating_cutoffs": dict(self.rating_cutoffs),
            "research_use_only": self.research_use_only,
            "clinical_validation": self.clinical_validation,
            "warning": (
                "Exploratory relative bands of self-reported ratings only; "
                "not a clinically validated stress-severity assessment."
            ),
        }
