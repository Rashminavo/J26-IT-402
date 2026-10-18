"""Inference wrapper for the provisional 15-second window-level stress model.

Research prototype only: predicts exploratory bands derived from self-reported
ratings, not clinically validated stress severity classes.
"""
from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "model_comparison"
    / "provisional_window_level_svc.joblib"
)
EXPECTED_CLASS_NAMES = {"Low", "Moderate", "High"}
EXPECTED_INPUT_LEVEL = "15-second window-level physiological features"


class WindowLevelStressClassifier:
    """Load and predict one 15-second feature window from the saved SVC bundle.

    The caller must first extract the same 32 named features used during training.
    Raw sensor samples are not accepted by this class.
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
            "input_level",
            "window_size_seconds",
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

        if bundle["input_level"] != EXPECTED_INPUT_LEVEL:
            raise ValueError(
                "Artifact input_level must identify 15-second window-level physiological features."
            )
        if bundle["window_size_seconds"] != 15:
            raise ValueError("Artifact window_size_seconds must be 15.")

        if not isinstance(bundle["class_names"], Mapping):
            raise ValueError("Artifact class_names must map class IDs to labels.")
        try:
            class_names = {int(key): str(value) for key, value in bundle["class_names"].items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Artifact class_names contains an invalid class ID.") from exc
        if set(class_names.keys()) != {0, 1, 2} or set(class_names.values()) != EXPECTED_CLASS_NAMES:
            raise ValueError("Artifact must map class IDs 0/1/2 to Low/Moderate/High.")

        cutoffs = bundle["rating_cutoffs"]
        if not isinstance(cutoffs, Mapping):
            raise ValueError("Artifact rating_cutoffs must be a mapping.")
        for key in ("lower_rating_cutoff", "upper_rating_cutoff"):
            if key not in cutoffs:
                raise ValueError(f"Artifact rating_cutoffs is missing {key}.")
        try:
            lower = float(cutoffs["lower_rating_cutoff"])
            upper = float(cutoffs["upper_rating_cutoff"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Artifact rating cutoffs must be numeric.") from exc
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("Artifact rating cutoffs must be finite and lower < upper.")

        self._pipeline = bundle["pipeline"]
        self.feature_columns = tuple(feature_columns)
        self.class_names = class_names
        self.model_name = str(bundle["model_name"])
        self.rating_cutoffs = dict(cutoffs)
        self.clinical_validation = bool(bundle["clinical_validation"])
        self.research_use_only = bool(bundle["research_use_only"])
        self.window_size_seconds = 15
        self.model_path = model_path

    @classmethod
    def load(cls, model_path: str | Path | None = None) -> "WindowLevelStressClassifier":
        """Load the persisted 15-second window-level model bundle from disk."""
        path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        if not path.is_file():
            raise FileNotFoundError(f"Saved window-level model not found: {path}")
        try:
            bundle = joblib.load(path)
        except Exception as exc:  # surface a clearer path for pickle/version errors
            raise RuntimeError(f"Could not load saved window-level model artifact: {path}") from exc
        return cls(bundle, model_path=path)

    def predict(self, features: Mapping[str, Any]) -> dict[str, Any]:
        """Predict one exploratory class from one 15-second feature mapping."""
        if not isinstance(features, Mapping):
            raise TypeError("features must be a mapping of feature names to numeric values.")

        missing = [name for name in self.feature_columns if name not in features]
        if missing:
            preview = ", ".join(missing[:8])
            suffix = "..." if len(missing) > 8 else ""
            raise ValueError(f"Missing required features ({len(missing)}): {preview}{suffix}")

        values: list[float] = []
        for name in self.feature_columns:
            value = features[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"Feature '{name}' must be a numeric value (not bool/string).")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"Feature '{name}' must be finite; got {value!r}.")
            values.append(numeric_value)

        input_array = np.asarray([values], dtype=float)
        try:
            prediction = self._pipeline.predict(input_array)
        except Exception as exc:
            raise RuntimeError("Saved classifier failed to predict this 15-second feature window.") from exc
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
            "input_level": EXPECTED_INPUT_LEVEL,
            "window_size_seconds": self.window_size_seconds,
            "rating_cutoffs": dict(self.rating_cutoffs),
            "research_use_only": self.research_use_only,
            "clinical_validation": self.clinical_validation,
            "warning": (
                "Exploratory relative bands of self-reported task ratings only; "
                "not a clinically validated stress-severity assessment."
            ),
        }
