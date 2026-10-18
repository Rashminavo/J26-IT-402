from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pytest

from physiological_analysis.window_stress_classifier import WindowLevelStressClassifier


class StubPipeline:
    def __init__(self, prediction=1):
        self.prediction = prediction
        self.last_input = None

    def predict(self, values):
        self.last_input = np.asarray(values).copy()
        return np.asarray([self.prediction])


def make_bundle(prediction=1, feature_columns=None, input_level="15-second window-level physiological features"):
    columns = feature_columns or ["heart_rate_mean", "eda_eda_mean", "temp_temp_mean"]
    return {
        "pipeline": StubPipeline(prediction),
        "feature_columns": columns,
        "class_names": {0: "Low", 1: "Moderate", 2: "High"},
        "model_name": "TestSVC",
        "rating_cutoffs": {"lower_rating_cutoff": 3.0, "upper_rating_cutoff": 5.0},
        "clinical_validation": False,
        "research_use_only": True,
        "input_level": input_level,
        "window_size_seconds": 15,
    }


def sample_features():
    return {"temp_temp_mean": 31.2, "heart_rate_mean": 72.5, "eda_eda_mean": 0.4}


def test_predict_maps_class_and_marks_exploratory():
    result = WindowLevelStressClassifier(make_bundle(2)).predict(sample_features())
    assert result["predicted_class"] == "High"
    assert result["class_id"] == 2
    assert result["window_size_seconds"] == 15
    assert result["research_use_only"] is True
    assert result["clinical_validation"] is False


def test_predict_orders_features_by_saved_schema():
    bundle = make_bundle(0)
    WindowLevelStressClassifier(bundle).predict(sample_features())
    assert bundle["pipeline"].last_input.tolist() == [[72.5, 0.4, 31.2]]


@pytest.mark.parametrize("class_id,label", [(0, "Low"), (1, "Moderate"), (2, "High")])
def test_all_class_ids_map(class_id, label):
    assert WindowLevelStressClassifier(make_bundle(class_id)).predict(sample_features())["predicted_class"] == label


def test_missing_feature_rejected():
    with pytest.raises(ValueError, match="Missing required features"):
        WindowLevelStressClassifier(make_bundle()).predict({"heart_rate_mean": 70.0})


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_feature_rejected(bad):
    features = sample_features()
    features["eda_eda_mean"] = bad
    with pytest.raises(ValueError, match="must be finite"):
        WindowLevelStressClassifier(make_bundle()).predict(features)


@pytest.mark.parametrize("bad", [True, "72.5", None])
def test_non_numeric_feature_rejected(bad):
    features = sample_features()
    features["heart_rate_mean"] = bad
    with pytest.raises(ValueError, match="numeric value"):
        WindowLevelStressClassifier(make_bundle()).predict(features)


def test_non_mapping_input_rejected():
    with pytest.raises(TypeError, match="features must be a mapping"):
        WindowLevelStressClassifier(make_bundle()).predict([1, 2, 3])


def test_unknown_class_id_rejected():
    with pytest.raises(RuntimeError, match="unknown class ID"):
        WindowLevelStressClassifier(make_bundle(9)).predict(sample_features())


def test_input_level_must_be_window_level():
    with pytest.raises(ValueError, match="input_level"):
        WindowLevelStressClassifier(make_bundle(input_level="task-level aggregated physiological features"))


def test_model_bundle_round_trip_load(tmp_path: Path):
    path = tmp_path / "model.joblib"
    joblib.dump(make_bundle(1), path)
    clf = WindowLevelStressClassifier.load(path)
    assert clf.predict(sample_features())["predicted_class"] == "Moderate"


def test_missing_model_file_rejected(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Saved window-level model not found"):
        WindowLevelStressClassifier.load(tmp_path / "absent.joblib")
