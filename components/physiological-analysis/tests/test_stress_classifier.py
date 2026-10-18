from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from physiological_analysis.stress_classifier import ExploratoryStressClassifier


class StubPipeline:
    def __init__(self, prediction=1):
        self.prediction = prediction
        self.last_frame = None

    def predict(self, frame):
        self.last_frame = frame.copy()
        return np.asarray([self.prediction])


def make_bundle(prediction=1, features=None):
    feature_columns = features or ["heart_rate_mean", "eda_eda_mean", "temp_temp_mean"]
    return {
        "pipeline": StubPipeline(prediction),
        "feature_columns": feature_columns,
        "class_names": {0: "Low", 1: "Moderate", 2: "High"},
        "model_name": "TestClassifier",
        "rating_cutoffs": {
            "lower_rating_cutoff": 3.0,
            "upper_rating_cutoff": 5.0,
            "rule": "exploratory test rule",
        },
        "clinical_validation": False,
        "research_use_only": True,
    }


def sample_features():
    # Deliberately supplied out of feature-column order to test stable ordering.
    return {
        "temp_temp_mean": 31.2,
        "heart_rate_mean": 72.5,
        "eda_eda_mean": 0.4,
    }


def test_predict_maps_class_id_and_marks_result_exploratory():
    bundle = make_bundle(prediction=1)
    classifier = ExploratoryStressClassifier(bundle)

    result = classifier.predict(sample_features())

    assert result["predicted_class"] == "Moderate"
    assert result["class_id"] == 1
    assert result["model_name"] == "TestClassifier"
    assert result["research_use_only"] is True
    assert result["clinical_validation"] is False
    assert "not a clinically validated" in result["warning"]


def test_predict_orders_features_using_saved_schema():
    bundle = make_bundle(prediction=0)
    classifier = ExploratoryStressClassifier(bundle)

    classifier.predict(sample_features())

    assert list(bundle["pipeline"].last_frame.columns) == [
        "heart_rate_mean", "eda_eda_mean", "temp_temp_mean"
    ]
    assert bundle["pipeline"].last_frame.iloc[0].tolist() == [72.5, 0.4, 31.2]


@pytest.mark.parametrize("prediction,expected", [(0, "Low"), (1, "Moderate"), (2, "High")])
def test_all_class_ids_map_to_expected_labels(prediction, expected):
    classifier = ExploratoryStressClassifier(make_bundle(prediction=prediction))
    assert classifier.predict(sample_features())["predicted_class"] == expected


def test_missing_feature_raises_clear_error():
    classifier = ExploratoryStressClassifier(make_bundle())
    with pytest.raises(ValueError, match="Missing required features"):
        classifier.predict({"heart_rate_mean": 70.0})


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_feature_raises_clear_error(bad_value):
    classifier = ExploratoryStressClassifier(make_bundle())
    features = sample_features()
    features["eda_eda_mean"] = bad_value
    with pytest.raises(ValueError, match="must be finite"):
        classifier.predict(features)


@pytest.mark.parametrize("bad_value", [True, "72.5", None])
def test_non_numeric_schema_values_are_rejected(bad_value):
    classifier = ExploratoryStressClassifier(make_bundle())
    features = sample_features()
    features["heart_rate_mean"] = bad_value
    with pytest.raises(ValueError, match="must be a numeric value"):
        classifier.predict(features)


def test_unknown_predicted_class_id_raises_error():
    classifier = ExploratoryStressClassifier(make_bundle(prediction=9))
    with pytest.raises(RuntimeError, match="unknown class ID"):
        classifier.predict(sample_features())


def test_non_mapping_input_is_rejected():
    classifier = ExploratoryStressClassifier(make_bundle())
    with pytest.raises(TypeError, match="features must be a mapping"):
        classifier.predict([1.0, 2.0, 3.0])


def test_invalid_bundle_is_rejected():
    with pytest.raises(ValueError, match="missing required keys"):
        ExploratoryStressClassifier({"pipeline": StubPipeline()})


def test_load_missing_model_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Saved exploratory model not found"):
        ExploratoryStressClassifier.load(tmp_path / "absent.joblib")
