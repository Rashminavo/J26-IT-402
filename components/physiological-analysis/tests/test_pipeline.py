import pytest

from physiological_analysis.pipeline import (
    analyze_physiology,
)


def test_physiological_pipeline():
    result = analyze_physiology(
        {
            "heart_rate": 78,
            "heart_rate_variability": 42,
            "body_temperature": 36.8,
            "sleep_duration": 7.2,
        }
    )

    assert result["stage"] == "physiological_reference"
    assert result["feature_count"] == 4
    assert "validated_data" in result
    assert "features" in result
    assert "standardized_features" in result
    assert result["model"]["name"] == "RandomForest"


def test_empty_measurements_are_rejected():
    with pytest.raises(ValueError):
        analyze_physiology({})
def test_heart_rate_series_pipeline():
    from physiological_analysis.pipeline import (
        analyze_heart_rate_series,
    )

    result = analyze_heart_rate_series(
        [60.0, 70.0, 80.0]
    )

    assert result["stage"] == "heart_rate_feature_extraction"
    assert result["feature_count"] == 4
    assert result["features"]["heart_rate_mean"] == 70.0
    assert result["model"]["status"] == (
        "pending_training_and_evaluation"
    )
    assert result["stress_score"]["status"] == (
        "PENDING_MODEL_IMPLEMENTATION"
    )


def test_heart_rate_series_pipeline_rejects_empty_input():
    from physiological_analysis.pipeline import (
        analyze_heart_rate_series,
    )

    with pytest.raises(ValueError):
        analyze_heart_rate_series([])
