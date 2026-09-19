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