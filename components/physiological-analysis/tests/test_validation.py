import pytest

from physiological_analysis.validation import (
    validate_measurements,
)


def test_validate_supported_measurements():
    result = validate_measurements(
        {
            "heart_rate": 78,
            "heart_rate_variability": 42,
            "body_temperature": 36.8,
        }
    )

    assert result["heart_rate"] == 78
    assert result["heart_rate_variability"] == 42
    assert result["body_temperature"] == 36.8


def test_unknown_measurements_are_ignored():
    result = validate_measurements(
        {
            "heart_rate": 78,
            "unknown_signal": 999,
        }
    )

    assert "heart_rate" in result
    assert "unknown_signal" not in result


def test_negative_measurement_is_rejected():
    with pytest.raises(ValueError):
        validate_measurements(
            {
                "heart_rate": -10,
            }
        )


def test_boolean_measurement_is_rejected():
    with pytest.raises(ValueError):
        validate_measurements(
            {
                "heart_rate": True,
            }
        )