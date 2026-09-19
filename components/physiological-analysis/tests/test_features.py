from physiological_analysis.features import (
    count_available_features,
    extract_features,
)


def test_extract_physiological_features():
    result = extract_features(
        {
            "heart_rate": 78.0,
            "body_temperature": 36.8,
        }
    )

    assert result["heart_rate"] == 78.0
    assert result["body_temperature"] == 36.8


def test_count_available_features():
    result = count_available_features(
        {
            "heart_rate": 78.0,
            "body_temperature": 36.8,
        }
    )

    assert result == 2