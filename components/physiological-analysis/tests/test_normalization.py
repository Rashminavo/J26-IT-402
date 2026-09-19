from physiological_analysis.normalization import (
    build_standardized_features,
)


def test_normalization_boundary():
    result = build_standardized_features(
        {
            "heart_rate": 78.0,
            "body_temperature": 36.8,
        }
    )

    assert result["status"] == "NORMALIZATION_PENDING"
    assert "features" in result