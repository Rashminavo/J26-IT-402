import pytest

from behavioural_tracking.features import extract_features


def test_extract_behavioural_features():
    result = extract_features(
        {
            "screen_time_minutes": 240,
            "app_usage_minutes": 180,
            "typing_speed": 38,
            "message_length": 42,
            "mobility_events": 6,
            "social_interaction_events": 12,
        }
    )

    assert result["screen_time_minutes"] == 240
    assert result["typing_speed"] == 38
    assert len(result) == 6


def test_unknown_features_are_ignored():
    result = extract_features(
        {
            "screen_time_minutes": 200,
            "unknown_signal": 999,
        }
    )

    assert "screen_time_minutes" in result
    assert "unknown_signal" not in result


def test_negative_feature_is_rejected():
    with pytest.raises(ValueError):
        extract_features(
            {
                "screen_time_minutes": -10,
            }
        )