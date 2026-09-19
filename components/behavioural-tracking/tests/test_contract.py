from behavioural_tracking.contract import build_behavioural_score


def test_build_behavioural_score():
    result = build_behavioural_score(
        raw_data={
            "screen_time_minutes": 260,
            "app_usage_minutes": 190,
        },
        historical_screen_times=[
            200,
            210,
            190,
            205,
        ],
        observation_id="OBS-BEH-001",
        user_id="USER-001",
        session_id="SESSION-001",
        timestamp="2026-09-19T10:00:00+00:00",
    )

    assert result["schema_version"] == "1.0.0"
    assert result["component"] == "behavioural_tracking"

    score = result["score"]

    assert score["unit"] == "baseline_deviation_reference"
    assert 0 <= score["value"] <= 1
    assert score["normalized_value"] == score["value"]

    assert result["risk_level"] is None
    assert result["confidence"] is None