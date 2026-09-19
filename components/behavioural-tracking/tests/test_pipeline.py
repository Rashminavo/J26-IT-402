from behavioural_tracking.pipeline import analyze_behaviour


def test_behavioural_pipeline():
    result = analyze_behaviour(
        raw_data={
            "screen_time_minutes": 260,
            "app_usage_minutes": 190,
            "typing_speed": 35,
            "message_length": 38,
        },
        historical_screen_times=[
            200,
            210,
            190,
            205,
        ],
    )

    assert result["stage"] == "behavioural_reference"
    assert "features" in result
    assert "baseline" in result
    assert "deviation" in result
    assert "models" in result

    assert (
        result["models"]["isolation_forest"]
        == "pending_component_2_implementation"
    )