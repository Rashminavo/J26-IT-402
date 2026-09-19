from physiological_analysis.contract import (
    build_physiological_score,
)


def test_build_physiological_score():
    result = build_physiological_score(
        measurements={
            "heart_rate": 78,
            "heart_rate_variability": 42,
        },
        observation_id="OBS-PHYS-001",
        user_id="USER-001",
        session_id="SESSION-001",
        timestamp="2026-09-19T10:00:00+00:00",
    )

    assert result["schema_version"] == "1.0.0"
    assert result["component"] == "physiological_analysis"

    score = result["score"]

    assert score["value"] is None
    assert score["normalized_value"] is None
    assert (
        score["unit"]
        == "physiological_stress_score_pending"
    )

    assert result["risk_level"] is None
    assert result["confidence"] is None