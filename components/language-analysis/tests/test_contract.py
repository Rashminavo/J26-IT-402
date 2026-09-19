from language_analysis.contract import build_language_score


def test_build_language_score():
    result = build_language_score(
        text="I feel happy and hopeful today.",
        observation_id="OBS-LANG-001",
        user_id="USER-001",
        session_id="SESSION-001",
        timestamp="2026-09-19T10:00:00+00:00",
    )

    assert result["schema_version"] == "1.0.0"
    assert result["component"] == "language_analysis"
    assert result["observation_id"] == "OBS-LANG-001"

    score = result["score"]

    assert score["unit"] == "vader_compound"
    assert -1 <= score["value"] <= 1
    assert 0 <= score["normalized_value"] <= 1

    assert result["risk_level"] is None
    assert result["confidence"] is None