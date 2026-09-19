from language_analysis.pipeline import analyze_text


def test_language_pipeline():
    result = analyze_text(
        "I am feeling happy and hopeful today."
    )

    assert result["stage"] == "vader_mvp"
    assert "preprocessing" in result
    assert "sentiment" in result

    assert result["preprocessing"]["token_count"] > 0

    sentiment = result["sentiment"]

    assert -1 <= sentiment["compound"] <= 1