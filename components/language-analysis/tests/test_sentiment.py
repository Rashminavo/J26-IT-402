from language_analysis.sentiment import analyze_sentiment


def test_positive_sentiment():
    result = analyze_sentiment(
        "I am very happy and excited about today."
    )

    assert result["positive"] > result["negative"]
    assert result["compound"] > 0


def test_negative_sentiment():
    result = analyze_sentiment(
        "This is terrible and I feel very unhappy."
    )

    assert result["negative"] > result["positive"]
    assert result["compound"] < 0


def test_sentiment_values_are_in_expected_ranges():
    result = analyze_sentiment(
        "This is an ordinary sentence."
    )

    assert 0 <= result["negative"] <= 1
    assert 0 <= result["neutral"] <= 1
    assert 0 <= result["positive"] <= 1
    assert -1 <= result["compound"] <= 1