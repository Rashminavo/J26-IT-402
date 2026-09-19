from nltk.sentiment import SentimentIntensityAnalyzer


_analyzer: SentimentIntensityAnalyzer | None = None


def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    """Create the VADER analyzer lazily."""
    global _analyzer

    if _analyzer is None:
        try:
            _analyzer = SentimentIntensityAnalyzer()
        except LookupError as exc:
            raise RuntimeError(
                "VADER lexicon is not installed. "
                "Run: python -m nltk.downloader vader_lexicon"
            ) from exc

    return _analyzer


def analyze_sentiment(text: str) -> dict[str, float]:
    """
    Return VADER sentiment evidence.

    This is a sentiment signal only. It is not a clinical diagnosis
    or the final Mental Health Risk Score.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    scores = get_sentiment_analyzer().polarity_scores(text)

    return {
        "negative": float(scores["neg"]),
        "neutral": float(scores["neu"]),
        "positive": float(scores["pos"]),
        "compound": float(scores["compound"]),
    }