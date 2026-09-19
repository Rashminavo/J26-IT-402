from typing import Any

from .preprocessing import preprocess_text
from .sentiment import analyze_sentiment


def analyze_text(text: str) -> dict[str, Any]:
    """
    Execute the current Component 1 MVP pipeline.

    Current stage:
        Text -> NLTK preprocessing -> VADER
    """
    preprocessing_result = preprocess_text(text)

    sentiment_result = analyze_sentiment(
        preprocessing_result["cleaned_text"]
    )

    return {
        "stage": "vader_mvp",
        "preprocessing": preprocessing_result,
        "sentiment": sentiment_result,
    }