
from typing import Any

from .preprocessing import preprocess_text
from .sentiment import analyze_sentiment
from .mentalbert import analyze_with_mentalbert


def analyze_text(
    text: str,
    include_mentalbert: bool = True,
) -> dict[str, Any]:
    """
    Run preprocessing and VADER sentiment analysis.
    Optionally include MentalBERT class probabilities.
    """
    preprocessing_result = preprocess_text(text)

    sentiment_result = analyze_sentiment(
        preprocessing_result["cleaned_text"]
    )

    result: dict[str, Any] = {
        "stage": "vader_mvp",
        "preprocessing": preprocessing_result,
        "sentiment": sentiment_result,
    }

    if include_mentalbert:
        result["mentalbert"] = analyze_with_mentalbert(
            preprocessing_result["cleaned_text"]
        )
        
    return result
