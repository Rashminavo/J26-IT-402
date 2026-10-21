from typing import Any

from .preprocessing import preprocess_text
from .sentiment import analyze_sentiment
from .mentalbert import analyze_with_mentalbert
from .negation import detect_negation
from .sarcasm import detect_sarcasm


def analyze_text(
    text: str,
    include_mentalbert: bool = True,
) -> dict[str, Any]:
    """
    Run preprocessing, VADER sentiment, negation detection,
    and experimental rule-based sarcasm detection.

    Optionally include MentalBERT class probabilities.
    """
    preprocessing_result = preprocess_text(text)
    cleaned_text = preprocessing_result["cleaned_text"]

    sentiment_result = analyze_sentiment(cleaned_text)
    negation_result = detect_negation(text)
    sarcasm_result = detect_sarcasm(text)

    result: dict[str, Any] = {
        "stage": "vader_mvp",
        "preprocessing": preprocessing_result,
        "sentiment": sentiment_result,
        "negation": negation_result,
        "sarcasm": sarcasm_result,
    }

    if include_mentalbert:
        result["mentalbert"] = analyze_with_mentalbert(cleaned_text)

    return result
