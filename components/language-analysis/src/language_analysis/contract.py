from datetime import datetime, timezone
from typing import Any

from .pipeline import analyze_text


def build_language_score(
    text: str,
    observation_id: str,
    user_id: str,
    session_id: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """
    Convert current VADER evidence into the shared component score contract.

    VADER sentiment is not the final Mental Health Risk Score.
    """

    analysis = analyze_text(text)

    compound = float(analysis["sentiment"]["compound"])

    # Map VADER's [-1, 1] range to the common [0, 1] integration range.
    normalized_value = (compound + 1.0) / 2.0

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "schema_version": "1.0.0",
        "observation_id": observation_id,
        "user_id": user_id,
        "session_id": session_id,
        "component": "language_analysis",
        "timestamp": timestamp,
        "data_status": "AVAILABLE",
        "score": {
            "value": compound,
            "min": -1,
            "max": 1,
            "normalized_value": normalized_value,
            "unit": "vader_compound"
        },
        "confidence": None,
        "risk_level": None,
        "risk_level_scheme": None,
        "data_quality": {
            "completeness": 1.0
        },
        "model": {
            "name": "VADER",
            "version": "3.10"
        },
        "metadata": {
            "stage": "vader_mvp",
            "sentiment": analysis["sentiment"],
            "token_count": analysis["preprocessing"]["token_count"]
        }
    }