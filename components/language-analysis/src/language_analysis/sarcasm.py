import re
from typing import Any


POSITIVE_CUES = {
    "great", "wonderful", "fantastic", "perfect", "brilliant",
    "amazing", "excellent", "lovely", "awesome"
}

NEGATIVE_CUES = {
    "failed", "fail", "terrible", "awful", "worst",
    "disaster", "broken", "ruined", "hate", "useless"
}

SARCASM_PHRASES = [
    "just what i needed",
    "what a wonderful day",
    "oh great",
    "oh, great",
    "oh fantastic",
    "oh, fantastic",
    "just perfect",
    "yeah right",
    "what could possibly go wrong",
    "best day ever"
]


def detect_sarcasm(text: str) -> dict[str, Any]:
    """
    Experimental rule-based sarcasm cue detector.

    This identifies possible sarcasm cues; it does not understand
    all contexts and is not a trained sarcasm classification model.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    cleaned = re.sub(r"\s+", " ", text.lower()).strip()

    positive_matches = [
        word for word in POSITIVE_CUES
        if re.search(r"\b" + re.escape(word) + r"\b", cleaned)
    ]

    negative_matches = [
        word for word in NEGATIVE_CUES
        if re.search(r"\b" + re.escape(word) + r"\b", cleaned)
    ]

    phrase_matches = [
        phrase for phrase in SARCASM_PHRASES
        if phrase in cleaned
    ]

    contradiction = bool(positive_matches and negative_matches)
    cue_detected = bool(phrase_matches or contradiction)

    if phrase_matches:
        reason = "sarcastic_phrase"
    elif contradiction:
        reason = "positive_negative_cue_mismatch"
    else:
        reason = "no_known_sarcasm_cues"

    return {
        "possible_sarcasm": cue_detected,
        "method": "rule_based_experimental",
        "confidence": None,
        "reason": reason,
        "matched_phrases": phrase_matches,
        "positive_cues": positive_matches,
        "negative_cues": negative_matches
    }