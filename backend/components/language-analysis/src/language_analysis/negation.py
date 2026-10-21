import re
from typing import Any

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "cannot",
    "can't", "couldn't", "didn't", "doesn't", "don't",
    "hadn't", "hasn't", "haven't", "isn't", "mustn't",
    "shouldn't", "wasn't", "weren't", "won't", "wouldn't",
    "hardly", "barely", "scarcely", "without"
}


def detect_negation(text: str) -> dict[str, Any]:
    """Detect common English negation words."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    matches = []

    for match in re.finditer(r"\b[\w]+(?:['’][\w]+)*\b", text.lower()):
        word = match.group(0).replace("’", "'")

        if word in NEGATION_WORDS:
            matches.append({
                "word": word,
                "start": match.start(),
                "end": match.end()
            })

    return {
        "detected": bool(matches),
        "count": len(matches),
        "words": [item["word"] for item in matches],
        "matches": matches,
        "method": "rule_based"
    }