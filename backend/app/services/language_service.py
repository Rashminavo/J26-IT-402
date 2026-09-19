from typing import Any

from language_analysis.contract import build_language_score


def analyze_language(
    text: str,
    observation_id: str,
    user_id: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    return build_language_score(
        text=text,
        observation_id=observation_id,
        user_id=user_id,
        session_id=session_id,
    )