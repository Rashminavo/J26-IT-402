from typing import Any

from physiological_analysis.contract import (
    build_physiological_score,
)


def analyze_physiological(
    measurements: dict[str, Any],
    observation_id: str,
    user_id: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run the current physiological reference pipeline."""
    return build_physiological_score(
        measurements=measurements,
        observation_id=observation_id,
        user_id=user_id,
        session_id=session_id,
    )