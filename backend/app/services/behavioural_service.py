from typing import Any

from behavioural_tracking.contract import build_behavioural_score


def analyze_behavioural(
    raw_data: dict[str, Any],
    historical_screen_times: list[float],
    observation_id: str,
    user_id: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run the current Behavioural component reference pipeline."""
    return build_behavioural_score(
        raw_data=raw_data,
        historical_screen_times=historical_screen_times,
        observation_id=observation_id,
        user_id=user_id,
        session_id=session_id,
    )