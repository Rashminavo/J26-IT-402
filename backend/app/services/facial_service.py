from typing import Any

from facial_analysis.contract import build_facial_score


def analyze_facial(
    frames: list[Any],
    bounding_boxes: list[
        tuple[int, int, int, int]
    ],
    observation_id: str,
    user_id: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run the current Facial component reference pipeline."""
    return build_facial_score(
        frames=frames,
        bounding_boxes=bounding_boxes,
        observation_id=observation_id,
        user_id=user_id,
        session_id=session_id,
    )