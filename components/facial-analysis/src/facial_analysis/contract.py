from datetime import datetime, timezone
from typing import Any

from .pipeline import analyze_facial_sequence


def build_facial_score(
    frames: list,
    bounding_boxes: list[
        tuple[int, int, int, int]
    ],
    observation_id: str,
    user_id: str,
    session_id: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """
    Convert current C3 processing evidence into the shared score contract.

    The actual micro-expression risk/evidence score remains pending until
    MobileNetV3-Large and temporal classification are implemented.
    """
    analysis = analyze_facial_sequence(
        frames,
        bounding_boxes,
    )

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "schema_version": "1.0.0",
        "observation_id": observation_id,
        "user_id": user_id,
        "session_id": session_id,
        "component": "facial_analysis",
        "timestamp": timestamp,
        "data_status": (
            "AVAILABLE"
            if analysis["sequence_count"] > 0
            else "PARTIAL"
        ),
        "score": {
            "value": None,
            "min": 0,
            "max": 1,
            "normalized_value": None,
            "unit": "facial_emotional_evidence_pending",
        },
        "confidence": None,
        "risk_level": None,
        "risk_level_scheme": None,
        "data_quality": {
            "sequence_count": analysis[
                "sequence_count"
            ],
        },
        "model": {
            "name": "MobileNetV3-Large",
            "version": None,
        },
        "metadata": {
            "stage": "facial_reference",
            "sequence_shapes": analysis[
                "sequence_shapes"
            ],
            "micro_expression_status": analysis[
                "micro_expression"
            ]["status"],
        },
    }