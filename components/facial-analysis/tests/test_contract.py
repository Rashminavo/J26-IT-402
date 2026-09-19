import numpy as np

from facial_analysis.contract import (
    build_facial_score,
)


def test_build_facial_score():
    frames = [
        np.zeros(
            (300, 300, 3),
            dtype=np.uint8,
        )
        for _ in range(16)
    ]

    boxes = [
        (50, 50, 100, 100)
        for _ in range(16)
    ]

    result = build_facial_score(
        frames=frames,
        bounding_boxes=boxes,
        observation_id="OBS-FACIAL-001",
        user_id="USER-001",
        session_id="SESSION-001",
        timestamp="2026-09-19T10:00:00+00:00",
    )

    assert result["schema_version"] == "1.0.0"
    assert result["component"] == "facial_analysis"

    assert (
        result["score"]["unit"]
        == "facial_emotional_evidence_pending"
    )

    assert result["score"]["value"] is None
    assert result["confidence"] is None