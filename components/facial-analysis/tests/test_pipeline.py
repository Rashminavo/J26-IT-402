import numpy as np

from facial_analysis.pipeline import (
    analyze_facial_sequence,
)


def test_facial_pipeline_reference():
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

    result = analyze_facial_sequence(
        frames,
        boxes,
    )

    assert result["stage"] == "facial_reference"
    assert result["sequence_count"] == 1

    assert (
        result["model"]["name"]
        == "MobileNetV3-Large"
    )

    assert (
        result["micro_expression"]["status"]
        == "pending_model_implementation"
    )