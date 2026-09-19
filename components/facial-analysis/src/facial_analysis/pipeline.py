from typing import Any

import numpy as np

from .face import extract_face_roi
from .model import MobileNetV3LargeFeatureExtractor
from .sequence import build_temporal_sequences


def prepare_face_sequence(
    frames: list[np.ndarray],
    bounding_boxes: list[
        tuple[int, int, int, int]
    ],
    sequence_length: int = 16,
    sequence_stride: int = 8,
) -> list[np.ndarray]:
    """
    Convert frame-level face bounding boxes into temporal face sequences.

    One bounding box is required for each input frame.
    """
    if len(frames) != len(bounding_boxes):
        raise ValueError(
            "frames and bounding_boxes must have equal length"
        )

    if not frames:
        return []

    rois = [
        extract_face_roi(
            frame,
            bounding_box,
        )
        for frame, bounding_box in zip(
            frames,
            bounding_boxes,
        )
    ]

    return build_temporal_sequences(
        rois,
        sequence_length=sequence_length,
        stride=sequence_stride,
    )


def analyze_facial_sequence(
    frames: list[np.ndarray],
    bounding_boxes: list[
        tuple[int, int, int, int]
    ],
) -> dict[str, Any]:
    """
    Current C3 reference pipeline.

    Current stage:
        frames
        -> face ROI
        -> temporal sequence
        -> MobileNetV3-Large interface
    """
    sequences = prepare_face_sequence(
        frames,
        bounding_boxes,
    )

    model = MobileNetV3LargeFeatureExtractor()

    return {
        "stage": "facial_reference",
        "sequence_count": len(sequences),
        "sequence_shapes": [
            tuple(sequence.shape)
            for sequence in sequences
        ],
        "model": {
            "name": model.model_name,
            "status": "interface_only",
        },
        "micro_expression": {
            "status": "pending_model_implementation",
        },
        "sequences": sequences,
    }