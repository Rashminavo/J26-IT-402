from pathlib import Path
from typing import Any

import numpy as np

from .face import (
    AmbiguousFaceDetectionError,
    DEFAULT_MIN_FACE_AREA_RATIO,
    FaceNotDetectedError,
    detect_faces,
    extract_face_roi,
    load_face_detector,
    select_face_box,
)
from .model import MobileNetV3LargeFeatureExtractor
from .image_sequence import read_image_sequence_frames
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

def prepare_face_sequence_from_directory(
    sequence_dir: str | Path,
    *,
    detector: Any | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
    sequence_length: int = 16,
    sequence_stride: int = 8,
    min_area_ratio: float = DEFAULT_MIN_FACE_AREA_RATIO,
) -> list[np.ndarray]:
    """
    Load an image sequence, select an acceptable face in every frame,
    extract RGB face ROIs, and build fixed-length temporal windows.

    Missing or ambiguous detections raise explicit errors instead of
    silently dropping frames and changing the temporal sequence.

    The area-ratio threshold is a provisional engineering heuristic,
    not a research-validated face-selection method.
    """
    frames = read_image_sequence_frames(
        sequence_dir,
        frame_stride=frame_stride,
        max_frames=max_frames,
    )

    if not frames:
        return []

    active_detector = (
        detector if detector is not None else load_face_detector()
    )

    bounding_boxes = []

    for frame_index, frame in enumerate(frames):
        candidates = detect_faces(
            frame,
            detector=active_detector,
        )

        try:
            selected_box = select_face_box(
                candidates,
                min_area_ratio=min_area_ratio,
            )
        except AmbiguousFaceDetectionError as exc:
            raise AmbiguousFaceDetectionError(
                f"Ambiguous face selection at frame index {frame_index} "
                f"in sequence {sequence_dir!s}: {exc}"
            ) from exc

        if selected_box is None:
            raise FaceNotDetectedError(
                f"No face detected at frame index {frame_index} "
                f"in image sequence {sequence_dir!s}"
            )

        bounding_boxes.append(selected_box)

    return prepare_face_sequence(
        frames,
        bounding_boxes,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
    )
