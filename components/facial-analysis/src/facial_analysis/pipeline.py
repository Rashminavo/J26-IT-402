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
from .model import (
    MOBILENET_V3_FEATURE_DIM,
    MOBILENET_V3_WEIGHTS,
    FacialFeatureExtractor,
    MobileNetV3LargeFeatureExtractor,
)
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


def extract_facial_sequence_features(
    sequences: list[np.ndarray],
    *,
    extractor: FacialFeatureExtractor | None = None,
) -> list[np.ndarray]:
    """Extract one 960-dimensional visual feature per frame and window.

    Input windows must contain RGB uint8 face ROIs with shape
    (T, 224, 224, 3). Output arrays have shape (T, 960) and dtype float32.

    These are ImageNet-pretrained visual features, not emotion labels,
    micro-expression probabilities, valence/arousal predictions, or
    mental-health risk scores. No temporal aggregation is performed.
    """
    if not sequences:
        return []

    # Validate every input before potentially expensive model initialization.
    for sequence_index, sequence in enumerate(sequences):
        if not isinstance(sequence, np.ndarray):
            raise TypeError(
                f"Sequence {sequence_index} must be a NumPy array"
            )

        if (
            sequence.ndim != 4
            or sequence.shape[0] <= 0
            or sequence.shape[1:] != (224, 224, 3)
            or sequence.dtype != np.uint8
        ):
            raise ValueError(
                f"Sequence {sequence_index} must be non-empty RGB uint8 "
                "with shape (T, 224, 224, 3)"
            )

    active_extractor = (
        extractor
        if extractor is not None
        else MobileNetV3LargeFeatureExtractor()
    )

    feature_sequences = []

    for sequence_index, sequence in enumerate(sequences):
        features = active_extractor.extract(sequence)

        if not isinstance(features, np.ndarray):
            raise TypeError(
                f"Features for sequence {sequence_index} must be a NumPy array"
            )

        expected_shape = (
            sequence.shape[0],
            MOBILENET_V3_FEATURE_DIM,
        )

        if features.shape != expected_shape:
            raise ValueError(
                f"Expected feature shape {expected_shape}; "
                f"received {features.shape}"
            )

        if features.dtype != np.float32:
            raise TypeError("Feature arrays must have dtype float32")

        if not np.isfinite(features).all():
            raise ValueError("Feature arrays must contain only finite values")

        feature_sequences.append(np.ascontiguousarray(features))

    return feature_sequences


def analyze_facial_sequence(
    frames: list[np.ndarray],
    bounding_boxes: list[
        tuple[int, int, int, int]
    ],
) -> dict[str, Any]:
    """
    Prepare face ROIs and temporal windows for the C3 reference pipeline.

    Feature extraction is available separately through
    extract_facial_sequence_features(). This function does not execute the
    pretrained model or generate emotion/risk predictions.
    """
    sequences = prepare_face_sequence(
        frames,
        bounding_boxes,
    )

    return {
        "stage": "facial_reference",
        "sequence_count": len(sequences),
        "sequence_shapes": [
            tuple(sequence.shape)
            for sequence in sequences
        ],
        "model": {
            "name": "MobileNetV3-Large",
            "status": "available_not_executed",
            "weights": MOBILENET_V3_WEIGHTS.name,
            "feature_dim": MOBILENET_V3_FEATURE_DIM,
        },
        "feature_extraction": {
            "status": "not_run",
            "output_shape_per_window": "(T, 960)",
            "output_dtype": "float32",
            "semantics": (
                "ImageNet-pretrained visual features only; "
                "not emotion labels or risk predictions"
            ),
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
