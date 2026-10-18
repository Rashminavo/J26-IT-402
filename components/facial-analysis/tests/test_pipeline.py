import numpy as np
import pytest

from facial_analysis.model import (
    FacialFeatureExtractor,
    MOBILENET_V3_FEATURE_DIM,
)
from facial_analysis.face import (
    AmbiguousFaceDetectionError,
    FaceNotDetectedError,
)
from facial_analysis.pipeline import (
    analyze_facial_sequence,
    extract_facial_sequence_features,
    prepare_face_sequence_from_directory,
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


class _FakeFeatureExtractor(FacialFeatureExtractor):
    def __init__(self, output_dim=MOBILENET_V3_FEATURE_DIM):
        self.output_dim = output_dim

    def extract(self, sequence):
        return np.zeros(
            (sequence.shape[0], self.output_dim),
            dtype=np.float32,
        )


def test_feature_helper_returns_one_vector_per_frame():
    sequences = [
        np.zeros((16, 224, 224, 3), dtype=np.uint8),
        np.zeros((8, 224, 224, 3), dtype=np.uint8),
    ]

    results = extract_facial_sequence_features(
        sequences,
        extractor=_FakeFeatureExtractor(),
    )

    assert [result.shape for result in results] == [
        (16, MOBILENET_V3_FEATURE_DIM),
        (8, MOBILENET_V3_FEATURE_DIM),
    ]
    assert all(result.dtype == np.float32 for result in results)
    assert all(result.flags.c_contiguous for result in results)
    assert all(np.isfinite(result).all() for result in results)


def test_feature_helper_rejects_wrong_feature_dimension():
    sequence = np.zeros((2, 224, 224, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Expected feature shape"):
        extract_facial_sequence_features(
            [sequence],
            extractor=_FakeFeatureExtractor(output_dim=1000),
        )


def test_feature_helper_rejects_invalid_input_before_model_initialization(
    monkeypatch,
):
    def unexpected_model_initialization():
        raise AssertionError(
            "Invalid input must be rejected before model initialization"
        )

    monkeypatch.setattr(
        "facial_analysis.pipeline.MobileNetV3LargeFeatureExtractor",
        unexpected_model_initialization,
    )

    invalid_sequence = np.zeros(
        (1, 224, 224, 3),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="non-empty RGB uint8"):
        extract_facial_sequence_features([invalid_sequence])


def test_feature_helper_empty_input_does_not_initialize_model(monkeypatch):
    def unexpected_model_initialization():
        raise AssertionError("Model must not initialize for empty input")

    monkeypatch.setattr(
        "facial_analysis.pipeline.MobileNetV3LargeFeatureExtractor",
        unexpected_model_initialization,
    )

    assert extract_facial_sequence_features([]) == []


def _mock_directory_preprocessor(monkeypatch, frames, detections):
    monkeypatch.setattr(
        "facial_analysis.pipeline.read_image_sequence_frames",
        lambda sequence_dir, frame_stride=1, max_frames=None: frames,
    )
    monkeypatch.setattr(
        "facial_analysis.pipeline.load_face_detector",
        lambda: object(),
    )
    monkeypatch.setattr(
        "facial_analysis.pipeline.detect_faces",
        lambda frame, detector=None: detections,
    )


def test_directory_preprocessing_selects_largest_detection(monkeypatch, tmp_path):
    frames = []

    for _ in range(16):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        frame[50:150, 50:150] = 30
        frame[250:270, 250:270] = 200
        frames.append(frame)

    detections = [
        (50, 50, 100, 100),
        (250, 250, 20, 20),
    ]
    _mock_directory_preprocessor(monkeypatch, frames, detections)

    sequences = prepare_face_sequence_from_directory(tmp_path)

    assert len(sequences) == 1
    assert sequences[0].shape == (16, 224, 224, 3)
    assert np.all(sequences[0] == 30)


def test_directory_preprocessing_rejects_missing_face(monkeypatch, tmp_path):
    frames = [np.zeros((300, 300, 3), dtype=np.uint8)]
    _mock_directory_preprocessor(monkeypatch, frames, [])

    with pytest.raises(FaceNotDetectedError, match="frame index 0"):
        prepare_face_sequence_from_directory(tmp_path)


def test_directory_preprocessing_rejects_ambiguous_face(monkeypatch, tmp_path):
    frames = [np.zeros((300, 300, 3), dtype=np.uint8)]
    detections = [
        (20, 20, 100, 100),
        (160, 160, 80, 80),
    ]
    _mock_directory_preprocessor(monkeypatch, frames, detections)

    with pytest.raises(
        AmbiguousFaceDetectionError,
        match="frame index 0",
    ):
        prepare_face_sequence_from_directory(tmp_path)
