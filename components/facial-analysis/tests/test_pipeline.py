import numpy as np
import pytest

from facial_analysis.face import (
    AmbiguousFaceDetectionError,
    FaceNotDetectedError,
)
from facial_analysis.pipeline import (
    analyze_facial_sequence,
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
