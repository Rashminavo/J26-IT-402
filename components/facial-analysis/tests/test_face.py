import cv2
import numpy as np
import pytest

from facial_analysis.face import (
    AmbiguousFaceDetectionError,
    extract_face_roi,
    load_face_detector,
    select_face_box,
)


def test_load_face_detector():
    detector = load_face_detector()

    assert not detector.empty()


def test_extract_face_roi():
    frame = np.zeros(
        (300, 300, 3),
        dtype=np.uint8,
    )

    roi = extract_face_roi(
        frame,
        (50, 50, 100, 100),
    )

    assert roi.shape == (
        224,
        224,
        3,
    )


def test_invalid_face_box():
    frame = np.zeros(
        (300, 300, 3),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        extract_face_roi(
            frame,
            (50, 50, 0, 100),
        )

def test_select_face_box_uses_largest_clear_detection():
    boxes = [
        (50, 50, 100, 100),
        (250, 250, 20, 20),
    ]

    assert select_face_box(boxes) == (50, 50, 100, 100)


def test_select_face_box_accepts_a_single_detection():
    box = (40, 30, 90, 90)

    assert select_face_box([box]) == box


def test_select_face_box_returns_none_without_detections():
    assert select_face_box([]) is None


def test_select_face_box_rejects_ambiguous_area_ratio():
    boxes = [
        (20, 20, 100, 100),
        (160, 160, 80, 80),
    ]

    with pytest.raises(AmbiguousFaceDetectionError):
        select_face_box(boxes)


def test_select_face_box_validates_area_ratio_threshold():
    with pytest.raises(ValueError, match="greater than 1"):
        select_face_box(
            [(10, 10, 100, 100)],
            min_area_ratio=1.0,
        )
