import cv2
import numpy as np
import pytest

from facial_analysis.face import (
    extract_face_roi,
    load_face_detector,
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