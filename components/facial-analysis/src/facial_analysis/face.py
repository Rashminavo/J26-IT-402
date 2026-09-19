from pathlib import Path

import cv2
import numpy as np


DEFAULT_CASCADE = (
    Path(cv2.data.haarcascades)
    / "haarcascade_frontalface_default.xml"
)


def load_face_detector() -> cv2.CascadeClassifier:
    """
    Load the OpenCV Haar Cascade face detector.

    Engineering Recommendation:
    This detector is a replaceable reference layer because the research
    proposal does not prescribe a specific face detector.
    """
    detector = cv2.CascadeClassifier(
        str(DEFAULT_CASCADE)
    )

    if detector.empty():
        raise RuntimeError(
            f"Unable to load face detector: {DEFAULT_CASCADE}"
        )

    return detector


def detect_faces(
    frame: np.ndarray,
    detector: cv2.CascadeClassifier | None = None,
) -> list[tuple[int, int, int, int]]:
    """
    Detect frontal faces in an OpenCV BGR frame.
    """
    if frame is None or frame.size == 0:
        raise ValueError("frame cannot be empty")

    active_detector = detector or load_face_detector()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    faces = active_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    return [
        (
            int(x),
            int(y),
            int(width),
            int(height),
        )
        for x, y, width, height in faces
    ]


def extract_face_roi(
    frame: np.ndarray,
    bounding_box: tuple[int, int, int, int],
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Crop, convert BGR -> RGB, and resize one face region.
    """
    if frame is None or frame.size == 0:
        raise ValueError("frame cannot be empty")

    x, y, width, height = bounding_box

    if width <= 0 or height <= 0:
        raise ValueError(
            "bounding box dimensions must be positive"
        )

    frame_height, frame_width = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_width, x + width)
    y2 = min(frame_height, y + height)

    if x1 >= x2 or y1 >= y2:
        raise ValueError(
            "bounding box is outside the frame"
        )

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        raise ValueError("face ROI is empty")

    roi_rgb = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB,
    )

    return cv2.resize(
        roi_rgb,
        target_size,
        interpolation=cv2.INTER_AREA,
    )