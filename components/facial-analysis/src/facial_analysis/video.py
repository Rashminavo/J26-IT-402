from pathlib import Path

import cv2
import numpy as np


def calculate_sample_indices(
    total_frames: int,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> list[int]:
    """
    Calculate deterministic frame indices for temporal sampling.
    """
    if total_frames < 0:
        raise ValueError("total_frames cannot be negative")

    if frame_stride <= 0:
        raise ValueError("frame_stride must be greater than zero")

    indices = list(range(0, total_frames, frame_stride))

    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be greater than zero")

        indices = indices[:max_frames]

    return indices


def read_video_frames(
    video_path: str | Path,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """
    Read selected frames from a video file.

    Frames are returned in OpenCV BGR format.
    """
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Video file not found: {path}"
        )

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise ValueError(
            f"Unable to open video: {path}"
        )

    frames: list[np.ndarray] = []

    try:
        frame_index = 0

        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame_index % frame_stride == 0:
                frames.append(frame)

                if (
                    max_frames is not None
                    and len(frames) >= max_frames
                ):
                    break

            frame_index += 1

    finally:
        capture.release()

    return frames