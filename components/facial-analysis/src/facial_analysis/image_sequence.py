from pathlib import Path
import re

import cv2
import numpy as np

from .video import calculate_sample_indices


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


def natural_sort_key(path: str | Path) -> tuple[tuple[int, int | str], ...]:
    """Return a key that sorts numbered filenames in natural order."""
    parts = re.split(r"(\d+)", Path(path).name)

    return tuple(
        (0, int(part)) if part.isdigit()
        else (1, part.casefold())
        for part in parts
    )


def list_sequence_images(
    sequence_dir: str | Path,
) -> list[Path]:
    """
    List supported images directly inside one sample directory.

    Images are returned in natural filename order. Subdirectories are
    not searched recursively because each CASME II sample has its own
    sequence directory.
    """
    directory = Path(sequence_dir)

    if not directory.exists():
        raise FileNotFoundError(
            f"Image-sequence directory not found: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Expected an image-sequence directory: {directory}"
        )

    images = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    ]

    images.sort(key=natural_sort_key)

    if not images:
        raise ValueError(
            f"No supported images found in sequence directory: {directory}"
        )

    return images


def read_image_sequence_frames(
    sequence_dir: str | Path,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """
    Read a sample's image sequence as OpenCV BGR frames.

    Sampling is deterministic and follows natural filename order.
    Frame selection uses the same index-calculation helper as video.py.
    """
    images = list_sequence_images(sequence_dir)

    selected_indices = calculate_sample_indices(
        total_frames=len(images),
        frame_stride=frame_stride,
        max_frames=max_frames,
    )

    frames: list[np.ndarray] = []

    for index in selected_indices:
        image_path = images[index]
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError(
                f"Unable to decode image-sequence frame: {image_path}"
            )

        frames.append(frame)

    return frames
