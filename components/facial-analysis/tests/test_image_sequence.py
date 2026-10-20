import cv2
import numpy as np
import pytest

from facial_analysis.image_sequence import (
    list_sequence_images,
    natural_sort_key,
    read_image_sequence_frames,
)


def write_image(path, value):
    image = np.full((8, 10, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_natural_numeric_sorting():
    names = ["img10.jpg", "img2.jpg", "img1.jpg"]

    result = sorted(names, key=natural_sort_key)

    assert result == ["img1.jpg", "img2.jpg", "img10.jpg"]


def test_lists_supported_images_and_ignores_other_files(tmp_path):
    (tmp_path / "img10.jpg").touch()
    (tmp_path / "img2.JPG").touch()
    (tmp_path / "img1.png").touch()
    (tmp_path / "notes.txt").touch()

    result = list_sequence_images(tmp_path)

    assert [path.name for path in result] == [
        "img1.png",
        "img2.JPG",
        "img10.jpg",
    ]


def test_reads_images_in_natural_order(tmp_path):
    write_image(tmp_path / "img10.jpg", 100)
    write_image(tmp_path / "img2.jpg", 20)
    write_image(tmp_path / "img1.jpg", 10)

    frames = read_image_sequence_frames(tmp_path)

    assert len(frames) == 3
    assert [int(frame[0, 0, 0]) for frame in frames] == [10, 20, 100]
    assert all(frame.shape == (8, 10, 3) for frame in frames)


def test_applies_stride_and_max_frames(tmp_path):
    for number in range(1, 6):
        write_image(tmp_path / f"img{number}.jpg", number * 20)

    frames = read_image_sequence_frames(
        tmp_path,
        frame_stride=2,
        max_frames=2,
    )

    assert len(frames) == 2
    assert [int(frame[0, 0, 0]) for frame in frames] == [20, 60]


def test_missing_directory_raises(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        list_sequence_images(missing)


def test_file_instead_of_directory_raises(tmp_path):
    image_path = tmp_path / "file.jpg"
    image_path.touch()

    with pytest.raises(NotADirectoryError):
        list_sequence_images(image_path)


def test_empty_sequence_raises(tmp_path):
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="No supported images"):
        list_sequence_images(tmp_path)


def test_invalid_stride_raises(tmp_path):
    write_image(tmp_path / "img1.jpg", 10)

    with pytest.raises(ValueError, match="frame_stride"):
        read_image_sequence_frames(tmp_path, frame_stride=0)


def test_invalid_max_frames_raises(tmp_path):
    write_image(tmp_path / "img1.jpg", 10)

    with pytest.raises(ValueError, match="max_frames"):
        read_image_sequence_frames(tmp_path, max_frames=0)


def test_corrupt_image_raises(tmp_path):
    (tmp_path / "img1.jpg").write_text(
        "not a JPEG image",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unable to decode"):
        read_image_sequence_frames(tmp_path)
