import pytest

from facial_analysis.video import (
    calculate_sample_indices,
)


def test_calculate_sample_indices():
    result = calculate_sample_indices(
        total_frames=10,
        frame_stride=2,
    )

    assert result == [
        0,
        2,
        4,
        6,
        8,
    ]


def test_max_frames_limits_sampling():
    result = calculate_sample_indices(
        total_frames=10,
        frame_stride=2,
        max_frames=3,
    )

    assert result == [0, 2, 4]


def test_invalid_stride():
    with pytest.raises(ValueError):
        calculate_sample_indices(
            total_frames=10,
            frame_stride=0,
        )