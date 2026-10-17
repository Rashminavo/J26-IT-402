import numpy as np
import pytest

from facial_analysis.sequence import (
    build_temporal_sequences,
    sequence_shape,
)


def test_build_temporal_sequences():
    frames = [
        np.zeros(
            (224, 224, 3),
            dtype=np.uint8,
        )
        for _ in range(20)
    ]

    result = build_temporal_sequences(
        frames,
        sequence_length=8,
        stride=4,
    )

    assert len(result) == 4
    assert result[0].shape == (
        8,
        224,
        224,
        3,
    )


def test_empty_sequence():
    result = build_temporal_sequences(
        [],
        sequence_length=8,
        stride=4,
    )

    assert result == []


def test_invalid_sequence_length():
    with pytest.raises(ValueError):
        build_temporal_sequences(
            [],
            sequence_length=0,
        )


def test_sequence_shape():
    sequence = np.zeros(
        (8, 224, 224, 3),
        dtype=np.uint8,
    )

    assert sequence_shape(sequence) == (
        8,
        224,
        224,
        3,
    )


def _indexed_frames(count: int) -> list[np.ndarray]:
    """Create tiny frames whose pixel value records the original frame index."""
    return [
        np.full((1, 1, 1), index, dtype=np.uint8)
        for index in range(count)
    ]


def _assert_windows_match_starts(
    windows: list[np.ndarray],
    expected_starts: list[int],
    sequence_length: int,
) -> None:
    assert len(windows) == len(expected_starts)
    assert len(expected_starts) == len(set(expected_starts))

    for window, start in zip(windows, expected_starts):
        assert window.shape == (sequence_length, 1, 1, 1)
        expected_indices = np.arange(
            start,
            start + sequence_length,
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(
            window[:, 0, 0, 0],
            expected_indices,
        )


def _covered_frame_indices(windows: list[np.ndarray]) -> set[int]:
    frame_values = np.concatenate(
        [window[:, 0, 0, 0] for window in windows]
    )
    return set(frame_values.tolist())


def test_tail_alignment_not_needed_when_regular_window_covers_final_frame():
    result = build_temporal_sequences(
        _indexed_frames(24),
        sequence_length=16,
        stride=8,
    )

    _assert_windows_match_starts(result, [0, 8], 16)
    assert _covered_frame_indices(result) == set(range(24))


def test_tail_aligned_window_covers_length_25_sequence():
    result = build_temporal_sequences(
        _indexed_frames(25),
        sequence_length=16,
        stride=8,
    )

    _assert_windows_match_starts(result, [0, 8, 9], 16)
    assert _covered_frame_indices(result) == set(range(25))


def test_tail_aligned_window_covers_length_41_sequence():
    result = build_temporal_sequences(
        _indexed_frames(41),
        sequence_length=16,
        stride=8,
    )

    _assert_windows_match_starts(result, [0, 8, 16, 24, 25], 16)
    assert _covered_frame_indices(result) == set(range(41))


def test_exact_length_sequence_produces_one_window():
    result = build_temporal_sequences(
        _indexed_frames(16),
        sequence_length=16,
        stride=8,
    )

    _assert_windows_match_starts(result, [0], 16)


def test_shorter_than_window_length_produces_no_windows():
    result = build_temporal_sequences(
        _indexed_frames(15),
        sequence_length=16,
        stride=8,
    )

    assert result == []


@pytest.mark.parametrize("sequence_length", [0, -1])
def test_nonpositive_sequence_length_is_rejected(sequence_length: int):
    with pytest.raises(
        ValueError,
        match="sequence_length must be greater than zero",
    ):
        build_temporal_sequences([], sequence_length=sequence_length)


@pytest.mark.parametrize("stride", [0, -1])
def test_nonpositive_stride_is_rejected(stride: int):
    with pytest.raises(
        ValueError,
        match="stride must be greater than zero",
    ):
        build_temporal_sequences([], sequence_length=16, stride=stride)
