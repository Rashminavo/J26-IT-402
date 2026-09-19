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