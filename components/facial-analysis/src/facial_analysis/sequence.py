import numpy as np


def build_temporal_sequences(
    frames: list[np.ndarray],
    sequence_length: int = 16,
    stride: int = 8,
) -> list[np.ndarray]:
    """
    Build fixed-length temporal face sequences.

    Each returned sequence has shape:
        (sequence_length, height, width, channels)
    """
    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be greater than zero"
        )

    if stride <= 0:
        raise ValueError(
            "stride must be greater than zero"
        )

    if not frames:
        return []

    sequences: list[np.ndarray] = []

    start = 0

    while start + sequence_length <= len(frames):
        window = frames[
            start:start + sequence_length
        ]

        sequences.append(
            np.stack(window, axis=0)
        )

        start += stride

    return sequences


def sequence_shape(
    sequence: np.ndarray,
) -> tuple[int, ...]:
    if sequence.ndim != 4:
        raise ValueError(
            "sequence must have 4 dimensions: "
            "(time, height, width, channels)"
        )

    return tuple(sequence.shape)