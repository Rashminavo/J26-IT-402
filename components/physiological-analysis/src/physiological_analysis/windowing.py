from __future__ import annotations

import math
from typing import TypeVar


T = TypeVar("T")


def segment_signal_into_windows(
    readings: list[T],
    sampling_rate: float,
    signal_start_offset_seconds: float = 0.0,
    window_size_seconds: float = 60.0,
    stride_seconds: float = 30.0,
    timeline_duration_seconds: float | None = None,
) -> list[dict[str, object]]:
    """Split regularly sampled readings into windows on a shared timeline.

    Window boundaries follow [start, end): start is included, end excluded.
    Windows are emitted only when the full window fits within the timeline
    and the signal contains at least one sample in that window.

    Sample timestamps are derived from the sampling rate and the signal's
    start offset. This function does not resample or interpolate data.
    """
    if not isinstance(readings, list):
        raise TypeError("readings must be a list")

    if not readings:
        raise ValueError("readings must not be empty")

    numeric_parameters = {
        "sampling_rate": sampling_rate,
        "signal_start_offset_seconds": signal_start_offset_seconds,
        "window_size_seconds": window_size_seconds,
        "stride_seconds": stride_seconds,
    }

    if timeline_duration_seconds is not None:
        numeric_parameters["timeline_duration_seconds"] = (
            timeline_duration_seconds
        )

    for name, value in numeric_parameters.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")

    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")

    if signal_start_offset_seconds < 0:
        raise ValueError("signal_start_offset_seconds must not be negative")

    if window_size_seconds <= 0:
        raise ValueError("window_size_seconds must be positive")

    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")

    signal_duration = len(readings) / float(sampling_rate)
    signal_end = signal_start_offset_seconds + signal_duration

    if timeline_duration_seconds is None:
        timeline_duration_seconds = signal_end
    elif timeline_duration_seconds <= 0:
        raise ValueError("timeline_duration_seconds must be positive")

    timeline_end = float(timeline_duration_seconds)

    if signal_start_offset_seconds >= timeline_end:
        return []

    windows: list[dict[str, object]] = []
    start = 0.0

    while start + window_size_seconds <= timeline_end:
        end = start + window_size_seconds

        first_index = max(
            0,
            math.ceil(
                (start - signal_start_offset_seconds) * sampling_rate
                - 1e-9
            ),
        )
        last_index = min(
            len(readings),
            math.ceil(
                (end - signal_start_offset_seconds) * sampling_rate
                - 1e-9
            ),
        )

        if first_index < last_index:
            windows.append(
                {
                    "start_seconds": start,
                    "end_seconds": end,
                    "readings": readings[first_index:last_index],
                    "sample_count": last_index - first_index,
                }
            )

        start += stride_seconds

    return windows

def validate_window_sample_count(
    window: dict[str, object],
    sampling_rate: float,
    window_size_seconds: float = 60.0,
) -> bool:
    """Return whether a window contains the expected number of samples."""
    if (
        isinstance(sampling_rate, bool)
        or not isinstance(sampling_rate, (int, float))
        or not math.isfinite(sampling_rate)
        or sampling_rate <= 0
    ):
        raise ValueError("sampling_rate must be a positive finite number")

    if (
        isinstance(window_size_seconds, bool)
        or not isinstance(window_size_seconds, (int, float))
        or not math.isfinite(window_size_seconds)
        or window_size_seconds <= 0
    ):
        raise ValueError(
            "window_size_seconds must be a positive finite number"
        )

    sample_count = window.get("sample_count")

    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 0
    ):
        raise ValueError("sample_count must be a non-negative integer")

    expected_sample_count = window_size_seconds * sampling_rate

    if not expected_sample_count.is_integer():
        raise ValueError(
            "window_size_seconds * sampling_rate must be a whole number"
        )

    return sample_count == int(expected_sample_count)
