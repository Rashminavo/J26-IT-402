import pytest

from physiological_analysis.windowing import (
    segment_signal_into_windows,
)


def test_creates_overlapping_windows():
    readings = list(range(120))

    windows = segment_signal_into_windows(
        readings,
        sampling_rate=1,
        window_size_seconds=60,
        stride_seconds=30,
    )

    assert len(windows) == 3
    assert windows[0]["readings"] == list(range(60))
    assert windows[1]["readings"] == list(range(30, 90))
    assert windows[2]["readings"] == list(range(60, 120))


def test_excludes_incomplete_final_window():
    readings = list(range(100))

    windows = segment_signal_into_windows(
        readings,
        sampling_rate=1,
        window_size_seconds=60,
        stride_seconds=30,
    )

    assert len(windows) == 2
    assert windows[-1]["end_seconds"] == 90.0


def test_accounts_for_signal_start_offset():
    readings = list(range(100))

    windows = segment_signal_into_windows(
        readings,
        sampling_rate=1,
        signal_start_offset_seconds=10,
        window_size_seconds=60,
        stride_seconds=30,
        timeline_duration_seconds=100,
    )

    assert windows[0]["readings"] == list(range(0, 50))
    assert windows[1]["readings"] == list(range(20, 80))
    assert len(windows) == 2
    assert windows[1]["start_seconds"] == 30.0
    assert windows[1]["end_seconds"] == 90.0


def test_rejects_empty_readings():
    with pytest.raises(ValueError, match="must not be empty"):
        segment_signal_into_windows(
            [],
            sampling_rate=1,
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("sampling_rate", 0),
        ("sampling_rate", -1),
        ("window_size_seconds", 0),
        ("stride_seconds", -1),
        ("signal_start_offset_seconds", -1),
        ("timeline_duration_seconds", 0),
    ],
)
def test_rejects_invalid_parameters(parameter, value):
    arguments = {
        "readings": list(range(120)),
        "sampling_rate": 1,
    }
    arguments[parameter] = value

    with pytest.raises(ValueError):
        segment_signal_into_windows(**arguments)

def test_complete_window_has_expected_sample_count():
    windows = segment_signal_into_windows(
        list(range(60)),
        sampling_rate=1,
        window_size_seconds=60,
        stride_seconds=30,
    )

    assert len(windows) == 1
    assert windows[0]["sample_count"] == 60
    assert len(windows[0]["readings"]) == 60


def test_offset_window_is_identified_as_incomplete():
    windows = segment_signal_into_windows(
        list(range(100)),
        sampling_rate=1,
        signal_start_offset_seconds=10,
        window_size_seconds=60,
        stride_seconds=30,
        timeline_duration_seconds=100,
    )

    assert windows[0]["sample_count"] == 50
    assert windows[0]["sample_count"] < 60
    assert windows[1]["sample_count"] == 60

from physiological_analysis.windowing import validate_window_sample_count


@pytest.mark.parametrize(
    ("sample_count", "sampling_rate", "expected"),
    [
        (60, 1, True),
        (50, 1, False),
        (240, 4, True),
        (3840, 64, True),
        (1920, 32, True),
    ],
)
def test_validate_window_sample_count(
    sample_count, sampling_rate, expected
):
    assert validate_window_sample_count(
        {"sample_count": sample_count},
        sampling_rate=sampling_rate,
        window_size_seconds=60,
    ) is expected
