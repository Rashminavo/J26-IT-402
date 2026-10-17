import csv
import math
from pathlib import Path


def load_heart_rate_csv(
    file_path: str | Path,
) -> list[float]:
    """
    Load HR readings from a wearable-dataset CSV.

    Expected format:
    - Row 1: initial timestamp
    - Row 2: sampling frequency
    - Remaining rows: heart-rate readings
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Heart-rate CSV not found: {path}"
        )

    readings: list[float] = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = csv.reader(file)

        # Skip timestamp and sampling-frequency rows.
        next(rows, None)
        next(rows, None)

        for row_number, row in enumerate(rows, start=3):
            if not row or not row[0].strip():
                continue

            try:
                value = float(row[0])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid HR value at row {row_number}"
                ) from exc

            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Invalid HR value at row {row_number}: {value}"
                )

            readings.append(value)

    if not readings:
        raise ValueError(
            "The HR CSV contains no valid readings"
        )

    return readings


def load_signal_csv(
    file_path: str | Path,
) -> dict[str, object]:
    """
    Load a wearable sensor CSV containing a timestamp,
    a sampling rate, and numeric signal readings.
    Supports single-channel and three-axis signals.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Signal CSV not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.reader(file))

    if len(rows) < 3:
        raise ValueError(
            "Signal CSV must contain a timestamp, "
            "sampling rate, and at least one reading"
        )

    timestamp = rows[0][0].strip()

    try:
        sampling_rates = [
            float(value) for value in rows[1]
        ]
    except ValueError as exc:
        raise ValueError(
            "Invalid sampling rate in signal CSV"
        ) from exc

    if (
        not timestamp
        or not sampling_rates
        or any(
            not math.isfinite(rate) or rate <= 0
            for rate in sampling_rates
        )
    ):
        raise ValueError(
            "Invalid timestamp or sampling rate"
        )

    readings = []

    for row_number, row in enumerate(rows[2:], start=3):
        if not row or all(not item.strip() for item in row):
            continue

        try:
            values = [float(item) for item in row]
        except ValueError as exc:
            raise ValueError(
                f"Invalid signal value at row {row_number}"
            ) from exc

        if any(not math.isfinite(value) for value in values):
            raise ValueError(
                f"Non-finite signal value at row {row_number}"
            )

        if len(values) != len(sampling_rates):
            raise ValueError(
                f"Unexpected number of channels at row {row_number}"
            )

        readings.append(values)

    if not readings:
        raise ValueError(
            "Signal CSV contains no readings"
        )

    return {
        "timestamp": timestamp,
        "sampling_rates": sampling_rates,
        "readings": readings,
        "sample_count": len(readings),
        "channel_count": len(sampling_rates),
    }


def load_tags_csv(
    file_path: str | Path,
) -> list[str]:
    """Load event timestamps from tags.csv."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Tags CSV not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        timestamps = [
            row[0].strip()
            for row in csv.reader(file)
            if row and row[0].strip()
        ]

    if not timestamps:
        raise ValueError(
            "Tags CSV contains no timestamps"
        )

    return timestamps

def load_ibi_csv(
    file_path: str | Path,
) -> dict[str, object]:
    """
    Load an Empatica IBI CSV.

    Expected format:
    - Row 1: recording start timestamp (metadata)
    - Remaining rows:
        Column 1: elapsed time in seconds
        Column 2: inter-beat interval duration in seconds

    Returns the recording timestamp, readings, and sample count.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"IBI CSV not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.reader(file))

    if len(rows) < 2 or not rows[0]:
        raise ValueError(
            "IBI CSV must contain a start timestamp "
            "and at least one reading"
        )

    start_timestamp = rows[0][0].strip()

    if not start_timestamp:
        raise ValueError(
            "IBI CSV contains an empty start timestamp"
        )

    readings: list[list[float]] = []

    for row_number, row in enumerate(rows[1:], start=2):
        if not row or all(not item.strip() for item in row):
            continue

        if len(row) != 2:
            raise ValueError(
                f"Expected 2 IBI columns at row {row_number}"
            )

        try:
            elapsed_time = float(row[0])
            ibi_duration = float(row[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid IBI value at row {row_number}"
            ) from exc

        if (
            not math.isfinite(elapsed_time)
            or not math.isfinite(ibi_duration)
        ):
            raise ValueError(
                f"Non-finite IBI value at row {row_number}"
            )

        if elapsed_time < 0 or ibi_duration <= 0:
            raise ValueError(
                f"Invalid IBI value at row {row_number}: "
                f"{elapsed_time}, {ibi_duration}"
            )

        readings.append([elapsed_time, ibi_duration])

    if not readings:
        raise ValueError(
            "IBI CSV contains no valid readings"
        )

    return {
        "start_timestamp": start_timestamp,
        "readings": readings,
        "sample_count": len(readings),
    }