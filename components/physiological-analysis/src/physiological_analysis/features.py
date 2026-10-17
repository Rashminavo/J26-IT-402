
from math import isfinite, sqrt
from statistics import mean, pstdev


def extract_features(
    validated_data: dict[str, float],
) -> dict[str, float]:
    """Build a feature dictionary from validated measurements."""
    if not isinstance(validated_data, dict):
        raise TypeError("validated_data must be a dictionary")

    return {
        key: float(value)
        for key, value in validated_data.items()
    }


def count_available_features(
    features: dict[str, float],
) -> int:
    return len(features)


def extract_heart_rate_features(
    heart_rate_readings: list[float],
) -> dict[str, float]:
    """Extract descriptive statistics from heart-rate readings."""
    if not isinstance(heart_rate_readings, list):
        raise TypeError("heart_rate_readings must be a list")

    if not heart_rate_readings:
        raise ValueError("At least one heart-rate reading is required")

    readings = []

    for value in heart_rate_readings:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Heart-rate readings must be numeric")

        value = float(value)

        if not isfinite(value) or value <= 0:
            raise ValueError(
                "Heart-rate readings must be finite and positive"
            )

        readings.append(value)

    return {
        "heart_rate_mean": mean(readings),
        "heart_rate_min": min(readings),
        "heart_rate_max": max(readings),
        "heart_rate_std": pstdev(readings),
    }


def extract_signal_features(
    signal_name: str,
    signal_data: dict[str, object],
) -> dict[str, float]:
    """
    Extract statistical features from a loaded wearable signal.

    Supports single-channel EDA, BVP, TEMP and three-axis ACC.
    Input data must follow the structure returned by load_signal_csv.
    """
    if not isinstance(signal_name, str):
        raise TypeError("signal_name must be a string")

    name = signal_name.strip().lower()

    if name not in {"eda", "bvp", "temp", "acc"}:
        raise ValueError("Unsupported signal name")

    if not isinstance(signal_data, dict):
        raise TypeError("signal_data must be a dictionary")

    readings = signal_data.get("readings")
    sampling_rates = signal_data.get("sampling_rates")

    if not isinstance(readings, list) or not readings:
        raise ValueError("Signal readings must be a non-empty list")

    if not isinstance(sampling_rates, list) or not sampling_rates:
        raise ValueError("Sampling rates must be a non-empty list")

    expected_channels = 3 if name == "acc" else 1

    if len(sampling_rates) != expected_channels:
        raise ValueError(
            f"{name.upper()} requires {expected_channels} channel(s)"
        )

    for rate in sampling_rates:
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not isfinite(float(rate))
            or rate <= 0
        ):
            raise ValueError("Sampling rates must be finite and positive")

    channels: list[list[float]] = [
        [] for _ in range(expected_channels)
    ]

    for row in readings:
        if not isinstance(row, list) or len(row) != expected_channels:
            raise ValueError("Signal reading has an unexpected channel count")

        for index, value in enumerate(row):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
            ):
                raise ValueError("Signal readings must be finite numbers")

            channels[index].append(float(value))

    features: dict[str, float] = {}

    if name == "acc":
        channel_names = ("x", "y", "z")
    else:
        channel_names = (name,)

    for channel_name, values in zip(channel_names, channels):
        features[f"{name}_{channel_name}_mean"] = mean(values)
        features[f"{name}_{channel_name}_std"] = pstdev(values)
        features[f"{name}_{channel_name}_min"] = min(values)
        features[f"{name}_{channel_name}_max"] = max(values)

    if name == "acc":
        magnitudes = [
            sqrt(x * x + y * y + z * z)
            for x, y, z in zip(*channels)
        ]

        features["acc_magnitude_mean"] = mean(magnitudes)
        features["acc_magnitude_std"] = pstdev(magnitudes)
        features["acc_magnitude_min"] = min(magnitudes)
        features["acc_magnitude_max"] = max(magnitudes)

    return features


def extract_ibi_features(
    ibi_readings: list[list[float]],
) -> dict[str, float]:
    """
    Extract basic IBI/HRV candidate features.

    Each reading contains:
    [elapsed_time_seconds, ibi_duration_seconds].

    RMSSD is calculated from consecutive IBI durations.
    These are signal features, not stress classifications.
    """
    if not isinstance(ibi_readings, list):
        raise TypeError("ibi_readings must be a list")

    if len(ibi_readings) < 2:
        raise ValueError("At least two IBI readings are required")

    elapsed_times: list[float] = []
    intervals: list[float] = []

    for row in ibi_readings:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(
                "Each IBI reading must contain elapsed time and interval"
            )

        elapsed, interval = row

        if (
            isinstance(elapsed, bool)
            or isinstance(interval, bool)
            or not isinstance(elapsed, (int, float))
            or not isinstance(interval, (int, float))
        ):
            raise ValueError("IBI values must be numeric")

        elapsed = float(elapsed)
        interval = float(interval)

        if (
            not isfinite(elapsed)
            or not isfinite(interval)
            or elapsed < 0
            or interval <= 0
        ):
            raise ValueError("IBI values must be finite and valid")

        elapsed_times.append(elapsed)
        intervals.append(interval)

    if any(
        current <= previous
        for previous, current in zip(elapsed_times, elapsed_times[1:])
    ):
        raise ValueError("IBI elapsed times must be strictly increasing")

    successive_differences = [
        current - previous
        for previous, current in zip(intervals, intervals[1:])
    ]

    rmssd = sqrt(
        mean(difference * difference for difference in successive_differences)
    )

    return {
        "ibi_mean_seconds": mean(intervals),
        "ibi_std_seconds": pstdev(intervals),
        "ibi_rmssd_seconds": rmssd,
    }
