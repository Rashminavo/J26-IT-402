from typing import Iterable


def calculate_baseline(
    historical_values: Iterable[float],
) -> dict[str, float]:
    """
    Calculate a simple reference baseline.

    This is NOT the final research baseline method.
    """
    values = [float(value) for value in historical_values]

    if not values:
        raise ValueError(
            "At least one historical value is required."
        )

    if any(value < 0 for value in values):
        raise ValueError(
            "Historical behavioural values cannot be negative."
        )

    mean = sum(values) / len(values)

    variance = sum(
        (value - mean) ** 2
        for value in values
    ) / len(values)

    return {
        "mean": mean,
        "std": variance ** 0.5,
        "sample_count": float(len(values)),
    }


def calculate_reference_deviation(
    current_value: float,
    baseline: dict[str, float],
) -> dict[str, float]:
    """
    Produce a deterministic reference deviation signal.

    The result is evidence of deviation from the personal baseline,
    not a clinical or final behavioural risk score.
    """
    mean = baseline["mean"]
    std = baseline["std"]

    if current_value < 0:
        raise ValueError(
            "Current behavioural value cannot be negative."
        )

    if std == 0:
        z_score = 0.0 if current_value == mean else 1.0
    else:
        z_score = abs(current_value - mean) / std

    deviation_score = z_score / (1.0 + z_score)

    return {
        "z_score": z_score,
        "deviation_score": deviation_score,
    }