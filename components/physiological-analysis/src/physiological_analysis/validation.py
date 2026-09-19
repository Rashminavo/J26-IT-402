from math import isfinite
from typing import Any


SUPPORTED_FIELDS = {
    "heart_rate",
    "heart_rate_variability",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "body_temperature",
    "sleep_duration",
}


def validate_measurements(
    measurements: dict[str, Any],
) -> dict[str, float]:
    """
    Validate the structural quality of incoming physiological measurements.

    This layer does not apply clinical thresholds. Device-specific units,
    sampling rates and physiological interpretation belong to the research
    implementation.
    """
    if not isinstance(measurements, dict):
        raise TypeError("measurements must be a JSON object")

    validated: dict[str, float] = {}

    for field, value in measurements.items():
        if field not in SUPPORTED_FIELDS:
            continue

        if value is None:
            continue

        if isinstance(value, bool):
            raise ValueError(
                f"Invalid boolean value for {field}"
            )

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field} must be numeric"
            ) from exc

        if not isfinite(numeric_value):
            raise ValueError(
                f"{field} must be finite"
            )

        if numeric_value < 0:
            raise ValueError(
                f"{field} cannot be negative"
            )

        validated[field] = numeric_value

    return validated