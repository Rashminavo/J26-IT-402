from typing import Any


def extract_features(
    validated_data: dict[str, float],
) -> dict[str, float]:
    """
    Build a standardized feature dictionary from validated measurements.

    The final Component 4 feature engineering strategy is research-defined
    and is intentionally not fixed in this reference MVP.
    """
    if not isinstance(validated_data, dict):
        raise TypeError(
            "validated_data must be a dictionary"
        )

    return {
        key: float(value)
        for key, value in validated_data.items()
    }


def count_available_features(
    features: dict[str, float],
) -> int:
    return len(features)