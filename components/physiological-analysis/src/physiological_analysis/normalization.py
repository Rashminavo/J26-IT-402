from typing import Any


def build_standardized_features(
    features: dict[str, float],
) -> dict[str, Any]:
    """
    Define the normalization boundary without inventing a final
    normalization formula.

    The actual method must account for device differences, units,
    sampling rates and research validation.
    """
    return {
        "status": "NORMALIZATION_PENDING",
        "features": features,
    }