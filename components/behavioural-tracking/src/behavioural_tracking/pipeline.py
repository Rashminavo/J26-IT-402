from typing import Any

from .baseline import (
    calculate_baseline,
    calculate_reference_deviation,
)
from .features import extract_features


def analyze_behaviour(
    raw_data: dict[str, Any],
    historical_screen_times: list[float],
) -> dict[str, Any]:
    """
    Execute the current behavioural reference pipeline.

    Current stage:

        Passive behavioural data
        -> feature extraction
        -> personalised baseline
        -> reference deviation analysis

    Isolation Forest, LSTM and Random Forest are deferred to the
    model-development stage.
    """
    features = extract_features(raw_data)

    if "screen_time_minutes" not in features:
        raise ValueError(
            "screen_time_minutes is required for the current reference MVP."
        )

    baseline = calculate_baseline(
        historical_screen_times
    )

    deviation = calculate_reference_deviation(
        current_value=features["screen_time_minutes"],
        baseline=baseline,
    )

    return {
        "stage": "behavioural_reference",
        "features": features,
        "baseline": baseline,
        "deviation": deviation,
        "models": {
            "isolation_forest": "pending_component_2_implementation",
            "lstm": "pending_component_2_implementation",
            "random_forest": "pending_component_2_implementation",
        },
    }