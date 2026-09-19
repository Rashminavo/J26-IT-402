from typing import Any


SUPPORTED_FEATURES = (
    "screen_time_minutes",
    "app_usage_minutes",
    "typing_speed",
    "message_length",
    "mobility_events",
    "social_interaction_events",
)


def extract_features(
    raw_data: dict[str, Any],
) -> dict[str, float]:
    """
    Extract the initial behavioural feature set.

    This is a reference engineering layer.
    Final feature definitions belong to Component 2 research work.
    """
    features: dict[str, float] = {}

    for feature_name in SUPPORTED_FEATURES:
        value = raw_data.get(feature_name)

        if value is None:
            continue

        if isinstance(value, bool):
            raise ValueError(
                f"Invalid boolean value for {feature_name}"
            )

        numeric_value = float(value)

        if numeric_value < 0:
            raise ValueError(
                f"{feature_name} cannot be negative"
            )

        features[feature_name] = numeric_value

    return features