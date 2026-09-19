from datetime import datetime, timezone
from typing import Any

from .pipeline import analyze_physiology


def build_physiological_score(
    measurements: dict[str, Any],
    observation_id: str,
    user_id: str,
    session_id: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """
    Convert the current physiological evidence into the shared contract.

    The physiological stress score remains pending until the Random Forest
    research implementation is available.
    """
    analysis = analyze_physiology(
        measurements
    )

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "schema_version": "1.0.0",
        "observation_id": observation_id,
        "user_id": user_id,
        "session_id": session_id,
        "component": "physiological_analysis",
        "timestamp": timestamp,
        "data_status": "AVAILABLE",
        "score": {
            "value": None,
            "min": 0,
            "max": 1,
            "normalized_value": None,
            "unit": "physiological_stress_score_pending",
        },
        "confidence": None,
        "risk_level": None,
        "risk_level_scheme": None,
        "data_quality": {
            "available_feature_count": analysis[
                "feature_count"
            ],
        },
        "model": {
            "name": "RandomForest",
            "version": None,
        },
        "metadata": {
            "stage": "physiological_reference",
            "validated_data": analysis[
                "validated_data"
            ],
            "standardization_status": analysis[
                "standardized_features"
            ]["status"],
            "model_status": analysis[
                "model"
            ]["status"],
        },
    }