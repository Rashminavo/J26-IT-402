from datetime import datetime, timezone
from typing import Any

from .pipeline import analyze_behaviour


def build_behavioural_score(
    raw_data: dict[str, Any],
    historical_screen_times: list[float],
    observation_id: str,
    user_id: str,
    session_id: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """
    Convert the current behavioural reference evidence into the
    shared component score contract.

    This output represents behavioural deviation evidence.
    It is not the final Behavioural Risk Score.
    """
    analysis = analyze_behaviour(
        raw_data=raw_data,
        historical_screen_times=historical_screen_times,
    )

    deviation_score = float(
        analysis["deviation"]["deviation_score"]
    )

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "schema_version": "1.0.0",
        "observation_id": observation_id,
        "user_id": user_id,
        "session_id": session_id,
        "component": "behavioural_tracking",
        "timestamp": timestamp,
        "data_status": "AVAILABLE",
        "score": {
            "value": deviation_score,
            "min": 0,
            "max": 1,
            "normalized_value": deviation_score,
            "unit": "baseline_deviation_reference",
        },
        "confidence": None,
        "risk_level": None,
        "risk_level_scheme": None,
        "data_quality": {
            "completeness": 1.0,
        },
        "model": {
            "name": "reference_baseline",
            "version": "0.1",
        },
        "metadata": {
            "stage": "behavioural_reference",
            "features": analysis["features"],
            "baseline": analysis["baseline"],
            "z_score": analysis["deviation"]["z_score"],
            "planned_models": [
                "IsolationForest",
                "LSTM",
                "RandomForest",
            ],
        },
    }