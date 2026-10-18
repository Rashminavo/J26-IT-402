from math import isfinite
from numbers import Real
from typing import Any

from flask import Blueprint, jsonify, request

from app.config import API_PREFIX
from app.services.physiological_service import analyze_physiological
from app.services.physiological_stress_prediction_service import (
    predict_window_stress,
)
from app.services.score_service import (
    ScoreAlreadyExistsError,
    ScoreValidationError,
    create_score,
)


physiological_bp = Blueprint(
    "physiological",
    __name__,
    url_prefix=f"{API_PREFIX}/physiological",
)


@physiological_bp.post("/analyze")
def analyze_physiological_data():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return (
            jsonify(
                {
                    "error": "Request body must be a JSON object."
                }
            ),
            400,
        )

    required_fields = [
        "measurements",
        "observation_id",
        "user_id",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in payload
    ]

    if missing_fields:
        return (
            jsonify(
                {
                    "error": "Missing required fields.",
                    "fields": missing_fields,
                }
            ),
            400,
        )

    if not isinstance(
        payload["measurements"],
        dict,
    ):
        return (
            jsonify(
                {
                    "error": "measurements must be a JSON object."
                }
            ),
            400,
        )

    try:
        score = analyze_physiological(
            measurements=payload["measurements"],
            observation_id=payload["observation_id"],
            user_id=payload["user_id"],
            session_id=payload.get("session_id"),
        )

        stored_score = create_score(score)

        return (
            jsonify(
                {
                    "message": "Physiological analysis completed.",
                    "data": stored_score,
                }
            ),
            201,
        )

    except ScoreValidationError as exc:
        return (
            jsonify(
                {
                    "error": "Generated score failed validation.",
                    "details": str(exc),
                }
            ),
            422,
        )

    except ScoreAlreadyExistsError as exc:
        return (
            jsonify(
                {
                    "error": str(exc),
                }
            ),
            409,
        )

    except (TypeError, ValueError) as exc:
        return (
            jsonify(
                {
                    "error": "Physiological analysis failed.",
                    "details": str(exc),
                }
            ),
            422,
        )


@physiological_bp.post("/predict-window-stress")
def predict_physiological_window_stress():
    """Return an exploratory Low/Moderate/High prediction for one 15s window.

    Separate from /analyze and the shared numeric score store: this exploratory
    class is not a validated numeric mental-health score.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    required_fields = [
        "features",
        "observation_id",
        "user_id",
        "window_start_seconds",
        "window_end_seconds",
    ]
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        return jsonify({
            "error": "Missing required fields.",
            "fields": missing_fields,
        }), 400

    if not isinstance(payload["features"], dict):
        return jsonify({"error": "features must be a JSON object."}), 400

    for field in ("observation_id", "user_id"):
        value = payload[field]
        if not isinstance(value, str) or not value.strip():
            return jsonify({"error": f"{field} must be a non-empty string."}), 400

    session_id = payload.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        return jsonify({"error": "session_id must be a string or null."}), 400

    start = payload["window_start_seconds"]
    end = payload["window_end_seconds"]
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, Real)
        or not isinstance(end, Real)
        or not isfinite(float(start))
        or not isfinite(float(end))
    ):
        return jsonify({"error": "Window bounds must be finite numbers."}), 422

    start = float(start)
    end = float(end)
    duration = end - start
    if start < 0 or duration <= 0 or abs(duration - 15.0) > 1e-6:
        return jsonify({
            "error": "Window must have non-negative start and exactly 15 seconds duration."
        }), 422

    try:
        prediction = predict_window_stress(payload["features"])
    except FileNotFoundError:
        return jsonify({
            "error": "Exploratory stress model artifact is unavailable.",
            "details": (
                "Provision the .joblib file and, if needed, set "
                "J26_PHYSIOLOGICAL_MODEL_PATH."
            ),
        }), 503
    except (TypeError, ValueError) as exc:
        return jsonify({
            "error": "Invalid physiological feature input.",
            "details": str(exc),
        }), 422
    except RuntimeError:
        return jsonify({
            "error": "Exploratory stress model could not complete prediction."
        }), 503

    return jsonify({
        "message": "Exploratory physiological window prediction completed.",
        "data": {
            "observation_id": payload["observation_id"],
            "user_id": payload["user_id"],
            "session_id": session_id,
            "component": "physiological_analysis",
            "window": {
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": duration,
            },
            "prediction": {
                "predicted_class": prediction["predicted_class"],
                "class_id": prediction["class_id"],
                "model_name": prediction["model_name"],
                "input_level": prediction["input_level"],
                "window_size_seconds": prediction["window_size_seconds"],
                "rating_cutoffs": prediction["rating_cutoffs"],
                "research_use_only": prediction["research_use_only"],
                "clinical_validation": prediction["clinical_validation"],
                "warning": prediction["warning"],
            },
        },
    }), 200
