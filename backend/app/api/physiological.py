from flask import Blueprint, jsonify, request

from app.config import API_PREFIX
from app.services.physiological_service import (
    analyze_physiological,
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