from flask import Blueprint, jsonify, request

from app.config import API_PREFIX
from app.services.behavioural_service import analyze_behavioural
from app.services.score_service import (
    ScoreAlreadyExistsError,
    ScoreValidationError,
    create_score,
)


behavioural_bp = Blueprint(
    "behavioural",
    __name__,
    url_prefix=f"{API_PREFIX}/behavioural",
)


@behavioural_bp.post("/analyze")
def analyze_behavioural_data():
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
        "raw_data",
        "historical_screen_times",
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

    if not isinstance(payload["raw_data"], dict):
        return (
            jsonify(
                {
                    "error": "raw_data must be a JSON object."
                }
            ),
            400,
        )

    if not isinstance(
        payload["historical_screen_times"],
        list,
    ):
        return (
            jsonify(
                {
                    "error": (
                        "historical_screen_times must "
                        "be a JSON array."
                    )
                }
            ),
            400,
        )

    try:
        score = analyze_behavioural(
            raw_data=payload["raw_data"],
            historical_screen_times=payload[
                "historical_screen_times"
            ],
            observation_id=payload["observation_id"],
            user_id=payload["user_id"],
            session_id=payload.get("session_id"),
        )

        stored_score = create_score(score)

        return (
            jsonify(
                {
                    "message": "Behavioural analysis completed.",
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
                    "error": "Behavioural analysis failed.",
                    "details": str(exc),
                }
            ),
            422,
        )