from flask import Blueprint, jsonify, request

from app.config import API_PREFIX
from app.services.language_service import analyze_language
from app.services.score_service import (
    ScoreAlreadyExistsError,
    ScoreValidationError,
    create_score,
)


language_bp = Blueprint(
    "language",
    __name__,
    url_prefix=f"{API_PREFIX}/language",
)


@language_bp.post("/analyze")
def analyze_language_text():
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
        "text",
        "observation_id",
        "user_id",
    ]

    missing_fields = [
        field
        for field in required_fields
        if not payload.get(field)
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

    try:
        score = analyze_language(
            text=payload["text"],
            observation_id=payload["observation_id"],
            user_id=payload["user_id"],
            session_id=payload.get("session_id"),
        )

        stored_score = create_score(score)

        return (
            jsonify(
                {
                    "message": "Language analysis completed.",
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