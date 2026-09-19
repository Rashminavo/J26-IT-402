from flask import Blueprint, jsonify, request

from app.config import API_PREFIX
from app.services.score_service import (
    ScoreAlreadyExistsError,
    ScoreNotFoundError,
    ScoreValidationError,
    create_score,
    get_score,
    list_scores,
)


scores_bp = Blueprint(
    "scores",
    __name__,
    url_prefix=f"{API_PREFIX}/scores",
)


@scores_bp.post("")
def create_component_score():
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

    try:
        score = create_score(payload)

        return jsonify(
            {
                "message": "Score accepted.",
                "data": score,
            }
        ), 201

    except ScoreValidationError as exc:
        return (
            jsonify(
                {
                    "error": "Score validation failed.",
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


@scores_bp.get("")
def get_all_scores():
    return jsonify(
        {
            "count": len(list_scores()),
            "data": list_scores(),
        }
    )


@scores_bp.get("/<observation_id>")
def get_component_score(observation_id: str):
    try:
        score = get_score(observation_id)

        return jsonify(
            {
                "data": score,
            }
        )

    except ScoreNotFoundError as exc:
        return (
            jsonify(
                {
                    "error": str(exc),
                }
            ),
            404,
        )