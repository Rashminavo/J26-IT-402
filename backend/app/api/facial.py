from flask import Blueprint, jsonify, request
import numpy as np

from app.config import API_PREFIX
from app.services.facial_service import analyze_facial
from app.services.score_service import (
    ScoreAlreadyExistsError,
    ScoreValidationError,
    create_score,
)


facial_bp = Blueprint(
    "facial",
    __name__,
    url_prefix=f"{API_PREFIX}/facial",
)


@facial_bp.post("/analyze")
def analyze_facial_data():
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
        "frames",
        "bounding_boxes",
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

    if not isinstance(payload["frames"], list):
        return (
            jsonify(
                {
                    "error": "frames must be a JSON array."
                }
            ),
            400,
        )

    if not isinstance(payload["bounding_boxes"], list):
        return (
            jsonify(
                {
                    "error": (
                        "bounding_boxes must be a JSON array."
                    )
                }
            ),
            400,
        )

    if len(payload["frames"]) != len(
        payload["bounding_boxes"]
    ):
        return (
            jsonify(
                {
                    "error": (
                        "frames and bounding_boxes must "
                        "have equal length."
                    )
                }
            ),
            400,
        )

    try:
        # The API receives serializable frame arrays and converts them
        # into NumPy arrays at the service boundary.
        frames = [
            np.asarray(frame, dtype=np.uint8)
            for frame in payload["frames"]
        ]

        bounding_boxes = [
            tuple(int(value) for value in box)
            for box in payload["bounding_boxes"]
        ]

        score = analyze_facial(
            frames=frames,
            bounding_boxes=bounding_boxes,
            observation_id=payload["observation_id"],
            user_id=payload["user_id"],
            session_id=payload.get("session_id"),
        )

        stored_score = create_score(score)

        return (
            jsonify(
                {
                    "message": "Facial analysis completed.",
                    "data": stored_score,
                }
            ),
            201,
        )

    except ScoreValidationError as exc:
        return (
            jsonify(
                {
                    "error": (
                        "Generated score failed validation."
                    ),
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
                    "error": "Facial analysis failed.",
                    "details": str(exc),
                }
            ),
            422,
        )