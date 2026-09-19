import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from app.config import SCORE_SCHEMA_PATH


class ScoreValidationError(ValueError):
    """Raised when a component score violates the shared contract."""


class ScoreAlreadyExistsError(ValueError):
    """Raised when an observation ID already exists."""


class ScoreNotFoundError(ValueError):
    """Raised when a requested observation does not exist."""


_scores: dict[str, dict[str, Any]] = {}


def _load_schema() -> dict[str, Any]:
    schema_path: Path = SCORE_SCHEMA_PATH

    with schema_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_score(payload: dict[str, Any]) -> None:
    schema = _load_schema()

    validator = Draft202012Validator(
        schema,
        format_checker=FormatChecker(),
    )

    errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: list(error.path),
    )

    if errors:
        error = errors[0]

        field_path = ".".join(str(item) for item in error.path)

        if field_path:
            message = f"{field_path}: {error.message}"
        else:
            message = error.message

        raise ScoreValidationError(message)


def create_score(payload: dict[str, Any]) -> dict[str, Any]:
    validate_score(payload)

    observation_id = payload["observation_id"]

    if observation_id in _scores:
        raise ScoreAlreadyExistsError(
            f"Observation '{observation_id}' already exists."
        )

    _scores[observation_id] = payload

    return payload


def get_score(observation_id: str) -> dict[str, Any]:
    score = _scores.get(observation_id)

    if score is None:
        raise ScoreNotFoundError(
            f"Observation '{observation_id}' was not found."
        )

    return score


def list_scores() -> list[dict[str, Any]]:
    return list(_scores.values())