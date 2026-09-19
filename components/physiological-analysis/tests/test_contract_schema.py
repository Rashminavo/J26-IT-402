import json

from jsonschema import Draft202012Validator, FormatChecker

from app.config import SCORE_SCHEMA_PATH
from physiological_analysis.contract import (
    build_physiological_score,
)


def test_physiological_score_matches_shared_schema():
    with SCORE_SCHEMA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        schema = json.load(file)

    payload = build_physiological_score(
        measurements={
            "heart_rate": 78,
            "heart_rate_variability": 42,
        },
        observation_id="OBS-PHYS-SCHEMA-001",
        user_id="USER-001",
        timestamp="2026-09-19T10:00:00+00:00",
    )

    validator = Draft202012Validator(
        schema,
        format_checker=FormatChecker(),
    )

    errors = list(
        validator.iter_errors(payload)
    )

    assert errors == []