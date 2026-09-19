import json

from jsonschema import Draft202012Validator, FormatChecker

from app.config import SCORE_SCHEMA_PATH
from language_analysis.contract import build_language_score


def test_language_score_matches_shared_schema():
    with SCORE_SCHEMA_PATH.open("r", encoding="utf-8") as file:
        schema = json.load(file)

    payload = build_language_score(
        text="I am feeling hopeful today.",
        observation_id="OBS-SCHEMA-001",
        user_id="USER-001",
        timestamp="2026-09-19T10:00:00+00:00",
    )

    validator = Draft202012Validator(
        schema,
        format_checker=FormatChecker(),
    )

    errors = list(validator.iter_errors(payload))

    assert errors == []