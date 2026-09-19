import json

from jsonschema import Draft202012Validator, FormatChecker

from app.config import SCORE_SCHEMA_PATH
from behavioural_tracking.contract import build_behavioural_score


def test_behavioural_score_matches_shared_schema():
    with SCORE_SCHEMA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        schema = json.load(file)

    payload = build_behavioural_score(
        raw_data={
            "screen_time_minutes": 260,
        },
        historical_screen_times=[
            200,
            210,
            190,
            205,
        ],
        observation_id="OBS-BEH-SCHEMA-001",
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