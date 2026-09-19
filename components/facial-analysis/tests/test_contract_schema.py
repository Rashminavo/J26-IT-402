import json

import numpy as np
from jsonschema import (
    Draft202012Validator,
    FormatChecker,
)

from app.config import SCORE_SCHEMA_PATH
from facial_analysis.contract import (
    build_facial_score,
)


def test_facial_score_matches_shared_schema():
    with SCORE_SCHEMA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        schema = json.load(file)

    frames = [
        np.zeros(
            (300, 300, 3),
            dtype=np.uint8,
        )
        for _ in range(16)
    ]

    boxes = [
        (50, 50, 100, 100)
        for _ in range(16)
    ]

    payload = build_facial_score(
        frames=frames,
        bounding_boxes=boxes,
        observation_id="OBS-FACIAL-SCHEMA-001",
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