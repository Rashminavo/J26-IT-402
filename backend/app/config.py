from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCORE_SCHEMA_PATH = (
    PROJECT_ROOT
    / "integration"
    / "score-interface"
    / "score-schema.json"
)

API_PREFIX = "/api/v1"