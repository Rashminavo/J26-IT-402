import pytest

from app.main import create_app


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client


def valid_score():
    return {
        "schema_version": "1.0.0",
        "observation_id": "OBS-TEST-001",
        "user_id": "USER-TEST-001",
        "session_id": "SESSION-TEST-001",
        "component": "language_analysis",
        "timestamp": "2026-09-19T08:00:00Z",
        "data_status": "AVAILABLE",
        "score": {
            "value": 72,
            "min": 0,
            "max": 100,
            "normalized_value": 0.72,
            "unit": "score"
        },
        "confidence": 0.86,
        "risk_level": "High Risk",
        "risk_level_scheme": "MHRS_0_100",
        "data_quality": {
            "completeness": 0.98
        },
        "model": {
            "name": "test-model",
            "version": "0.1"
        },
        "metadata": {
            "environment": "test"
        }
    }


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "ok"
    assert data["service"] == "j26-it-402-backend"


def test_create_valid_score(client):
    response = client.post(
        "/api/v1/scores",
        json=valid_score(),
    )

    assert response.status_code == 201

    data = response.get_json()

    assert data["message"] == "Score accepted."
    assert data["data"]["observation_id"] == "OBS-TEST-001"


def test_reject_invalid_score(client):
    payload = valid_score()
    del payload["component"]

    response = client.post(
        "/api/v1/scores",
        json=payload,
    )

    assert response.status_code == 422


def test_get_score(client):
    client.post(
        "/api/v1/scores",
        json=valid_score(),
    )

    response = client.get(
        "/api/v1/scores/OBS-TEST-001"
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["data"]["observation_id"] == "OBS-TEST-001"


def test_score_not_found(client):
    response = client.get(
        "/api/v1/scores/UNKNOWN-OBSERVATION"
    )

    assert response.status_code == 404