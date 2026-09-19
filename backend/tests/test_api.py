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

def test_language_analysis_endpoint(client):
    response = client.post(
        "/api/v1/language/analyze",
        json={
            "text": "I am feeling happy and hopeful today.",
            "observation_id": "OBS-LANGUAGE-001",
            "user_id": "USER-TEST-001",
            "session_id": "SESSION-TEST-001",
        },
    )

    assert response.status_code == 201

    data = response.get_json()

    assert data["message"] == "Language analysis completed."

    score = data["data"]

    assert score["component"] == "language_analysis"
    assert score["score"]["unit"] == "vader_compound"

def test_behavioural_analysis_endpoint(client):
    response = client.post(
        "/api/v1/behavioural/analyze",
        json={
            "raw_data": {
                "screen_time_minutes": 260,
                "app_usage_minutes": 190,
                "typing_speed": 35,
            },
            "historical_screen_times": [
                200,
                210,
                190,
                205,
            ],
            "observation_id": "OBS-BEHAVIOUR-001",
            "user_id": "USER-TEST-001",
            "session_id": "SESSION-TEST-001",
        },
    )

    assert response.status_code == 201

    data = response.get_json()

    assert (
        data["message"]
        == "Behavioural analysis completed."
    )

    score = data["data"]

    assert score["component"] == "behavioural_tracking"
    assert (
        score["score"]["unit"]
        == "baseline_deviation_reference"
    )


def test_physiological_analysis_endpoint(client):
    response = client.post(
        "/api/v1/physiological/analyze",
        json={
            "measurements": {
                "heart_rate": 78,
                "heart_rate_variability": 42,
                "body_temperature": 36.8
            },
            "observation_id": "OBS-PHYSIOLOGICAL-001",
            "user_id": "USER-TEST-001",
            "session_id": "SESSION-TEST-001",
        },
    )

    assert response.status_code == 201

    data = response.get_json()

    assert (
        data["message"]
        == "Physiological analysis completed."
    )

    score = data["data"]

    assert score["component"] == "physiological_analysis"
    assert (
        score["score"]["unit"]
        == "physiological_stress_score_pending"
    )
    assert score["score"]["value"] is None