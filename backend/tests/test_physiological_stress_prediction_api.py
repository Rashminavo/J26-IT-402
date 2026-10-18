import pytest

from app.main import create_app


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def valid_payload():
    feature_names = [
        "heart_rate_mean", "heart_rate_min", "heart_rate_max", "heart_rate_std",
        "eda_eda_mean", "eda_eda_std", "eda_eda_min", "eda_eda_max",
        "bvp_bvp_mean", "bvp_bvp_std", "bvp_bvp_min", "bvp_bvp_max",
        "temp_temp_mean", "temp_temp_std", "temp_temp_min", "temp_temp_max",
        "acc_x_mean", "acc_x_std", "acc_x_min", "acc_x_max",
        "acc_y_mean", "acc_y_std", "acc_y_min", "acc_y_max",
        "acc_z_mean", "acc_z_std", "acc_z_min", "acc_z_max",
        "acc_magnitude_mean", "acc_magnitude_std", "acc_magnitude_min", "acc_magnitude_max",
    ]
    return {
        "features": {name: float(index + 1) for index, name in enumerate(feature_names)},
        "observation_id": "OBS-WINDOW-TEST-001",
        "user_id": "USER-TEST-001",
        "session_id": "SESSION-TEST-001",
        "window_start_seconds": 15.0,
        "window_end_seconds": 30.0,
    }


def fake_prediction():
    return {
        "predicted_class": "Low",
        "class_id": 0,
        "model_name": "SVC_RBF",
        "input_level": "15-second window-level physiological features",
        "window_size_seconds": 15,
        "rating_cutoffs": {"lower_rating_cutoff": 3.0, "upper_rating_cutoff": 5.0},
        "research_use_only": True,
        "clinical_validation": False,
        "warning": "Exploratory self-report bands only; not clinically validated.",
    }


def test_window_prediction_endpoint_returns_exploratory_prediction(client, monkeypatch):
    from app.api import physiological

    monkeypatch.setattr(
        physiological,
        "predict_window_stress",
        lambda features: fake_prediction(),
    )
    response = client.post("/api/v1/physiological/predict-window-stress", json=valid_payload())

    assert response.status_code == 200
    body = response.get_json()
    assert body["data"]["prediction"]["predicted_class"] == "Low"
    assert body["data"]["prediction"]["research_use_only"] is True
    assert body["data"]["prediction"]["clinical_validation"] is False
    assert body["data"]["window"]["duration_seconds"] == 15.0


def test_window_prediction_endpoint_rejects_missing_fields(client):
    payload = valid_payload()
    del payload["features"]
    response = client.post("/api/v1/physiological/predict-window-stress", json=payload)
    assert response.status_code == 400
    assert "features" in response.get_json()["fields"]


def test_window_prediction_endpoint_rejects_non_object_features(client):
    payload = valid_payload()
    payload["features"] = []
    response = client.post("/api/v1/physiological/predict-window-stress", json=payload)
    assert response.status_code == 400


def test_window_prediction_endpoint_rejects_wrong_window_duration(client):
    payload = valid_payload()
    payload["window_end_seconds"] = 31.0
    response = client.post("/api/v1/physiological/predict-window-stress", json=payload)
    assert response.status_code == 422


def test_window_prediction_endpoint_returns_503_when_model_missing(client, monkeypatch):
    from app.api import physiological

    def missing_model(_features):
        raise FileNotFoundError("model artifact missing")

    monkeypatch.setattr(physiological, "predict_window_stress", missing_model)
    response = client.post("/api/v1/physiological/predict-window-stress", json=valid_payload())
    assert response.status_code == 503
    assert "artifact is unavailable" in response.get_json()["error"]


def test_window_prediction_endpoint_rejects_bad_feature_values(client, monkeypatch):
    from app.api import physiological

    def bad_features(_features):
        raise ValueError("Feature 'heart_rate_mean' must be finite")

    monkeypatch.setattr(physiological, "predict_window_stress", bad_features)
    response = client.post("/api/v1/physiological/predict-window-stress", json=valid_payload())
    assert response.status_code == 422
    assert "Invalid physiological feature input" in response.get_json()["error"]


def test_existing_numeric_analyze_route_remains_available(client):
    # The original endpoint remains separate and its contract is unchanged.
    response = client.post(
        "/api/v1/physiological/analyze",
        json={
            "measurements": {"heart_rate": 78, "heart_rate_variability": 42, "body_temperature": 36.8},
            "observation_id": "OBS-PHYSIOLOGICAL-API-REGRESSION-001",
            "user_id": "USER-TEST-001",
            "session_id": "SESSION-TEST-001",
        },
    )
    assert response.status_code == 201
    score = response.get_json()["data"]
    assert score["component"] == "physiological_analysis"
    assert score["score"]["unit"] == "physiological_stress_score_pending"
    assert score["score"]["value"] is None
