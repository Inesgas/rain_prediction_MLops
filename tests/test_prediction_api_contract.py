import os

from fastapi.testclient import TestClient
import pytest

from src.prediction_api.main import MODEL_PATH, app


VALID_PAYLOAD = {
    "location": "Albury",
    "humidity_3pm": 50.0,
    "rain_today": "No",
    "wind_gust_speed": 40.0,
    "rainfall": 0.0,
    "pressure_3pm": 1015.0,
    "humidity_9am": 70.0,
}

ADMIN_USERNAME = os.getenv("FASTAPI_ADMIN_USER") or os.getenv("NGINX_ADMIN_USER")
USER_HEADERS = {"X-Forwarded-User": "ines"}


def get_admin_headers():
    if not ADMIN_USERNAME:
        pytest.skip("Set FASTAPI_ADMIN_USER or NGINX_ADMIN_USER to run admin endpoint contract checks.")
    return {"X-Forwarded-User": ADMIN_USERNAME}


@pytest.fixture(scope="module")
def client():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model artifact is not available locally: {MODEL_PATH}")
    with TestClient(app) as test_client:
        yield test_client


def test_health_loads_final_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["features_required"] > 0


def test_locations_contract(client):
    response = client.get("/locations")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 54
    assert "Albury" in body["locations"]


def test_prediction_requires_forwarded_user(client):
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 401


def test_prediction_contract_with_nginx_forwarded_user(client):
    response = client.post("/predict", json=VALID_PAYLOAD, headers=USER_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert body["rain_tomorrow"] in ["Yes", "No"]
    assert "confidence" in body
    assert "timestamp" in body


def test_admin_model_info_contract(client):
    response = client.get("/model/info", headers=get_admin_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["features_count"] > 0
    assert body["model_path"].replace("\\", "/") == "models/final_winner/winner_model.joblib"


def test_user_cannot_access_admin_model_info(client):
    response = client.get("/model/info", headers=USER_HEADERS)

    assert response.status_code == 403
