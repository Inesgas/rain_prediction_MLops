from __future__ import annotations

from fastapi.testclient import TestClient

from src.models.api import create_app
from src.models.inference import WeatherInferenceService


class FakeModel:
    def predict_proba(self, frame):
        probability = 0.7 if frame.iloc[0]["rain_today"] == "Yes" else 0.1
        return [[1.0 - probability, probability]]


def make_service() -> WeatherInferenceService:
    return WeatherInferenceService(
        artifact={
            "model": FakeModel(),
            "model_name": "api_test_model",
            "model_role": "unit test double",
            "features": ["humidity_3pm", "rain_today"],
            "threshold": 0.58,
            "metrics": {},
        }
    )


def test_api_health_and_predict_endpoints() -> None:
    client = TestClient(create_app(make_service()))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["service"] == "weather-winner-inference-api"
    assert health.json()["model_name"] == "api_test_model"

    model_info = client.get("/model-info")
    assert model_info.status_code == 200
    assert model_info.json()["service"] == "weather-winner-inference-api"
    assert model_info.json()["required_features"] == ["humidity_3pm", "rain_today"]

    prediction = client.post("/predict", json={"humidity_3pm": 70, "rain_today": "Yes"})
    assert prediction.status_code == 200
    assert prediction.json()["service"] == "weather-winner-inference-api"
    assert prediction.json()["predicted_label"] == "Rain"
    assert prediction.json()["probability_rain"] == 0.7
    assert "X-Request-ID" in prediction.headers


def test_api_returns_structured_payload_errors() -> None:
    client = TestClient(create_app(make_service()))

    response = client.post("/predict", json={"humidity_3pm": 70})

    assert response.status_code == 422
    error = response.json()
    assert error["status"] == "error"
    assert error["error"] == "invalid_payload"
    assert error["missing_features"] == ["rain_today"]


def test_api_rejects_oversized_request_body() -> None:
    client = TestClient(create_app(make_service()))

    response = client.post(
        "/predict",
        content=b"{}",
        headers={"content-length": "999999"},
    )

    assert response.status_code == 413
    assert response.json()["error"] == "request_too_large"
