from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.models.api import create_server
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
    server = create_server("127.0.0.1", 0, make_service())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        with urlopen(f"{base_url}/health", timeout=5) as response:
            health = json.loads(response.read().decode("utf-8"))
        assert health["status"] == "ok"
        assert health["service"] == "weather-winner-inference-api"
        assert health["model_name"] == "api_test_model"

        with urlopen(f"{base_url}/model-info", timeout=5) as response:
            model_info = json.loads(response.read().decode("utf-8"))
        assert model_info["service"] == "weather-winner-inference-api"
        assert model_info["required_features"] == ["humidity_3pm", "rain_today"]

        request = Request(
            f"{base_url}/predict",
            data=json.dumps({"humidity_3pm": 70, "rain_today": "Yes"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            prediction = json.loads(response.read().decode("utf-8"))
        assert prediction["service"] == "weather-winner-inference-api"
        assert prediction["predicted_label"] == "Rain"
        assert prediction["probability_rain"] == 0.7
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_api_returns_structured_payload_errors() -> None:
    server = create_server("127.0.0.1", 0, make_service())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        request = Request(
            f"{base_url}/predict",
            data=json.dumps({"humidity_3pm": 70}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(request, timeout=5)
            raise AssertionError("Expected HTTP 422 for missing features")
        except HTTPError as exc:
            assert exc.code == 422
            error = json.loads(exc.read().decode("utf-8"))

        assert error["status"] == "error"
        assert error["error"] == "invalid_payload"
        assert error["missing_features"] == ["rain_today"]
    finally:
        server.shutdown()
        thread.join(timeout=5)
