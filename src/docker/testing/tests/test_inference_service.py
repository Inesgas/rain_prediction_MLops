from __future__ import annotations

import json

import pytest

from src.config.paths import FINAL_WINNER_MODEL_ARTIFACT, FINAL_WINNER_SAMPLE_INPUT_PATH
from src.models.inference import PayloadValidationError, WeatherInferenceService


class FakeModel:
    def predict_proba(self, frame):
        humidity = float(frame.iloc[0]["humidity_3pm"])
        probability = 0.8 if humidity >= 60 else 0.2
        return [[1.0 - probability, probability]]


def make_service() -> WeatherInferenceService:
    return WeatherInferenceService(
        artifact={
            "model": FakeModel(),
            "model_name": "test_model",
            "model_role": "unit test double",
            "features": ["humidity_3pm", "rain_today"],
            "threshold": 0.58,
            "metrics": {"f1": 0.5},
        }
    )


def test_predict_one_returns_rain_when_probability_crosses_threshold() -> None:
    service = make_service()

    prediction = service.predict_one({"humidity_3pm": 75, "rain_today": "Yes"})

    assert prediction["predicted_label"] == "Rain"
    assert prediction["probability_rain"] == pytest.approx(0.8)
    assert prediction["threshold"] == pytest.approx(0.58)


def test_predict_one_rejects_missing_required_features() -> None:
    service = make_service()

    with pytest.raises(PayloadValidationError, match="Payload is missing required features") as exc_info:
        service.predict_one({"humidity_3pm": 75})

    assert exc_info.value.missing_features == ["rain_today"]


def test_model_info_exposes_serving_contract() -> None:
    service = make_service()

    info = service.model_info()

    assert info["model_name"] == "test_model"
    assert info["feature_count"] == 2
    assert info["features"] == ["humidity_3pm", "rain_today"]
    assert info["required_features"] == ["humidity_3pm", "rain_today"]
    assert info["positive_class"] == "Rain"
    assert info["negative_class"] == "No rain"


@pytest.mark.skipif(not FINAL_WINNER_MODEL_ARTIFACT.exists(), reason="DVC model artifact is not present")
def test_default_service_loads_final_winner_artifact() -> None:
    service = WeatherInferenceService()
    payload = json.loads(FINAL_WINNER_SAMPLE_INPUT_PATH.read_text(encoding="utf-8"))

    prediction = service.predict_one(payload)

    assert service.model_name == "final_hybrid_catboost"
    assert service.model_family == "CatBoost binary classification"
    assert len(service.features) == 68
    assert prediction["model_name"] == "final_hybrid_catboost"
    assert 0.0 <= prediction["probability_rain"] <= 1.0
