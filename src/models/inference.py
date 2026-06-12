from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config.paths import FINAL_WINNER_MODEL_ARTIFACT


class InferenceError(RuntimeError):
    """Raised when a prediction cannot be produced."""


class PayloadValidationError(ValueError):
    """Raised when an inference payload does not match the model contract."""

    def __init__(self, message: str, missing_features: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing_features = missing_features or []


def load_artifact(path: Path = FINAL_WINNER_MODEL_ARTIFACT) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}.")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact or "features" not in artifact:
        raise InferenceError(f"Invalid model artifact: {path}")
    return artifact


class WeatherInferenceService:
    def __init__(self, artifact: dict[str, Any] | None = None, artifact_path: Path = FINAL_WINNER_MODEL_ARTIFACT) -> None:
        self.artifact = artifact if artifact is not None else load_artifact(artifact_path)
        self.model = self.artifact["model"]
        self.features = list(self.artifact["features"])
        self.categorical_features = list(self.artifact.get("categorical_features", []))
        self.numeric_features = list(self.artifact.get("numeric_features", []))
        self.numeric_fill_values = dict(self.artifact.get("numeric_fill_values", {}))
        self.threshold = float(self.artifact.get("threshold", 0.5))
        self.model_name = str(self.artifact.get("model_name", "weather_model"))
        self.model_role = str(self.artifact.get("model_role", "prediction_model"))
        self.model_family = str(self.artifact.get("model_family", "classification"))
        self.metrics = self.artifact.get("metrics", {})

    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_role": self.model_role,
            "model_family": self.model_family,
            "feature_count": len(self.features),
            "features": self.features,
            "required_features": self.features,
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
            "threshold": self.threshold,
            "metrics": self.metrics,
            "positive_class": "Rain",
            "negative_class": "No rain",
        }

    def validate_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise PayloadValidationError("Payload must be a JSON object.")
        missing = [feature for feature in self.features if feature not in payload]
        if missing:
            raise PayloadValidationError("Payload is missing required features.", missing_features=missing)
        return {feature: payload.get(feature) for feature in self.features}

    def _prepare_frame(self, features: dict[str, Any]) -> pd.DataFrame:
        frame = pd.DataFrame([features], columns=self.features)
        for column in self.numeric_features:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
                fill_value = self.numeric_fill_values.get(column)
                if fill_value is not None:
                    frame[column] = frame[column].fillna(fill_value)
        for column in self.categorical_features:
            if column in frame.columns:
                frame[column] = frame[column].fillna("Missing").astype(str)
        return frame

    def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        features = self.validate_payload(payload)
        frame = self._prepare_frame(features)
        try:
            probability_rain = float(self.model.predict_proba(frame)[0][1])
        except Exception as exc:  # pragma: no cover - defensive wrapper for model internals
            raise InferenceError(f"Model prediction failed: {exc}") from exc

        predicted_label = "Rain" if probability_rain >= self.threshold else "No rain"
        return {
            "model_name": self.model_name,
            "model_role": self.model_role,
            "model_family": self.model_family,
            "predicted_label": predicted_label,
            "probability_rain": probability_rain,
            "probability_no_rain": 1.0 - probability_rain,
            "threshold": self.threshold,
            "feature_count": len(self.features),
        }
