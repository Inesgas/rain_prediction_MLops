from __future__ import annotations

import argparse
import os
import time
import uuid
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.models.inference import InferenceError, PayloadValidationError, WeatherInferenceService


API_NAME = "weather-winner-inference-api"
API_VERSION = "1.2.0"
DEFAULT_CORS_ORIGINS = "http://localhost:8501,http://127.0.0.1:8501"
MAX_REQUEST_BYTES = int(os.environ.get("PREDICTION_API_MAX_REQUEST_BYTES", "65536"))


def _cors_origins() -> list[str]:
    raw = os.environ.get("PREDICTION_API_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or DEFAULT_CORS_ORIGINS.split(",")


def _error_payload(error: str, message: str, **details: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "error",
        "error": error,
        "message": message,
    }
    payload.update(details)
    return payload


def _get_service(app: FastAPI) -> WeatherInferenceService:
    service = getattr(app.state, "service", None)
    if service is None:
        service = WeatherInferenceService()
        app.state.service = service
    return service


def create_app(service: WeatherInferenceService | None = None) -> FastAPI:
    app = FastAPI(
        title="Rain Prediction API",
        description="FastAPI service for the final rain prediction model.",
        version=API_VERSION,
        docs_url=os.environ.get("PREDICTION_API_DOCS_URL", "/docs"),
        redoc_url=None,
    )
    app.state.service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_guard(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_REQUEST_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content=_error_payload(
                            "request_too_large",
                            f"Request body must be at most {MAX_REQUEST_BYTES} bytes.",
                        ),
                        headers={"X-Request-ID": request_id},
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content=_error_payload("invalid_content_length", "Content-Length must be an integer."),
                    headers={"X-Request-ID": request_id},
                )

        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
        return response

    @app.exception_handler(PayloadValidationError)
    async def payload_validation_handler(_: Request, exc: PayloadValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                "invalid_payload",
                str(exc),
                missing_features=exc.missing_features,
            ),
        )

    @app.exception_handler(InferenceError)
    async def inference_error_handler(_: Request, exc: InferenceError) -> JSONResponse:
        return JSONResponse(status_code=500, content=_error_payload("inference_error", str(exc)))

    @app.get("/health")
    def health() -> dict[str, Any]:
        service_obj = _get_service(app)
        return {
            "status": "ok",
            "service": API_NAME,
            "api_version": API_VERSION,
            "model_name": service_obj.model_name,
        }

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        service_obj = _get_service(app)
        return {
            "status": "ready",
            "service": API_NAME,
            "api_version": API_VERSION,
            "model_name": service_obj.model_name,
            "feature_count": len(service_obj.features),
        }

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        service_obj = _get_service(app)
        payload = {"service": API_NAME, "api_version": API_VERSION}
        payload.update(service_obj.model_info())
        return payload

    @app.post("/predict")
    def predict(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=422,
                detail=_error_payload("invalid_payload", "Payload must be a JSON object."),
            )
        service_obj = _get_service(app)
        prediction = service_obj.predict_one(payload)
        return {"service": API_NAME, "api_version": API_VERSION, **prediction}

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Weather prediction FastAPI service.")
    parser.add_argument("--host", default=os.environ.get("PREDICTION_API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PREDICTION_API_PORT", "8000")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("PREDICTION_API_WORKERS", "1")))
    args = parser.parse_args()

    uvicorn.run(
        "src.models.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
