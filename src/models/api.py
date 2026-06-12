from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from src.models.inference import InferenceError, PayloadValidationError, WeatherInferenceService


API_NAME = "weather-winner-inference-api"
API_VERSION = "1.1.0"


def _json_response(handler: BaseHTTPRequestHandler, status_code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _error_response(
    handler: BaseHTTPRequestHandler,
    status_code: int,
    error: str,
    message: str,
    **details: Any,
) -> None:
    payload: dict[str, Any] = {
        "status": "error",
        "error": error,
        "message": message,
    }
    payload.update(details)
    _json_response(handler, status_code, payload)


def make_handler(service: WeatherInferenceService) -> type[BaseHTTPRequestHandler]:
    class WeatherPredictionAPIHandler(BaseHTTPRequestHandler):
        server_version = API_NAME

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                _json_response(
                    self,
                    200,
                    {
                        "status": "ok",
                        "service": API_NAME,
                        "api_version": API_VERSION,
                        "model_name": service.model_name,
                    },
                )
                return
            if path == "/model-info":
                payload = {"service": API_NAME, "api_version": API_VERSION}
                payload.update(service.model_info())
                _json_response(self, 200, payload)
                return
            _error_response(self, 404, "not_found", "Endpoint not found.")

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/predict":
                _error_response(self, 404, "not_found", "Endpoint not found.")
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                _error_response(self, 400, "invalid_content_length", "Content-Length must be an integer.")
                return

            try:
                raw_body = self.rfile.read(content_length)
                payload = json.loads(raw_body.decode("utf-8"))
                prediction = service.predict_one(payload)
            except json.JSONDecodeError:
                _error_response(self, 400, "invalid_json", "Request body must be valid JSON.")
                return
            except PayloadValidationError as exc:
                _error_response(
                    self,
                    422,
                    "invalid_payload",
                    str(exc),
                    missing_features=exc.missing_features,
                )
                return
            except InferenceError as exc:
                _error_response(self, 500, "inference_error", str(exc))
                return

            _json_response(self, 200, {"service": API_NAME, "api_version": API_VERSION, **prediction})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return WeatherPredictionAPIHandler


def create_server(host: str, port: int, service: WeatherInferenceService | None = None) -> ThreadingHTTPServer:
    active_service = service if service is not None else WeatherInferenceService()
    return ThreadingHTTPServer((host, port), make_handler(active_service))


def main() -> None:
    parser = argparse.ArgumentParser(description="Weather prediction API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = create_server(args.host, args.port)
    print(f"Weather prediction API running on http://{args.host}:{args.port}")
    print("Endpoints: GET /health, GET /model-info, POST /predict")
    server.serve_forever()


if __name__ == "__main__":
    main()
