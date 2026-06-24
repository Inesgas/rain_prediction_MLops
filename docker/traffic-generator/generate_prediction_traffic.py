from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


API_URL = os.environ.get("PREDICTION_API_URL", "http://fastapi:8502/predict")
BATCH_API_URL = os.environ.get("PREDICTION_BATCH_API_URL", "http://fastapi:8502/predict/batch")
HEALTH_URL = os.environ.get("PREDICTION_HEALTH_URL", "http://fastapi:8502/health")
LOCATIONS_URL = os.environ.get("PREDICTION_LOCATIONS_URL", "http://fastapi:8502/locations")
INTERVAL_SECONDS = float(os.environ.get("PREDICTION_TRAFFIC_INTERVAL_SECONDS", "60"))
FORWARDED_USER = os.environ.get("PREDICTION_TRAFFIC_USER", "ines")


FALLBACK_LOCATIONS = ["Albury", "Sydney", "Melbourne", "Brisbane"]


def build_sample(location: str, index: int) -> dict:
    humid_3pm = 25 + (index * 7) % 65
    rain_today = "Yes" if index % 3 == 0 else "No"
    rainfall = round(((index * 1.7) % 12), 1) if rain_today == "Yes" else round((index % 4) * 0.2, 1)
    return {
        "location": location,
        "humidity_3pm": float(humid_3pm),
        "rain_today": rain_today,
        "wind_gust_speed": float(25 + (index * 3) % 45),
        "rainfall": float(rainfall),
        "pressure_3pm": float(995 + (index * 1.9) % 35),
        "humidity_9am": float(min(100, humid_3pm + 12)),
    }


def request_json(url: str, payload: dict | None = None) -> tuple[int, str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Forwarded-User": FORWARDED_USER,
        },
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, response.read().decode("utf-8")


def wait_for_api() -> None:
    while True:
        try:
            status, _ = request_json(HEALTH_URL)
            if status == 200:
                print("FastAPI health check passed. Starting prediction traffic.", flush=True)
                return
        except Exception as exc:
            print(f"Waiting for FastAPI: {exc}", flush=True)
        time.sleep(5)


def load_locations() -> list[str]:
    try:
        _, response = request_json(LOCATIONS_URL)
        payload = json.loads(response)
        locations = payload.get("locations") or []
        if locations:
            return sorted(str(location) for location in locations)
    except Exception as exc:
        print(f"Could not load locations from FastAPI, using fallback list: {exc}", flush=True)
    return FALLBACK_LOCATIONS


def main() -> None:
    wait_for_api()
    locations = load_locations()
    samples = [build_sample(location, index) for index, location in enumerate(locations)]
    batch_payload = {"samples": samples}
    print(f"Loaded {len(samples)} locations for automatic dashboard traffic.", flush=True)

    while True:
        try:
            status, response = request_json(BATCH_API_URL, batch_payload)
            print(
                f"prediction batch sent samples={len(samples)} status={status} response={response}",
                flush=True,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            print(f"batch prediction failed status={exc.code} detail={detail}", flush=True)
            for payload in samples:
                try:
                    status, response = request_json(API_URL, payload)
                    print(
                        f"prediction sent location={payload['location']} status={status} response={response}",
                        flush=True,
                    )
                except Exception as item_exc:
                    print(
                        f"prediction failed location={payload['location']} error={item_exc}",
                        flush=True,
                    )
        except Exception as exc:
            print(f"batch prediction failed error={exc}", flush=True)
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
