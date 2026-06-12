from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen

from src.config.paths import FINAL_WINNER_SAMPLE_INPUT_PATH


def _read_json_response(url: str) -> dict:
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the running prediction API.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    health = _read_json_response(f"{args.base_url}/health")
    model_info = _read_json_response(f"{args.base_url}/model-info")
    sample_payload = FINAL_WINNER_SAMPLE_INPUT_PATH.read_text(encoding="utf-8")
    request = Request(
        f"{args.base_url}/predict",
        data=sample_payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        prediction = json.loads(response.read().decode("utf-8"))

    print(
        json.dumps(
            {
                "health": health,
                "model": {
                    "model_name": model_info["model_name"],
                    "feature_count": model_info["feature_count"],
                    "threshold": model_info["threshold"],
                },
                "prediction": prediction,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

