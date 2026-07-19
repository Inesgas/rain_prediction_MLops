from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from src.config.paths import BASE_DIR, RAW_BASE

DEFAULT_API_URL = os.environ.get("RAIN_PREDICTION_API_URL", "http://rain-prediction-api:8502")
DEFAULT_API_USER = os.environ.get("RAIN_PREDICTION_API_USER", "admin")
DEFAULT_LOG_DIR = BASE_DIR / "data" / "monitoring" / "model_performance"

# Maps the /predict payload fields to the corresponding weatherAUS.csv columns.
FEATURE_COLUMN_MAP = {
    "humidity_3pm": "Humidity3pm",
    "rain_today": "RainToday",
    "wind_gust_speed": "WindGustSpeed",
    "rainfall": "Rainfall",
    "pressure_3pm": "Pressure3pm",
    "humidity_9am": "Humidity9am",
}


def parse_date(value: str | None) -> date:
    if value:
        return date.fromisoformat(value)
    # daily_weather_ingestion uses end_lag_days=1, so the newest row in
    # weatherAUS.csv is always yesterday's date, not today's. To have both
    # the target day's features AND its following day's actual outcome
    # already present, we go back 2 days instead of 1.
    return date.today() - timedelta(days=2)


def build_payload(row: pd.Series) -> dict:
    payload = {"location": row["Location"]}
    for api_field, csv_column in FEATURE_COLUMN_MAP.items():
        value = row[csv_column]
        if api_field == "rain_today":
            payload[api_field] = "Yes" if str(value).strip().lower() in ("yes", "1", "1.0") else "No"
        else:
            payload[api_field] = float(value)
    return payload


def call_predict(session: requests.Session, api_url: str, api_user: str, payload: dict) -> dict:
    # X-Forwarded-User mirrors the header Nginx normally injects for
    # authenticated users. The API only checks that the name exists in
    # its USERS dict, so no password is needed for this internal,
    # in-cluster call.
    response = session.post(
        f"{api_url}/predict",
        json=payload,
        headers={"X-Forwarded-User": api_user},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def run_backfill(target_date: date, api_url: str, api_user: str, log_dir: Path, raw_path: Path) -> Path:
    df = pd.read_csv(raw_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    day_rows = df[df["Date"] == target_date].dropna(subset=list(FEATURE_COLUMN_MAP.values()))

    if day_rows.empty:
        raise RuntimeError(
            f"No usable rows found in {raw_path} for date {target_date.isoformat()}. "
            "Make sure daily_weather_ingestion has already ingested this date."
        )

    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"backfill_predictions_{target_date.isoformat()}.jsonl"

    session = requests.Session()
    written = 0
    failures: list[dict] = []

    with out_path.open("w", encoding="utf-8") as f:
        for _, row in day_rows.iterrows():
            payload = build_payload(row)
            try:
                result = call_predict(session, api_url, api_user, payload)
            except Exception as exc:
                failures.append({"location": row["Location"], "error": str(exc)})
                continue

            record = {
                "for_date": target_date.isoformat(),
                "location": row["Location"],
                "predicted_rain_tomorrow": result["rain_tomorrow"],
                "predicted_value": result["prediction"],
                "confidence": result.get("confidence"),
                "requested_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    print(f"Backfilled {written} prediction(s) for {target_date.isoformat()} -> {out_path}")
    if failures:
        print(f"Warning: {len(failures)} location(s) failed: {failures[:5]}")
    if written == 0:
        raise RuntimeError(f"No predictions were successfully backfilled for {target_date.isoformat()}.")

    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a past day's weather rows through the /predict API.")
    parser.add_argument("--date", default=None, help="Target date (YYYY-MM-DD). Defaults to yesterday.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-user", default=DEFAULT_API_USER)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--raw-path", type=Path, default=RAW_BASE)
    # Accepted for symmetry with other Airflow tasks in this project; not used directly.
    parser.add_argument("--run-id", default=os.environ.get("AIRFLOW_CTX_RUN_ID"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target_date = parse_date(args.date)
    run_backfill(target_date, args.api_url, args.api_user, args.log_dir, args.raw_path)


if __name__ == "__main__":
    main()