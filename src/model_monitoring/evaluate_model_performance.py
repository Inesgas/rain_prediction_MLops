from __future__ import annotations

import argparse
import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from src.config.paths import BASE_DIR, RAW_BASE

DEFAULT_LOG_DIR = BASE_DIR / "data" / "monitoring" / "model_performance"
DEFAULT_PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "pushgateway:9091")
# Own job name so this never collides with the drift_monitoring DAG's
# Pushgateway job (which reports rain_dataset_drift_detected).
PUSHGATEWAY_JOB = "model_performance_and_drift"


def parse_date(value: str | None) -> date:
    if value:
        return date.fromisoformat(value)
    # daily_weather_ingestion uses end_lag_days=1, so the newest row in
    # weatherAUS.csv is always yesterday's date, not today's. To have both
    # the target day's features AND its following day's actual outcome
    # already present, we go back 2 days instead of 1.
    return date.today() - timedelta(days=2)


def load_predictions(log_dir: Path, target_date: date) -> pd.DataFrame:
    path = log_dir / f"backfill_predictions_{target_date.isoformat()}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No backfilled predictions found for {target_date.isoformat()}: {path}. "
            "Run backfill_predictions.py for this date first."
        )
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(records)


# Matches RAIN_THRESHOLD_MM in src/data/extract_open_meteo_daily.py, so the
# "actual" label here is computed the same way the daily ingestion labels
# RainToday/RainTomorrow for freshly-fetched rows.
RAIN_THRESHOLD_MM = 1.0


def load_actuals(raw_path: Path, target_date: date) -> pd.DataFrame:
    # weatherAUS.csv's own RainTomorrow column is computed at ingestion time
    # via a per-batch shift(-1) on Rainfall (see extract_open_meteo_daily.py).
    # For rows added by the daily upsert, the following day usually isn't in
    # the same batch yet, so RainTomorrow is left as NaN and never backfilled
    # once the next day's row does arrive. To get a reliable ground truth we
    # instead look up the *next* day's own Rainfall value directly and derive
    # RainTomorrow ourselves, the same way the ingestion script does.
    df = pd.read_csv(raw_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    next_day = target_date + timedelta(days=1)
    next_day_rows = df[df["Date"] == next_day][["Location", "Rainfall"]].dropna()

    if next_day_rows.empty:
        raise RuntimeError(
            f"No rows found for {next_day.isoformat()} (the day after {target_date.isoformat()}) "
            f"in {raw_path}. Need the following day's data to know whether it actually rained."
        )

    next_day_rows = next_day_rows.rename(columns={"Location": "location"})
    next_day_rows["actual_rain_tomorrow"] = next_day_rows["Rainfall"].apply(
        lambda mm: "Yes" if float(mm) > RAIN_THRESHOLD_MM else "No"
    )
    return next_day_rows[["location", "actual_rain_tomorrow"]]


def to_binary(value: str) -> int:
    return 1 if str(value).strip().lower() == "yes" else 0


def compute_metrics(merged: pd.DataFrame) -> dict:
    y_true = merged["actual_rain_tomorrow"].map(to_binary).to_numpy()
    y_pred = merged["predicted_rain_tomorrow"].map(to_binary).to_numpy()

    errors = y_true - y_pred
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"rmse": rmse, "mae": mae, "r2": r2, "n_samples": int(len(merged))}


def push_metrics(metrics: dict, pushgateway_url: str) -> None:
    registry = CollectorRegistry()
    Gauge("model_rmse_score", "Model RMSE against actual next-day rainfall", registry=registry).set(metrics["rmse"])
    Gauge("model_mae_score", "Model MAE against actual next-day rainfall", registry=registry).set(metrics["mae"])
    r2_value = metrics["r2"]
    if r2_value == r2_value:  # skip pushing NaN (single-class day, ss_tot == 0)
        Gauge("model_r2_score", "Model R2 against actual next-day rainfall", registry=registry).set(r2_value)
    push_to_gateway(pushgateway_url, job=PUSHGATEWAY_JOB, registry=registry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare backfilled predictions to actual outcomes and push RMSE/MAE/R2 to Pushgateway."
    )
    parser.add_argument("--date", default=None, help="Target date (YYYY-MM-DD). Defaults to yesterday.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--raw-path", type=Path, default=RAW_BASE)
    parser.add_argument("--pushgateway-url", default=DEFAULT_PUSHGATEWAY_URL)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target_date = parse_date(args.date)

    predictions = load_predictions(args.log_dir, target_date)
    actuals = load_actuals(args.raw_path, target_date)

    merged = predictions.merge(actuals, on="location", how="inner")
    if merged.empty:
        raise RuntimeError(
            f"No overlapping locations between predictions and actual outcomes for {target_date.isoformat()}."
        )

    metrics = compute_metrics(merged)
    print(f"Metrics for {target_date.isoformat()}: {metrics}")

    push_metrics(metrics, args.pushgateway_url)
    print(f"Pushed metrics to Pushgateway ({args.pushgateway_url}, job={PUSHGATEWAY_JOB})")


if __name__ == "__main__":
    main()
