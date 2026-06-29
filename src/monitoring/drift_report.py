from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config.paths import BASE_DIR, RAIN_MODEL_DATASET_ALIGNED, REFERENCE_DATASET_PATH, DATE_COLUMN

REPORT_DIR = BASE_DIR / "reports" / "monitoring"

MONITORED_COLUMNS = [
    "humidity_3pm", "humidity_9am",
    "pressure_3pm", "pressure_9am",
    "temp_3pm", "temp_9am",
    "wind_gust_speed",
]


def safe_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_reference() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_DATASET_PATH)


def load_current_window(days_back: int) -> pd.DataFrame:
    df = pd.read_csv(RAIN_MODEL_DATASET_ALIGNED, parse_dates=[DATE_COLUMN])
    cutoff = df[DATE_COLUMN].max() - pd.Timedelta(days=days_back)
    return df[df[DATE_COLUMN] > cutoff]


def build_snapshot(reference: pd.DataFrame, current: pd.DataFrame):
    columns = [c for c in MONITORED_COLUMNS if c in reference.columns and c in current.columns]
    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=current[columns], reference_data=reference[columns])
    return snapshot


def extract_summary(snapshot) -> dict:
    result = snapshot.dict()
    summary = {"per_column": {}}
    for metric in result["metrics"]:
        config = metric.get("config", {})
        metric_type = config.get("type", "")
        if metric_type == "evidently:metric_v2:DriftedColumnsCount":
            summary["number_of_drifted_columns"] = metric["value"]["count"]
            summary["share_of_drifted_columns"] = metric["value"]["share"]
            summary["dataset_drift"] = metric["value"]["share"] >= config.get("drift_share", 0.5)
        elif metric_type == "evidently:metric_v2:ValueDrift":
            summary["per_column"][config["column"]] = metric["value"]
    return summary

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-back", type=int, default=14)
    parser.add_argument("--log-to-mlflow", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reference = load_reference()
    current = load_current_window(args.days_back)

    snapshot = build_snapshot(reference, current)
    timestamp = safe_timestamp()
    html_path = REPORT_DIR / f"drift_{timestamp}.html"
    summary_path = REPORT_DIR / f"drift_{timestamp}_summary.json"

    snapshot.save_html(str(html_path))
    summary = extract_summary(snapshot)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))

    if args.log_to_mlflow:
        import mlflow

        mlflow.set_experiment("rain_prediction_monitoring")
        if mlflow.active_run() is not None:
            mlflow.end_run()
        with mlflow.start_run(run_name=f"drift_check_{timestamp}"):
            mlflow.log_artifact(str(html_path))


if __name__ == "__main__":
    main()