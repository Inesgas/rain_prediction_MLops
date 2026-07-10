from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
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
    return pd.read_csv(REFERENCE_DATASET_PATH, low_memory=False)


def load_current_window(days_back: int) -> pd.DataFrame:
    df = pd.read_csv(RAIN_MODEL_DATASET_ALIGNED, parse_dates=[DATE_COLUMN])
    cutoff = df[DATE_COLUMN].max() - pd.Timedelta(days=days_back)
    return df[df[DATE_COLUMN] > cutoff]

def build_snapshot(reference: pd.DataFrame, current: pd.DataFrame):
    columns = [c for c in MONITORED_COLUMNS if c in reference.columns and c in current.columns]
    if not columns:
        raise ValueError(
            "None of the monitored columns "
            f"{MONITORED_COLUMNS} are present in both the reference and current datasets."
        )
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


def push_summary_to_gateway(summary: dict) -> None:
    """
    Pushes drift summary metrics to the Prometheus Pushgateway so that
    Grafana can alert on drift results from this batch job.
    """
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    gateway_url = os.environ.get("PUSHGATEWAY_URL", "pushgateway:9091")
    registry = CollectorRegistry()

    dataset_drift_gauge = Gauge(
        "rain_dataset_drift_detected",
        "Whether dataset-level drift was detected in the latest check (1=drift, 0=no drift)",
        registry=registry,
    )
    drifted_columns_count_gauge = Gauge(
        "rain_drifted_columns_count",
        "Number of columns flagged as drifted in the latest check",
        registry=registry,
    )
    drifted_columns_share_gauge = Gauge(
        "rain_drifted_columns_share",
        "Share of columns flagged as drifted in the latest check",
        registry=registry,
    )
    last_run_timestamp_gauge = Gauge(
        "rain_drift_check_last_run_timestamp_seconds",
        "Unix timestamp of the last completed drift check",
        registry=registry,
    )

    dataset_drift_gauge.set(1 if summary.get("dataset_drift") else 0)
    drifted_columns_count_gauge.set(summary.get("number_of_drifted_columns", 0))
    drifted_columns_share_gauge.set(summary.get("share_of_drifted_columns", 0.0))
    last_run_timestamp_gauge.set(datetime.now(timezone.utc).timestamp())

    push_to_gateway(gateway_url, job="drift_monitoring", registry=registry)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # 365 days by default so the current window spans a full seasonal cycle,
    # matching the reference dataset's year-round composition. A short window
    # (e.g. 14 days) picks up one season only and looks like drift even when
    # nothing is actually wrong — see reports/monitoring notes from 2026-07-02.
    parser.add_argument("--days-back", type=int, default=365)
    parser.add_argument("--log-to-mlflow", action="store_true")
    parser.add_argument("--push-to-gateway", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.days_back <= 0:
        parser.error("--days-back must be a positive integer")

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

    if args.push_to_gateway:
        push_summary_to_gateway(summary)


if __name__ == "__main__":
    main()
