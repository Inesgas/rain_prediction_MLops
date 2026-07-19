from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = os.environ.get("AIRFLOW_PROJECT_DIR", "/opt/airflow/project")


def project_command(command: str) -> str:
    return (
        f"cd {PROJECT_DIR} && "
        f"export PYTHONPATH={PROJECT_DIR}:$PYTHONPATH && "
        f"{command}"
    )


default_args = {
    "owner": "andrey-model-monitoring",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="model_performance_and_drift",
    description=(
        "Replays yesterday's weatherAUS.csv rows through /predict and compares "
        "them to the actual RainTomorrow outcome, pushing RMSE/MAE/R2 to Pushgateway "
        "for the 'Model Performance & Drift' Grafana dashboard."
    ),
    start_date=datetime(2026, 1, 1),
    # Runs after daily_weather_ingestion (03:00) so yesterday's row is present.
    schedule="30 5 * * *",
    catchup=False,
    default_args=default_args,
    tags=["model-monitoring", "grafana", "pushgateway"],
) as dag:
    backfill_predictions = BashOperator(
        task_id="backfill_predictions_for_yesterday",
        bash_command=project_command(
            "python -m src.model_monitoring.backfill_predictions --run-id '{{ run_id }}'"
        ),
        execution_timeout=timedelta(minutes=20),
    )

    evaluate_and_push_metrics = BashOperator(
        task_id="evaluate_and_push_metrics",
        bash_command=project_command(
            "python -m src.model_monitoring.evaluate_model_performance"
        ),
        execution_timeout=timedelta(minutes=10),
    )

    backfill_predictions >> evaluate_and_push_metrics