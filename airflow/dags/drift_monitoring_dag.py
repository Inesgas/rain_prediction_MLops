from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator


PROJECT_DIR = os.environ.get("AIRFLOW_PROJECT_DIR", "/opt/airflow/project")
REFERENCE_DATASET_TARGET = "data/monitoring/reference_dataset.csv"


def project_command(command: str) -> str:
    return (
        f"cd {PROJECT_DIR} && "
        f"export PYTHONPATH={PROJECT_DIR}:$PYTHONPATH && "
        f"{command}"
    )


def drift_monitoring_schedule() -> str | None:
    value = os.environ.get("DRIFT_MONITORING_SCHEDULE", "0 6 * * *").strip()
    return value or None


default_args = {
    "owner": "rain-prediction-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


with DAG(
    dag_id="drift_monitoring",
    description="Daily Evidently data drift check between the training reference dataset and the recent input window.",
    start_date=datetime(2026, 1, 1),
    schedule=drift_monitoring_schedule(),
    catchup=False,
    default_args=default_args,
    tags=["monitoring", "evidently", "drift", "mlflow"],
) as dag:
    start = EmptyOperator(task_id="start")

    verify_reference_dataset = BashOperator(
        task_id="verify_reference_dataset_present",
        bash_command=project_command(
            f"python -m src.versioning.dvc_versioning verify-local --target {REFERENCE_DATASET_TARGET}"
        ),
    )

    run_drift_report = BashOperator(
        task_id="run_drift_report",
        bash_command=project_command(
            "python -m src.monitoring.drift_report "
            "--days-back ${DRIFT_MONITORING_DAYS_BACK:-365} "
            # change -14 to -365 to span a full seasonal cycle, matching the reference dataset's year-round composition. A short window (e.g. 14 days) picks up one season only and looks like drift even when nothing is actually wrong — see reports/monitoring notes from 2026-07-02.
            "--log-to-mlflow"
        ),
        execution_timeout=timedelta(minutes=30),
    )

    local_only_notice = BashOperator(
        task_id="local_only_no_remote_push",
        bash_command="echo 'Drift monitoring run complete. No GitHub or DVC remote push was performed.'",
    )

    end = EmptyOperator(task_id="end")

    (
        start
        >> verify_reference_dataset
        >> run_drift_report
        >> local_only_notice
        >> end
    )