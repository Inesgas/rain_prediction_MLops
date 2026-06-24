from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator


PROJECT_DIR = os.environ.get("AIRFLOW_PROJECT_DIR", "/opt/airflow/project")


def project_command(command: str) -> str:
    return (
        f"cd {PROJECT_DIR} && "
        f"export PYTHONPATH={PROJECT_DIR}:$PYTHONPATH && "
        f"{command}"
    )


default_args = {
    "owner": "rain-prediction-mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=15),
}


with DAG(
    dag_id="daily_weather_ingestion",
    description="Daily local-only Open-Meteo ingestion into the WeatherAUS raw schema.",
    start_date=datetime(2026, 1, 1),
    schedule="0 3 * * *",
    catchup=False,
    default_args=default_args,
    tags=["ingestion", "daily", "open-meteo", "dvc"],
) as dag:
    start = EmptyOperator(task_id="start")

    fetch_and_upsert_daily_weather = BashOperator(
        task_id="fetch_and_upsert_daily_weather",
        bash_command=project_command(
            "python -m src.data.extract_weather_data "
            "--daily-provider open-meteo "
            "--mode ${WEATHER_AUS_EXTRACT_MODE:-upsert} "
            "--source-min-rows ${WEATHER_AUS_SOURCE_MIN_ROWS:-1} "
            "--run-id '{{ run_id }}'"
        ),
        execution_timeout=timedelta(hours=1),
    )

    version_raw_weather_data = BashOperator(
        task_id="version_raw_weather_data",
        bash_command=project_command(
            "python -m src.versioning.dvc_versioning dvc-add --target data/raw/weatherAUS.csv"
        ),
    )

    check_data_freshness = BashOperator(
        task_id="check_data_freshness",
        bash_command=project_command(
            "python -m src.data.check_data_freshness "
            "--max-lag-days ${WEATHER_AUS_MAX_LAG_DAYS:-2} "
            "--warn-only "
            "--run-id '{{ run_id }}'"
        ),
    )

    snapshot_daily_ingest = BashOperator(
        task_id="snapshot_daily_ingest",
        bash_command=project_command(
            "python -m src.versioning.dvc_versioning snapshot "
            "--phase inputs --run-id '{{ run_id }}'"
        ),
    )

    dvc_status = BashOperator(
        task_id="record_dvc_status",
        bash_command=project_command(
            "python -m src.versioning.dvc_versioning dvc-status --run-id '{{ run_id }}'"
        ),
    )

    local_only_notice = BashOperator(
        task_id="local_only_no_remote_push",
        bash_command="echo 'Daily ingest complete. No GitHub or DagsHub push was performed.'",
    )

    end = EmptyOperator(task_id="end")

    (
        start
        >> fetch_and_upsert_daily_weather
        >> version_raw_weather_data
        >> check_data_freshness
        >> snapshot_daily_ingest
        >> dvc_status
        >> local_only_notice
        >> end
    )
