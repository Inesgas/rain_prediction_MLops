from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator


PROJECT_DIR = os.environ.get("AIRFLOW_PROJECT_DIR", "/opt/airflow/project")
RAW_DATA_TARGET = "data/raw/weatherAUS.csv"


def project_command(command: str) -> str:
    return (
        f"cd {PROJECT_DIR} && "
        f"export PYTHONPATH={PROJECT_DIR}:$PYTHONPATH && "
        f"{command}"
    )


def optional_schedule() -> str | None:
    value = os.environ.get("MLOPS_E2E_SCHEDULE", "").strip()
    return value or None


default_args = {
    "owner": "rain-prediction-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


with DAG(
    dag_id="end_to_end_mlops_pipeline",
    description="Complete local MLOps orchestration: extract, version, train, validate, and snapshot.",
    start_date=datetime(2026, 1, 1),
    schedule=optional_schedule(),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "end-to-end", "airflow", "dvc", "fastapi"],
) as dag:
    start = EmptyOperator(task_id="start")

    extract_raw_weather_data = BashOperator(
        task_id="extract_raw_weather_data",
        bash_command=project_command(
            "python -m src.data.extract_weather_data "
            "--daily-provider ${WEATHER_AUS_DAILY_PROVIDER:-open-meteo} "
            "--mode ${WEATHER_AUS_EXTRACT_MODE:-upsert} "
            "--source-min-rows ${WEATHER_AUS_SOURCE_MIN_ROWS:-1} "
            "--run-id '{{ run_id }}'"
        ),
        execution_timeout=timedelta(hours=1),
    )

    version_raw_weather_data = BashOperator(
        task_id="version_raw_weather_data",
        bash_command=project_command(
            f"python -m src.versioning.dvc_versioning dvc-add --target {RAW_DATA_TARGET}"
        ),
    )

    verify_local_versioned_inputs = BashOperator(
        task_id="verify_local_versioned_inputs",
        bash_command=project_command("python -m src.versioning.dvc_versioning verify-local"),
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

    snapshot_input_versions = BashOperator(
        task_id="snapshot_input_versions",
        bash_command=project_command(
            "python -m src.versioning.dvc_versioning snapshot "
            "--phase inputs --run-id '{{ run_id }}'"
        ),
    )

    train_winner_model = BashOperator(
        task_id="train_winner_model",
        bash_command=project_command("python -m src.models.train_winner"),
        execution_timeout=timedelta(hours=2),
    )

    version_model_artifact = BashOperator(
        task_id="version_model_artifact",
        bash_command=project_command("python -m src.versioning.dvc_versioning dvc-add-model"),
    )

    snapshot_output_versions = BashOperator(
        task_id="snapshot_output_versions",
        bash_command=project_command(
            "python -m src.versioning.dvc_versioning snapshot "
            "--phase outputs --run-id '{{ run_id }}'"
        ),
    )

    validate_official_fastapi_contract = BashOperator(
        task_id="validate_official_fastapi_contract",
        bash_command=project_command("python -m pytest tests/test_prediction_api_contract.py"),
        execution_timeout=timedelta(minutes=30),
    )

    record_dvc_status = BashOperator(
        task_id="record_dvc_status",
        bash_command=project_command(
            "python -m src.versioning.dvc_versioning dvc-status --run-id '{{ run_id }}'"
        ),
    )

    local_only_notice = BashOperator(
        task_id="local_only_no_remote_push",
        bash_command="echo 'End-to-end pipeline complete. No GitHub or DagsHub push was performed.'",
    )

    end = EmptyOperator(task_id="end")

    (
        start
        >> extract_raw_weather_data
        >> version_raw_weather_data
        >> verify_local_versioned_inputs
        >> check_data_freshness
        >> snapshot_input_versions
        >> train_winner_model
        >> version_model_artifact
        >> snapshot_output_versions
        >> validate_official_fastapi_contract
        >> record_dvc_status
        >> local_only_notice
        >> end
    )
