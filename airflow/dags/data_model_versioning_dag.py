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


def model_versioning_schedule() -> str | None:
    value = os.environ.get("MODEL_VERSIONING_SCHEDULE", "0 */6 * * *").strip()
    return value or None


default_args = {
    "owner": "rain-prediction-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="data_model_versioning",
    description="DVC-backed data/model versioning plus MLflow tracking for the rain prediction project.",
    start_date=datetime(2026, 1, 1),
    schedule=model_versioning_schedule(),
    catchup=False,
    default_args=default_args,
    tags=["versioning", "dvc", "mlflow", "model"],
) as dag:
    start = EmptyOperator(task_id="start")

    extract_raw_weather_data = BashOperator(
        task_id="extract_raw_weather_data",
        bash_command=project_command(
            "python -m src.data.extract_weather_data --run-id '{{ run_id }}'"
        ),
    )

    version_raw_weather_data = BashOperator(
        task_id="version_raw_weather_data",
        bash_command=project_command(
            "python -m src.versioning.dvc_versioning dvc-add --target data/raw/weatherAUS.csv"
        ),
    )

    verify_local_inputs = BashOperator(
        task_id="verify_local_versioned_inputs",
        bash_command=project_command("python -m src.versioning.dvc_versioning verify-local"),
    )

    snapshot_inputs = BashOperator(
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

    push_model_artifact = BashOperator(
        task_id="push_model_artifact_to_dagshub",
        bash_command=project_command("python -m src.versioning.dvc_versioning dvc-push-model"),
        execution_timeout=timedelta(minutes=30),
    )

    snapshot_outputs = BashOperator(
        task_id="snapshot_output_versions",
        bash_command=project_command(
            "python -m src.versioning.dvc_versioning snapshot "
            "--phase outputs --run-id '{{ run_id }}'"
        ),
    )

    log_model_to_mlflow = BashOperator(
        task_id="log_model_to_mlflow",
        bash_command=project_command(
            "python -m src.versioning.mlflow_tracking log-model "
            "--run-id '{{ run_id }}'"
        ),
        execution_timeout=timedelta(minutes=30),
    )

    dvc_status = BashOperator(
        task_id="record_dvc_status",
        bash_command=project_command(
            "python -m src.versioning.dvc_versioning dvc-status --run-id '{{ run_id }}'"
        ),
    )

    restart_api_deployment = BashOperator(
        task_id="restart_api_deployment",
        bash_command=project_command(
            "python -m src.versioning.kubernetes_rollout "
            "--deployment rain-prediction-api --skip-outside-kubernetes"
        ),
    )

    end = EmptyOperator(task_id="end")

    (
        start
        >> extract_raw_weather_data
        >> version_raw_weather_data
        >> verify_local_inputs
        >> snapshot_inputs
        >> train_winner_model
        >> version_model_artifact
        >> push_model_artifact
        >> snapshot_outputs
        >> log_model_to_mlflow
        >> dvc_status
        >> restart_api_deployment
        >> end
    )
