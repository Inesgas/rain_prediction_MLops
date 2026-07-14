from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCKER_SCHEDULES = {
    "MLOPS_E2E_SCHEDULE": "${MLOPS_E2E_SCHEDULE:-0 6 * * *}",
    "MODEL_VERSIONING_SCHEDULE": "${MODEL_VERSIONING_SCHEDULE:-0 4 * * *}",
    "DRIFT_MONITORING_SCHEDULE": "${DRIFT_MONITORING_SCHEDULE:-0 7 * * *}",
}

KUBERNETES_SCHEDULES = {
    "MLOPS_E2E_SCHEDULE": "0 6 * * *",
    "MODEL_VERSIONING_SCHEDULE": "0 4 * * *",
    "DRIFT_MONITORING_SCHEDULE": "0 7 * * *",
}

DOCKER_AIRFLOW_SETTINGS = {
    "PUSHGATEWAY_URL": "pushgateway:9091",
}

KUBERNETES_AIRFLOW_SETTINGS = {
    "AIRFLOW__WEBSERVER__WEB_SERVER_URL_PREFIX": "/airflow",
    "AIRFLOW__WEBSERVER__ENABLE_PROXY_FIX": "true",
    "PUSHGATEWAY_URL": "pushgateway:9091",
}

DAG_SCHEDULES = {
    "daily_weather_ingestion_dag.py": ("daily_weather_ingestion", "0 3 * * *"),
    "data_model_versioning_dag.py": ("data_model_versioning", "model_versioning_schedule"),
    "end_to_end_mlops_dag.py": ("end_to_end_mlops_pipeline", "optional_schedule"),
    "drift_monitoring_dag.py": ("drift_monitoring", "drift_monitoring_schedule"),
}


def simple_yaml_values(path: Path, keys: set[str]) -> dict[str, str]:
    values: dict[str, str] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        if key in keys:
            values[key] = value.strip().strip("\"'")

    return values


def dag_kwargs(path: Path) -> dict[str, ast.expr]:
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue

        for item in node.items:
            call = item.context_expr
            if isinstance(call, ast.Call) and call_name(call.func) == "DAG":
                return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}

    raise AssertionError(f"No DAG(...) declaration found in {path}")


def call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def schedule_value(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call):
        return call_name(node.func)
    return None


def test_docker_airflow_schedules_are_automated():
    values = simple_yaml_values(PROJECT_ROOT / "docker-compose.yml", set(DOCKER_SCHEDULES))

    assert values == DOCKER_SCHEDULES


def test_docker_airflow_pushgateway_dependency_is_configured():
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    values = simple_yaml_values(PROJECT_ROOT / "docker-compose.yml", set(DOCKER_AIRFLOW_SETTINGS))

    assert values == DOCKER_AIRFLOW_SETTINGS
    assert "pushgateway:" in compose
    assert "prom/pushgateway:latest" in compose


def test_kubernetes_airflow_schedules_are_automated():
    values = simple_yaml_values(
        PROJECT_ROOT / "kubernetes" / "airflow-configmap.yaml",
        set(KUBERNETES_SCHEDULES),
    )

    assert values == KUBERNETES_SCHEDULES


def test_kubernetes_airflow_prefix_and_pushgateway_are_configured():
    kubernetes_values = simple_yaml_values(
        PROJECT_ROOT / "kubernetes" / "airflow-configmap.yaml",
        set(KUBERNETES_AIRFLOW_SETTINGS),
    )

    assert kubernetes_values == KUBERNETES_AIRFLOW_SETTINGS


def test_kubernetes_includes_airflow_pushgateway_dependency():
    kustomization = (PROJECT_ROOT / "kubernetes" / "kustomization.yaml").read_text(encoding="utf-8")

    assert "- pushgateway-deployment.yaml" in kustomization
    assert "- pushgateway-service.yaml" in kustomization


def test_airflow_dags_have_expected_schedule_wiring():
    for filename, (dag_id, expected_schedule) in DAG_SCHEDULES.items():
        kwargs = dag_kwargs(PROJECT_ROOT / "airflow" / "dags" / filename)

        assert isinstance(kwargs["dag_id"], ast.Constant)
        assert kwargs["dag_id"].value == dag_id
        assert schedule_value(kwargs["schedule"]) == expected_schedule
