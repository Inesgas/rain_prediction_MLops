from __future__ import annotations

from pathlib import Path

# adding evidently and plotly pins to ensure compatibility with drift monitoring 02.07.26gn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_MLFLOW_PIN = "mlflow-skinny==2.17.2"
ALLOWED_EVIDENTLY_PIN = "evidently==0.7.21"
ALLOWED_PLOTLY_PIN = "plotly==5.24.1"
ALLOWED_CRYPTOGRAPHY_RANGE = "cryptography>=43.0.1,<45"
ALLOWED_CFFI_RANGE = "cffi<2.0.0"
MIN_PROMETHEUS_CLIENT = "prometheus-client>=0.20.0"
KUBERNETES_CLIENT_RANGE = "kubernetes>=30.0.0,<32.0.0"
REQUIREMENT_FILES = [
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / "docker" / "airflow" / "airflow_requirements.txt",
    PROJECT_ROOT / "src" / "docker" / "airflow" / "airflow_requirements.txt",
]
AIRFLOW_REQUIREMENT_FILES = [
    PROJECT_ROOT / "docker" / "airflow" / "airflow_requirements.txt",
    PROJECT_ROOT / "src" / "docker" / "airflow" / "airflow_requirements.txt",
]


def requirement_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_mlflow_dependency_pin_is_consistent():
    invalid_lines: list[str] = []
    duplicate_files: list[str] = []

    for path in REQUIREMENT_FILES:
        mlflow_lines = [
            line
            for line in requirement_lines(path)
            if line.lower().startswith(("mlflow==", "mlflow>=", "mlflow-skinny"))
        ]

        if len(mlflow_lines) > 1:
            duplicate_files.append(str(path.relative_to(PROJECT_ROOT)))

        invalid_lines.extend(
            f"{path.relative_to(PROJECT_ROOT)}: {line}"
            for line in mlflow_lines
            if line != ALLOWED_MLFLOW_PIN
        )

    assert not duplicate_files, f"Duplicate MLflow requirement pins: {duplicate_files}"
    assert not invalid_lines, f"Use only {ALLOWED_MLFLOW_PIN}: {invalid_lines}"


def test_evidently_dependency_pin_is_present_wherever_drift_monitoring_runs():
    missing_files = [
        str(path.relative_to(PROJECT_ROOT))
        for path in REQUIREMENT_FILES
        if ALLOWED_EVIDENTLY_PIN not in requirement_lines(path)
    ]

    assert not missing_files, (
        f"src.monitoring.drift_report needs '{ALLOWED_EVIDENTLY_PIN}' installed, "
        f"but it is missing from: {missing_files}"
    )


def test_plotly_pin_stays_compatible_with_evidently():
    invalid_lines: list[str] = []

    for path in REQUIREMENT_FILES:
        lines = requirement_lines(path)
        if not any(line.startswith("evidently==") for line in lines):
            continue

        plotly_lines = [line for line in lines if line.startswith("plotly==")]
        invalid_lines.extend(
            f"{path.relative_to(PROJECT_ROOT)}: {line}"
            for line in plotly_lines
            if line != ALLOWED_PLOTLY_PIN
        )
        if not plotly_lines:
            invalid_lines.append(f"{path.relative_to(PROJECT_ROOT)}: missing {ALLOWED_PLOTLY_PIN}")

    assert not invalid_lines, f"evidently 0.7.21 requires plotly<6: {invalid_lines}"


def test_airflow_evidently_transitive_bounds_keep_pyopenssl_compatible():
    invalid_lines: list[str] = []

    for path in AIRFLOW_REQUIREMENT_FILES:
        lines = requirement_lines(path)
        cryptography_lines = [line for line in lines if line.startswith("cryptography")]
        cffi_lines = [line for line in lines if line.startswith("cffi")]

        if cryptography_lines != [ALLOWED_CRYPTOGRAPHY_RANGE]:
            invalid_lines.append(
                f"{path.relative_to(PROJECT_ROOT)}: expected {ALLOWED_CRYPTOGRAPHY_RANGE}, "
                f"found {cryptography_lines or 'missing'}"
            )
        if cffi_lines != [ALLOWED_CFFI_RANGE]:
            invalid_lines.append(
                f"{path.relative_to(PROJECT_ROOT)}: expected {ALLOWED_CFFI_RANGE}, "
                f"found {cffi_lines or 'missing'}"
            )

    assert not invalid_lines, (
        "Airflow's pyOpenSSL stack breaks with latest cryptography/cffi: "
        f"{invalid_lines}"
    )


def test_airflow_pushgateway_client_dependency_is_present():
    missing_files = [
        str(path.relative_to(PROJECT_ROOT))
        for path in AIRFLOW_REQUIREMENT_FILES
        if MIN_PROMETHEUS_CLIENT not in requirement_lines(path)
    ]

    assert not missing_files, (
        "Airflow drift monitoring pushes metrics to Pushgateway and needs "
        f"{MIN_PROMETHEUS_CLIENT}: {missing_files}"
    )


def test_airflow_kubernetes_client_dependency_is_present():
    missing_files = [
        str(path.relative_to(PROJECT_ROOT))
        for path in AIRFLOW_REQUIREMENT_FILES
        if KUBERNETES_CLIENT_RANGE not in requirement_lines(path)
    ]

    assert not missing_files, (
        "Airflow restarts the Kubernetes API deployment after model publishing and needs "
        f"{KUBERNETES_CLIENT_RANGE}: {missing_files}"
    )
