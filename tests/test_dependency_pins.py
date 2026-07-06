from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_MLFLOW_PIN = "mlflow-skinny==2.17.2"
ALLOWED_EVIDENTLY_PIN = "evidently==0.7.21"
ALLOWED_PLOTLY_PIN = "plotly==5.24.1"
REQUIREMENT_FILES = [
    PROJECT_ROOT / "requirements.txt",
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
