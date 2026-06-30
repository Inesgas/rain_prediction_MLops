from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_MLFLOW_PIN = "mlflow-skinny==2.17.2"
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
