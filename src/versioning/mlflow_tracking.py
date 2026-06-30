from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ARTIFACT = PROJECT_ROOT / "models" / "final_winner" / "winner_model.joblib"
DEFAULT_MODEL_METADATA = PROJECT_ROOT / "models" / "final_winner" / "metadata.json"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "models" / "final_winner" / "model_config.json"
DEFAULT_MODEL_SAMPLE = PROJECT_ROOT / "models" / "final_winner" / "sample_input.json"
DEFAULT_EXPERIMENT_NAME = "rain_prediction_winner"
DEFAULT_REGISTERED_MODEL_NAME = "rain_prediction_final_winner"


def str_to_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required MLflow metadata file is missing: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return loaded


def tracking_uri_from_env() -> str:
    configured = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if configured:
        return configured
    return (PROJECT_ROOT / "mlruns").resolve().as_uri()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def mlflow_param_value(value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True)


def numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if is_number(value)}


def build_tracking_payload(
    metadata: dict[str, Any],
    config: dict[str, Any],
    run_id: str | None,
) -> dict[str, dict[str, Any]]:
    metrics = dict(metadata.get("metrics") or {})
    model_params = dict(config.get("params") or metadata.get("params") or {})
    feature_count = len(metadata.get("features") or config.get("features") or [])

    params: dict[str, Any] = {
        "airflow_run_id": run_id or os.environ.get("AIRFLOW_CTX_RUN_ID", ""),
        "model_name": metadata.get("model_name") or config.get("model_name", ""),
        "model_role": metadata.get("model_role", ""),
        "model_family": metadata.get("model_family") or config.get("algorithm", ""),
        "feature_set_name": metadata.get("feature_set_name") or config.get("feature_set_name", ""),
        "feature_count": feature_count,
        "target": metadata.get("target", ""),
        "threshold": metadata.get("threshold") or config.get("threshold", ""),
        "artifact_path": metadata.get("artifact_path", ""),
        "config_path": metadata.get("config_path", ""),
        "test_start_date": metrics.get("test_start_date", ""),
        "test_end_date": metrics.get("test_end_date", ""),
    }
    params.update({f"catboost_{key}": value for key, value in model_params.items()})

    tags = {
        "project": "rain_prediction_mlops",
        "orchestrator": "airflow",
        "dag_id": os.environ.get("AIRFLOW_CTX_DAG_ID", ""),
        "task_id": os.environ.get("AIRFLOW_CTX_TASK_ID", ""),
        "model_name": str(params["model_name"]),
    }

    return {
        "params": {key: mlflow_param_value(value) for key, value in params.items()},
        "metrics": numeric_metrics(metrics),
        "tags": tags,
    }


def dry_run_summary(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(args.metadata_path)
    config = read_json(args.config_path)
    payload = build_tracking_payload(metadata=metadata, config=config, run_id=args.run_id)
    return {
        "dry_run": True,
        "tracking_uri": tracking_uri_from_env(),
        "experiment_name": args.experiment_name,
        "registered_model_name": args.registered_model_name if args.register_model else None,
        "model_artifact_exists": args.model_artifact.exists(),
        "artifact_paths": {
            "model": str(args.model_artifact),
            "metadata": str(args.metadata_path),
            "config": str(args.config_path),
            "sample_input": str(args.sample_input_path),
        },
        "payload": payload,
    }


def log_model_to_mlflow(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model_artifact.exists():
        raise FileNotFoundError(f"Trained model artifact is missing: {args.model_artifact}")

    metadata = read_json(args.metadata_path)
    config = read_json(args.config_path)
    payload = build_tracking_payload(metadata=metadata, config=config, run_id=args.run_id)

    import joblib
    import mlflow
    import mlflow.catboost

    tracking_uri = tracking_uri_from_env()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    run_name = str(metadata.get("model_name") or config.get("model_name") or "final_hybrid_catboost")
    if args.run_id:
        run_name = f"{run_name}-{args.run_id}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({key: value for key, value in payload["tags"].items() if value})
        mlflow.log_params(payload["params"])
        if payload["metrics"]:
            mlflow.log_metrics(payload["metrics"])

        for path in (args.metadata_path, args.config_path, args.sample_input_path):
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="model_metadata")

        artifact_bundle = joblib.load(args.model_artifact)
        model = artifact_bundle.get("model") if isinstance(artifact_bundle, dict) else artifact_bundle
        mlflow.catboost.log_model(model, artifact_path="model")

        registered_version = None
        if args.register_model:
            model_uri = f"runs:/{run.info.run_id}/model"
            registered = mlflow.register_model(model_uri, args.registered_model_name)
            registered_version = getattr(registered, "version", None)

        result = {
            "tracking_uri": tracking_uri,
            "experiment_name": args.experiment_name,
            "run_id": run.info.run_id,
            "run_name": run_name,
            "registered_model_name": args.registered_model_name if args.register_model else None,
            "registered_model_version": registered_version,
        }
        print(json.dumps(result, indent=2))
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Log the trained winner model to MLflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    log_parser = subparsers.add_parser("log-model", help="Log model metadata, metrics, and artifact to MLflow.")
    log_parser.add_argument("--run-id", default=os.environ.get("AIRFLOW_CTX_RUN_ID"))
    log_parser.add_argument("--model-artifact", type=Path, default=DEFAULT_MODEL_ARTIFACT)
    log_parser.add_argument("--metadata-path", type=Path, default=DEFAULT_MODEL_METADATA)
    log_parser.add_argument("--config-path", type=Path, default=DEFAULT_MODEL_CONFIG)
    log_parser.add_argument("--sample-input-path", type=Path, default=DEFAULT_MODEL_SAMPLE)
    log_parser.add_argument(
        "--experiment-name",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME),
    )
    log_parser.add_argument(
        "--registered-model-name",
        default=os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", DEFAULT_REGISTERED_MODEL_NAME),
    )
    log_parser.add_argument(
        "--register-model",
        action=argparse.BooleanOptionalAction,
        default=str_to_bool(os.environ.get("MLFLOW_REGISTER_MODEL"), default=True),
    )
    log_parser.add_argument("--dry-run", action="store_true", help="Validate and print the MLflow payload only.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "log-model":
        if args.dry_run:
            print(json.dumps(dry_run_summary(args), indent=2))
        else:
            log_model_to_mlflow(args)
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
