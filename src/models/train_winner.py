from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#update mlflow to 2.9.1 to avoid the error: AttributeError: module 'mlflow' has no attribute 'catboost'
import joblib
import mlflow
import mlflow.catboost
import pandas as pd

from src.config.paths import (
    BASE_DIR,
    DATE_COLUMN,
    FINAL_WINNER_CONFIG_PATH,
    FINAL_WINNER_DIR,
    FINAL_WINNER_METADATA_PATH,
    FINAL_WINNER_MODEL_ARTIFACT,
    FINAL_WINNER_SAMPLE_INPUT_PATH,
    TARGET_COLUMN,
)
from src.models.experiments.geo_climate_context_extension.experiment import prepare_variant_frames
from src.models.ines_feature_modeling import TARGET
from src.models.ines_modeling_core import make_catboost_classifier, prepare_catboost_frames, score_predictions


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _relative(path: Path) -> str:
    return str(path.relative_to(BASE_DIR))


def load_winner_config(path: Path = FINAL_WINNER_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Winner model configuration not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric_fill_values(frame: pd.DataFrame, numeric_features: list[str]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for column in numeric_features:
        median = pd.to_numeric(frame[column], errors="coerce").median()
        values[column] = None if pd.isna(median) else float(median)
    return values


def train_winner(
    config_path: Path = FINAL_WINNER_CONFIG_PATH,
    output_dir: Path = FINAL_WINNER_DIR,
) -> dict[str, Path]:
    config = load_winner_config(config_path)
    features = list(config["features"])
    params = dict(config["params"])
    threshold = float(config["threshold"])

    train_df, valid_df, test_df, _, _ = prepare_variant_frames()
    missing_features = [feature for feature in features if feature not in train_df.columns]
    if missing_features:
        raise ValueError(f"Winner feature frame is missing columns: {missing_features}")

    combined = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)
    X_train = combined[features].copy()
    y_train = combined[TARGET].astype(int)
    X_test = test_df[features].copy()
    y_test = test_df[TARGET].astype(int)

    X_train_ready, X_test_ready, categorical_features = prepare_catboost_frames(X_train, X_test)
    numeric_features = [feature for feature in features if feature not in categorical_features]
    numeric_fill_values = _numeric_fill_values(X_train, numeric_features)

    model = make_catboost_classifier(y_train, params=params)
    if model is None:
        raise RuntimeError("CatBoost is not available in this environment.")

    mlflow.set_experiment("rain_prediction_winner")
    with mlflow.start_run(run_name=config.get("model_name", "final_hybrid_catboost")):
        model.fit(X_train_ready, y_train, cat_features=categorical_features)

        probabilities = model.predict_proba(X_test_ready)[:, 1]
        metrics = {
            key: float(value)
            for key, value in score_predictions(y_test, probabilities, threshold=threshold).items()
        }
        metrics.update(
            {
                "threshold": threshold,
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "test_start_date": pd.to_datetime(test_df[DATE_COLUMN]).min().date().isoformat(),
                "test_end_date": pd.to_datetime(test_df[DATE_COLUMN]).max().date().isoformat(),
            }
        )

        mlflow.log_params(params)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("feature_set_name", config.get("feature_set_name", ""))
        date_fields = {"test_start_date", "test_end_date"}
        numeric_metrics = {k: v for k, v in metrics.items() if k not in date_fields}
        mlflow.log_metrics(numeric_metrics)
        mlflow.log_params({k: metrics[k] for k in date_fields})
        mlflow.catboost.log_model(
        model,
        artifact_path="model",
        registered_model_name="rain_prediction_catboost",
)
        
        
        

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "model_name": config.get("model_name", "final_hybrid_catboost"),
        "model_role": "final rain prediction model",
        "model_family": "CatBoost binary classification",
        "feature_set_name": config.get("feature_set_name", "hybrid_regime_keep_location_plus_core"),
        "features": features,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "numeric_fill_values": numeric_fill_values,
        "target": TARGET_COLUMN,
        "threshold": threshold,
        "params": params,
        "metrics": metrics,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(artifact, FINAL_WINNER_MODEL_ARTIFACT)

    metadata = {key: value for key, value in artifact.items() if key != "model"}
    metadata["artifact_path"] = _relative(FINAL_WINNER_MODEL_ARTIFACT)
    metadata["config_path"] = _relative(config_path)
    FINAL_WINNER_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    sample_row = X_test.iloc[0].to_dict()
    sample_input = {feature: _json_safe(sample_row.get(feature)) for feature in features}
    FINAL_WINNER_SAMPLE_INPUT_PATH.write_text(json.dumps(sample_input, indent=2), encoding="utf-8")

    return {
        "artifact_path": FINAL_WINNER_MODEL_ARTIFACT,
        "metadata_path": FINAL_WINNER_METADATA_PATH,
        "sample_input_path": FINAL_WINNER_SAMPLE_INPUT_PATH,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the final winner rain prediction model.")
    parser.add_argument("--config", type=Path, default=FINAL_WINNER_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=FINAL_WINNER_DIR)
    args = parser.parse_args()

    outputs = train_winner(config_path=args.config, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
