from src.versioning.mlflow_tracking import build_tracking_payload, str_to_bool


def test_build_tracking_payload_keeps_dates_as_params_and_numbers_as_metrics():
    metadata = {
        "model_name": "final_hybrid_catboost",
        "model_family": "CatBoost binary classification",
        "feature_set_name": "hybrid",
        "features": ["location", "rainfall"],
        "target": "rain_tomorrow",
        "threshold": 0.58,
        "artifact_path": "models/final_winner/winner_model.joblib",
        "config_path": "models/final_winner/model_config.json",
        "metrics": {
            "roc_auc": 0.9,
            "f1": 0.7,
            "threshold": 0.58,
            "train_rows": 100,
            "test_start_date": "2026-01-01",
            "test_end_date": "2026-01-31",
        },
    }
    config = {"params": {"depth": 8, "learning_rate": 0.05}}

    payload = build_tracking_payload(metadata=metadata, config=config, run_id="airflow-run")

    assert payload["params"]["airflow_run_id"] == "airflow-run"
    assert payload["params"]["feature_count"] == 2
    assert payload["params"]["test_start_date"] == "2026-01-01"
    assert payload["params"]["catboost_depth"] == 8
    assert payload["metrics"] == {"roc_auc": 0.9, "f1": 0.7, "train_rows": 100.0, "threshold": 0.58}


def test_str_to_bool_accepts_airflow_friendly_values():
    assert str_to_bool("true") is True
    assert str_to_bool("0", default=True) is False
    assert str_to_bool("", default=True) is True
