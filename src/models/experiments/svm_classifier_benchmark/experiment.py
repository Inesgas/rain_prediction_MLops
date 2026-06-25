from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import mlflow

from src.models.experiments.hybrid_imputation_breakthrough.experiment import tune_threshold_from_validation
from src.models.ines_feature_modeling import TARGET, fit_model_by_name, predict_proba_for_model, score_predictions
from src.utils.validation import BEST_FEATURE_SET_NAME, PipelineConfig, prepare_standard_split_frames


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "models" / "svm_classifier_benchmark"
RESULTS_PATH = RESULTS_DIR / "svm_candidate_results.csv"
SUMMARY_PATH = RESULTS_DIR / "svm_summary.json"
NOTES_PATH = RESULTS_DIR / "notes.md"
WINNER_SUMMARY_PATH = PROJECT_ROOT / "models" / "winner_model_calibration" / "winner_calibration_summary.json"

CANDIDATE_CONFIGS: list[dict[str, Any]] = [
    {
        "candidate": "svm_keep_location_c05",
        "model_label": "Linear SVM",
        "drop_location": False,
        "C": 0.5,
    },
    {
        "candidate": "svm_keep_location_c1",
        "model_label": "Linear SVM",
        "drop_location": False,
        "C": 1.0,
    },
    {
        "candidate": "svm_keep_location_c2",
        "model_label": "Linear SVM",
        "drop_location": False,
        "C": 2.0,
    },
    {
        "candidate": "svm_drop_location_c1",
        "model_label": "Linear SVM",
        "drop_location": True,
        "C": 1.0,
    },
]
SELECTION_SORT_COLUMNS = [
    "validation_f1",
    "validation_roc_auc",
    "validation_precision",
    "validation_recall",
    "test_f1",
    "candidate",
]
SELECTION_SORT_ASCENDING = [False, False, False, False, False, True]


def write_notes() -> None:
    text = """# SVM Classifier Benchmark

## Goal

The SVM benchmark is a standalone classifier experiment, separate from the HCA clustering work.

## Design

- Reuse the same hybrid-plus-core winner feature space.
- Keep the same chronological train / validation / test rule.
- Tune threshold on validation, refit on train + validation, then score once on test.
- Test a small SVM grid around the regularization strength `C`.
- Include one `drop_location` variant to check whether raw station identity matters for the SVM.

## Important Constraint

This benchmark uses a linear SVM so the full dataset and validation-first design remain tractable in the current
CPU-only repository.
"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text(text, encoding="utf-8")


def load_locked_winner_metrics() -> dict[str, float]:
    if not WINNER_SUMMARY_PATH.exists():
        return {}
    summary = json.loads(WINNER_SUMMARY_PATH.read_text(encoding="utf-8"))
    raw = summary.get("uncalibrated", {})
    return {
        "winner_test_roc_auc": float(raw.get("test_roc_auc", np.nan)),
        "winner_test_f1": float(raw.get("test_f1", np.nan)),
        "winner_test_precision": float(raw.get("test_precision", np.nan)),
        "winner_test_recall": float(raw.get("test_recall", np.nan)),
    }


def evaluate_svm_feature_set(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    svm_params = params or {"C": 1.0}
    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df[TARGET].astype(int)
    y_valid = valid_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    fitted = fit_model_by_name("Linear SVM", X_train, y_train, params=svm_params)
    valid_scores = predict_proba_for_model("Linear SVM", fitted, X_valid)
    best_threshold, valid_metrics = tune_threshold_from_validation(valid_scores, y_valid)

    combined = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
    X_combined = combined[features].copy()
    y_combined = combined[TARGET].astype(int)
    final_model = fit_model_by_name("Linear SVM", X_combined, y_combined, params=svm_params)
    test_scores = predict_proba_for_model("Linear SVM", final_model, X_test)
    test_metrics = score_predictions(y_test, test_scores, threshold=best_threshold)

    return {
        "feature_count": int(len(features)),
        "validation_threshold": float(best_threshold),
        "validation_f1": float(valid_metrics["f1"]),
        "validation_roc_auc": float(valid_metrics["roc_auc"]),
        "validation_precision": float(valid_metrics["precision"]),
        "validation_recall": float(valid_metrics["recall"]),
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "test_f1": float(test_metrics["f1"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "params": json.dumps(svm_params, sort_keys=True),
    }


def run_experiment() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_notes()
    
    mlflow.set_experiment("rain_prediction_comparisons")

    config = PipelineConfig(name="hybrid_default")
    train_df, valid_df, test_df, feature_sets = prepare_standard_split_frames(config=config, keep_date=False)
    base_features = list(feature_sets[BEST_FEATURE_SET_NAME])
    winner_metrics = load_locked_winner_metrics()

    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATE_CONFIGS:
        if mlflow.active_run() is not None:
            mlflow.end_run()

        with mlflow.start_run(run_name=str(candidate["candidate"])):
            features = list(base_features)
            if candidate.get("drop_location", False):
                features = [feature for feature in features if feature != "location"]

            metrics = evaluate_svm_feature_set(
                train_df,
                valid_df,
                test_df,
                features,
                params={"C": float(candidate["C"])},
            )
            row = {
                "candidate": str(candidate["candidate"]),
                "candidate_label": str(candidate["candidate"]).replace("_", " "),
                "model": str(candidate["model_label"]),
                "drop_location": bool(candidate["drop_location"]),
                "C": float(candidate["C"]),
                "train_support": int(len(train_df)),
                "validation_support": int(len(valid_df)),
                "test_support": int(len(test_df)),
                **metrics,
            }
            if winner_metrics:
                row["winner_test_roc_auc_gap"] = float(row["test_roc_auc"] - winner_metrics["winner_test_roc_auc"])
                row["winner_test_f1_gap"] = float(row["test_f1"] - winner_metrics["winner_test_f1"])
                row["winner_test_precision_gap"] = float(row["test_precision"] - winner_metrics["winner_test_precision"])
                row["winner_test_recall_gap"] = float(row["test_recall"] - winner_metrics["winner_test_recall"])
            rows.append(row)

            mlflow.log_param("C", float(candidate["C"]))
            mlflow.log_param("drop_location", bool(candidate["drop_location"]))
            mlflow.log_param("model", str(candidate["model_label"]))
        numeric_metrics = {k: v for k, v in row.items() if isinstance(v, (int, float)) and k != "C"}
        mlflow.log_metrics(numeric_metrics)

    results = (
        pd.DataFrame(rows)
        .sort_values(SELECTION_SORT_COLUMNS, ascending=SELECTION_SORT_ASCENDING)
        .reset_index(drop=True)
    )
    results.to_csv(RESULTS_PATH, index=False)

    best_row = results.iloc[0].to_dict()
    summary: dict[str, Any] = {
        "experiment": "svm_classifier_benchmark",
        "model": "Linear SVM",
        "selection_basis": "validation_first_chronological_split",
        "feature_set_name": BEST_FEATURE_SET_NAME,
        "candidate_count": int(len(results)),
        "best_result": best_row,
        "results_path": str(RESULTS_PATH),
        "notes_path": str(NOTES_PATH),
    }
    if winner_metrics:
        summary["winner_comparison"] = {
            **winner_metrics,
            "svm_minus_winner_roc_auc": float(best_row["test_roc_auc"] - winner_metrics["winner_test_roc_auc"]),
            "svm_minus_winner_f1": float(best_row["test_f1"] - winner_metrics["winner_test_f1"]),
            "svm_minus_winner_precision": float(best_row["test_precision"] - winner_metrics["winner_test_precision"]),
            "svm_minus_winner_recall": float(best_row["test_recall"] - winner_metrics["winner_test_recall"]),
        }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))

