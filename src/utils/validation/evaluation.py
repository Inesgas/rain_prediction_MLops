from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.models.experiments.daily_zonal_baseline.experiment import align_feature_columns
from src.models.experiments.hybrid_imputation_breakthrough.experiment import tune_threshold_from_validation
from src.models.ines_feature_modeling import TARGET, fit_model_by_name, predict_proba_for_model, score_predictions

from .config import BEST_FEATURE_SET_NAME, load_best_hybrid_selection
from .frames import month_to_season


def _add_targeted_features(frame: pd.DataFrame, temp_cutoff: float, pressure_cutoff: float) -> pd.DataFrame:
    result = frame.copy()
    result["season_label"] = result["month"].astype(int).map(month_to_season)
    result["summer_flag"] = (result["season_label"] == "Summer").astype(int)
    result["high_elevation_flag"] = (result["elevation"].fillna(0) >= 500).astype(int)
    result["very_dry_3pm_flag"] = (result["humidity_3pm"].fillna(100) < 40).astype(int)
    result["low_cloud_3pm_flag"] = (result["cloud_3pm"].fillna(8) < 2).astype(int)
    result["warm_afternoon_flag"] = (result["temp_3pm"].fillna(0) >= temp_cutoff).astype(int)
    result["high_pressure_3pm_flag"] = (result["pressure_3pm"].fillna(0) >= pressure_cutoff).astype(int)
    result["summer_dry_low_cloud_flag"] = (
        (result["summer_flag"] == 1)
        & (result["very_dry_3pm_flag"] == 1)
        & (result["low_cloud_3pm_flag"] == 1)
    ).astype(int)
    result["summer_high_elevation_flag"] = (
        (result["summer_flag"] == 1) & (result["high_elevation_flag"] == 1)
    ).astype(int)
    result["warm_dry_high_pressure_flag"] = (
        (result["warm_afternoon_flag"] == 1)
        & (result["very_dry_3pm_flag"] == 1)
        & (result["high_pressure_3pm_flag"] == 1)
    ).astype(int)
    return result


def add_targeted_false_negative_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    temp_cutoff = float(train_df["temp_3pm"].quantile(0.75))
    pressure_cutoff = float(train_df["pressure_3pm"].quantile(0.75))

    train_enriched = _add_targeted_features(train_df, temp_cutoff, pressure_cutoff)
    valid_enriched = _add_targeted_features(valid_df, temp_cutoff, pressure_cutoff)
    test_enriched = _add_targeted_features(test_df, temp_cutoff, pressure_cutoff)

    baseline_features = [col for col in train_df.columns if col != TARGET]
    targeted_columns = [
        "summer_dry_low_cloud_flag",
        "summer_high_elevation_flag",
        "warm_dry_high_pressure_flag",
    ]
    feature_sets = {
        BEST_FEATURE_SET_NAME: [col for col in baseline_features if col in train_enriched.columns],
        "hybrid_targeted_small_adjustments": list(
            dict.fromkeys([col for col in baseline_features if col in train_enriched.columns] + targeted_columns)
        ),
    }
    train_enriched, valid_enriched, test_enriched = align_feature_columns(train_enriched, valid_enriched, test_enriched)
    return train_enriched, valid_enriched, test_enriched, feature_sets


def evaluate_catboost_feature_set(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = params or load_best_hybrid_selection()["params"]
    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df[TARGET].astype(int)
    y_valid = valid_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    fitted = fit_model_by_name("CatBoost", X_train, y_train, params=params)
    valid_proba = predict_proba_for_model("CatBoost", fitted, X_valid)
    best_threshold, valid_metrics = tune_threshold_from_validation(valid_proba, y_valid)

    combined = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)
    X_combined = combined[features].copy()
    y_combined = combined[TARGET].astype(int)
    final_model = fit_model_by_name("CatBoost", X_combined, y_combined, params=params)
    test_proba = predict_proba_for_model("CatBoost", final_model, X_test)
    test_metrics = score_predictions(y_test, test_proba, threshold=best_threshold)

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
        "params": json.dumps(params, sort_keys=True),
    }

