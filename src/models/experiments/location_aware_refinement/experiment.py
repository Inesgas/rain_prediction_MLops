from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.experiments.daily_zonal_baseline.experiment import (
    METADATA_PATH,
    SEARCH_RESULTS_PATH as BASELINE_RESULTS_PATH,
    THRESHOLDS,
    TARGET_MODELS,
    add_metadata,
    align_feature_columns,
    chronological_split,
    enrich_metadata_with_rainfall_zone,
    load_raw,
    daily_zonal_pipeline,
    tune_threshold_from_validation,
)
from src.models.ines_feature_modeling import (
    TARGET,
    fit_model_by_name,
    predict_proba_for_model,
    score_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "models" / "location_aware_refinement"
SEARCH_RESULTS_PATH = RESULTS_DIR / "location_refinement_results.csv"
SUMMARY_PATH = RESULTS_DIR / "location_refinement_summary.json"
NOTES_PATH = RESULTS_DIR / "notes.md"


def add_core_project_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if {"max_temp", "min_temp"}.issubset(result.columns):
        result["temp_range"] = result["max_temp"] - result["min_temp"]

    if {"temp_3pm", "humidity_3pm"}.issubset(result.columns):
        result["humidity_temp_3pm_interaction"] = result["temp_3pm"] * (result["humidity_3pm"] / 100.0)

    if {"pressure_3pm", "humidity_3pm"}.issubset(result.columns):
        result["pressure_humidity_3pm_ratio"] = result["pressure_3pm"] / (result["humidity_3pm"] + 1.0)

    if {"cloud_3pm", "humidity_3pm"}.issubset(result.columns):
        result["cloud_humidity_3pm_interaction"] = result["cloud_3pm"] * result["humidity_3pm"]

    if {"dewpoint_spread_3pm", "humidity_3pm"}.issubset(result.columns):
        result["moisture_stability_3pm"] = result["dewpoint_spread_3pm"] * (100.0 - result["humidity_3pm"])

    if {"humidity_3pm", "humidity_9am"}.issubset(result.columns):
        result["humidity_rising_fast"] = (result["humidity_3pm"] > (result["humidity_9am"] * 1.1)).astype(int)

    if {"temp_3pm", "temp_9am"}.issubset(result.columns):
        result["warming_day"] = (result["temp_3pm"] > result["temp_9am"]).astype(int)

    return result


def add_observation_helpers(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    obs_index = result.groupby(["date", "location"], dropna=False).cumcount().astype(str)
    result["_obs_key"] = (
        result["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + result["location"].astype(str)
        + "|"
        + obs_index
    )
    result["_source_month"] = result["date"].dt.month.astype("int16").astype(str).str.zfill(2)
    return result


def fit_location_month_climatology(train_df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    keep_cols = [col for col in columns if col in train_df.columns]
    return {
        "columns": keep_cols,
        "location_month": train_df.groupby(["location", "_source_month"], dropna=False)[keep_cols].median().reset_index(),
        "location": train_df.groupby(["location"], dropna=False)[keep_cols].median().reset_index(),
        "global": {col: float(train_df[col].median()) for col in keep_cols},
    }


def _merge_lookup(df: pd.DataFrame, lookup: pd.DataFrame, keys: list[str], columns: list[str], suffix: str) -> pd.DataFrame:
    renamed = lookup.rename(columns={col: f"{col}{suffix}" for col in columns})
    return df.merge(renamed, on=keys, how="left")


def add_anomaly_features(df: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    result = df.copy()
    columns = stats["columns"]
    result = _merge_lookup(result, stats["location_month"], ["location", "_source_month"], columns, "__loc_month")
    result = _merge_lookup(result, stats["location"], ["location"], columns, "__location")

    for col in columns:
        baseline = result.get(f"{col}__loc_month", pd.Series(np.nan, index=result.index))
        if f"{col}__location" in result.columns:
            baseline = baseline.fillna(result[f"{col}__location"])
        baseline = baseline.fillna(stats["global"][col])
        result[f"{col}_anomaly"] = result[col] - baseline

    if {"pressure_3pm_anomaly", "humidity_3pm_anomaly"}.issubset(result.columns):
        result["pressure_humidity_anomaly_interaction"] = (
            result["pressure_3pm_anomaly"] * result["humidity_3pm_anomaly"]
        )
    if {"cloud_3pm_anomaly", "humidity_3pm_anomaly"}.issubset(result.columns):
        result["cloud_humidity_anomaly_interaction"] = (
            result["cloud_3pm_anomaly"] * result["humidity_3pm_anomaly"]
        )

    helper_cols = [col for col in result.columns if col.endswith("__loc_month") or col.endswith("__location")]
    return result.drop(columns=helper_cols, errors="ignore")


def build_refinement_feature_sets(
    base_features: list[str],
    core_features: list[str],
    anomaly_features: list[str],
) -> dict[str, list[str]]:
    return {
        "location_aware_base": base_features,
        "location_aware_plus_core": list(dict.fromkeys(base_features + core_features)),
        "location_aware_plus_core_anomalies": list(dict.fromkeys(base_features + core_features + anomaly_features)),
    }


def prepare_variant_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    metadata_path = enrich_metadata_with_rainfall_zone()
    raw_df = load_raw()
    raw_df = raw_df[raw_df[TARGET].notna()].drop_duplicates().reset_index(drop=True)
    train_full, test_df = chronological_split(raw_df, test_size=0.20)
    train_df, valid_df = chronological_split(train_full, test_size=0.20)

    train_df = add_observation_helpers(train_df)
    valid_df = add_observation_helpers(valid_df)
    test_df = add_observation_helpers(test_df)
    train_full = add_observation_helpers(train_full)

    train_ready = daily_zonal_pipeline(train_df, train_df, metadata_path, drop_location=False)
    valid_ready = daily_zonal_pipeline(valid_df, train_df, metadata_path, drop_location=False)
    test_ready = daily_zonal_pipeline(test_df, train_full, metadata_path, drop_location=False)

    excluded = {TARGET, "_obs_key", "_source_month"}
    base_features = [col for col in train_ready.columns if col not in excluded]

    train_ready = add_core_project_features(train_ready)
    valid_ready = add_core_project_features(valid_ready)
    test_ready = add_core_project_features(test_ready)
    core_features = [
        col for col in [
            "temp_range",
            "humidity_temp_3pm_interaction",
            "pressure_humidity_3pm_ratio",
            "cloud_humidity_3pm_interaction",
            "moisture_stability_3pm",
            "humidity_rising_fast",
            "warming_day",
        ] if col in train_ready.columns
    ]

    anomaly_stats = fit_location_month_climatology(
        train_ready,
        ["humidity_3pm", "pressure_3pm", "cloud_3pm", "temp_3pm", "sunshine"],
    )

    train_ready = add_anomaly_features(train_ready, anomaly_stats)
    valid_ready = add_anomaly_features(valid_ready, anomaly_stats)
    test_ready = add_anomaly_features(test_ready, anomaly_stats)
    anomaly_features = [col for col in train_ready.columns if col.endswith("_anomaly")]
    anomaly_features += [
        col for col in [
            "pressure_humidity_anomaly_interaction",
            "cloud_humidity_anomaly_interaction",
        ] if col in train_ready.columns
    ]

    feature_sets = build_refinement_feature_sets(base_features, core_features, anomaly_features)
    train_ready, valid_ready, test_ready = align_feature_columns(train_ready, valid_ready, test_ready)
    return train_ready, valid_ready, test_ready, feature_sets


def evaluate_feature_set(
    feature_set_name: str,
    features: list[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_models = [(model_name, params) for model_name, params in TARGET_MODELS if model_name in {"XGBoost", "CatBoost"}]
    for model_name, params in candidate_models:
        print(f"Running {model_name} | {feature_set_name}", flush=True)
        X_train = train_df[features].copy()
        X_valid = valid_df[features].copy()
        X_test = test_df[features].copy()
        y_train = train_df[TARGET].astype(int)
        y_valid = valid_df[TARGET].astype(int)
        y_test = test_df[TARGET].astype(int)

        fitted = fit_model_by_name(model_name, X_train, y_train, params=params)
        valid_proba = predict_proba_for_model(model_name, fitted, X_valid)
        best_threshold, valid_metrics = tune_threshold_from_validation(valid_proba, y_valid)

        combined = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)
        X_combined = combined[features].copy()
        y_combined = combined[TARGET].astype(int)
        final_model = fit_model_by_name(model_name, X_combined, y_combined, params=params)
        test_proba = predict_proba_for_model(model_name, final_model, X_test)
        test_metrics = score_predictions(y_test, test_proba, threshold=best_threshold)

        rows.append(
            {
                "model": model_name,
                "feature_set": feature_set_name,
                "feature_count": int(len(features)),
                "validation_threshold": best_threshold,
                "validation_f1": float(valid_metrics["f1"]),
                "validation_roc_auc": float(valid_metrics["roc_auc"]),
                "validation_precision": float(valid_metrics["precision"]),
                "validation_recall": float(valid_metrics["recall"]),
                "test_roc_auc": float(test_metrics["roc_auc"]),
                "test_f1": float(test_metrics["f1"]),
                "test_precision": float(test_metrics["precision"]),
                "test_recall": float(test_metrics["recall"]),
                "params": json.dumps(params, sort_keys=True) if params is not None else "default",
            }
        )
    return rows


def write_notes(feature_sets: dict[str, list[str]]) -> None:
    notes = """# Location-Aware Refinement Notes

## Goal

Refine the current best location-aware baseline (`keep_location`) with a very small set of extra features drawn from the earlier Ines pipeline.

## Why This Design

We did not want another broad feature explosion. The refinement only adds features that were already strong in earlier experiments and that have clear meteorological meaning.

## Step-by-Step Choices

1. Start from the winning location-aware keep-location baseline.
2. Add only the strongest proven interaction and weather-state helpers:
   - `temp_range`
   - `humidity_temp_3pm_interaction`
   - `pressure_humidity_3pm_ratio`
   - `cloud_humidity_3pm_interaction`
   - `moisture_stability_3pm`
   - `humidity_rising_fast`
   - `warming_day`
3. Add a small anomaly block based on location-month climatology for selected columns.
4. Re-run only `XGBoost` and `CatBoost`, because they were already the strongest families.
5. Keep the same chronological split and validation-based threshold tuning.

## Feature Set Sizes

"""
    for name, features in feature_sets.items():
        notes += f"- `{name}`: {len(features)} features\n"
    NOTES_PATH.write_text(notes, encoding="utf-8")


def load_previous_best() -> dict[str, Any]:
    baseline_results = pd.read_csv(BASELINE_RESULTS_PATH)
    best_prev = baseline_results.sort_values(["test_f1", "test_roc_auc"], ascending=[False, False]).iloc[0].to_dict()
    return best_prev


def run_experiment() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_ready, valid_ready, test_ready, feature_sets = prepare_variant_frames()
    write_notes(feature_sets)

    rows: list[dict[str, Any]] = []
    for feature_set_name, features in feature_sets.items():
        rows.extend(evaluate_feature_set(feature_set_name, features, train_ready, valid_ready, test_ready))
        pd.DataFrame(rows).to_csv(SEARCH_RESULTS_PATH, index=False)

    results = pd.DataFrame(rows).sort_values(
        ["test_f1", "test_roc_auc", "validation_f1"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    results.to_csv(SEARCH_RESULTS_PATH, index=False)

    previous_best = load_previous_best()
    best_result = results.iloc[0].to_dict()
    summary = {
        "notes_path": str(NOTES_PATH),
        "feature_sets": {name: len(features) for name, features in feature_sets.items()},
        "previous_best_baseline": previous_best,
        "best_result": best_result,
        "improvement_vs_previous_best": {
            "previous_test_f1": float(previous_best["test_f1"]),
            "previous_test_roc_auc": float(previous_best["test_roc_auc"]),
            "delta_test_f1": float(best_result["test_f1"] - float(previous_best["test_f1"])),
            "delta_test_roc_auc": float(best_result["test_roc_auc"] - float(previous_best["test_roc_auc"])),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = run_experiment()
    print(json.dumps(summary["best_result"], indent=2))
    print(json.dumps(summary["improvement_vs_previous_best"], indent=2))
    print(f"Saved results to: {SEARCH_RESULTS_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")
    print(f"Saved notes to: {NOTES_PATH}")


if __name__ == "__main__":
    main()

