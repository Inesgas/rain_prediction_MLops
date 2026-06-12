from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.experiments.daily_zonal_baseline.experiment import align_feature_columns
from src.features.feature_pipeline import (
    add_koppen_zone,
    add_longitude_strips,
    add_ncc_zones,
    add_season_zone_interactions,
    add_seasonal_rainfall_zone,
    add_temperature_humidity_zone,
)
from src.models.experiments.hybrid_imputation_breakthrough.experiment import tune_threshold_from_validation
from src.models.ines_feature_modeling import TARGET, fit_model_by_name, predict_proba_for_model, score_predictions
from src.utils.validation import (
    BEST_FEATURE_SET_NAME,
    PipelineConfig,
    load_best_hybrid_selection,
    load_modeling_base_table,
    prepare_standard_split_frames,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "models" / "geo_climate_context_extension"
RESULTS_PATH = RESULTS_DIR / "geo_climate_context_results.csv"
SUMMARY_PATH = RESULTS_DIR / "geo_climate_context_summary.json"
NOTES_PATH = RESULTS_DIR / "notes.md"

RAW_MISSINGNESS_COLUMNS = [
    "wind_gust_speed",
    "humidity_9am",
    "pressure_9am",
]

SELECTION_SORT_COLUMNS = [
    "validation_f1",
    "validation_roc_auc",
    "validation_precision",
    "validation_recall",
    "feature_count",
    "variant",
]
SELECTION_SORT_ASCENDING = [False, False, False, False, True, True]


def write_notes() -> None:
    text = """# Geo-Climate Context Extension Notes

## Goal

Test whether the current hybrid CatBoost winner can improve by adding only a small,
neutral set of extra regional and climate-context signals.

## Design Rules

1. Keep the current hybrid baseline unchanged as the control.
2. Reuse the same chronological train / validation / test protocol.
3. Reuse the same CatBoost hyperparameters as the locked hybrid baseline.
4. Tune the decision threshold on validation only.
5. Add context in small blocks so any lift is easy to explain.

## Context Blocks

- missingness context: a few raw missing-value indicators not present in the baseline
- regional context: finer regional labels such as NCC zone and longitude strip
- climate-zone context: gridded climate labels derived from station coordinates
- seasonal climate context: season-by-zone interaction labels
"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text(text, encoding="utf-8")


def build_missingness_lookup() -> pd.DataFrame:
    raw_df = load_modeling_base_table().copy()
    keep_cols = ["date", "location", *RAW_MISSINGNESS_COLUMNS]
    lookup = raw_df[keep_cols].drop_duplicates(subset=["date", "location"]).copy()
    for column in RAW_MISSINGNESS_COLUMNS:
        lookup[f"{column}_missing_context"] = lookup[column].isna().astype(int)
    return lookup[["date", "location", *[f"{column}_missing_context" for column in RAW_MISSINGNESS_COLUMNS]]]


def add_missingness_context(frame: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    result = frame.merge(lookup, on=["date", "location"], how="left")
    for column in RAW_MISSINGNESS_COLUMNS:
        flag_col = f"{column}_missing_context"
        result[flag_col] = result[flag_col].fillna(0).astype(int)
    return result


def add_geo_climate_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result = add_ncc_zones(result)
    result = add_longitude_strips(result)
    result = add_koppen_zone(result)
    result = add_seasonal_rainfall_zone(result)
    result = add_temperature_humidity_zone(result)
    result = add_season_zone_interactions(result)

    if "season_label" in result.columns:
        result = result.drop(columns=["season_label"])

    dummy_columns = [
        column
        for column in [
            "ncc_zone",
            "land_strip",
            "koppen_zone",
            "seasonal_rainfall_zone",
            "temperature_humidity_zone",
            "season_ncc_zone",
            "season_koppen_zone",
        ]
        if column in result.columns
    ]
    if dummy_columns:
        result = pd.get_dummies(result, columns=dummy_columns, dtype=int)
    return result


def collect_context_feature_blocks(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "missingness_context": sorted([col for col in frame.columns if col.endswith("_missing_context")]),
        "regional_context": sorted(
            [
                col
                for col in frame.columns
                if col.startswith("ncc_zone_") or col.startswith("land_strip_")
            ]
        ),
        "climate_zone_context": sorted(
            [
                col
                for col in frame.columns
                if col.startswith("koppen_zone_")
                or col.startswith("seasonal_rainfall_zone_")
                or col.startswith("temperature_humidity_zone_")
            ]
        ),
        "seasonal_context": sorted(
            [
                col
                for col in frame.columns
                if col.startswith("season_ncc_zone_") or col.startswith("season_koppen_zone_")
            ]
        ),
    }


def prepare_variant_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]], dict[str, list[str]]]:
    train_df, valid_df, test_df, feature_sets = prepare_standard_split_frames(
        PipelineConfig(name="hybrid_default"),
        keep_date=True,
    )
    baseline_features = list(feature_sets[BEST_FEATURE_SET_NAME])

    missingness_lookup = build_missingness_lookup()
    train_context = add_geo_climate_context(add_missingness_context(train_df, missingness_lookup))
    valid_context = add_geo_climate_context(add_missingness_context(valid_df, missingness_lookup))
    test_context = add_geo_climate_context(add_missingness_context(test_df, missingness_lookup))
    train_context, valid_context, test_context = align_feature_columns(train_context, valid_context, test_context)

    context_blocks = collect_context_feature_blocks(train_context)
    feature_sets: dict[str, list[str]] = {"current_hybrid_baseline": baseline_features}
    candidate_specs = [
        (
            "hybrid_plus_geo_context",
            context_blocks["missingness_context"] + context_blocks["regional_context"],
        ),
        (
            "hybrid_plus_geo_climate_context",
            context_blocks["missingness_context"]
            + context_blocks["regional_context"]
            + context_blocks["climate_zone_context"],
        ),
        (
            "hybrid_plus_geo_climate_season_context",
            context_blocks["missingness_context"]
            + context_blocks["regional_context"]
            + context_blocks["climate_zone_context"]
            + context_blocks["seasonal_context"],
        ),
    ]

    seen_feature_signatures = {tuple(baseline_features)}
    for variant_name, extra_columns in candidate_specs:
        if not extra_columns:
            continue
        variant_features = list(dict.fromkeys(baseline_features + extra_columns))
        signature = tuple(variant_features)
        if signature in seen_feature_signatures:
            continue
        feature_sets[variant_name] = variant_features
        seen_feature_signatures.add(signature)
    return train_context, valid_context, test_context, feature_sets, context_blocks


def evaluate_variant(
    variant: str,
    features: list[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict[str, float | int],
) -> dict[str, Any]:
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
        "variant": variant,
        "model": "CatBoost",
        "feature_count": int(len(features)),
        "validation_threshold": float(best_threshold),
        "validation_f1": float(valid_metrics["f1"]),
        "validation_roc_auc": float(valid_metrics["roc_auc"]),
        "validation_precision": float(valid_metrics["precision"]),
        "validation_recall": float(valid_metrics["recall"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "test_f1": float(test_metrics["f1"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "params": json.dumps(params, sort_keys=True),
    }


def rank_results_for_selection(results: pd.DataFrame) -> pd.DataFrame:
    return results.sort_values(SELECTION_SORT_COLUMNS, ascending=SELECTION_SORT_ASCENDING).reset_index(drop=True)


def run_experiment() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_notes()

    selection = load_best_hybrid_selection()
    params = dict(selection["params"])

    train_df, valid_df, test_df, feature_sets, context_blocks = prepare_variant_frames()
    rows = [
        evaluate_variant(variant, features, train_df, valid_df, test_df, params)
        for variant, features in feature_sets.items()
    ]
    results = rank_results_for_selection(pd.DataFrame(rows))
    results.to_csv(RESULTS_PATH, index=False)

    baseline_row = results.loc[results["variant"] == "current_hybrid_baseline"].iloc[0].to_dict()
    best_row = results.iloc[0].to_dict()
    summary = {
        "experiment": "geo_climate_context_extension",
        "selection_basis": "validation_first_chronological_split",
        "baseline_variant": "current_hybrid_baseline",
        "baseline_result": {key: (float(value) if isinstance(value, (int, float)) else value) for key, value in baseline_row.items()},
        "best_result": {key: (float(value) if isinstance(value, (int, float)) else value) for key, value in best_row.items()},
        "new_context_beats_baseline": bool(best_row["variant"] != "current_hybrid_baseline"),
        "improvement_vs_baseline": {
            "delta_validation_f1": float(best_row["validation_f1"] - baseline_row["validation_f1"]),
            "delta_validation_roc_auc": float(best_row["validation_roc_auc"] - baseline_row["validation_roc_auc"]),
            "delta_test_f1": float(best_row["test_f1"] - baseline_row["test_f1"]),
            "delta_test_roc_auc": float(best_row["test_roc_auc"] - baseline_row["test_roc_auc"]),
            "delta_test_precision": float(best_row["test_precision"] - baseline_row["test_precision"]),
            "delta_test_recall": float(best_row["test_recall"] - baseline_row["test_recall"]),
        },
        "context_block_sizes": {key: int(len(value)) for key, value in context_blocks.items()},
        "results_path": str(RESULTS_PATH),
        "notes_path": str(NOTES_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["best_result"], indent=2))
    return summary


if __name__ == "__main__":
    run_experiment()

