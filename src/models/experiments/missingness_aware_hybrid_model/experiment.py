from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.experiments.hybrid_imputation_breakthrough.experiment import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    add_regime_bins,
    apply_categorical_lookup_fill,
    apply_numeric_lookup_fill,
    build_neighbor_map,
    fit_categorical_lookup_tables,
    fit_numeric_lookup_tables,
    fit_regime_edges,
    spatial_same_date_categorical_fill,
    spatial_same_date_numeric_fill,
    tune_threshold_from_validation,
)
from src.models.experiments.daily_zonal_baseline.experiment import (
    WIND_COLUMNS,
    add_24h_diff,
    add_calendar_parts,
    add_daily_diffs,
    add_dewpoint_features,
    add_metadata,
    add_yesterday_lag,
    align_feature_columns,
    enrich_metadata_with_rainfall_zone,
)
from src.models.experiments.location_aware_refinement.experiment import add_core_project_features
from src.utils.validation import BEST_FEATURE_SET_NAME, PipelineConfig, load_best_hybrid_selection, prepare_standard_split_frames
from src.utils.validation.hybrid_pipeline import (
    apply_numeric_lookup_fill_simple,
    fit_numeric_lookup_tables_simple,
    load_modeling_base_table,
    month_to_season,
)
from src.models.ines_feature_modeling import TARGET, fit_model_by_name, predict_proba_for_model, score_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "models" / "missingness_aware_hybrid_model"
DIAGNOSTIC_SUMMARY_PATH = RESULTS_DIR / "missingness_variable_summary.csv"
DIAGNOSTIC_SEASON_PATH = RESULTS_DIR / "missingness_by_season.csv"
DIAGNOSTIC_LOCATION_PATH = RESULTS_DIR / "missingness_by_location.csv"
RESULTS_PATH = RESULTS_DIR / "missing_family_design_results.csv"
SUMMARY_PATH = RESULTS_DIR / "missing_family_design_summary.json"
SEASON_FN_PATH = RESULTS_DIR / "variant_fn_by_season.csv"
LOCATION_FN_PATH = RESULTS_DIR / "variant_fn_by_location.csv"
BURDEN_FN_PATH = RESULTS_DIR / "variant_fn_by_missingness_burden.csv"
NOTES_PATH = RESULTS_DIR / "notes.md"
SELECTION_SORT_COLUMNS = [
    "validation_f1",
    "validation_roc_auc",
    "validation_precision",
    "validation_recall",
    "variant",
]
SELECTION_SORT_ASCENDING = [False, False, False, False, True]

OBSERVATIONAL_NUMERIC = ["evaporation", "sunshine", "cloud_9am", "cloud_3pm"]
DIRECTIONAL_COLUMNS = list(WIND_COLUMNS)
EXPANDED_FLAG_COLUMNS = [
    "min_temp",
    "max_temp",
    "rainfall",
    "evaporation",
    "sunshine",
    "wind_gust_speed",
    "wind_speed_9am",
    "wind_speed_3pm",
    "humidity_9am",
    "humidity_3pm",
    "pressure_9am",
    "pressure_3pm",
    "cloud_9am",
    "cloud_3pm",
    "temp_9am",
    "temp_3pm",
    "rain_today",
    "wind_gust_dir",
    "wind_dir_9am",
    "wind_dir_3pm",
]
STABLE_NUMERIC = [col for col in NUMERIC_COLUMNS if col not in OBSERVATIONAL_NUMERIC]


def write_notes() -> None:
    text = """# Variable-Specific Missingness Design Notes

## Goal

Test whether a more scientific missing-data strategy improves predictive quality more than adding more model complexity.

## Core Idea

Stop treating all missing variables the same.

Instead, split them into logical groups:

- stable physical numeric variables: temperature, pressure, humidity, rainfall, wind speed
- observational numeric variables: sunshine, evaporation, cloud
- directional categorical variables: wind direction columns

## Experiments

1. Current hybrid imputer baseline
2. Current hybrid imputer plus expanded missingness features
3. Variable-specific missing-data design by group

## Important Constraint

The current CatBoost wrapper still median-fills numeric NaNs before fitting, so this experiment uses conservative fallbacks rather than a raw leave-missing numeric path. That limitation is documented explicitly so the result is still interpretable.
"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text(text, encoding="utf-8")


def add_expanded_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in [col for col in EXPANDED_FLAG_COLUMNS if col in result.columns]:
        result[f"{column}_missing_family"] = result[column].isna().astype(int)
    return result


def build_missingness_diagnostics(raw_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frame = raw_df.copy()
    frame["season_label"] = frame["date"].dt.month.astype(int).map(month_to_season)
    rainy_mask = frame[TARGET] == "Yes"
    non_rainy_mask = frame[TARGET] == "No"
    variables = [col for col in EXPANDED_FLAG_COLUMNS if col in frame.columns]

    overall_rows: list[dict[str, Any]] = []
    season_rows: list[dict[str, Any]] = []
    location_rows: list[dict[str, Any]] = []

    for column in variables:
        overall_rows.append(
            {
                "variable": column,
                "missing_pct_overall": float(frame[column].isna().mean()),
                "missing_pct_rainy_days": float(frame.loc[rainy_mask, column].isna().mean()),
                "missing_pct_non_rainy_days": float(frame.loc[non_rainy_mask, column].isna().mean()),
                "rainy_minus_non_rainy_missing_pct": float(
                    frame.loc[rainy_mask, column].isna().mean() - frame.loc[non_rainy_mask, column].isna().mean()
                ),
            }
        )
        for season_value, group in frame.groupby("season_label", dropna=False):
            season_rows.append(
                {
                    "variable": column,
                    "season_label": season_value,
                    "support": int(len(group)),
                    "missing_pct": float(group[column].isna().mean()),
                }
            )
        for location_value, group in frame.groupby("location", dropna=False):
            location_rows.append(
                {
                    "variable": column,
                    "location": location_value,
                    "support": int(len(group)),
                    "missing_pct": float(group[column].isna().mean()),
                }
            )

    overall = pd.DataFrame(overall_rows).sort_values("rainy_minus_non_rainy_missing_pct", ascending=False).reset_index(drop=True)
    by_season = pd.DataFrame(season_rows).sort_values(["variable", "season_label"]).reset_index(drop=True)
    by_location = pd.DataFrame(location_rows).sort_values(["variable", "missing_pct"], ascending=[True, False]).reset_index(drop=True)
    return {
        "overall": overall,
        "by_season": by_season,
        "by_location": by_location,
    }


def build_feature_sets(base_features: list[str], core_features: list[str]) -> dict[str, list[str]]:
    feature_sets = {"hybrid_regime_keep_location_base": base_features}
    if core_features:
        feature_sets[BEST_FEATURE_SET_NAME] = list(dict.fromkeys(base_features + core_features))
    return feature_sets


def prepare_base_frame(df: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    result = add_metadata(df.copy(), metadata_path)
    return add_calendar_parts(result)


def prepare_hybrid_expanded_missing_frames(
    keep_date: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    metadata_path = enrich_metadata_with_rainfall_zone()
    neighbors, neighbor_weights = build_neighbor_map(metadata_path)
    raw_df = load_modeling_base_table()
    train_full, test_raw = raw_df.sort_values("date").iloc[: int(len(raw_df) * 0.8)].copy(), raw_df.sort_values("date").iloc[int(len(raw_df) * 0.8):].copy()
    train_raw, valid_raw = train_full.iloc[: int(len(train_full) * 0.8)].copy(), train_full.iloc[int(len(train_full) * 0.8):].copy()

    train_df = add_expanded_missing_flags(prepare_base_frame(train_raw, metadata_path))
    valid_df = add_expanded_missing_flags(prepare_base_frame(valid_raw, metadata_path))
    test_df = add_expanded_missing_flags(prepare_base_frame(test_raw, metadata_path))

    train_df = spatial_same_date_numeric_fill(train_df, NUMERIC_COLUMNS, neighbors, neighbor_weights)
    valid_df = spatial_same_date_numeric_fill(valid_df, NUMERIC_COLUMNS, neighbors, neighbor_weights)
    test_df = spatial_same_date_numeric_fill(test_df, NUMERIC_COLUMNS, neighbors, neighbor_weights)
    train_df = spatial_same_date_categorical_fill(train_df, CATEGORICAL_COLUMNS, neighbors)
    valid_df = spatial_same_date_categorical_fill(valid_df, CATEGORICAL_COLUMNS, neighbors)
    test_df = spatial_same_date_categorical_fill(test_df, CATEGORICAL_COLUMNS, neighbors)

    regime_edges = fit_regime_edges(train_df)
    train_df = add_regime_bins(train_df, regime_edges)
    valid_df = add_regime_bins(valid_df, regime_edges)
    test_df = add_regime_bins(test_df, regime_edges)

    numeric_stats = fit_numeric_lookup_tables(train_df, NUMERIC_COLUMNS)
    train_df = apply_numeric_lookup_fill(train_df, numeric_stats)
    valid_df = apply_numeric_lookup_fill(valid_df, numeric_stats)
    test_df = apply_numeric_lookup_fill(test_df, numeric_stats)

    categorical_stats = fit_categorical_lookup_tables(train_df, CATEGORICAL_COLUMNS)
    train_df = apply_categorical_lookup_fill(train_df, categorical_stats)
    valid_df = apply_categorical_lookup_fill(valid_df, categorical_stats)
    test_df = apply_categorical_lookup_fill(test_df, categorical_stats)

    train_ready = engineer_features_hybrid_style(train_df)
    valid_ready = engineer_features_hybrid_style(valid_df)
    test_ready = engineer_features_hybrid_style(test_df)
    base_features = [col for col in train_ready.columns if col != TARGET]

    train_ready = add_core_project_features(train_ready)
    valid_ready = add_core_project_features(valid_ready)
    test_ready = add_core_project_features(test_ready)
    core_features = [col for col in [
        "temp_range",
        "humidity_temp_3pm_interaction",
        "pressure_humidity_3pm_ratio",
        "cloud_humidity_3pm_interaction",
        "moisture_stability_3pm",
        "humidity_rising_fast",
        "warming_day",
    ] if col in train_ready.columns]
    feature_sets = build_feature_sets(base_features, core_features)
    if keep_date:
        train_ready["date"] = pd.to_datetime(train_df.loc[train_ready.index, "date"])
        valid_ready["date"] = pd.to_datetime(valid_df.loc[valid_ready.index, "date"])
        test_ready["date"] = pd.to_datetime(test_df.loc[test_ready.index, "date"])
    train_ready, valid_ready, test_ready = align_feature_columns(train_ready, valid_ready, test_ready)
    return train_ready, valid_ready, test_ready, feature_sets


def engineer_features_hybrid_style(df: pd.DataFrame) -> pd.DataFrame:
    from src.models.experiments.hybrid_imputation_breakthrough.experiment import engineer_features

    return engineer_features(df, drop_location=False)


def engineer_features_variable_specific(df: pd.DataFrame) -> pd.DataFrame:
    result = add_daily_diffs(df, ["temp", "humidity", "pressure", "cloud"])
    result = add_dewpoint_features(result)
    result["year_cycle_sin"] = np.sin(2 * np.pi * result["date"].dt.dayofyear / 365.25)
    result["year_cycle_cos"] = np.cos(2 * np.pi * result["date"].dt.dayofyear / 365.25)
    result = add_24h_diff(result, ["max_temp", "pressure_3pm", "humidity_3pm", "wind_gust_speed"])
    result = add_yesterday_lag(result, ["rainfall", "max_temp"])

    for column in [col for col in DIRECTIONAL_COLUMNS if col in result.columns]:
        result[f"{column}_explicit_missing"] = result[column].isna().astype(int)
        result[column] = result[column].fillna("Missing").astype(str)

    result["wind_missing_count"] = result[[col for col in result.columns if col.endswith("_explicit_missing")]].sum(axis=1)
    result = result.dropna(subset=[TARGET, "rain_today", "rainfall", "rainfall_yest", "max_temp_yest"]).copy()
    for col in ["rain_today", TARGET]:
        result[col] = result[col].map({"Yes": 1, "No": 0}).astype(int)

    result = pd.get_dummies(result, columns=["rainfall_zone"], drop_first=True)
    result = result.drop(columns=["date"], errors="ignore")
    return result


def prepare_variable_specific_frames(
    keep_date: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    metadata_path = enrich_metadata_with_rainfall_zone()
    neighbors, neighbor_weights = build_neighbor_map(metadata_path)
    raw_df = load_modeling_base_table()
    ordered = raw_df.sort_values("date").reset_index(drop=True)
    split_test = int(len(ordered) * 0.8)
    train_full = ordered.iloc[:split_test].copy()
    test_raw = ordered.iloc[split_test:].copy()
    split_valid = int(len(train_full) * 0.8)
    train_raw = train_full.iloc[:split_valid].copy()
    valid_raw = train_full.iloc[split_valid:].copy()

    train_df = add_expanded_missing_flags(prepare_base_frame(train_raw, metadata_path))
    valid_df = add_expanded_missing_flags(prepare_base_frame(valid_raw, metadata_path))
    test_df = add_expanded_missing_flags(prepare_base_frame(test_raw, metadata_path))

    train_df = spatial_same_date_numeric_fill(train_df, STABLE_NUMERIC, neighbors, neighbor_weights)
    valid_df = spatial_same_date_numeric_fill(valid_df, STABLE_NUMERIC, neighbors, neighbor_weights)
    test_df = spatial_same_date_numeric_fill(test_df, STABLE_NUMERIC, neighbors, neighbor_weights)

    regime_edges = fit_regime_edges(train_df)
    train_df = add_regime_bins(train_df, regime_edges)
    valid_df = add_regime_bins(valid_df, regime_edges)
    test_df = add_regime_bins(test_df, regime_edges)

    stable_stats = fit_numeric_lookup_tables(train_df, STABLE_NUMERIC)
    train_df = apply_numeric_lookup_fill(train_df, stable_stats)
    valid_df = apply_numeric_lookup_fill(valid_df, stable_stats)
    test_df = apply_numeric_lookup_fill(test_df, stable_stats)

    obs_stats = fit_numeric_lookup_tables_simple(train_df, OBSERVATIONAL_NUMERIC)
    train_df = apply_numeric_lookup_fill_simple(train_df, obs_stats)
    valid_df = apply_numeric_lookup_fill_simple(valid_df, obs_stats)
    test_df = apply_numeric_lookup_fill_simple(test_df, obs_stats)

    rain_today_stats = fit_categorical_lookup_tables(train_df, ["rain_today"])
    train_df = apply_categorical_lookup_fill(train_df, rain_today_stats)
    valid_df = apply_categorical_lookup_fill(valid_df, rain_today_stats)
    test_df = apply_categorical_lookup_fill(test_df, rain_today_stats)

    train_ready = engineer_features_variable_specific(train_df)
    valid_ready = engineer_features_variable_specific(valid_df)
    test_ready = engineer_features_variable_specific(test_df)
    base_features = [col for col in train_ready.columns if col != TARGET]

    train_ready = add_core_project_features(train_ready)
    valid_ready = add_core_project_features(valid_ready)
    test_ready = add_core_project_features(test_ready)
    core_features = [col for col in [
        "temp_range",
        "humidity_temp_3pm_interaction",
        "pressure_humidity_3pm_ratio",
        "cloud_humidity_3pm_interaction",
        "moisture_stability_3pm",
        "humidity_rising_fast",
        "warming_day",
    ] if col in train_ready.columns]
    feature_sets = {"variable_specific_missing_design": list(dict.fromkeys(base_features + core_features))}
    if keep_date:
        train_ready["date"] = pd.to_datetime(train_df.loc[train_ready.index, "date"])
        valid_ready["date"] = pd.to_datetime(valid_df.loc[valid_ready.index, "date"])
        test_ready["date"] = pd.to_datetime(test_df.loc[test_ready.index, "date"])
    train_ready, valid_ready, test_ready = align_feature_columns(train_ready, valid_ready, test_ready)
    return train_ready, valid_ready, test_ready, feature_sets


def burden_bucket(value: float) -> str:
    if value <= 0:
        return "0"
    if value <= 2:
        return "1-2"
    if value <= 4:
        return "3-4"
    return "5+"


def grouped_error_summary(df: pd.DataFrame, group_column: str, min_actual_rainy: int = 1) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_value, group in df.groupby(group_column, dropna=False):
        actual_rainy = int((group["y_true"] == 1).sum())
        if actual_rainy < min_actual_rainy:
            continue
        tp = int(((group["y_true"] == 1) & (group["y_pred"] == 1)).sum())
        fn = int(((group["y_true"] == 1) & (group["y_pred"] == 0)).sum())
        rows.append(
            {
                group_column: group_value,
                "support": int(len(group)),
                "actual_rainy": actual_rainy,
                "tp": tp,
                "fn": fn,
                "fn_rate_among_rainy": float(fn / actual_rainy) if actual_rainy else np.nan,
                "recall_among_rainy": float(tp / actual_rainy) if actual_rainy else np.nan,
                "mean_probability": float(group["proba"].mean()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["fn_rate_among_rainy", "actual_rainy"], ascending=[False, False]).reset_index(drop=True)


def evaluate_variant(
    variant_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
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
    test_pred = (test_proba >= best_threshold).astype(int)

    flag_columns = [col for col in test_df.columns if "_missing_" in col or col.endswith("_explicit_missing")]
    analysis = pd.DataFrame(
        {
            "variant": variant_name,
            "location": test_df["location"].astype(str) if "location" in test_df.columns else "Unknown",
            "season_label": test_df["month"].astype(int).map(month_to_season) if "month" in test_df.columns else "Unknown",
            "y_true": y_test.to_numpy(),
            "y_pred": test_pred,
            "proba": test_proba,
            "missingness_burden": test_df[flag_columns].sum(axis=1) if flag_columns else 0,
        }
    )
    analysis["missingness_burden_bucket"] = analysis["missingness_burden"].map(burden_bucket)

    row = {
        "variant": variant_name,
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
    return row, analysis


def rank_results_for_selection(results: pd.DataFrame) -> pd.DataFrame:
    return results.sort_values(SELECTION_SORT_COLUMNS, ascending=SELECTION_SORT_ASCENDING).reset_index(drop=True)


def load_missingness_winner_selection() -> dict[str, Any]:
    if SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return run_experiment()


def prepare_final_winner_frames(
    keep_date: bool = False,
) -> tuple[str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]], list[str]]:
    summary = load_missingness_winner_selection()
    best_result = summary.get("best_result", {})
    winner_variant = str(best_result.get("variant", "current_hybrid_baseline"))

    if winner_variant == "hybrid_plus_expanded_missingness":
        train_df, valid_df, test_df, feature_sets = prepare_hybrid_expanded_missing_frames(keep_date=keep_date)
        feature_set_name = BEST_FEATURE_SET_NAME
    elif winner_variant == "variable_specific_missing_design":
        train_df, valid_df, test_df, feature_sets = prepare_variable_specific_frames(keep_date=keep_date)
        feature_set_name = "variable_specific_missing_design"
    else:
        winner_variant = "current_hybrid_baseline"
        train_df, valid_df, test_df, feature_sets = prepare_standard_split_frames(
            PipelineConfig(name="hybrid_default"),
            keep_date=keep_date,
        )
        feature_set_name = BEST_FEATURE_SET_NAME

    features = feature_sets[feature_set_name]
    return winner_variant, feature_set_name, train_df, valid_df, test_df, feature_sets, features


def run_experiment() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_notes()
    raw_df = load_modeling_base_table()
    diagnostics = build_missingness_diagnostics(raw_df)
    diagnostics["overall"].to_csv(DIAGNOSTIC_SUMMARY_PATH, index=False)
    diagnostics["by_season"].to_csv(DIAGNOSTIC_SEASON_PATH, index=False)
    diagnostics["by_location"].to_csv(DIAGNOSTIC_LOCATION_PATH, index=False)

    best_selection = load_best_hybrid_selection()
    params = best_selection["params"]

    rows: list[dict[str, Any]] = []
    season_summaries: list[pd.DataFrame] = []
    location_summaries: list[pd.DataFrame] = []
    burden_summaries: list[pd.DataFrame] = []

    baseline_train, baseline_valid, baseline_test, baseline_feature_sets = prepare_standard_split_frames(PipelineConfig(name="hybrid_default"))
    baseline_features = baseline_feature_sets[BEST_FEATURE_SET_NAME]
    row, analysis = evaluate_variant("current_hybrid_baseline", baseline_train, baseline_valid, baseline_test, baseline_features, params)
    rows.append(row)
    season_frame = grouped_error_summary(analysis, "season_label")
    location_frame = grouped_error_summary(analysis, "location", min_actual_rainy=20)
    burden_frame = grouped_error_summary(analysis, "missingness_burden_bucket")
    season_frame["variant"] = "current_hybrid_baseline"
    location_frame["variant"] = "current_hybrid_baseline"
    burden_frame["variant"] = "current_hybrid_baseline"
    season_summaries.append(season_frame)
    location_summaries.append(location_frame)
    burden_summaries.append(burden_frame)

    expanded_train, expanded_valid, expanded_test, expanded_feature_sets = prepare_hybrid_expanded_missing_frames()
    expanded_features = expanded_feature_sets[BEST_FEATURE_SET_NAME]
    row, analysis = evaluate_variant("hybrid_plus_expanded_missingness", expanded_train, expanded_valid, expanded_test, expanded_features, params)
    rows.append(row)
    season_frame = grouped_error_summary(analysis, "season_label")
    location_frame = grouped_error_summary(analysis, "location", min_actual_rainy=20)
    burden_frame = grouped_error_summary(analysis, "missingness_burden_bucket")
    season_frame["variant"] = "hybrid_plus_expanded_missingness"
    location_frame["variant"] = "hybrid_plus_expanded_missingness"
    burden_frame["variant"] = "hybrid_plus_expanded_missingness"
    season_summaries.append(season_frame)
    location_summaries.append(location_frame)
    burden_summaries.append(burden_frame)

    family_train, family_valid, family_test, family_feature_sets = prepare_variable_specific_frames()
    family_features = family_feature_sets["variable_specific_missing_design"]
    row, analysis = evaluate_variant("variable_specific_missing_design", family_train, family_valid, family_test, family_features, params)
    rows.append(row)
    season_frame = grouped_error_summary(analysis, "season_label")
    location_frame = grouped_error_summary(analysis, "location", min_actual_rainy=20)
    burden_frame = grouped_error_summary(analysis, "missingness_burden_bucket")
    season_frame["variant"] = "variable_specific_missing_design"
    location_frame["variant"] = "variable_specific_missing_design"
    burden_frame["variant"] = "variable_specific_missing_design"
    season_summaries.append(season_frame)
    location_summaries.append(location_frame)
    burden_summaries.append(burden_frame)

    results = rank_results_for_selection(pd.DataFrame(rows))
    results.to_csv(RESULTS_PATH, index=False)
    pd.concat(season_summaries, ignore_index=True).to_csv(SEASON_FN_PATH, index=False)
    pd.concat(location_summaries, ignore_index=True).to_csv(LOCATION_FN_PATH, index=False)
    pd.concat(burden_summaries, ignore_index=True).to_csv(BURDEN_FN_PATH, index=False)

    best_row = results.iloc[0].to_dict()
    summary = {
        "experiment": "variable_specific_missingness_design",
        "selection_basis": "validation_first_chronological_split",
        "best_result": {key: (float(value) if isinstance(value, (int, float)) else value) for key, value in best_row.items()},
        "top_predictive_missingness": diagnostics["overall"].head(5).to_dict(orient="records"),
        "results_path": str(RESULTS_PATH),
        "diagnostic_summary_path": str(DIAGNOSTIC_SUMMARY_PATH),
        "notes_path": str(NOTES_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))

