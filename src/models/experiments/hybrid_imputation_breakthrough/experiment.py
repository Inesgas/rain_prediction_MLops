from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.experiments.daily_zonal_baseline.experiment import (
    THRESHOLDS,
    WIND_COLUMNS,
    add_24h_diff,
    add_calendar_parts,
    add_daily_diffs,
    add_dewpoint_features,
    add_metadata,
    add_wind_shift,
    add_yesterday_lag,
    align_feature_columns,
    chronological_split,
    enrich_metadata_with_rainfall_zone,
    load_raw,
)
from src.models.experiments.location_aware_refinement.experiment import add_core_project_features
from src.models.ines_feature_modeling import (
    BEST_CATBOOST_PARAMS,
    BEST_XGB_PARAMS,
    TARGET,
    fit_model_by_name,
    predict_proba_for_model,
    score_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "models" / "hybrid_imputation_breakthrough"
SEARCH_RESULTS_PATH = RESULTS_DIR / "hybrid_regime_imputer_results.csv"
SUMMARY_PATH = RESULTS_DIR / "hybrid_regime_imputer_summary.json"
NOTES_PATH = RESULTS_DIR / "notes.md"
REFERENCES_PATH = RESULTS_DIR / "references.json"
PREVIOUS_SUMMARY_PATH = PROJECT_ROOT / "models" / "location_aware_refinement" / "location_refinement_summary.json"

TARGET_MODELS = [
    ("XGBoost", BEST_XGB_PARAMS),
    ("CatBoost", BEST_CATBOOST_PARAMS),
]
SELECTION_SORT_COLUMNS = [
    "validation_f1",
    "validation_roc_auc",
    "validation_precision",
    "validation_recall",
    "model",
    "feature_set",
]
SELECTION_SORT_ASCENDING = [False, False, False, False, True, True]

NUMERIC_COLUMNS = [
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
]
CATEGORICAL_COLUMNS = ["rain_today", "wind_gust_dir", "wind_dir_9am", "wind_dir_3pm"]
MISSING_FLAG_COLUMNS = ["rainfall", "evaporation", "sunshine", "cloud_9am", "cloud_3pm", "pressure_3pm", "humidity_3pm"]
REGIME_COLUMNS = ["humidity_9am", "pressure_9am", "temp_9am"]
NEIGHBOR_COUNT = 4

METHOD_REFERENCES = {
    "label": "Hybrid spatio-temporal weather-regime imputation",
    "idea_summary": (
        "This experiment uses a custom hybrid imputer: same-date nearby-station donor fill first, "
        "then train-only weather-regime medians by location/season as fallback."
    ),
    "source_links": [
        "C:/Users/user 1/Downloads/australia-rain-location-based-null-imputation.ipynb"
    ],
    "why_it_is_different": (
        "Unlike standard mean/median/KNN/Iterative imputers, this approach uses station geography, "
        "same-day cross-station agreement, and train-only seasonal weather regimes."
    ),
}


def tune_threshold_from_validation(proba: np.ndarray, y_true: pd.Series) -> tuple[float, dict[str, float]]:
    rows = []
    for threshold in THRESHOLDS:
        rows.append({"threshold": float(threshold), **score_predictions(y_true, proba, threshold=float(threshold))})
    frame = pd.DataFrame(rows)
    best_idx = frame["f1"].idxmax()
    best_threshold = float(frame.loc[best_idx, "threshold"])
    return best_threshold, {key: float(value) for key, value in frame.loc[best_idx].to_dict().items()}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in MISSING_FLAG_COLUMNS:
        if column in result.columns:
            result[f"{column}_missing_hybrid"] = result[column].isna().astype(int)
    return result


def build_neighbor_map(metadata_path: Path) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
    meta = pd.read_csv(metadata_path)
    neighbors: dict[str, list[str]] = {}
    weights: dict[str, list[float]] = {}
    for _, row in meta.iterrows():
        location = row["location"]
        distances: list[tuple[str, float]] = []
        for _, other in meta.iterrows():
            other_location = other["location"]
            if other_location == location:
                continue
            distance = haversine_km(float(row["lat"]), float(row["lon"]), float(other["lat"]), float(other["lon"]))
            distances.append((other_location, distance))
        distances.sort(key=lambda item: item[1])
        top = distances[:NEIGHBOR_COUNT]
        neighbors[location] = [item[0] for item in top]
        weights[location] = [1.0 / max(item[1], 1.0) for item in top]
    return neighbors, weights


def spatial_same_date_numeric_fill(
    frame: pd.DataFrame,
    columns: list[str],
    neighbors: dict[str, list[str]],
    neighbor_weights: dict[str, list[float]],
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            continue
        pivot = result.pivot_table(index="date", columns="location", values=column, aggfunc="mean")
        missing_mask = result[column].isna()
        if not missing_mask.any():
            continue
        for location, location_neighbors in neighbors.items():
            row_mask = missing_mask & (result["location"] == location)
            if not row_mask.any():
                continue
            available_neighbors = [neighbor for neighbor in location_neighbors if neighbor in pivot.columns]
            if not available_neighbors:
                continue
            dates = result.loc[row_mask, "date"]
            neighbor_frame = pivot.reindex(index=dates, columns=available_neighbors)
            weight_series = pd.Series(
                neighbor_weights[location][: len(available_neighbors)],
                index=available_neighbors,
                dtype="float64",
            )
            weighted_num = neighbor_frame.mul(weight_series, axis=1).sum(axis=1, skipna=True)
            weighted_den = neighbor_frame.notna().mul(weight_series, axis=1).sum(axis=1)
            fill_values = weighted_num.div(weighted_den.where(weighted_den > 0, np.nan))
            fill_series = pd.Series(fill_values.to_numpy(), index=result.loc[row_mask].index)
            result.loc[row_mask, column] = result.loc[row_mask, column].fillna(fill_series)
    return result


def spatial_same_date_categorical_fill(frame: pd.DataFrame, columns: list[str], neighbors: dict[str, list[str]]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            continue
        pivot = result.pivot_table(index="date", columns="location", values=column, aggfunc="first")
        missing_mask = result[column].isna()
        if not missing_mask.any():
            continue
        for location, location_neighbors in neighbors.items():
            row_mask = missing_mask & (result["location"] == location)
            if not row_mask.any():
                continue
            dates = result.loc[row_mask, "date"]
            fill_values = []
            for date in dates:
                vote = None
                for neighbor in location_neighbors:
                    if neighbor not in pivot.columns:
                        continue
                    value = pivot.at[date, neighbor] if date in pivot.index else np.nan
                    if pd.notna(value):
                        vote = value
                        break
                fill_values.append(vote)
            result.loc[row_mask, column] = result.loc[row_mask, column].fillna(pd.Series(fill_values, index=result.loc[row_mask].index))
    return result


def fit_regime_edges(train_df: pd.DataFrame) -> dict[str, list[float]]:
    edges: dict[str, list[float]] = {}
    for column in REGIME_COLUMNS:
        if column not in train_df.columns:
            continue
        series = train_df[column].dropna()
        if series.empty:
            continue
        quantiles = series.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
        deduped = sorted(set(float(x) for x in quantiles))
        if len(deduped) < 2:
            deduped = [float(series.min()) - 1.0, float(series.max()) + 1.0]
        else:
            deduped[0] -= 1e-6
            deduped[-1] += 1e-6
        edges[column] = deduped
    return edges


def add_regime_bins(df: pd.DataFrame, edges: dict[str, list[float]]) -> pd.DataFrame:
    result = df.copy()
    for column, bins in edges.items():
        if column not in result.columns:
            continue
        labels = list(range(len(bins) - 1))
        result[f"{column}_bin"] = (
            pd.cut(result[column], bins=bins, labels=labels, include_lowest=True)
            .astype("object")
            .where(lambda s: s.notna(), "missing")
            .astype(str)
        )
    return result


def fit_numeric_lookup_tables(train_df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    keep_cols = [col for col in columns if col in train_df.columns]
    return {
        "columns": keep_cols,
        "loc_month_regime": train_df.groupby(
            ["location", "month", "humidity_9am_bin", "pressure_9am_bin", "temp_9am_bin"], dropna=False
        )[keep_cols].median().reset_index(),
        "zone_month_regime": train_df.groupby(
            ["rainfall_zone", "month", "humidity_9am_bin", "pressure_9am_bin", "temp_9am_bin"], dropna=False
        )[keep_cols].median().reset_index(),
        "location_month": train_df.groupby(["location", "month"], dropna=False)[keep_cols].median().reset_index(),
        "zone_month": train_df.groupby(["rainfall_zone", "month"], dropna=False)[keep_cols].median().reset_index(),
        "global": {col: float(train_df[col].median()) for col in keep_cols},
    }


def fit_categorical_lookup_tables(train_df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    keep_cols = [col for col in columns if col in train_df.columns]

    def mode_agg(series: pd.Series) -> Any:
        modes = series.mode(dropna=True)
        return modes.iloc[0] if not modes.empty else np.nan

    return {
        "columns": keep_cols,
        "location_month": train_df.groupby(["location", "month"], dropna=False)[keep_cols].agg(mode_agg).reset_index(),
        "zone_month": train_df.groupby(["rainfall_zone", "month"], dropna=False)[keep_cols].agg(mode_agg).reset_index(),
        "global": {col: mode_agg(train_df[col]) for col in keep_cols},
    }


def _merge_lookup(df: pd.DataFrame, lookup: pd.DataFrame, keys: list[str], columns: list[str], suffix: str) -> pd.DataFrame:
    renamed = lookup.rename(columns={col: f"{col}{suffix}" for col in columns})
    return df.merge(renamed, on=keys, how="left")


def apply_numeric_lookup_fill(df: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    result = df.copy()
    columns = stats["columns"]
    result = _merge_lookup(
        result,
        stats["loc_month_regime"],
        ["location", "month", "humidity_9am_bin", "pressure_9am_bin", "temp_9am_bin"],
        columns,
        "__lmr",
    )
    result = _merge_lookup(
        result,
        stats["zone_month_regime"],
        ["rainfall_zone", "month", "humidity_9am_bin", "pressure_9am_bin", "temp_9am_bin"],
        columns,
        "__zmr",
    )
    result = _merge_lookup(result, stats["location_month"], ["location", "month"], columns, "__lm")
    result = _merge_lookup(result, stats["zone_month"], ["rainfall_zone", "month"], columns, "__zm")

    for column in columns:
        fill_values = result[f"{column}__lmr"]
        fill_values = fill_values.fillna(result[f"{column}__zmr"])
        fill_values = fill_values.fillna(result[f"{column}__lm"])
        fill_values = fill_values.fillna(result[f"{column}__zm"])
        fill_values = fill_values.fillna(stats["global"][column])
        result[column] = result[column].fillna(fill_values)

    helper_cols = [col for col in result.columns if any(col.endswith(suffix) for suffix in ["__lmr", "__zmr", "__lm", "__zm"])]
    return result.drop(columns=helper_cols, errors="ignore")


def apply_categorical_lookup_fill(df: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    result = df.copy()
    columns = stats["columns"]
    result = _merge_lookup(result, stats["location_month"], ["location", "month"], columns, "__lm")
    result = _merge_lookup(result, stats["zone_month"], ["rainfall_zone", "month"], columns, "__zm")

    for column in columns:
        fill_values = result[f"{column}__lm"].fillna(result[f"{column}__zm"]).fillna(stats["global"][column])
        result[column] = result[column].fillna(fill_values)

    helper_cols = [col for col in result.columns if any(col.endswith(suffix) for suffix in ["__lm", "__zm"])]
    return result.drop(columns=helper_cols, errors="ignore")


def add_circular_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["year_cycle_sin"] = np.sin(2 * np.pi * result["date"].dt.dayofyear / 365.25)
    result["year_cycle_cos"] = np.cos(2 * np.pi * result["date"].dt.dayofyear / 365.25)

    wd_map = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
        "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
        "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
        "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
    }
    for col in WIND_COLUMNS:
        if col in result.columns:
            degrees = result[col].map(wd_map)
            radians = np.radians(degrees)
            result[f"{col}_x"] = np.sin(radians)
            result[f"{col}_y"] = np.cos(radians)
            result = result.drop(columns=[col], errors="ignore")
    return result


def engineer_features(df: pd.DataFrame, drop_location: bool = False) -> pd.DataFrame:
    result = add_daily_diffs(df, ["temp", "humidity", "pressure"])
    result = add_dewpoint_features(result)
    result = add_circular_features(result)
    result = add_wind_shift(result)
    result = add_24h_diff(result)
    result = add_yesterday_lag(result, ["rainfall", "max_temp"])

    result = result.dropna(subset=[TARGET, "rain_today", "rainfall", "rainfall_yest", "max_temp_yest"]).copy()
    for col in ["rain_today", TARGET]:
        result[col] = result[col].map({"Yes": 1, "No": 0}).astype(int)

    result = pd.get_dummies(result, columns=["rainfall_zone"], drop_first=True)
    result = result.drop(columns=["date"], errors="ignore")
    if drop_location:
        result = result.drop(columns=["location"], errors="ignore")
    return result


def build_feature_sets(base_features: list[str], core_features: list[str]) -> dict[str, list[str]]:
    return {
        "hybrid_regime_keep_location_base": base_features,
        "hybrid_regime_keep_location_plus_core": list(dict.fromkeys(base_features + core_features)),
    }


def prepare_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    metadata_path = enrich_metadata_with_rainfall_zone()
    neighbors, neighbor_weights = build_neighbor_map(metadata_path)

    raw_df = load_raw()
    raw_df = raw_df[raw_df[TARGET].notna()].drop_duplicates().reset_index(drop=True)
    train_full, test_df = chronological_split(raw_df, test_size=0.20)
    train_df, valid_df = chronological_split(train_full, test_size=0.20)

    train_df = add_metadata(train_df, metadata_path)
    valid_df = add_metadata(valid_df, metadata_path)
    test_df = add_metadata(test_df, metadata_path)

    train_df = add_calendar_parts(train_df)
    valid_df = add_calendar_parts(valid_df)
    test_df = add_calendar_parts(test_df)

    train_df = add_missing_flags(train_df)
    valid_df = add_missing_flags(valid_df)
    test_df = add_missing_flags(test_df)

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
    categorical_stats = fit_categorical_lookup_tables(train_df, CATEGORICAL_COLUMNS)
    train_df = apply_numeric_lookup_fill(train_df, numeric_stats)
    valid_df = apply_numeric_lookup_fill(valid_df, numeric_stats)
    test_df = apply_numeric_lookup_fill(test_df, numeric_stats)
    train_df = apply_categorical_lookup_fill(train_df, categorical_stats)
    valid_df = apply_categorical_lookup_fill(valid_df, categorical_stats)
    test_df = apply_categorical_lookup_fill(test_df, categorical_stats)

    train_ready = engineer_features(train_df, drop_location=False)
    valid_ready = engineer_features(valid_df, drop_location=False)
    test_ready = engineer_features(test_df, drop_location=False)
    base_features = [col for col in train_ready.columns if col != TARGET]

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
    feature_sets = build_feature_sets(base_features, core_features)

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
    for model_name, params in TARGET_MODELS:
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
                "params": json.dumps(params, sort_keys=True),
            }
        )
    return rows


def write_notes(feature_sets: dict[str, list[str]]) -> None:
    notes = """# Hybrid Regime Imputer Notes

## Goal

Test a more innovative imputation strategy that does not mainly depend on the earlier daily-zonal median logic.

## Core Idea

This experiment uses a two-stage hybrid imputer:

1. Same-date nearby-station donor fill using inverse-distance weights.
2. Train-only weather-regime fallback medians by location, month, and coarse humidity / pressure / temperature regime bins.

## Why This Is Different

- It uses station geography directly.
- It uses same-day cross-station agreement when available.
- It uses learned weather regimes rather than one generic fill rule.
- It keeps the downstream modeling comparison stable with the stronger keep-location setup.

## Step-by-Step Design

1. Keep the chronological train / validation / test split.
2. Merge station metadata and rainfall-zone context.
3. Add missingness indicators for the most informative weather variables.
4. Build a nearest-station map from station latitude and longitude.
5. Fill numeric gaps from nearby stations on the same date with inverse-distance weighting.
6. Fill categorical gaps from the nearest available same-date station.
7. Learn train-only regime bins from `humidity_9am`, `pressure_9am`, and `temp_9am`.
8. Fill remaining gaps with hierarchical train-only medians:
   - location + month + regime
   - rainfall_zone + month + regime
   - location + month
   - rainfall_zone + month
   - global median / mode
9. Rebuild the engineered weather features and compare `XGBoost` and `CatBoost`.

## Feature Set Sizes

"""
    for name, features in feature_sets.items():
        notes += f"- `{name}`: {len(features)} features\n"
    NOTES_PATH.write_text(notes, encoding="utf-8")


def load_previous_best() -> dict[str, Any]:
    summary = json.loads(PREVIOUS_SUMMARY_PATH.read_text(encoding="utf-8"))
    return summary["best_result"]


def rank_results_for_selection(results: pd.DataFrame) -> pd.DataFrame:
    return results.sort_values(SELECTION_SORT_COLUMNS, ascending=SELECTION_SORT_ASCENDING).reset_index(drop=True)


def run_experiment() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCES_PATH.write_text(json.dumps(METHOD_REFERENCES, indent=2), encoding="utf-8")

    train_ready, valid_ready, test_ready, feature_sets = prepare_frames()
    write_notes(feature_sets)

    rows: list[dict[str, Any]] = []
    for feature_set_name, features in feature_sets.items():
        rows.extend(evaluate_feature_set(feature_set_name, features, train_ready, valid_ready, test_ready))
        pd.DataFrame(rows).to_csv(SEARCH_RESULTS_PATH, index=False)

    results = rank_results_for_selection(pd.DataFrame(rows))
    results.to_csv(SEARCH_RESULTS_PATH, index=False)

    previous_best = load_previous_best()
    best_result = results.iloc[0].to_dict()
    summary = {
        "notes_path": str(NOTES_PATH),
        "references_path": str(REFERENCES_PATH),
        "feature_sets": {name: len(features) for name, features in feature_sets.items()},
        "selection_basis": "validation_first_chronological_split",
        "previous_best": previous_best,
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

