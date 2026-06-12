from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.models.experiments.daily_zonal_baseline.experiment import (
    add_calendar_parts,
    add_metadata,
    align_feature_columns,
    chronological_split,
    enrich_metadata_with_rainfall_zone,
    load_raw,
)
from src.models.experiments.hybrid_imputation_breakthrough.experiment import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    add_missing_flags,
    add_regime_bins,
    apply_categorical_lookup_fill,
    apply_numeric_lookup_fill,
    build_neighbor_map,
    engineer_features,
    fit_categorical_lookup_tables,
    fit_numeric_lookup_tables,
    fit_regime_edges,
    spatial_same_date_categorical_fill,
    spatial_same_date_numeric_fill,
)
from src.models.experiments.location_aware_refinement.experiment import add_core_project_features
from src.models.ines_feature_modeling import TARGET

from .config import BEST_FEATURE_SET_NAME, CORE_FEATURE_COLUMNS, PipelineConfig


def load_modeling_base_table() -> pd.DataFrame:
    raw_df = load_raw()
    return raw_df[raw_df[TARGET].notna()].drop_duplicates().reset_index(drop=True)


def fit_numeric_lookup_tables_simple(train_df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    keep_cols = [col for col in columns if col in train_df.columns]
    return {
        "columns": keep_cols,
        "location_month": train_df.groupby(["location", "month"], dropna=False)[keep_cols].median().reset_index(),
        "zone_month": train_df.groupby(["rainfall_zone", "month"], dropna=False)[keep_cols].median().reset_index(),
        "global": {col: float(train_df[col].median()) for col in keep_cols},
    }


def _merge_lookup(
    df: pd.DataFrame,
    lookup: pd.DataFrame,
    keys: list[str],
    columns: list[str],
    suffix: str,
) -> pd.DataFrame:
    renamed = lookup.rename(columns={col: f"{col}{suffix}" for col in columns})
    return df.merge(renamed, on=keys, how="left")


def apply_numeric_lookup_fill_simple(df: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    result = df.copy()
    columns = stats["columns"]
    result = _merge_lookup(result, stats["location_month"], ["location", "month"], columns, "__lm")
    result = _merge_lookup(result, stats["zone_month"], ["rainfall_zone", "month"], columns, "__zm")

    for column in columns:
        fill_values = result[f"{column}__lm"].fillna(result[f"{column}__zm"]).fillna(stats["global"][column])
        result[column] = result[column].fillna(fill_values)

    helper_cols = [col for col in result.columns if col.endswith("__lm") or col.endswith("__zm")]
    return result.drop(columns=helper_cols, errors="ignore")


def build_feature_sets(base_features: list[str], core_features: list[str]) -> dict[str, list[str]]:
    feature_sets = {"hybrid_regime_keep_location_base": base_features}
    if core_features:
        feature_sets[BEST_FEATURE_SET_NAME] = list(dict.fromkeys(base_features + core_features))
    return feature_sets


def best_feature_list(feature_sets: dict[str, list[str]]) -> list[str]:
    if BEST_FEATURE_SET_NAME in feature_sets:
        return feature_sets[BEST_FEATURE_SET_NAME]
    return next(iter(feature_sets.values()))


def _prepare_partition(df: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    result = add_metadata(df.copy(), metadata_path)
    return add_calendar_parts(result)


def prepare_configured_frames(
    train_raw: pd.DataFrame,
    valid_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    config: PipelineConfig,
    keep_date: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    metadata_path = enrich_metadata_with_rainfall_zone()
    neighbors, neighbor_weights = build_neighbor_map(metadata_path)

    train_df = _prepare_partition(train_raw, metadata_path)
    valid_df = _prepare_partition(valid_raw, metadata_path)
    test_df = _prepare_partition(test_raw, metadata_path)

    if config.add_missing_indicators:
        train_df = add_missing_flags(train_df)
        valid_df = add_missing_flags(valid_df)
        test_df = add_missing_flags(test_df)

    if config.use_spatial_fill:
        train_df = spatial_same_date_numeric_fill(train_df, NUMERIC_COLUMNS, neighbors, neighbor_weights)
        valid_df = spatial_same_date_numeric_fill(valid_df, NUMERIC_COLUMNS, neighbors, neighbor_weights)
        test_df = spatial_same_date_numeric_fill(test_df, NUMERIC_COLUMNS, neighbors, neighbor_weights)

        train_df = spatial_same_date_categorical_fill(train_df, CATEGORICAL_COLUMNS, neighbors)
        valid_df = spatial_same_date_categorical_fill(valid_df, CATEGORICAL_COLUMNS, neighbors)
        test_df = spatial_same_date_categorical_fill(test_df, CATEGORICAL_COLUMNS, neighbors)

    if config.use_regime_fill:
        regime_edges = fit_regime_edges(train_df)
        train_df = add_regime_bins(train_df, regime_edges)
        valid_df = add_regime_bins(valid_df, regime_edges)
        test_df = add_regime_bins(test_df, regime_edges)
        numeric_stats = fit_numeric_lookup_tables(train_df, NUMERIC_COLUMNS)
        train_df = apply_numeric_lookup_fill(train_df, numeric_stats)
        valid_df = apply_numeric_lookup_fill(valid_df, numeric_stats)
        test_df = apply_numeric_lookup_fill(test_df, numeric_stats)
    else:
        numeric_stats = fit_numeric_lookup_tables_simple(train_df, NUMERIC_COLUMNS)
        train_df = apply_numeric_lookup_fill_simple(train_df, numeric_stats)
        valid_df = apply_numeric_lookup_fill_simple(valid_df, numeric_stats)
        test_df = apply_numeric_lookup_fill_simple(test_df, numeric_stats)

    categorical_stats = fit_categorical_lookup_tables(train_df, CATEGORICAL_COLUMNS)
    train_df = apply_categorical_lookup_fill(train_df, categorical_stats)
    valid_df = apply_categorical_lookup_fill(valid_df, categorical_stats)
    test_df = apply_categorical_lookup_fill(test_df, categorical_stats)

    train_ready = engineer_features(train_df, drop_location=config.drop_location)
    valid_ready = engineer_features(valid_df, drop_location=config.drop_location)
    test_ready = engineer_features(test_df, drop_location=config.drop_location)
    base_features = [col for col in train_ready.columns if col != TARGET]

    core_features: list[str] = []
    if config.add_core_features:
        train_ready = add_core_project_features(train_ready)
        valid_ready = add_core_project_features(valid_ready)
        test_ready = add_core_project_features(test_ready)
        core_features = [col for col in CORE_FEATURE_COLUMNS if col in train_ready.columns]

    feature_sets = build_feature_sets(base_features, core_features)
    if keep_date:
        train_ready["date"] = pd.to_datetime(train_df.loc[train_ready.index, "date"])
        valid_ready["date"] = pd.to_datetime(valid_df.loc[valid_ready.index, "date"])
        test_ready["date"] = pd.to_datetime(test_df.loc[test_ready.index, "date"])

    train_ready, valid_ready, test_ready = align_feature_columns(train_ready, valid_ready, test_ready)
    return train_ready, valid_ready, test_ready, feature_sets


def prepare_standard_split_frames(
    config: PipelineConfig | None = None,
    keep_date: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    config = config or PipelineConfig(name="hybrid_default")
    raw_df = load_modeling_base_table()
    train_full, test_df = chronological_split(raw_df, test_size=0.20)
    train_df, valid_df = chronological_split(train_full, test_size=0.20)
    return prepare_configured_frames(train_df, valid_df, test_df, config, keep_date=keep_date)


def make_expanding_windows(
    raw_df: pd.DataFrame,
    initial_train_frac: float = 0.50,
    valid_frac: float = 0.10,
    test_frac: float = 0.10,
    step_frac: float = 0.10,
) -> list[dict[str, Any]]:
    ordered = raw_df.sort_values("date").reset_index(drop=True)
    total_rows = len(ordered)
    train_end = max(1, int(total_rows * initial_train_frac))
    valid_size = max(1, int(total_rows * valid_frac))
    test_size = max(1, int(total_rows * test_frac))
    step_size = max(1, int(total_rows * step_frac))

    windows: list[dict[str, Any]] = []
    split_number = 1
    while train_end + valid_size + test_size <= total_rows:
        valid_end = train_end + valid_size
        test_end = valid_end + test_size
        train_df = ordered.iloc[:train_end].copy()
        valid_df = ordered.iloc[train_end:valid_end].copy()
        test_df = ordered.iloc[valid_end:test_end].copy()
        windows.append(
            {
                "split_name": f"rolling_split_{split_number}",
                "train_df": train_df,
                "valid_df": valid_df,
                "test_df": test_df,
                "train_start": str(train_df["date"].min().date()),
                "train_end": str(train_df["date"].max().date()),
                "valid_start": str(valid_df["date"].min().date()),
                "valid_end": str(valid_df["date"].max().date()),
                "test_start": str(test_df["date"].min().date()),
                "test_end": str(test_df["date"].max().date()),
            }
        )
        train_end += step_size
        split_number += 1
    return windows


def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Summer"
    if month in (3, 4, 5):
        return "Autumn"
    if month in (6, 7, 8):
        return "Winter"
    return "Spring"

