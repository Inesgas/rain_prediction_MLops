from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.ines_feature_modeling import (
    BEST_CATBOOST_PARAMS,
    BEST_XGB_PARAMS,
    TARGET,
    fit_model_by_name,
    predict_proba_for_model,
    score_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "weatherAUS.csv"
BASE_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "locations_metadata.csv"
RESULTS_DIR = PROJECT_ROOT / "reports" / "model_evidence" / "daily_zonal_baseline"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "daily_zonal_locations_metadata.csv"
SEARCH_RESULTS_PATH = RESULTS_DIR / "daily_zonal_results.csv"
SUMMARY_PATH = RESULTS_DIR / "daily_zonal_summary.json"

RENAME_MAP = {
    "Date": "date",
    "Location": "location",
    "MinTemp": "min_temp",
    "MaxTemp": "max_temp",
    "Rainfall": "rainfall",
    "Evaporation": "evaporation",
    "Sunshine": "sunshine",
    "WindGustDir": "wind_gust_dir",
    "WindGustSpeed": "wind_gust_speed",
    "WindDir9am": "wind_dir_9am",
    "WindDir3pm": "wind_dir_3pm",
    "WindSpeed9am": "wind_speed_9am",
    "WindSpeed3pm": "wind_speed_3pm",
    "Humidity9am": "humidity_9am",
    "Humidity3pm": "humidity_3pm",
    "Pressure9am": "pressure_9am",
    "Pressure3pm": "pressure_3pm",
    "Cloud9am": "cloud_9am",
    "Cloud3pm": "cloud_3pm",
    "Temp9am": "temp_9am",
    "Temp3pm": "temp_3pm",
    "RainToday": "rain_today",
    "RainTomorrow": "rain_tomorrow",
}
RAW_COLUMNS = list(RENAME_MAP)
WIND_COLUMNS = ["wind_dir_9am", "wind_dir_3pm", "wind_gust_dir"]
TARGET_MODELS = [
    ("XGBoost", BEST_XGB_PARAMS),
    ("Random Forest", None),
    ("CatBoost", BEST_CATBOOST_PARAMS),
]
THRESHOLDS = np.arange(0.30, 0.71, 0.02)

MANUAL_RAINFALL_ZONE_BY_LOCATION = {
    "Adelaide": "Winter",
    "Albany": "Winter dominant",
    "Albury": "Uniform",
    "AliceSprings": "Arid",
    "BadgerysCreek": "Uniform",
    "Ballarat": "Winter",
    "Bendigo": "Uniform",
    "Brisbane": "Summer",
    "Cairns": "Summer dominant",
    "Canberra": "Uniform",
    "Cobar": "Arid",
    "CoffsHarbour": "Uniform",
    "Dartmoor": "Winter",
    "Darwin": "Summer dominant",
    "GoldCoast": "Summer",
    "Hobart": "Winter",
    "Katherine": "Summer dominant",
    "Launceston": "Winter",
    "Melbourne": "Uniform",
    "MelbourneAirport": "Uniform",
    "Mildura": "Arid",
    "Moree": "Summer",
    "MountGambier": "Winter",
    "MountGinini": "Winter",
    "Newcastle": "Uniform",
    "Nhil": "Winter",
    "NorahHead": "Uniform",
    "NorfolkIsland": "Summer",
    "Nuriootpa": "Winter",
    "PearceRAAF": "Winter dominant",
    "Penrith": "Uniform",
    "Perth": "Winter dominant",
    "PerthAirport": "Winter dominant",
    "Portland": "Winter",
    "Richmond": "Uniform",
    "Sale": "Uniform",
    "SalmonGums": "Winter",
    "Sydney": "Uniform",
    "SydneyAirport": "Uniform",
    "Townsville": "Summer dominant",
    "Tuggeranong": "Uniform",
    "Uluru": "Arid",
    "WaggaWagga": "Uniform",
    "Walpole": "Winter dominant",
    "Watsonia": "Uniform",
    "Williamtown": "Uniform",
    "Witchcliffe": "Winter dominant",
    "Wollongong": "Uniform",
    "Woomera": "Arid",
}


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, usecols=RAW_COLUMNS).rename(columns=RENAME_MAP)
    df["date"] = pd.to_datetime(df["date"])
    return df


def chronological_split(df: pd.DataFrame, test_size: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(ordered) * (1.0 - test_size))
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def enrich_metadata_with_rainfall_zone() -> Path:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(BASE_METADATA_PATH)
    if "rainfall_zone" in meta.columns:
        meta.to_csv(METADATA_PATH, index=False)
        return METADATA_PATH

    meta["rainfall_zone"] = meta["location"].map(MANUAL_RAINFALL_ZONE_BY_LOCATION).fillna("Unknown")
    meta.to_csv(METADATA_PATH, index=False)
    return METADATA_PATH


def add_metadata(df: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path)
    keep_cols = ["location", "lat", "lon", "elevation", "rainfall_zone"]
    return df.merge(meta[keep_cols], on="location", how="left")


def add_calendar_parts(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["month"] = result["date"].dt.month.astype("int16")
    result["day"] = result["date"].dt.day.astype("int16")
    result["year"] = result["date"].dt.year.astype("int16")
    return result


def fill_with_daily_zonal_stats(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    columns: list[str],
    zone_col: str = "rainfall_zone",
    strategy: str = "median",
    threshold: float = 0.1,
) -> pd.DataFrame:
    result = df.copy()
    reference = reference_df.copy()
    result["date"] = pd.to_datetime(result["date"])
    reference["date"] = pd.to_datetime(reference["date"])

    def get_stats(grouped: pd.core.groupby.generic.SeriesGroupBy, strat: str, min_samples: int = 10) -> pd.Series:
        count = grouped.count()
        valid_index = count[count >= min_samples].index
        if strat == "mean":
            stats = grouped.mean()
        elif strat == "mode":
            stats = grouped.apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        else:
            stats = grouped.median()
        return stats.loc[stats.index.intersection(valid_index)]

    for col in columns:
        if col not in result.columns or col not in reference.columns:
            continue

        nan_rate = reference[col].isna().mean()
        if nan_rate > threshold:
            result[f"{col}_missing_daily_zonal"] = result[col].isna().astype(int)

        daily_stats = get_stats(reference.groupby([zone_col, "date"])[col], strategy)
        result = result.set_index([zone_col, "date"])
        result[col] = result[col].fillna(daily_stats)
        result = result.reset_index()

        if result[col].isna().any():
            result["month"] = result["date"].dt.month
            reference["month"] = reference["date"].dt.month
            seasonal_stats = get_stats(reference.groupby([zone_col, "month"])[col], strategy)
            result = result.set_index([zone_col, "month"])
            result[col] = result[col].fillna(seasonal_stats)
            result = result.reset_index()
            result = result.drop(columns=["month"], errors="ignore")
            reference = reference.drop(columns=["month"], errors="ignore")

    return result


def add_daily_diffs(df: pd.DataFrame, prefixes: list[str] | None = None) -> pd.DataFrame:
    result = df.copy()
    prefixes = prefixes or ["humidity", "temp", "wind_speed", "pressure", "cloud"]
    for prefix in prefixes:
        c3 = f"{prefix}_3pm"
        c9 = f"{prefix}_9am"
        if c3 in result.columns and c9 in result.columns:
            result[f"{prefix}_day_diff"] = result[c3] - result[c9]
    return result


def add_overnight_diffs(df: pd.DataFrame, prefixes: list[str] | None = None) -> pd.DataFrame:
    result = df.copy()
    prefixes = prefixes or ["humidity", "temp", "wind_speed", "pressure", "cloud"]
    for prefix in prefixes:
        c9 = f"{prefix}_9am"
        c3 = f"{prefix}_3pm"
        if c9 not in result.columns or c3 not in result.columns:
            continue
        temp = result[["date", "location", c3]].copy()
        temp["date"] = temp["date"] + pd.Timedelta(days=1)
        temp = temp.rename(columns={c3: f"{prefix}_3pm_yest"})
        result = result.merge(temp, on=["date", "location"], how="left")
        result[f"{prefix}_overnight_diff"] = result[c9] - result[f"{prefix}_3pm_yest"]
        result = result.drop(columns=[f"{prefix}_3pm_yest"], errors="ignore")
    return result


def add_yesterday_lag(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        temp = result[["date", "location", col]].copy()
        temp["date"] = temp["date"] + pd.Timedelta(days=1)
        temp = temp.rename(columns={col: f"{col}_yest"})
        result = result.merge(temp, on=["date", "location"], how="left")
    return result


def add_24h_diff(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    result = df.copy()
    columns = columns or ["max_temp", "pressure_3pm", "humidity_3pm", "wind_gust_speed"]
    for col in columns:
        if col not in result.columns:
            continue
        temp = result[["date", "location", col]].copy()
        temp["date"] = temp["date"] + pd.Timedelta(days=1)
        temp = temp.rename(columns={col: f"{col}_yest"})
        result = result.merge(temp, on=["date", "location"], how="left")
        result[f"{col}_24h_diff"] = result[col] - result[f"{col}_yest"]
        result = result.drop(columns=[f"{col}_yest"], errors="ignore")
    return result


def add_dewpoint_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    a = 17.27
    b = 237.7
    for time in ["9am", "3pm"]:
        t_col = f"temp_{time}"
        h_col = f"humidity_{time}"
        dp_col = f"dewpoint_{time}"
        s_col = f"dewpoint_spread_{time}"
        if t_col in result.columns and h_col in result.columns:
            safe_h = result[h_col].clip(lower=0.01, upper=100.0)
            alpha = ((a * result[t_col]) / (b + result[t_col])) + np.log(safe_h / 100.0)
            result[dp_col] = (b * alpha) / (a - alpha)
            result[s_col] = result[t_col] - result[dp_col]
    return result


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


def add_wind_shift(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    cols = ["wind_dir_9am_x", "wind_dir_3pm_x", "wind_dir_9am_y", "wind_dir_3pm_y"]
    if all(col in result.columns for col in cols):
        result["wind_shift_score"] = (
            result["wind_dir_9am_x"] * result["wind_dir_3pm_x"]
            + result["wind_dir_9am_y"] * result["wind_dir_3pm_y"]
        )
    return result


def daily_zonal_pipeline(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    metadata_path: Path,
    drop_location: bool = True,
) -> pd.DataFrame:
    result = add_metadata(df, metadata_path)
    reference = add_metadata(reference_df, metadata_path)

    result = add_calendar_parts(result)
    reference = add_calendar_parts(reference)

    numeric_fill_cols = [
        col for col in result.columns
        if col not in ["rain_today", "date", "location", "rainfall", "rain_tomorrow", "rainfall_zone"]
        and col not in WIND_COLUMNS
        and pd.api.types.is_numeric_dtype(result[col])
    ]
    result = fill_with_daily_zonal_stats(result, reference, numeric_fill_cols, zone_col="rainfall_zone", strategy="median")
    result = fill_with_daily_zonal_stats(result, reference, [col for col in WIND_COLUMNS if col in result.columns], zone_col="rainfall_zone", strategy="mode")

    result = add_daily_diffs(result, ["temp", "humidity", "pressure"])
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


def tune_threshold_from_validation(proba: np.ndarray, y_true: pd.Series) -> tuple[float, dict[str, float]]:
    rows = []
    for threshold in THRESHOLDS:
        rows.append({"threshold": float(threshold), **score_predictions(y_true, proba, threshold=float(threshold))})
    frame = pd.DataFrame(rows)
    best_idx = frame["f1"].idxmax()
    best_threshold = float(frame.loc[best_idx, "threshold"])
    return best_threshold, {key: float(value) for key, value in frame.loc[best_idx].to_dict().items()}


def align_feature_columns(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_cols = [TARGET]
    common = sorted(set(train_df.columns) | set(valid_df.columns) | set(test_df.columns))

    def align(df: pd.DataFrame) -> pd.DataFrame:
        aligned = df.copy()
        for col in common:
            if col not in aligned.columns:
                aligned[col] = 0
        return aligned[common].copy()

    train_aligned = align(train_df)
    valid_aligned = align(valid_df)
    test_aligned = align(test_df)
    # make sure target is last for convenience
    ordered = [c for c in common if c != TARGET] + target_cols
    return train_aligned[ordered], valid_aligned[ordered], test_aligned[ordered]


def run_experiment() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = enrich_metadata_with_rainfall_zone()

    raw_df = load_raw()
    raw_df = raw_df[raw_df[TARGET].notna()].drop_duplicates().reset_index(drop=True)
    train_full, test_df = chronological_split(raw_df, test_size=0.20)
    train_df, valid_df = chronological_split(train_full, test_size=0.20)

    rows: list[dict[str, Any]] = []
    variant_counts: dict[str, int] = {}
    for feature_set_name, drop_location in [
        ("daily_zonal_drop_location", True),
        ("daily_zonal_keep_location", False),
    ]:
        train_ready = daily_zonal_pipeline(train_df, train_df, metadata_path, drop_location=drop_location)
        valid_ready = daily_zonal_pipeline(valid_df, train_df, metadata_path, drop_location=drop_location)
        test_ready = daily_zonal_pipeline(test_df, train_full, metadata_path, drop_location=drop_location)
        train_ready, valid_ready, test_ready = align_feature_columns(train_ready, valid_ready, test_ready)

        feature_cols = [col for col in train_ready.columns if col != TARGET]
        variant_counts[feature_set_name] = int(len(feature_cols))
        for model_name, params in TARGET_MODELS:
            print(f"Running {model_name} | {feature_set_name}", flush=True)
            X_train = train_ready[feature_cols].copy()
            X_valid = valid_ready[feature_cols].copy()
            X_test = test_ready[feature_cols].copy()
            y_train = train_ready[TARGET].astype(int)
            y_valid = valid_ready[TARGET].astype(int)
            y_test = test_ready[TARGET].astype(int)

            fitted = fit_model_by_name(model_name, X_train, y_train, params=params)
            valid_proba = predict_proba_for_model(model_name, fitted, X_valid)
            best_threshold, valid_metrics = tune_threshold_from_validation(valid_proba, y_valid)

            combined = pd.concat([train_ready, valid_ready], axis=0).reset_index(drop=True)
            X_combined = combined[feature_cols].copy()
            y_combined = combined[TARGET].astype(int)
            final_model = fit_model_by_name(model_name, X_combined, y_combined, params=params)
            test_proba = predict_proba_for_model(model_name, final_model, X_test)
            test_metrics = score_predictions(y_test, test_proba, threshold=best_threshold)

            rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "feature_count": int(len(feature_cols)),
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
            pd.DataFrame(rows).to_csv(SEARCH_RESULTS_PATH, index=False)

    results = pd.DataFrame(rows).sort_values(
        ["test_f1", "test_roc_auc", "validation_f1"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    results.to_csv(SEARCH_RESULTS_PATH, index=False)

    summary = {
        "metadata_path": str(metadata_path),
        "rainfall_zone_source": "manual_station_mapping_approximation",
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "feature_counts": variant_counts,
        "best_result": results.iloc[0].to_dict(),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = run_experiment()
    print(json.dumps(summary["best_result"], indent=2))
    print(f"Saved results to: {SEARCH_RESULTS_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

