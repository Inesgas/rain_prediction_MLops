from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
except ImportError:  # pragma: no cover
    RandomOverSampler = None
    SMOTE = None
    SMOTENC = None

from src.models.ines_modeling_core import (
    ALIGNED_CATBOOST_PARAMS,
    BEST_XGB_PARAMS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_VALIDATION_SHARE,
    TARGET,
    TIME_COL,
    add_calendar_parts,
    build_model_registry,
    chronological_split,
    chronological_validation_split,
    fit_standard_model,
    is_categorical,
    load_feature_table as load_shared_feature_table,
    make_catboost_classifier as make_catboost_classifier_core,
    make_model_dataset,
    predict_proba_for_plain_model,
    prepare_catboost_frames as prepare_simple_catboost_frames,
    rank_raw_features,
    score_predictions,
    select_top_features,
    split_xy,
    tune_threshold as tune_threshold_core,
)


CATBOOST_IMPUTATION_STRATEGY = "location_then_ncc_median"
DEFAULT_RESAMPLING_STRATEGIES = ["none", "random_oversample", "smote"]

BASE_FEATURES = [
    "location",
    "ncc_zone",
    "rain_today",
    "max_temp",
    "rainfall",
    "sunshine",
    "wind_gust_speed",
    "humidity_3pm",
    "pressure_3pm",
    "cloud_9am",
    "cloud_3pm",
    "temp_3pm",
    "humidity_day_diff",
    "pressure_day_diff",
    "temp_day_diff",
    "wind_speed_day_diff",
    "cloud_day_diff",
    "humidity_overnight_change",
    "pressure_overnight_change",
    "temp_overnight_change",
    "rainfall_yesterday",
    "pressure_3pm_yesterday",
    "cloud_3pm_yesterday",
    "dew_point_spread_9am",
    "dew_point_spread_3pm",
    "wind_dir_9am_x",
    "wind_dir_9am_y",
    "wind_dir_3pm_x",
    "wind_dir_3pm_y",
    "wind_shift_score",
    "day_of_year_sin",
    "day_of_year_cos",
    "pressure_fall",
    "humidity_rising_fast",
    "warming_day",
    "sunshine_missing",
    "evaporation_missing",
    "cloud_9am_missing",
    "cloud_3pm_missing",
]


EXPANDED_FEATURES = BASE_FEATURES + [
    "min_temp",
    "evaporation",
    "wind_speed_9am",
    "wind_speed_3pm",
    "humidity_9am",
    "pressure_9am",
    "temp_9am",
    "dewpoint_9am",
    "dewpoint_3pm",
    "temp_range",
    "humidity_temp_3pm_interaction",
    "humidity_temp_9am_interaction",
    "temp_3pm_vs_max_gap",
    "pressure_humidity_9am_ratio",
    "pressure_humidity_3pm_ratio",
    "cloud_humidity_3pm_interaction",
    "moisture_stability_3pm",
    "rainfall_prev_1d",
    "rainfall_roll3_mean",
    "rainfall_roll7_sum",
    "rainfall_roll7_max",
    "days_since_rain_1mm",
    "rain_today_streak_3",
    "rain_today_streak_7",
    "humidity_3pm_roll3_mean",
    "humidity_3pm_roll7_mean",
    "pressure_3pm_roll3_mean",
    "pressure_3pm_roll7_mean",
    "temp_3pm_roll3_mean",
    "temp_3pm_roll7_mean",
    "wind_gust_speed_roll3_mean",
    "wind_gust_speed_roll7_mean",
    "lat",
    "lon",
    "elevation",
    "land_strip",
    "rainfall_missing",
    "wind_gust_speed_missing",
    "humidity_9am_missing",
    "humidity_3pm_missing",
    "pressure_9am_missing",
    "pressure_3pm_missing",
]


def load_feature_table() -> pd.DataFrame:
    return load_shared_feature_table(profile="aligned")


def make_catboost_classifier(y_train: pd.Series) -> object | None:
    return make_catboost_classifier_core(y_train, params=ALIGNED_CATBOOST_PARAMS)


def available_resampling_strategies() -> list[str]:
    strategies = ["none", "random_oversample"]
    if SMOTE is not None or SMOTENC is not None:
        strategies.append("smote")
    return strategies


def _manual_random_oversample(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    target_name = y_train.name or TARGET
    train_frame = X_train.copy()
    train_frame[target_name] = y_train.to_numpy()

    class_counts = y_train.value_counts()
    if class_counts.empty or class_counts.nunique() == 1:
        return X_train.copy(), y_train.copy()

    majority_count = int(class_counts.max())
    sampled_frames: list[pd.DataFrame] = []
    for _, group in train_frame.groupby(target_name):
        if len(group) < majority_count:
            sampled = group.sample(
                n=majority_count,
                replace=True,
                random_state=DEFAULT_RANDOM_STATE,
            )
            sampled_frames.append(sampled)
        else:
            sampled_frames.append(group.copy())

    balanced = (
        pd.concat(sampled_frames, axis=0)
        .sample(frac=1.0, random_state=DEFAULT_RANDOM_STATE)
        .reset_index(drop=True)
    )
    y_resampled = balanced[target_name].astype(y_train.dtype)
    X_resampled = balanced.drop(columns=[target_name])
    return X_resampled, y_resampled


def apply_resampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "none",
) -> tuple[pd.DataFrame, pd.Series]:
    if strategy == "none":
        return X_train.copy(), y_train.copy()

    if strategy == "random_oversample":
        if RandomOverSampler is not None:
            sampler = RandomOverSampler(random_state=DEFAULT_RANDOM_STATE)
            X_resampled, y_resampled = sampler.fit_resample(X_train.copy(), y_train.copy())
            if not isinstance(X_resampled, pd.DataFrame):
                X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
            if not isinstance(y_resampled, pd.Series):
                y_resampled = pd.Series(y_resampled, name=y_train.name or TARGET)
            return X_resampled.reset_index(drop=True), y_resampled.reset_index(drop=True)
        return _manual_random_oversample(X_train, y_train)

    if strategy == "smote":
        cat_cols = [column for column in X_train.columns if is_categorical(X_train[column])]
        if cat_cols and SMOTENC is not None:
            smote_input = X_train.copy()
            for column in cat_cols:
                smote_input[column] = smote_input[column].astype("category")
            sampler = SMOTENC(
                categorical_features=[smote_input.columns.get_loc(column) for column in cat_cols],
                random_state=DEFAULT_RANDOM_STATE,
                k_neighbors=5,
            )
            X_resampled, y_resampled = sampler.fit_resample(smote_input, y_train.copy())
            if not isinstance(X_resampled, pd.DataFrame):
                X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
            for column in cat_cols:
                X_resampled[column] = X_resampled[column].astype(str)
            if not isinstance(y_resampled, pd.Series):
                y_resampled = pd.Series(y_resampled, name=y_train.name or TARGET)
            return X_resampled.reset_index(drop=True), y_resampled.reset_index(drop=True)

        if not cat_cols and SMOTE is not None:
            sampler = SMOTE(random_state=DEFAULT_RANDOM_STATE, k_neighbors=5)
            X_resampled, y_resampled = sampler.fit_resample(X_train.copy(), y_train.copy())
            if not isinstance(X_resampled, pd.DataFrame):
                X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
            if not isinstance(y_resampled, pd.Series):
                y_resampled = pd.Series(y_resampled, name=y_train.name or TARGET)
            return X_resampled.reset_index(drop=True), y_resampled.reset_index(drop=True)

        return apply_resampling(X_train, y_train, strategy="random_oversample")

    raise ValueError(f"Unsupported resampling strategy: {strategy}")


def _build_model_registry(y_train: pd.Series) -> dict[str, object]:
    return build_model_registry(y_train, catboost_params=ALIGNED_CATBOOST_PARAMS)


def build_numeric_imputer_bundle(
    X: pd.DataFrame,
    strategy: str = CATBOOST_IMPUTATION_STRATEGY,
) -> tuple[dict[str, Any], list[str], list[str]]:
    cat_cols = [column for column in X.columns if is_categorical(X[column])]
    num_cols = [column for column in X.columns if column not in cat_cols]

    global_fill_values: dict[str, float] = {}
    for column in num_cols:
        series = pd.to_numeric(X[column], errors="coerce")
        global_fill_values[column] = float(series.median()) if series.notna().any() else 0.0

    bundle: dict[str, Any] = {
        "strategy": strategy,
        "global_fill_values": global_fill_values,
        "group_fill_values": {},
    }

    if strategy in {"ncc_zone_median", "location_then_ncc_median"} and "ncc_zone" in X.columns:
        zone_frame = X.groupby("ncc_zone", dropna=True)[num_cols].median(numeric_only=True)
        bundle["group_fill_values"]["ncc_zone"] = {
            str(index): {column: float(value) for column, value in row.dropna().items()}
            for index, row in zone_frame.iterrows()
        }

    if strategy == "location_then_ncc_median" and "location" in X.columns:
        location_frame = X.groupby("location", dropna=True)[num_cols].median(numeric_only=True)
        bundle["group_fill_values"]["location"] = {
            str(index): {column: float(value) for column, value in row.dropna().items()}
            for index, row in location_frame.iterrows()
        }

    return bundle, num_cols, cat_cols


def _apply_numeric_imputer_bundle(
    X: pd.DataFrame,
    imputer_bundle: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    ready = X.copy()
    cat_cols = [column for column in ready.columns if is_categorical(ready[column])]
    num_cols = [column for column in ready.columns if column not in cat_cols]

    global_fill_values = dict(imputer_bundle.get("global_fill_values", {}))
    group_fill_values = dict(imputer_bundle.get("group_fill_values", {}))

    for column in num_cols:
        filled = pd.to_numeric(ready[column], errors="coerce")
        if filled.isna().any() and "location" in ready.columns and "location" in group_fill_values:
            location_map = {
                key: value_map.get(column)
                for key, value_map in group_fill_values["location"].items()
                if column in value_map
            }
            if location_map:
                filled = filled.fillna(ready["location"].astype(str).map(location_map))
        if filled.isna().any() and "ncc_zone" in ready.columns and "ncc_zone" in group_fill_values:
            zone_map = {
                key: value_map.get(column)
                for key, value_map in group_fill_values["ncc_zone"].items()
                if column in value_map
            }
            if zone_map:
                filled = filled.fillna(ready["ncc_zone"].astype(str).map(zone_map))
        filled = filled.fillna(global_fill_values.get(column, 0.0))
        ready[column] = filled

    for column in cat_cols:
        ready[column] = ready[column].fillna("Missing").astype(str)

    return ready, cat_cols


def prepare_catboost_frames(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    imputer_bundle: dict[str, Any] | None = None,
    return_imputer: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]] | tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    resolved_imputer = imputer_bundle
    if resolved_imputer is None:
        resolved_imputer, _, _ = build_numeric_imputer_bundle(X_train)

    X_train_ready, cat_cols = _apply_numeric_imputer_bundle(X_train, resolved_imputer)
    X_test_ready, _ = _apply_numeric_imputer_bundle(X_test, resolved_imputer)

    if return_imputer:
        return X_train_ready, X_test_ready, cat_cols, resolved_imputer
    return X_train_ready, X_test_ready, cat_cols


def fit_named_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    resampling_strategy: str = "none",
) -> object:
    X_fit, y_fit = apply_resampling(X_train, y_train, strategy=resampling_strategy)

    if model_name == "CatBoost":
        estimator = make_catboost_classifier(y_fit)
        if estimator is None:
            raise ValueError("CatBoost is not available in this environment.")
        X_train_ready, _, cat_cols, imputer_bundle = prepare_catboost_frames(
            X_fit,
            X_fit.iloc[:0].copy(),
            return_imputer=True,
        )
        estimator.fit(X_train_ready, y_fit, cat_features=cat_cols)
        return {
            "model_name": "CatBoost",
            "model": estimator,
            "cat_features": cat_cols,
            "numeric_imputer": imputer_bundle,
            "resampling": resampling_strategy,
        }

    return fit_standard_model(model_name, X_fit, y_fit)


def compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    resampling_strategies: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    model_names = list(_build_model_registry(y_train).keys())
    strategies = available_resampling_strategies() if resampling_strategies is None else resampling_strategies

    rows: list[dict[str, float | str]] = []
    fitted: dict[str, object] = {}

    for name in model_names:
        for strategy in strategies:
            fitted_model = fit_named_model(name, X_train, y_train, resampling_strategy=strategy)
            proba = predict_proba_for_model(name, fitted_model, X_test)
            rows.append(
                {
                    "model": name,
                    "resampling": strategy,
                    **score_predictions(y_test, proba),
                }
            )
            fitted[f"{name}__{strategy}"] = fitted_model

    results = (
        pd.DataFrame(rows)
        .sort_values(["roc_auc", "f1"], ascending=False)
        .reset_index(drop=True)
    )
    return results, fitted


def predict_proba_for_model(model_name: str, fitted_model: object, X: pd.DataFrame) -> np.ndarray:
    if model_name == "CatBoost":
        if isinstance(fitted_model, dict):
            X_ready, _ = _apply_numeric_imputer_bundle(X, fitted_model["numeric_imputer"])
            return fitted_model["model"].predict_proba(X_ready)[:, 1]
        X_ready, _, _ = prepare_simple_catboost_frames(X, X)
        return fitted_model.predict_proba(X_ready)[:, 1]
    return predict_proba_for_plain_model(model_name, fitted_model, X)


def tune_threshold(
    pipeline: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "XGBoost",
    thresholds: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    return tune_threshold_core(
        pipeline,
        X_test,
        y_test,
        model_name=model_name,
        thresholds=thresholds,
        predict_fn=predict_proba_for_model,
    )


def crossvalidation_resample_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "XGBoost",
    resampling_strategies: list[str] | None = None,
    n_splits: int = 5,
    thresholds: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategies = available_resampling_strategies() if resampling_strategies is None else resampling_strategies
    X_ordered = X.reset_index(drop=True).copy()
    y_ordered = y.reset_index(drop=True).copy()

    fold_rows: list[dict[str, float | int | str]] = []
    for strategy in strategies:
        for fold_id, (train_idx, valid_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X_ordered), start=1):
            X_train_fold = X_ordered.iloc[train_idx].copy()
            X_valid_fold = X_ordered.iloc[valid_idx].copy()
            y_train_fold = y_ordered.iloc[train_idx].copy()
            y_valid_fold = y_ordered.iloc[valid_idx].copy()

            fitted_model = fit_named_model(
                model_name,
                X_train_fold,
                y_train_fold,
                resampling_strategy=strategy,
            )
            best_threshold, _ = tune_threshold(
                fitted_model,
                X_valid_fold,
                y_valid_fold,
                model_name=model_name,
                thresholds=thresholds,
            )
            proba = predict_proba_for_model(model_name, fitted_model, X_valid_fold)
            fold_rows.append(
                {
                    "model": model_name,
                    "resampling": strategy,
                    "fold": fold_id,
                    "train_rows": int(len(train_idx)),
                    "valid_rows": int(len(valid_idx)),
                    "positive_rate_train": float(y_train_fold.mean()),
                    "positive_rate_valid": float(y_valid_fold.mean()),
                    "best_threshold": float(best_threshold),
                    **score_predictions(y_valid_fold, proba, threshold=best_threshold),
                }
            )

    fold_frame = pd.DataFrame(fold_rows)
    summary = (
        fold_frame.groupby(["model", "resampling"], as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_threshold=("best_threshold", "mean"),
            folds=("fold", "count"),
        )
        .sort_values(["mean_roc_auc", "mean_f1"], ascending=False)
        .reset_index(drop=True)
    )
    return summary, fold_frame


def benchmark_feature_sets(top_n: int = 25) -> pd.DataFrame:
    df_feat = add_calendar_parts(load_feature_table())
    base_dataset = make_model_dataset(df_feat, BASE_FEATURES, keep_date=True)
    train_df, test_df = chronological_split(base_dataset)
    X_train, _, y_train, _ = split_xy(train_df, test_df)

    selected_features = select_top_features(
        X_train.drop(columns=[TIME_COL]),
        y_train,
        top_n=top_n,
    )
    selected_dataset = make_model_dataset(df_feat, selected_features, keep_date=True)

    experiments = {
        "manual_base": evaluate_feature_set(base_dataset),
        "expanded_manual": evaluate_feature_set(make_model_dataset(df_feat, EXPANDED_FEATURES, keep_date=True)),
        f"top_{top_n}_from_train": evaluate_feature_set(selected_dataset),
    }
    return pd.DataFrame(experiments).T.sort_values(["roc_auc", "f1"], ascending=False)


def evaluate_feature_set(
    df_model: pd.DataFrame,
    validation_share: float = DEFAULT_VALIDATION_SHARE,
) -> pd.Series:
    train_df, test_df = chronological_split(df_model)
    model_train_df, valid_df = chronological_validation_split(train_df, valid_size=validation_share)

    X_model_train, X_valid, y_model_train, y_valid = split_xy(model_train_df, valid_df)
    X_train_full, X_test, y_train_full, y_test = split_xy(train_df, test_df)

    X_model_train = X_model_train.drop(columns=[TIME_COL], errors="ignore")
    X_valid = X_valid.drop(columns=[TIME_COL], errors="ignore")
    X_train_full = X_train_full.drop(columns=[TIME_COL], errors="ignore")
    X_test = X_test.drop(columns=[TIME_COL], errors="ignore")

    validation_results, fitted_models = compare_models(X_model_train, X_valid, y_model_train, y_valid)
    best_name = validation_results.iloc[0]["model"]
    best_resampling = validation_results.iloc[0]["resampling"]
    best_threshold, threshold_df = tune_threshold(
        fitted_models[f"{best_name}__{best_resampling}"],
        X_valid,
        y_valid,
        model_name=best_name,
    )

    final_model = fit_named_model(
        best_name,
        X_train_full,
        y_train_full,
        resampling_strategy=best_resampling,
    )
    proba = predict_proba_for_model(best_name, final_model, X_test)
    metrics = score_predictions(y_test, proba, threshold=best_threshold)

    return pd.Series(
        {
            "best_model": best_name,
            "best_resampling": best_resampling,
            "feature_count": X_train_full.shape[1],
            **metrics,
            "best_threshold": best_threshold,
            "validation_roc_auc": float(validation_results.iloc[0]["roc_auc"]),
            "threshold_f1_at_0_5": float(
                threshold_df.loc[(threshold_df["threshold"] - 0.5).abs().idxmin(), "f1"]
            ),
        }
    )


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    df_feat = add_calendar_parts(load_feature_table())
    base_dataset = make_model_dataset(df_feat, BASE_FEATURES, keep_date=True)
    train_df, _ = chronological_split(base_dataset)
    X_train, _, y_train, _ = split_xy(train_df, train_df.iloc[:1].copy())
    ranking = rank_raw_features(X_train.drop(columns=[TIME_COL]), y_train)

    print("Top ranked raw features from the training fold:")
    print(ranking.head(20).round(4).to_string(index=False))
    print()
    print("Feature-set benchmark:")
    print(benchmark_feature_sets(top_n=25).round(4).to_string())

