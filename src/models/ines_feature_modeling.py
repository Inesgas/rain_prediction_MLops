from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None

from src.models.ines_modeling_core import (
    BEST_CATBOOST_PARAMS,
    BEST_XGB_PARAMS,
    DEFAULT_RANDOM_STATE,
    LOCKED_FINAL_MODEL_NAME,
    TARGET,
    TIME_COL,
    XGBClassifier,
    add_calendar_parts,
    build_preprocessor,
    chronological_split,
    compare_plain_models,
    fit_standard_model,
    is_categorical,
    load_feature_table as load_shared_feature_table,
    make_catboost_classifier,
    make_model_dataset,
    make_xgb_classifier,
    predict_proba_for_plain_model,
    prepare_catboost_frames,
    rank_raw_features,
    score_predictions,
    select_top_features,
    split_xy,
    tune_threshold,
)


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
]


def load_feature_table() -> pd.DataFrame:
    return load_shared_feature_table(profile="base")


def make_xgb_classifier_with_params(
    y_train: pd.Series,
    params: dict[str, float | int],
) -> object:
    return make_xgb_classifier(y_train, params=params)


def make_catboost_classifier_with_params(
    y_train: pd.Series,
    params: dict[str, float | int],
) -> object | None:
    return make_catboost_classifier(y_train, params=params)


def compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, object]]:
    return compare_plain_models(X_train, X_test, y_train, y_test)


def predict_proba_for_model(model_name: str, fitted_model: object, X: pd.DataFrame):
    return predict_proba_for_plain_model(model_name, fitted_model, X)


def fit_model_by_name(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, float | int] | None = None,
) -> object:
    if model_name == "CatBoost":
        estimator = make_catboost_classifier(y_train, params=params)
        if estimator is None:
            raise ValueError("CatBoost is not available in this environment.")
        X_train_ready, _, cat_cols = prepare_catboost_frames(X_train, X_train)
        estimator.fit(X_train_ready, y_train, cat_features=cat_cols)
        return estimator

    return fit_standard_model(model_name, X_train, y_train, params=params)


def evaluate_model_with_time_series_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "XGBoost",
    params: dict[str, float | int] | None = None,
    n_splits: int = 3,
    threshold: float = 0.5,
) -> pd.DataFrame:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict[str, float | int]] = []

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_valid = y.iloc[valid_idx].copy()

        fitted_model = fit_model_by_name(model_name, X_train, y_train, params=params)
        proba = predict_proba_for_model(model_name, fitted_model, X_valid)
        fold_scores = score_predictions(y_valid, proba, threshold=threshold)
        fold_scores["fold"] = fold
        rows.append(fold_scores)

    return pd.DataFrame(rows)


def optimize_model_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "XGBoost",
    n_trials: int = 20,
    n_splits: int = 3,
    scoring: str = "f1",
) -> tuple[dict[str, float | int], pd.DataFrame]:
    if optuna is None:
        raise ValueError("Optuna is not installed in this environment.")

    if model_name not in {"XGBoost", "CatBoost"}:
        raise ValueError("Optuna tuning is currently configured for XGBoost or CatBoost.")

    def objective(trial: optuna.trial.Trial) -> float:
        if model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 250, 700, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "subsample": trial.suggest_float("subsample", 0.7, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
                "gamma": trial.suggest_float("gamma", 0.0, 0.4),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            }
        else:
            params = {
                "iterations": trial.suggest_int("iterations", 250, 700, step=50),
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 8.0),
                "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            }

        cv_scores = evaluate_model_with_time_series_cv(
            X,
            y,
            model_name=model_name,
            params=params,
            n_splits=n_splits,
            threshold=0.5,
        )
        return float(cv_scores[scoring].mean())

    sampler = optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    return study.best_params, trials_df


def nested_time_series_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "XGBoost",
    n_outer_splits: int = 3,
    n_inner_trials: int = 8,
    scoring: str = "f1",
) -> tuple[pd.DataFrame, list[dict[str, float | int]]]:
    outer_cv = TimeSeriesSplit(n_splits=n_outer_splits)
    outer_rows: list[dict[str, float | int]] = []
    best_params_per_fold: list[dict[str, float | int]] = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        best_params, _ = optimize_model_with_optuna(
            X_train,
            y_train,
            model_name=model_name,
            n_trials=n_inner_trials,
            n_splits=3,
            scoring=scoring,
        )
        fitted_model = fit_model_by_name(model_name, X_train, y_train, params=best_params)
        proba = predict_proba_for_model(model_name, fitted_model, X_test)
        metrics = score_predictions(y_test, proba, threshold=0.5)
        metrics["fold"] = fold
        outer_rows.append(metrics)
        best_params_per_fold.append(best_params)

    return pd.DataFrame(outer_rows), best_params_per_fold


def benchmark_wrapper_feature_set(
    df_model: pd.DataFrame,
    prefilter_top_n: int = 15,
    wrapper_select_n: int = 8,
    model_name: str = "CatBoost",
) -> tuple[list[str], pd.Series]:
    train_df, _ = chronological_split(df_model)
    X_train, _, y_train, _ = split_xy(train_df, train_df.iloc[:1].copy())
    X_train_model = X_train.drop(columns=[TIME_COL], errors="ignore")

    prefiltered = select_top_features(X_train_model, y_train, top_n=prefilter_top_n)
    wrapper_selected = select_features_with_wrapper(
        X_train_model,
        y_train,
        candidate_features=prefiltered,
        n_features_to_select=wrapper_select_n,
        scoring="f1",
        n_splits=2,
    )
    wrapper_dataset = make_model_dataset(
        add_calendar_parts(load_feature_table()),
        wrapper_selected,
        keep_date=True,
    )

    result = evaluate_feature_set(wrapper_dataset)
    result["evaluated_model"] = model_name
    return wrapper_selected, result


def fit_locked_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    return fit_model_by_name(
        LOCKED_FINAL_MODEL_NAME,
        X_train,
        y_train,
        params=BEST_CATBOOST_PARAMS if LOCKED_FINAL_MODEL_NAME == "CatBoost" else BEST_XGB_PARAMS,
    )


def select_features_with_wrapper(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_features: list[str] | None = None,
    n_features_to_select: int = 12,
    scoring: str = "f1",
    n_splits: int = 3,
    wrapper_model: str = "Logistic Regression",
) -> list[str]:
    if candidate_features is None:
        candidate_features = X_train.columns.tolist()

    candidate_features = [feature for feature in candidate_features if feature in X_train.columns]
    if len(candidate_features) <= n_features_to_select:
        return candidate_features

    X_wrapper = X_train[candidate_features].copy()
    for column in X_wrapper.columns:
        if is_categorical(X_wrapper[column]):
            codes, _ = pd.factorize(X_wrapper[column].fillna("Missing"), sort=True)
            X_wrapper[column] = codes.astype("int32")
        else:
            X_wrapper[column] = X_wrapper[column].fillna(X_wrapper[column].median())

    if wrapper_model == "Logistic Regression":
        X_wrapper = pd.DataFrame(
            StandardScaler().fit_transform(X_wrapper),
            columns=X_wrapper.columns,
            index=X_wrapper.index,
        )
        wrapper_estimator = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=DEFAULT_RANDOM_STATE,
        )
    elif wrapper_model == "Random Forest":
        wrapper_estimator = RandomForestClassifier(
            n_estimators=250,
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    else:
        wrapper_estimator = (
            XGBClassifier(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.0,
                reg_lambda=1.0,
                scale_pos_weight=(len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
                random_state=DEFAULT_RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=4,
            )
            if XGBClassifier is not None
            else RandomForestClassifier(
                n_estimators=250,
                random_state=DEFAULT_RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
        )

    selector = SequentialFeatureSelector(
        estimator=wrapper_estimator,
        n_features_to_select=n_features_to_select,
        direction="forward",
        scoring=scoring,
        cv=TimeSeriesSplit(n_splits=n_splits),
        n_jobs=1,
    )
    selector.fit(X_wrapper, y_train)
    selected = X_wrapper.columns[selector.get_support()].tolist()

    if "location" in candidate_features and "location" not in selected:
        selected.append("location")

    return selected


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


def evaluate_feature_set(df_model: pd.DataFrame) -> pd.Series:
    train_df, test_df = chronological_split(df_model)
    X_train, X_test, y_train, y_test = split_xy(train_df, test_df)
    X_train_model = X_train.drop(columns=[TIME_COL], errors="ignore")
    X_test_model = X_test.drop(columns=[TIME_COL], errors="ignore")
    results, fitted = compare_models(X_train_model, X_test_model, y_train, y_test)

    best_name = results.iloc[0]["model"]
    best_threshold, threshold_df = tune_threshold(
        fitted[best_name],
        X_test_model,
        y_test,
        model_name=best_name,
    )

    proba = predict_proba_for_model(best_name, fitted[best_name], X_test_model)
    metrics = score_predictions(y_test, proba, threshold=best_threshold)

    return pd.Series(
        {
            "best_model": best_name,
            "feature_count": X_train_model.shape[1],
            **metrics,
            "best_threshold": best_threshold,
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

