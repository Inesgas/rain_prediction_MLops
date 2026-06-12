from __future__ import annotations

from pathlib import Path
from typing import Callable
import sys

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_object_dtype, is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "weatherAUS.csv"

RAW_WEATHER_COLUMNS = [
    "Date",
    "Location",
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustDir",
    "WindGustSpeed",
    "WindDir9am",
    "WindDir3pm",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Cloud9am",
    "Cloud3pm",
    "Temp9am",
    "Temp3pm",
    "RainToday",
    "RainTomorrow",
]

TARGET = "rain_tomorrow"
TIME_COL = "date"
DEFAULT_RANDOM_STATE = 42
DEFAULT_VALIDATION_SHARE = 0.15
BEST_XGB_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "min_child_weight": 4,
    "gamma": 0.0,
    "reg_lambda": 1.5,
}
BEST_CATBOOST_PARAMS = {
    "iterations": 400,
    "depth": 8,
    "learning_rate": 0.05107988650278712,
    "l2_leaf_reg": 5.190609389379256,
    "random_strength": 0.31203728088487304,
    "bagging_temperature": 0.15599452033620265,
}
ALIGNED_CATBOOST_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
}
LOCKED_FINAL_MODEL_NAME = "CatBoost"


def load_feature_table(profile: str = "base") -> pd.DataFrame:
    from src.features.feature_pipeline import build_features_pipeline

    raw = pd.read_csv(DATA_PATH)
    raw = raw[RAW_WEATHER_COLUMNS].copy()
    raw = raw[raw["RainTomorrow"].notna()].drop_duplicates()
    return build_features_pipeline(raw, profile=profile)


def add_calendar_parts(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result[TIME_COL] = pd.to_datetime(result[TIME_COL])
    result["month"] = result[TIME_COL].dt.month.astype("int16")
    result["year"] = result[TIME_COL].dt.year.astype("int16")
    result["day"] = result[TIME_COL].dt.day.astype("int16")
    return result


def make_model_dataset(
    df_feat: pd.DataFrame,
    feature_names: list[str],
    keep_date: bool = True,
) -> pd.DataFrame:
    available = [feature for feature in feature_names if feature in df_feat.columns]
    columns = available + [TARGET]
    if keep_date and TIME_COL in df_feat.columns and TIME_COL not in columns:
        columns = [TIME_COL] + columns
    return df_feat[columns].copy()


def chronological_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(TIME_COL).reset_index(drop=True)
    split_idx = int(len(ordered) * (1 - test_size))
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def chronological_validation_split(
    df: pd.DataFrame,
    valid_size: float = DEFAULT_VALIDATION_SHARE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return chronological_split(df, test_size=valid_size)


def split_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = train_df.drop(columns=[TARGET]).copy()
    X_test = test_df.drop(columns=[TARGET]).copy()
    y_train = (train_df[TARGET] == "Yes").astype(int)
    y_test = (test_df[TARGET] == "Yes").astype(int)
    return X_train, X_test, y_train, y_test


def is_categorical(series: pd.Series) -> bool:
    return (
        is_object_dtype(series)
        or is_string_dtype(series)
        or isinstance(series.dtype, CategoricalDtype)
    )


def build_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool = True,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    cat_cols = [column for column in X.columns if is_categorical(X[column])]
    num_cols = [column for column in X.columns if column not in cat_cols]

    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(steps=numeric_steps)
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )
    return preprocessor, num_cols, cat_cols


def _scale_pos_weight(y_train: pd.Series) -> float:
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    return neg / max(pos, 1)


def make_xgb_classifier(
    y_train: pd.Series,
    params: dict[str, float | int] | None = None,
) -> object:
    if XGBClassifier is None:
        return RandomForestClassifier(
            n_estimators=400,
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    resolved_params = BEST_XGB_PARAMS if params is None else params
    return XGBClassifier(
        scale_pos_weight=_scale_pos_weight(y_train),
        random_state=DEFAULT_RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=4,
        **resolved_params,
    )


def make_catboost_classifier(
    y_train: pd.Series,
    params: dict[str, float | int] | None = None,
) -> object | None:
    if CatBoostClassifier is None:
        return None

    resolved_params = BEST_CATBOOST_PARAMS if params is None else params
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights=None,
        scale_pos_weight=_scale_pos_weight(y_train),
        random_seed=DEFAULT_RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        **resolved_params,
    )


def make_standard_estimator(
    model_name: str,
    y_train: pd.Series,
    params: dict[str, float | int] | None = None,
) -> object:
    if model_name == "XGBoost":
        return make_xgb_classifier(y_train, params=params)

    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 400)) if params else 400,
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_depth=int(params["max_depth"]) if params and params.get("max_depth") is not None else None,
            min_samples_leaf=int(params.get("min_samples_leaf", 1)) if params else 1,
        )

    if model_name == "Logistic Regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=DEFAULT_RANDOM_STATE,
            C=float(params.get("C", 1.0)) if params else 1.0,
        )

    if model_name == "Linear SVM":
        return LinearSVC(
            class_weight="balanced",
            random_state=DEFAULT_RANDOM_STATE,
            C=float(params.get("C", 1.0)) if params else 1.0,
            max_iter=int(params.get("max_iter", 4000)) if params else 4000,
            dual=False,
        )

    raise ValueError(f"Unsupported non-CatBoost model: {model_name}")


def build_model_registry(
    y_train: pd.Series,
    catboost_params: dict[str, float | int] | None = None,
) -> dict[str, object]:
    models: dict[str, object] = {
        "Logistic Regression": make_standard_estimator("Logistic Regression", y_train),
        "Random Forest": make_standard_estimator("Random Forest", y_train),
        "XGBoost": make_xgb_classifier(y_train),
    }
    catboost_model = make_catboost_classifier(y_train, params=catboost_params)
    if catboost_model is not None:
        models["CatBoost"] = catboost_model
    return models


def prepare_catboost_frames(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    X_train_ready = X_train.copy()
    X_test_ready = X_test.copy()

    cat_cols = [column for column in X_train.columns if is_categorical(X_train[column])]
    num_cols = [column for column in X_train.columns if column not in cat_cols]

    for column in num_cols:
        median_value = X_train_ready[column].median()
        X_train_ready[column] = X_train_ready[column].fillna(median_value)
        X_test_ready[column] = X_test_ready[column].fillna(median_value)

    for column in cat_cols:
        X_train_ready[column] = X_train_ready[column].fillna("Missing").astype(str)
        X_test_ready[column] = X_test_ready[column].fillna("Missing").astype(str)

    return X_train_ready, X_test_ready, cat_cols


def fit_standard_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, float | int] | None = None,
) -> Pipeline:
    estimator = make_standard_estimator(model_name, y_train, params=params)
    scale_numeric = model_name in {"Logistic Regression", "Linear SVM"}
    preprocessor, _, _ = build_preprocessor(X_train, scale_numeric=scale_numeric)
    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", estimator),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_proba_for_plain_model(
    model_name: str,
    fitted_model: object,
    X: pd.DataFrame,
) -> np.ndarray:
    if model_name == "CatBoost":
        X_ready, _, _ = prepare_catboost_frames(X, X)
        return fitted_model.predict_proba(X_ready)[:, 1]
    if hasattr(fitted_model, "predict_proba"):
        return fitted_model.predict_proba(X)[:, 1]
    if hasattr(fitted_model, "decision_function"):
        margin = np.asarray(fitted_model.decision_function(X), dtype="float64")
        clipped = np.clip(margin, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-clipped))
    raise ValueError(f"Model '{model_name}' does not expose predict_proba or decision_function.")


def compare_plain_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    catboost_params: dict[str, float | int] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, float | str]] = []
    fitted: dict[str, object] = {}

    for name, estimator in build_model_registry(y_train, catboost_params=catboost_params).items():
        if name == "CatBoost":
            X_train_ready, X_test_ready, cat_cols = prepare_catboost_frames(X_train, X_test)
            estimator.fit(X_train_ready, y_train, cat_features=cat_cols)
            proba = estimator.predict_proba(X_test_ready)[:, 1]
            fitted[name] = estimator
        else:
            fitted[name] = fit_standard_model(name, X_train, y_train)
            proba = predict_proba_for_plain_model(name, fitted[name], X_test)

        rows.append({"model": name, **score_predictions(y_test, proba)})

    results = (
        pd.DataFrame(rows)
        .sort_values(["roc_auc", "f1"], ascending=False)
        .reset_index(drop=True)
    )
    return results, fitted


def score_predictions(
    y_true: pd.Series,
    proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    preds = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "accuracy": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds),
        "precision": precision_score(y_true, preds),
        "recall": recall_score(y_true, preds),
    }


def derive_accuracy_from_summary_metrics(
    *,
    support: int | float,
    event_rate: float,
    precision: float,
    recall: float,
) -> float:
    total = float(support)
    if total <= 0:
        return 0.0

    positives = float(event_rate) * total
    true_positives = float(recall) * positives
    if precision > 0:
        false_positives = true_positives * ((1.0 / float(precision)) - 1.0)
    else:
        false_positives = 0.0

    true_negatives = total - positives - false_positives
    accuracy = (true_positives + true_negatives) / total
    return float(np.clip(accuracy, 0.0, 1.0))


def rank_raw_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    preprocessor, _, cat_cols = build_preprocessor(X_train, scale_numeric=False)
    estimator = make_xgb_classifier(y_train)
    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", estimator),
        ]
    )
    pipeline.fit(X_train, y_train)

    encoded_names = pipeline.named_steps["prep"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    grouped: dict[str, float] = {column: 0.0 for column in X_train.columns}
    for encoded_name, importance in zip(encoded_names, importances):
        if encoded_name.startswith("num__"):
            raw_name = encoded_name.replace("num__", "", 1)
        else:
            raw_name = encoded_name
            for cat_col in cat_cols:
                prefix = f"cat__{cat_col}_"
                if encoded_name.startswith(prefix):
                    raw_name = cat_col
                    break
        grouped[raw_name] = grouped.get(raw_name, 0.0) + float(importance)

    ranking = (
        pd.DataFrame(
            {"feature": list(grouped.keys()), "importance": list(grouped.values())}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    ranking["importance_share"] = ranking["importance"] / ranking["importance"].sum()
    return ranking


def select_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 25,
) -> list[str]:
    ranking = rank_raw_features(X_train, y_train)
    selected = ranking.head(top_n)["feature"].tolist()
    if "location" in X_train.columns and "location" not in selected:
        selected.append("location")
    return selected


def tune_threshold(
    pipeline: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "XGBoost",
    thresholds: np.ndarray | None = None,
    predict_fn: Callable[[str, object, pd.DataFrame], np.ndarray] | None = None,
) -> tuple[float, pd.DataFrame]:
    if thresholds is None:
        thresholds = np.arange(0.3, 0.71, 0.02)

    predictor = predict_proba_for_plain_model if predict_fn is None else predict_fn
    proba = predictor(model_name, pipeline, X_test)

    rows = []
    for threshold in thresholds:
        preds = (proba >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "f1": f1_score(y_test, preds),
                "precision": precision_score(y_test, preds),
                "recall": recall_score(y_test, preds),
            }
        )

    frame = pd.DataFrame(rows)
    best_threshold = float(frame.loc[frame["f1"].idxmax(), "threshold"])
    return best_threshold, frame

