from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.paths import ALIGNED_TOP25_FEATURES, DATE_COLUMN, RAIN_MODEL_DATASET_ALIGNED, TARGET_COLUMN


def load_feature_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_baseline_pipeline(categorical_features: list[str], numeric_features: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            ),
        ]
    )


def train_baseline(
    dataset_path: Path = RAIN_MODEL_DATASET_ALIGNED,
    features_path: Path = ALIGNED_TOP25_FEATURES,
    threshold: float = 0.58,
) -> dict[str, Any]:
    feature_names = load_feature_names(features_path)
    required_columns = [DATE_COLUMN, TARGET_COLUMN, *feature_names]

    df = pd.read_csv(dataset_path, usecols=lambda column: column in required_columns)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df.dropna(subset=[DATE_COLUMN, TARGET_COLUMN]).copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN).reset_index(drop=True)

    y = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    valid_target = y.notna()
    df = df.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].astype(int).reset_index(drop=True)

    split_index = int(len(df) * 0.8)
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Dataset is too small for chronological baseline evaluation.")

    X = df[feature_names]
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_features = [feature for feature in feature_names if feature not in categorical_features]
    model = build_baseline_pipeline(categorical_features, numeric_features)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "model_name": "logistic_regression_baseline",
        "model_role": "baseline evaluation model",
        "feature_count": len(feature_names),
        "threshold": float(threshold),
        "metrics": {
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1": float(f1_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
        },
        "rows": {
            "train": int(len(X_train)),
            "test": int(len(X_test)),
        },
        "features": feature_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the Phase 1 baseline model.")
    parser.add_argument("--dataset", type=Path, default=RAIN_MODEL_DATASET_ALIGNED)
    parser.add_argument("--features", type=Path, default=ALIGNED_TOP25_FEATURES)
    parser.add_argument("--threshold", type=float, default=0.58)
    parser.add_argument("--output", type=Path, default=Path("references/phase1_baseline_metrics.json"))
    args = parser.parse_args()

    result = train_baseline(args.dataset, args.features, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
