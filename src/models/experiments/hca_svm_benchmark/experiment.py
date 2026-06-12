from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from src.models.experiments.hybrid_imputation_breakthrough.experiment import tune_threshold_from_validation
from src.models.ines_feature_modeling import TARGET, fit_model_by_name, predict_proba_for_model, score_predictions
from src.utils.validation import BEST_FEATURE_SET_NAME, PipelineConfig, prepare_standard_split_frames


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "models" / "hca_svm_benchmark"
RESULTS_PATH = RESULTS_DIR / "hca_svm_candidate_results.csv"
SUMMARY_PATH = RESULTS_DIR / "hca_svm_summary.json"
NOTES_PATH = RESULTS_DIR / "notes.md"
PROFILE_PATH = RESULTS_DIR / "hca_station_profiles.csv"
CLUSTER_PATH = RESULTS_DIR / "hca_station_clusters.csv"
WINNER_SUMMARY_PATH = PROJECT_ROOT / "models" / "winner_model_calibration" / "winner_calibration_summary.json"

PROFILE_NUMERIC_COLUMNS = [
    "lat",
    "lon",
    "elevation",
    "rainfall",
    "humidity_3pm",
    "pressure_3pm",
    "temp_3pm",
    "wind_gust_speed",
    TARGET,
    "rainfall_missing_hybrid",
    "evaporation_missing_hybrid",
    "sunshine_missing_hybrid",
    "cloud_9am_missing_hybrid",
    "cloud_3pm_missing_hybrid",
    "pressure_3pm_missing_hybrid",
    "humidity_3pm_missing_hybrid",
]
ZONE_PREFIX = "rainfall_zone_"
CLUSTER_COUNTS = [4, 6, 8]
CANDIDATE_CONFIGS: list[dict[str, Any]] = [
    {
        "candidate": "svm_baseline_keep_location",
        "model_label": "Linear SVM",
        "cluster_count": None,
        "use_cluster_feature": False,
        "drop_location": False,
        "C": 1.0,
    },
    {
        "candidate": "svm_hca_4_add_location",
        "model_label": "Linear SVM",
        "cluster_count": 4,
        "use_cluster_feature": True,
        "drop_location": False,
        "C": 1.0,
    },
    {
        "candidate": "svm_hca_6_add_location",
        "model_label": "Linear SVM",
        "cluster_count": 6,
        "use_cluster_feature": True,
        "drop_location": False,
        "C": 1.0,
    },
    {
        "candidate": "svm_hca_8_add_location",
        "model_label": "Linear SVM",
        "cluster_count": 8,
        "use_cluster_feature": True,
        "drop_location": False,
        "C": 1.0,
    },
    {
        "candidate": "svm_hca_6_cluster_only_c1",
        "model_label": "Linear SVM",
        "cluster_count": 6,
        "use_cluster_feature": True,
        "drop_location": True,
        "C": 1.0,
    },
    {
        "candidate": "svm_hca_6_cluster_only_c2",
        "model_label": "Linear SVM",
        "cluster_count": 6,
        "use_cluster_feature": True,
        "drop_location": True,
        "C": 2.0,
    },
]
SELECTION_SORT_COLUMNS = [
    "validation_f1",
    "validation_roc_auc",
    "validation_precision",
    "validation_recall",
    "test_f1",
    "candidate",
]
SELECTION_SORT_ASCENDING = [False, False, False, False, False, True]


def write_notes() -> None:
    text = """# HCA + SVM Benchmark

## Goal

Test the mentor-requested combination of:

- HCA (hierarchical clustering analysis) on stations / locations
- an SVM classifier on the same hybrid-plus-core winner representation

## Design

- Reuse the existing chronological hybrid preprocessing pipeline so the comparison stays fair.
- Build train-only station profiles from geography, climatology, target rate, and missingness burden.
- Run agglomerative hierarchical clustering (Ward linkage) with 4, 6, and 8 clusters.
- Add the resulting station cluster as a categorical feature for the SVM benchmark.
- Compare a plain SVM baseline against clustered variants and a cluster-only variant that removes raw location.

## Important Constraint

This benchmark uses a linear SVM rather than an exact nonlinear kernel SVM.
That choice keeps the full chronological split, the full feature space, and the full sample size tractable in the
current CPU-only repository without collapsing the benchmark into a heavy downsampled toy run.

## Evaluation Rule

- train on the earliest block
- tune threshold on the validation block
- refit on train + validation
- evaluate once on the untouched test block
- rank candidates by validation-first evidence
"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text(text, encoding="utf-8")


def load_locked_winner_metrics() -> dict[str, float]:
    if not WINNER_SUMMARY_PATH.exists():
        return {}
    summary = json.loads(WINNER_SUMMARY_PATH.read_text(encoding="utf-8"))
    raw = summary.get("uncalibrated", {})
    return {
        "winner_test_roc_auc": float(raw.get("test_roc_auc", np.nan)),
        "winner_test_f1": float(raw.get("test_f1", np.nan)),
        "winner_test_precision": float(raw.get("test_precision", np.nan)),
        "winner_test_recall": float(raw.get("test_recall", np.nan)),
    }


def build_station_profiles(train_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    zone_columns = [column for column in train_df.columns if column.startswith(ZONE_PREFIX)]
    feature_columns = [column for column in PROFILE_NUMERIC_COLUMNS if column in train_df.columns] + zone_columns
    grouped = train_df.groupby("location", dropna=False)
    profiles = grouped[feature_columns].mean().reset_index()
    profiles["train_support"] = grouped.size().astype(int).to_numpy()
    profile_features = [column for column in profiles.columns if column not in {"location"}]
    profiles[profile_features] = profiles[profile_features].apply(pd.to_numeric, errors="coerce")
    profiles[profile_features] = profiles[profile_features].fillna(profiles[profile_features].median())
    return profiles, profile_features


def build_hca_assignments(
    profiles: pd.DataFrame,
    feature_columns: list[str],
    cluster_counts: list[int],
) -> tuple[pd.DataFrame, dict[int, dict[str, str]]]:
    scaled = StandardScaler().fit_transform(profiles[feature_columns].to_numpy(dtype="float64"))
    assignments = pd.DataFrame({"location": profiles["location"].astype(str)})
    cluster_maps: dict[int, dict[str, str]] = {}

    for cluster_count in cluster_counts:
        model = AgglomerativeClustering(n_clusters=int(cluster_count), linkage="ward")
        labels = model.fit_predict(scaled)
        column = f"station_cluster_hca_{cluster_count}"
        assignments[column] = [f"hca_{cluster_count}_cluster_{int(label)}" for label in labels]
        cluster_maps[cluster_count] = dict(zip(assignments["location"], assignments[column]))

    return assignments, cluster_maps


def add_cluster_feature(
    frame: pd.DataFrame,
    cluster_map: dict[str, str],
    cluster_count: int,
) -> tuple[pd.DataFrame, str, int]:
    result = frame.copy()
    column = f"station_cluster_hca_{cluster_count}"
    mapped = result["location"].astype(str).map(cluster_map)
    unknown_count = int(mapped.isna().sum())
    result[column] = mapped.fillna(f"hca_{cluster_count}_unknown").astype(str)
    return result, column, unknown_count


def evaluate_svm_feature_set(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    svm_params = params or {"C": 1.0}
    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df[TARGET].astype(int)
    y_valid = valid_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    fitted = fit_model_by_name("Linear SVM", X_train, y_train, params=svm_params)
    valid_scores = predict_proba_for_model("Linear SVM", fitted, X_valid)
    best_threshold, valid_metrics = tune_threshold_from_validation(valid_scores, y_valid)

    combined = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
    X_combined = combined[features].copy()
    y_combined = combined[TARGET].astype(int)
    final_model = fit_model_by_name("Linear SVM", X_combined, y_combined, params=svm_params)
    test_scores = predict_proba_for_model("Linear SVM", final_model, X_test)
    test_metrics = score_predictions(y_test, test_scores, threshold=best_threshold)

    return {
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
        "params": json.dumps(svm_params, sort_keys=True),
    }


def run_experiment() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_notes()

    config = PipelineConfig(name="hybrid_default")
    train_df, valid_df, test_df, feature_sets = prepare_standard_split_frames(config=config, keep_date=False)
    base_features = list(feature_sets[BEST_FEATURE_SET_NAME])

    station_profiles, profile_feature_columns = build_station_profiles(train_df)
    station_profiles.to_csv(PROFILE_PATH, index=False)
    cluster_assignments, cluster_maps = build_hca_assignments(station_profiles, profile_feature_columns, CLUSTER_COUNTS)
    cluster_assignments.to_csv(CLUSTER_PATH, index=False)

    rows: list[dict[str, Any]] = []
    winner_metrics = load_locked_winner_metrics()

    for candidate in CANDIDATE_CONFIGS:
        candidate_train = train_df.copy()
        candidate_valid = valid_df.copy()
        candidate_test = test_df.copy()
        features = list(base_features)
        cluster_feature_name = ""
        unknown_valid_count = 0
        unknown_test_count = 0

        cluster_count = candidate.get("cluster_count")
        if cluster_count is not None and candidate.get("use_cluster_feature", False):
            cluster_map = cluster_maps[int(cluster_count)]
            candidate_train, cluster_feature_name, _ = add_cluster_feature(candidate_train, cluster_map, int(cluster_count))
            candidate_valid, _, unknown_valid_count = add_cluster_feature(candidate_valid, cluster_map, int(cluster_count))
            candidate_test, _, unknown_test_count = add_cluster_feature(candidate_test, cluster_map, int(cluster_count))
            if cluster_feature_name not in features:
                features.append(cluster_feature_name)

        if candidate.get("drop_location", False):
            features = [feature for feature in features if feature != "location"]

        metrics = evaluate_svm_feature_set(
            candidate_train,
            candidate_valid,
            candidate_test,
            features,
            params={"C": float(candidate["C"])},
        )

        row = {
            "candidate": str(candidate["candidate"]),
            "candidate_label": str(candidate["candidate"]).replace("_", " "),
            "model": str(candidate["model_label"]),
            "cluster_count": int(cluster_count) if cluster_count is not None else 0,
            "use_cluster_feature": bool(candidate["use_cluster_feature"]),
            "drop_location": bool(candidate["drop_location"]),
            "cluster_feature_name": cluster_feature_name,
            "profile_feature_count": int(len(profile_feature_columns)),
            "station_count": int(len(station_profiles)),
            "train_support": int(len(train_df)),
            "validation_support": int(len(valid_df)),
            "test_support": int(len(test_df)),
            "unknown_validation_locations": int(unknown_valid_count),
            "unknown_test_locations": int(unknown_test_count),
            "C": float(candidate["C"]),
            **metrics,
        }
        if winner_metrics:
            row["winner_test_roc_auc_gap"] = float(row["test_roc_auc"] - winner_metrics["winner_test_roc_auc"])
            row["winner_test_f1_gap"] = float(row["test_f1"] - winner_metrics["winner_test_f1"])
            row["winner_test_precision_gap"] = float(row["test_precision"] - winner_metrics["winner_test_precision"])
            row["winner_test_recall_gap"] = float(row["test_recall"] - winner_metrics["winner_test_recall"])
        rows.append(row)

    results = (
        pd.DataFrame(rows)
        .sort_values(SELECTION_SORT_COLUMNS, ascending=SELECTION_SORT_ASCENDING)
        .reset_index(drop=True)
    )
    results.to_csv(RESULTS_PATH, index=False)

    best_row = results.iloc[0].to_dict()
    summary: dict[str, Any] = {
        "experiment": "hca_svm_benchmark",
        "model": "Linear SVM",
        "selection_basis": "validation_first_chronological_split",
        "feature_set_name": BEST_FEATURE_SET_NAME,
        "candidate_count": int(len(results)),
        "cluster_counts_tested": CLUSTER_COUNTS,
        "best_result": best_row,
        "results_path": str(RESULTS_PATH),
        "notes_path": str(NOTES_PATH),
        "profile_path": str(PROFILE_PATH),
        "cluster_path": str(CLUSTER_PATH),
    }
    if winner_metrics:
        summary["winner_comparison"] = {
            **winner_metrics,
            "svm_minus_winner_roc_auc": float(best_row["test_roc_auc"] - winner_metrics["winner_test_roc_auc"]),
            "svm_minus_winner_f1": float(best_row["test_f1"] - winner_metrics["winner_test_f1"]),
            "svm_minus_winner_precision": float(best_row["test_precision"] - winner_metrics["winner_test_precision"]),
            "svm_minus_winner_recall": float(best_row["test_recall"] - winner_metrics["winner_test_recall"]),
        }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))

