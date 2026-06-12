from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.models.ines_feature_modeling import TARGET
from src.utils.validation import PipelineConfig, prepare_standard_split_frames


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "models" / "hca_station_benchmark"
RESULTS_PATH = RESULTS_DIR / "hca_cluster_results.csv"
SUMMARY_PATH = RESULTS_DIR / "hca_cluster_summary.json"
NOTES_PATH = RESULTS_DIR / "notes.md"
PROFILE_PATH = RESULTS_DIR / "hca_station_profiles.csv"
CLUSTER_PATH = RESULTS_DIR / "hca_station_clusters.csv"

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
SELECTION_SORT_COLUMNS = [
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "min_cluster_size",
]
SELECTION_SORT_ASCENDING = [False, False, True, False]


def write_notes() -> None:
    text = """# HCA Station Benchmark

## Goal

Run a standalone hierarchical clustering analysis (HCA) on stations so the clustering itself can be evaluated as a
separate benchmark rather than mixing it directly into a classifier comparison.

## Design

- Reuse the hybrid preprocessing pipeline to build the train partition only.
- Aggregate each station into a train-only station profile using:
  - geography
  - elevation
  - rainfall-zone indicators
  - average weather values
  - target rate
  - missingness burden
- Standardize the station profiles.
- Run agglomerative clustering with Ward linkage for 4, 6, and 8 clusters.
- Score each clustering using silhouette, Calinski-Harabasz, and Davies-Bouldin criteria.

## Why Separate It

The mentor requested HCA and SVM as separate benchmarks. This experiment therefore asks only one question:
do station clusters show usable structure on their own?
"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text(text, encoding="utf-8")


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


def run_experiment() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_notes()

    config = PipelineConfig(name="hybrid_default")
    train_df, _, _, _ = prepare_standard_split_frames(config=config, keep_date=False)

    profiles, feature_columns = build_station_profiles(train_df)
    profiles.to_csv(PROFILE_PATH, index=False)

    scaled = StandardScaler().fit_transform(profiles[feature_columns].to_numpy(dtype="float64"))
    cluster_assignments = pd.DataFrame({"location": profiles["location"].astype(str)})
    rows: list[dict[str, Any]] = []

    for cluster_count in CLUSTER_COUNTS:
        model = AgglomerativeClustering(n_clusters=int(cluster_count), linkage="ward")
        labels = model.fit_predict(scaled)
        label_names = [f"hca_{cluster_count}_cluster_{int(label)}" for label in labels]
        column = f"station_cluster_hca_{cluster_count}"
        cluster_assignments[column] = label_names

        unique_labels, counts = np.unique(labels, return_counts=True)
        row = {
            "cluster_count": int(cluster_count),
            "station_count": int(len(profiles)),
            "profile_feature_count": int(len(feature_columns)),
            "silhouette_score": float(silhouette_score(scaled, labels)),
            "calinski_harabasz_score": float(calinski_harabasz_score(scaled, labels)),
            "davies_bouldin_score": float(davies_bouldin_score(scaled, labels)),
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
            "mean_cluster_size": float(counts.mean()),
            "cluster_size_summary": json.dumps(
                {f"cluster_{int(label)}": int(count) for label, count in zip(unique_labels, counts)}
            ),
        }
        rows.append(row)

    cluster_assignments.to_csv(CLUSTER_PATH, index=False)
    results = (
        pd.DataFrame(rows)
        .sort_values(SELECTION_SORT_COLUMNS, ascending=SELECTION_SORT_ASCENDING)
        .reset_index(drop=True)
    )
    results.to_csv(RESULTS_PATH, index=False)

    best_row = results.iloc[0].to_dict()
    summary = {
        "experiment": "hca_station_benchmark",
        "selection_basis": "unsupervised_cluster_quality",
        "cluster_counts_tested": CLUSTER_COUNTS,
        "station_count": int(len(profiles)),
        "profile_feature_count": int(len(feature_columns)),
        "best_result": best_row,
        "results_path": str(RESULTS_PATH),
        "notes_path": str(NOTES_PATH),
        "profile_path": str(PROFILE_PATH),
        "cluster_path": str(CLUSTER_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))

