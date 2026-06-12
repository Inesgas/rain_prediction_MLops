from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.validation import (
    BEST_FEATURE_SET_NAME,
    PipelineConfig,
    best_feature_list,
    evaluate_catboost_feature_set,
    load_best_hybrid_selection,
    prepare_standard_split_frames,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "models" / "hybrid_pipeline_ablation"
RESULTS_CSV_PATH = RESULTS_DIR / "hybrid_ablation_results.csv"
SUMMARY_PATH = RESULTS_DIR / "hybrid_ablation_summary.json"
NOTES_PATH = RESULTS_DIR / "notes.md"


def write_notes() -> None:
    text = """# Hybrid Pipeline Ablation Notes

## Goal

Identify which parts of the current hybrid winner are genuinely earning their keep.

## Ablations Included

- Full hybrid pipeline plus core features
- No spatial same-date donor fill
- No regime-based numeric fill
- No missingness indicators
- No core engineered feature block
- Simplified candidate with only simple lookup fill and no missingness indicators

## Why This Matters

The current pipeline is effective, but it has grown more complex. This ablation study helps us decide which components should stay in the final GitHub version and which ones can be removed without hurting performance.
"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text(text, encoding="utf-8")


def _variant_feature_list(
    variant_name: str,
    feature_sets: dict[str, list[str]],
    train_df: pd.DataFrame,
) -> list[str]:
    features = feature_sets.get(BEST_FEATURE_SET_NAME, best_feature_list(feature_sets))
    if variant_name == "no_missing_indicators":
        return [col for col in features if not col.endswith("_missing_hybrid")]
    if variant_name == "no_core_features":
        return feature_sets.get("hybrid_regime_keep_location_base", best_feature_list(feature_sets))
    if variant_name == "simplified_candidate":
        removable = {"humidity_rising_fast", "warming_day", "moisture_stability_3pm", "cloud_humidity_3pm_interaction"}
        return [col for col in features if col in train_df.columns and not col.endswith("_missing_hybrid") and col not in removable]
    return features


def run_experiment() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_notes()
    best_selection = load_best_hybrid_selection()

    variants = [
        ("full_hybrid_plus_core", PipelineConfig(name="full_hybrid")),
        ("no_spatial_fill", PipelineConfig(name="no_spatial_fill", use_spatial_fill=False)),
        ("no_regime_fill", PipelineConfig(name="no_regime_fill", use_regime_fill=False)),
        ("no_missing_indicators", PipelineConfig(name="no_missing_indicators", add_missing_indicators=False)),
        ("no_core_features", PipelineConfig(name="no_core_features", add_core_features=False)),
        (
            "simplified_candidate",
            PipelineConfig(
                name="simplified_candidate",
                use_spatial_fill=False,
                use_regime_fill=False,
                add_missing_indicators=False,
                add_core_features=True,
            ),
        ),
    ]

    rows: list[dict[str, Any]] = []
    for variant_name, config in variants:
        train_df, valid_df, test_df, feature_sets = prepare_standard_split_frames(config)
        features = _variant_feature_list(variant_name, feature_sets, train_df)
        metrics = evaluate_catboost_feature_set(
            train_df,
            valid_df,
            test_df,
            features,
            params=best_selection["params"],
        )
        rows.append(
            {
                "variant": variant_name,
                "use_spatial_fill": int(config.use_spatial_fill),
                "use_regime_fill": int(config.use_regime_fill),
                "add_missing_indicators": int(config.add_missing_indicators),
                "add_core_features": int(config.add_core_features),
                "feature_count": int(len(features)),
                **{key: value for key, value in metrics.items() if key != "feature_count"},
            }
        )

    results = pd.DataFrame(rows).sort_values(["test_f1", "test_roc_auc"], ascending=False).reset_index(drop=True)
    results.to_csv(RESULTS_CSV_PATH, index=False)
    best_row = results.iloc[0].to_dict()

    summary = {
        "experiment": "hybrid_ablation_study",
        "best_result": {key: (float(value) if isinstance(value, (int, float)) else value) for key, value in best_row.items()},
        "results_path": str(RESULTS_CSV_PATH),
        "notes_path": str(NOTES_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))

