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
    load_modeling_base_table,
    make_expanding_windows,
    prepare_configured_frames,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "models" / "rolling_robustness_validation"
RESULTS_CSV_PATH = RESULTS_DIR / "rolling_robustness_results.csv"
SUMMARY_PATH = RESULTS_DIR / "rolling_robustness_summary.json"
NOTES_PATH = RESULTS_DIR / "notes.md"


def write_notes() -> None:
    text = """# Rolling Robustness Validation Notes

## Goal

Test whether the current hybrid-regime CatBoost result is stable across multiple chronological windows rather than only one split.

## Why This Matters

The mentor feedback was clear: the current gain is promising, but we need to show it holds across time and not just in one favorable period.

## Design Choices

- Reuse the current best hybrid-regime pipeline instead of inventing a new one.
- Keep the model family fixed to CatBoost so the question is stability, not model search.
- Use expanding-window chronological validation:
  - train on earlier periods
  - validate on the next block
  - test on the following block
- Report per-split metrics plus mean and standard deviation.

## Interpretation Plan

- If the score stays tight across splits, the pipeline looks more credible.
- If the score swings a lot, the current single-split result is less trustworthy and we should simplify before adding more ideas.
"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text(text, encoding="utf-8")


def run_experiment() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_notes()

    raw_df = load_modeling_base_table()
    windows = make_expanding_windows(raw_df)
    config = PipelineConfig(name="hybrid_default")
    best_selection = load_best_hybrid_selection()
    rows: list[dict[str, Any]] = []

    for window in windows:
        train_df, valid_df, test_df, feature_sets = prepare_configured_frames(
            window["train_df"],
            window["valid_df"],
            window["test_df"],
            config,
        )
        features = feature_sets.get(BEST_FEATURE_SET_NAME, best_feature_list(feature_sets))
        metrics = evaluate_catboost_feature_set(
            train_df,
            valid_df,
            test_df,
            features,
            params=best_selection["params"],
        )
        rows.append(
            {
                "split_name": window["split_name"],
                "train_start": window["train_start"],
                "train_end": window["train_end"],
                "valid_start": window["valid_start"],
                "valid_end": window["valid_end"],
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "feature_set": BEST_FEATURE_SET_NAME,
                **metrics,
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_CSV_PATH, index=False)

    summary = {
        "experiment": "rolling_robustness_validation",
        "model": "CatBoost",
        "feature_set": BEST_FEATURE_SET_NAME,
        "split_count": int(len(results)),
        "mean_test_f1": float(results["test_f1"].mean()),
        "std_test_f1": float(results["test_f1"].std(ddof=0)),
        "mean_test_roc_auc": float(results["test_roc_auc"].mean()),
        "std_test_roc_auc": float(results["test_roc_auc"].std(ddof=0)),
        "min_test_f1": float(results["test_f1"].min()),
        "max_test_f1": float(results["test_f1"].max()),
        "notes_path": str(NOTES_PATH),
        "results_path": str(RESULTS_CSV_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))

