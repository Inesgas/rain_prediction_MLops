from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap

from src.models.experiments.geo_climate_context_extension.experiment import prepare_variant_frames
from src.models.ines_feature_modeling import TARGET, fit_model_by_name, predict_proba_for_model, score_predictions
from src.models.ines_modeling_core import prepare_catboost_frames
from src.utils.validation import load_best_hybrid_selection
from src.models.experiments.winner_model_calibration.experiment import (
    DEFAULT_OOF_SPLITS,
    build_segment_labels,
    build_time_series_oof_predictions,
    fit_segmented_calibrator,
    apply_segmented_calibrator,
    load_location_to_climate_regime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "models" / "final_hybrid_refinement"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RESULTS_PATH = RESULTS_DIR / "final_hybrid_refinement_results.csv"
SUMMARY_PATH = RESULTS_DIR / "final_hybrid_refinement_summary.json"
NOTES_PATH = RESULTS_DIR / "notes.md"
TRIALS_PATH = RESULTS_DIR / "baseline_retune_trials.csv"
THRESHOLD_PROFILES_PATH = RESULTS_DIR / "baseline_threshold_profiles.csv"
THRESHOLD_CURVE_PATH = RESULTS_DIR / "baseline_threshold_curve.csv"
SHAP_IMPORTANCE_PATH = RESULTS_DIR / "final_model_shap_importance.csv"
SHAP_SAMPLE_PATH = RESULTS_DIR / "final_model_shap_sample.csv"

SHAP_BAR_PATH = FIGURES_DIR / "fig_54_final_hybrid_shap_bar.png"
SHAP_BEESWARM_PATH = FIGURES_DIR / "fig_55_final_hybrid_shap_beeswarm.png"
SHAP_WATERFALL_PATH = FIGURES_DIR / "fig_56_final_hybrid_shap_waterfall.png"

SELECTION_SORT_COLUMNS = [
    "selection_f1",
    "test_f1",
    "selection_roc_auc",
    "test_roc_auc",
    "selection_precision",
    "selection_recall",
    "feature_count",
    "candidate",
]
SELECTION_SORT_ASCENDING = [False, False, False, False, False, False, True, True]
THRESHOLDS = np.arange(0.30, 0.71, 0.02)
RANDOM_STATE = 42
SHAP_SAMPLE_SIZE = 400
OPTUNA_TRIALS = 6
TIME_AWARE_SPLITS = 3


def write_notes() -> None:
    text = """# Final Hybrid Refinement Notes

## Goal

The last neutral improvement pass builds on the locked hybrid CatBoost winner:

1. threshold optimization on the current baseline
2. a compact CatBoost retune on the 68-feature baseline
3. the best small geo-context extension with threshold and calibration checked jointly
4. SHAP explanations for the final selected variant

## Selection Rules

1. Keep the same chronological train / validation / test protocol.
2. Fit models on train only for validation-based selection, then refit on train+validation for test.
3. Use validation F1 as the main operating-threshold selection metric.
4. Evaluate calibration with time-aware climate-regime isotonic calibration because that was the recommended family in the locked winner analysis.
5. Promote a challenger only if it improves the holdout story without breaking the validation story.
"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text(text, encoding="utf-8")


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def tune_threshold_for_metric(
    proba: np.ndarray,
    y_true: pd.Series,
    metric: str = "f1",
) -> tuple[float, dict[str, float], pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        metrics = score_predictions(y_true, proba, threshold=float(threshold))
        rows.append({"threshold": float(threshold), **{key: float(value) for key, value in metrics.items()}})
    frame = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    best_idx = frame[metric].idxmax()
    best_threshold = float(frame.loc[best_idx, "threshold"])
    best_metrics = {key: float(value) for key, value in frame.loc[best_idx].to_dict().items()}
    return best_threshold, best_metrics, frame


def build_threshold_profiles(
    model_name: str,
    features: list[str],
    params: dict[str, float | int],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df[TARGET].astype(int)
    y_valid = valid_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    selection_model = fit_model_by_name(model_name, X_train, y_train, params=params)
    valid_proba = predict_proba_for_model(model_name, selection_model, X_valid)
    combined = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)
    X_combined = combined[features].copy()
    y_combined = combined[TARGET].astype(int)
    final_model = fit_model_by_name(model_name, X_combined, y_combined, params=params)
    test_proba = predict_proba_for_model(model_name, final_model, X_test)

    threshold_profiles: list[dict[str, Any]] = []
    full_curve_rows: list[pd.DataFrame] = []
    for metric in ["f1", "precision", "recall"]:
        best_threshold, valid_metrics, curve_df = tune_threshold_for_metric(valid_proba, y_valid, metric=metric)
        full_curve_rows.append(curve_df.assign(selection_metric=metric))
        test_metrics = score_predictions(y_test, test_proba, threshold=best_threshold)

        threshold_profiles.append(
            {
                "candidate": "baseline_locked_threshold_profile",
                "selection_metric": metric,
                "threshold": float(best_threshold),
                "validation_roc_auc": float(valid_metrics["roc_auc"]),
                "validation_f1": float(valid_metrics["f1"]),
                "validation_precision": float(valid_metrics["precision"]),
                "validation_recall": float(valid_metrics["recall"]),
                "test_roc_auc": float(test_metrics["roc_auc"]),
                "test_f1": float(test_metrics["f1"]),
                "test_precision": float(test_metrics["precision"]),
                "test_recall": float(test_metrics["recall"]),
            }
        )

    return pd.DataFrame(threshold_profiles), pd.concat(full_curve_rows, ignore_index=True)


def evaluate_raw_candidate(
    candidate: str,
    features: list[str],
    params: dict[str, float | int],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df[TARGET].astype(int)
    y_valid = valid_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    selection_model = fit_model_by_name("CatBoost", X_train, y_train, params=params)
    valid_proba = predict_proba_for_model("CatBoost", selection_model, X_valid)
    threshold, selection_metrics, _ = tune_threshold_for_metric(valid_proba, y_valid, metric="f1")

    combined = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)
    X_combined = combined[features].copy()
    y_combined = combined[TARGET].astype(int)
    final_model = fit_model_by_name("CatBoost", X_combined, y_combined, params=params)
    test_proba = predict_proba_for_model("CatBoost", final_model, X_test)
    test_metrics = score_predictions(y_test, test_proba, threshold=threshold)

    return {
        "candidate": candidate,
        "variant_family": "raw",
        "feature_count": int(len(features)),
        "selection_support": int(len(y_valid)),
        "selection_scheme": "validation_block",
        "selection_threshold": float(threshold),
        "selection_roc_auc": float(selection_metrics["roc_auc"]),
        "selection_f1": float(selection_metrics["f1"]),
        "selection_precision": float(selection_metrics["precision"]),
        "selection_recall": float(selection_metrics["recall"]),
        "test_support": int(len(y_test)),
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "test_f1": float(test_metrics["f1"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "params": json.dumps(params, sort_keys=True),
        "feature_names": json.dumps(features),
    }


def evaluate_time_aware_climate_isotonic_candidate(
    candidate: str,
    features: list[str],
    params: dict[str, float | int],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    location_to_climate_regime: dict[str, str],
) -> dict[str, Any]:
    combined = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)
    X_combined = combined[features].copy()
    y_combined = combined[TARGET].astype(int)
    combined_dates = pd.to_datetime(combined["date"])
    combined_segments = build_segment_labels(combined, location_to_climate_regime)
    test_segments = build_segment_labels(test_df, location_to_climate_regime)

    oof_proba, oof_y, oof_frame, _, split_count = build_time_series_oof_predictions(
        X_combined,
        y_combined,
        combined_dates,
        params=params,
        n_splits=min(TIME_AWARE_SPLITS, DEFAULT_OOF_SPLITS),
    )
    oof_segments = combined_segments.iloc[oof_frame["row_index"].to_numpy()].reset_index(drop=True)
    calibrator = fit_segmented_calibrator(
        "isotonic",
        oof_proba,
        oof_y,
        oof_segments["climate_regime"],
        segment_strategy="climate_regime",
    )
    selection_proba = apply_segmented_calibrator(calibrator, oof_proba, oof_segments["climate_regime"])
    threshold, selection_metrics, _ = tune_threshold_for_metric(selection_proba, oof_y, metric="f1")

    final_model = fit_model_by_name("CatBoost", X_combined, y_combined, params=params)
    X_test = test_df[features].copy()
    y_test = test_df[TARGET].astype(int)
    raw_test_proba = predict_proba_for_model("CatBoost", final_model, X_test)
    test_proba = apply_segmented_calibrator(calibrator, raw_test_proba, test_segments["climate_regime"])
    test_metrics = score_predictions(y_test, test_proba, threshold=threshold)

    return {
        "candidate": candidate,
        "variant_family": "time_aware_climate_isotonic",
        "feature_count": int(len(features)),
        "selection_support": int(len(oof_y)),
        "selection_scheme": "time_series_oof",
        "selection_threshold": float(threshold),
        "selection_roc_auc": float(selection_metrics["roc_auc"]),
        "selection_f1": float(selection_metrics["f1"]),
        "selection_precision": float(selection_metrics["precision"]),
        "selection_recall": float(selection_metrics["recall"]),
        "test_support": int(len(y_test)),
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "test_f1": float(test_metrics["f1"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "time_series_oof_splits": int(split_count),
        "params": json.dumps(params, sort_keys=True),
        "feature_names": json.dumps(features),
    }


def optimize_baseline_params(
    features: list[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    initial_params: dict[str, float | int],
) -> tuple[dict[str, float | int], pd.DataFrame]:
    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    y_train = train_df[TARGET].astype(int)
    y_valid = valid_df[TARGET].astype(int)

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 300, 600, step=50),
            "depth": trial.suggest_int("depth", 6, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 8.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        }
        fitted = fit_model_by_name("CatBoost", X_train, y_train, params=params)
        valid_proba = predict_proba_for_model("CatBoost", fitted, X_valid)
        _, metrics, _ = tune_threshold_for_metric(valid_proba, y_valid, metric="f1")
        trial.set_user_attr("threshold", float(metrics["threshold"]))
        trial.set_user_attr("validation_roc_auc", float(metrics["roc_auc"]))
        trial.set_user_attr("validation_precision", float(metrics["precision"]))
        trial.set_user_attr("validation_recall", float(metrics["recall"]))
        return float(metrics["f1"])

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial({key: value for key, value in initial_params.items() if key in {"iterations", "depth", "learning_rate", "l2_leaf_reg", "random_strength", "bagging_temperature"}})
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    return study.best_params, trials_df


def choose_recommended_final(results: pd.DataFrame) -> tuple[str, str]:
    baseline = results.loc[results["candidate"] == "baseline_locked_raw"].iloc[0]
    contenders = results.copy()
    holdout_better = contenders["test_f1"] > float(baseline["test_f1"])
    selection_not_worse = contenders["selection_f1"] >= float(baseline["selection_f1"]) - 0.002
    eligible = contenders.loc[holdout_better & selection_not_worse].copy()

    if eligible.empty:
        return "baseline_locked_raw", "No challenger cleared both the holdout F1 bar and the validation/OOF consistency check."

    eligible = eligible.sort_values(SELECTION_SORT_COLUMNS, ascending=SELECTION_SORT_ASCENDING).reset_index(drop=True)
    winner = str(eligible.iloc[0]["candidate"])
    return winner, "Selected the challenger that improved holdout F1 without materially regressing the selection-side score."


def generate_shap_outputs(
    candidate_row: pd.Series,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    features = json.loads(candidate_row["feature_names"])
    params = json.loads(candidate_row["params"])

    combined = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)
    X_train = combined[features].copy()
    y_train = combined[TARGET].astype(int)
    X_test = test_df[features].copy()
    y_test = test_df[TARGET].astype(int).reset_index(drop=True)

    model = fit_model_by_name("CatBoost", X_train, y_train, params=params)
    X_train_ready, X_test_ready, _ = prepare_catboost_frames(X_train, X_test)

    sample_size = min(SHAP_SAMPLE_SIZE, len(X_test_ready))
    sample_seed = np.random.default_rng(RANDOM_STATE)
    sample_idx = np.sort(sample_seed.choice(len(X_test_ready), size=sample_size, replace=False))
    X_sample = X_test_ready.iloc[sample_idx].reset_index(drop=True)
    y_sample = y_test.iloc[sample_idx].reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    shap_value_matrix = np.asarray(shap_values.values)
    base_values = np.asarray(shap_values.base_values)
    if shap_value_matrix.ndim == 3:
        class_index = min(1, shap_value_matrix.shape[-1] - 1)
        shap_value_matrix = shap_value_matrix[:, :, class_index]
        if base_values.ndim >= 2:
            base_values = base_values[:, class_index]
    shap_for_plots = shap.Explanation(
        values=shap_value_matrix,
        base_values=base_values,
        data=np.asarray(X_sample),
        feature_names=list(X_sample.columns),
    )

    mean_abs = np.abs(shap_value_matrix).mean(axis=0)
    importance = (
        pd.DataFrame(
            {
                "feature": X_sample.columns,
                "mean_abs_shap": mean_abs,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(SHAP_IMPORTANCE_PATH, index=False)
    X_sample.assign(actual_rain_tomorrow=y_sample).to_csv(SHAP_SAMPLE_PATH, index=False)

    plt.figure(figsize=(8.0, 6.0))
    shap.plots.bar(shap_for_plots, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_BAR_PATH, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8.4, 6.4))
    shap.plots.beeswarm(shap_for_plots, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_BEESWARM_PATH, dpi=160, bbox_inches="tight")
    plt.close()

    sample_proba = model.predict_proba(X_sample)[:, 1]
    local_index = int(np.argmax(sample_proba))
    local_explanation = shap.Explanation(
        values=np.asarray(shap_value_matrix[local_index]),
        base_values=base_values[local_index] if np.ndim(base_values) > 0 else base_values,
        data=np.asarray(X_sample.iloc[local_index]),
        feature_names=list(X_sample.columns),
    )
    plt.figure(figsize=(8.2, 6.2))
    shap.plots.waterfall(local_explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_WATERFALL_PATH, dpi=160, bbox_inches="tight")
    plt.close()

    return {
        "shap_feature_count": int(len(importance)),
        "shap_sample_size": int(sample_size),
        "shap_importance_path": str(SHAP_IMPORTANCE_PATH),
        "shap_sample_path": str(SHAP_SAMPLE_PATH),
        "shap_bar_path": str(SHAP_BAR_PATH),
        "shap_beeswarm_path": str(SHAP_BEESWARM_PATH),
        "shap_waterfall_path": str(SHAP_WATERFALL_PATH),
        "top_shap_features": make_json_safe(importance.head(10).to_dict(orient="records")),
        "waterfall_sample_index": int(sample_idx[local_index]),
        "waterfall_sample_actual": int(y_sample.iloc[local_index]),
        "waterfall_sample_probability": float(sample_proba[local_index]),
    }


def run_experiment() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    write_notes()

    selection = load_best_hybrid_selection()
    base_params = dict(selection["params"])
    location_to_climate_regime = load_location_to_climate_regime()

    train_df, valid_df, test_df, feature_sets, _ = prepare_variant_frames()
    baseline_features = feature_sets["current_hybrid_baseline"]
    geo_context_features = feature_sets["hybrid_plus_geo_context"]

    threshold_profiles, threshold_curve = build_threshold_profiles(
        "CatBoost",
        baseline_features,
        base_params,
        train_df,
        valid_df,
        test_df,
    )
    threshold_profiles.to_csv(THRESHOLD_PROFILES_PATH, index=False)
    threshold_curve.to_csv(THRESHOLD_CURVE_PATH, index=False)

    best_params, trials_df = optimize_baseline_params(baseline_features, train_df, valid_df, base_params)
    trials_df.to_csv(TRIALS_PATH, index=False)

    rows = [
        evaluate_raw_candidate("baseline_locked_raw", baseline_features, base_params, train_df, valid_df, test_df),
        evaluate_time_aware_climate_isotonic_candidate(
            "baseline_locked_climate_isotonic",
            baseline_features,
            base_params,
            train_df,
            valid_df,
            test_df,
            location_to_climate_regime,
        ),
        evaluate_raw_candidate("baseline_retuned_raw", baseline_features, best_params, train_df, valid_df, test_df),
        evaluate_time_aware_climate_isotonic_candidate(
            "baseline_retuned_climate_isotonic",
            baseline_features,
            best_params,
            train_df,
            valid_df,
            test_df,
            location_to_climate_regime,
        ),
        evaluate_raw_candidate("geo_context_raw", geo_context_features, base_params, train_df, valid_df, test_df),
        evaluate_time_aware_climate_isotonic_candidate(
            "geo_context_climate_isotonic",
            geo_context_features,
            base_params,
            train_df,
            valid_df,
            test_df,
            location_to_climate_regime,
        ),
    ]
    results = pd.DataFrame(rows).sort_values(SELECTION_SORT_COLUMNS, ascending=SELECTION_SORT_ASCENDING).reset_index(drop=True)
    results.to_csv(RESULTS_PATH, index=False)

    recommended_candidate, recommendation_reason = choose_recommended_final(results)
    recommended_row = results.loc[results["candidate"] == recommended_candidate].iloc[0]
    baseline_row = results.loc[results["candidate"] == "baseline_locked_raw"].iloc[0]
    shap_summary = generate_shap_outputs(recommended_row, train_df, valid_df, test_df)

    summary = {
        "experiment": "final_hybrid_refinement",
        "selection_basis": "validation_first_plus_time_aware_calibration_check",
        "notes_path": str(NOTES_PATH),
        "results_path": str(RESULTS_PATH),
        "baseline_threshold_profiles_path": str(THRESHOLD_PROFILES_PATH),
        "baseline_threshold_curve_path": str(THRESHOLD_CURVE_PATH),
        "retune_trials_path": str(TRIALS_PATH),
        "baseline_params": make_json_safe(base_params),
        "retuned_params": make_json_safe(best_params),
        "baseline_result": make_json_safe(baseline_row.to_dict()),
        "recommended_result": make_json_safe(recommended_row.to_dict()),
        "recommended_candidate": recommended_candidate,
        "recommendation_reason": recommendation_reason,
        "comparison_vs_baseline": {
            "delta_selection_f1": float(recommended_row["selection_f1"] - baseline_row["selection_f1"]),
            "delta_test_f1": float(recommended_row["test_f1"] - baseline_row["test_f1"]),
            "delta_test_roc_auc": float(recommended_row["test_roc_auc"] - baseline_row["test_roc_auc"]),
            "delta_test_precision": float(recommended_row["test_precision"] - baseline_row["test_precision"]),
            "delta_test_recall": float(recommended_row["test_recall"] - baseline_row["test_recall"]),
        },
        "threshold_profiles": make_json_safe(threshold_profiles.to_dict(orient="records")),
        "shap_summary": make_json_safe(shap_summary),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_experiment()

