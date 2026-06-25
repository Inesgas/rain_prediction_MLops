from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

from src.models.experiments.daily_zonal_baseline.experiment import enrich_metadata_with_rainfall_zone
from src.models.experiments.missingness_aware_hybrid_model.experiment import prepare_final_winner_frames
from src.models.ines_feature_modeling import fit_model_by_name, predict_proba_for_model, score_predictions
from src.utils.validation import load_best_hybrid_selection
from src.utils.validation.hybrid_pipeline import month_to_season


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "models" / "winner_model_calibration"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
THRESHOLDS = np.arange(0.30, 0.71, 0.02)
RELIABILITY_BINS = 15
DEFAULT_OOF_SPLITS = 5
SEGMENT_MIN_SUPPORT = 500
SEGMENT_MIN_CLASS_SUPPORT = 25

SUMMARY_PATH = RESULTS_DIR / "winner_calibration_summary.json"
METHODS_PATH = RESULTS_DIR / "winner_calibration_methods.csv"
OOF_FOLDS_PATH = RESULTS_DIR / "winner_time_series_calibration_folds.csv"
SEGMENT_SUPPORT_PATH = RESULTS_DIR / "winner_segmented_calibration_support.csv"

RAW_CURVE_PATH = RESULTS_DIR / "winner_uncalibrated_curve.csv"
SIGMOID_CURVE_PATH = RESULTS_DIR / "winner_sigmoid_curve.csv"
ISOTONIC_CURVE_PATH = RESULTS_DIR / "winner_isotonic_curve.csv"
VALIDATION_RAW_CURVE_PATH = RESULTS_DIR / "winner_validation_raw_curve.csv"
VALIDATION_SIGMOID_CURVE_PATH = RESULTS_DIR / "winner_validation_sigmoid_curve.csv"
VALIDATION_ISOTONIC_CURVE_PATH = RESULTS_DIR / "winner_validation_isotonic_curve.csv"
VALIDATION_SEASON_SIGMOID_CURVE_PATH = RESULTS_DIR / "winner_validation_season_sigmoid_curve.csv"
VALIDATION_SEASON_ISOTONIC_CURVE_PATH = RESULTS_DIR / "winner_validation_season_isotonic_curve.csv"
VALIDATION_CLIMATE_SIGMOID_CURVE_PATH = RESULTS_DIR / "winner_validation_climate_regime_sigmoid_curve.csv"
VALIDATION_CLIMATE_ISOTONIC_CURVE_PATH = RESULTS_DIR / "winner_validation_climate_regime_isotonic_curve.csv"
VALIDATION_HIERARCHICAL_SIGMOID_CURVE_PATH = RESULTS_DIR / "winner_validation_climate_regime_season_sigmoid_curve.csv"
VALIDATION_HIERARCHICAL_ISOTONIC_CURVE_PATH = RESULTS_DIR / "winner_validation_climate_regime_season_isotonic_curve.csv"
TIME_SERIES_SEASON_SIGMOID_CURVE_PATH = RESULTS_DIR / "winner_season_sigmoid_curve.csv"
TIME_SERIES_SEASON_ISOTONIC_CURVE_PATH = RESULTS_DIR / "winner_season_isotonic_curve.csv"
TIME_SERIES_CLIMATE_SIGMOID_CURVE_PATH = RESULTS_DIR / "winner_climate_regime_sigmoid_curve.csv"
TIME_SERIES_CLIMATE_ISOTONIC_CURVE_PATH = RESULTS_DIR / "winner_climate_regime_isotonic_curve.csv"
TIME_SERIES_HIERARCHICAL_SIGMOID_CURVE_PATH = RESULTS_DIR / "winner_climate_regime_season_sigmoid_curve.csv"
TIME_SERIES_HIERARCHICAL_ISOTONIC_CURVE_PATH = RESULTS_DIR / "winner_climate_regime_season_isotonic_curve.csv"

RAW_THRESHOLD_PATH = RESULTS_DIR / "winner_uncalibrated_threshold_curve.csv"
SIGMOID_THRESHOLD_PATH = RESULTS_DIR / "winner_sigmoid_threshold_curve.csv"
ISOTONIC_THRESHOLD_PATH = RESULTS_DIR / "winner_isotonic_threshold_curve.csv"
VALIDATION_RAW_THRESHOLD_PATH = RESULTS_DIR / "winner_validation_raw_threshold_curve.csv"
VALIDATION_SIGMOID_THRESHOLD_PATH = RESULTS_DIR / "winner_validation_sigmoid_threshold_curve.csv"
VALIDATION_ISOTONIC_THRESHOLD_PATH = RESULTS_DIR / "winner_validation_isotonic_threshold_curve.csv"
VALIDATION_SEASON_SIGMOID_THRESHOLD_PATH = RESULTS_DIR / "winner_validation_season_sigmoid_threshold_curve.csv"
VALIDATION_SEASON_ISOTONIC_THRESHOLD_PATH = RESULTS_DIR / "winner_validation_season_isotonic_threshold_curve.csv"
VALIDATION_CLIMATE_SIGMOID_THRESHOLD_PATH = RESULTS_DIR / "winner_validation_climate_regime_sigmoid_threshold_curve.csv"
VALIDATION_CLIMATE_ISOTONIC_THRESHOLD_PATH = RESULTS_DIR / "winner_validation_climate_regime_isotonic_threshold_curve.csv"
VALIDATION_HIERARCHICAL_SIGMOID_THRESHOLD_PATH = RESULTS_DIR / "winner_validation_climate_regime_season_sigmoid_threshold_curve.csv"
VALIDATION_HIERARCHICAL_ISOTONIC_THRESHOLD_PATH = RESULTS_DIR / "winner_validation_climate_regime_season_isotonic_threshold_curve.csv"
TIME_SERIES_SEASON_SIGMOID_THRESHOLD_PATH = RESULTS_DIR / "winner_season_sigmoid_threshold_curve.csv"
TIME_SERIES_SEASON_ISOTONIC_THRESHOLD_PATH = RESULTS_DIR / "winner_season_isotonic_threshold_curve.csv"
TIME_SERIES_CLIMATE_SIGMOID_THRESHOLD_PATH = RESULTS_DIR / "winner_climate_regime_sigmoid_threshold_curve.csv"
TIME_SERIES_CLIMATE_ISOTONIC_THRESHOLD_PATH = RESULTS_DIR / "winner_climate_regime_isotonic_threshold_curve.csv"
TIME_SERIES_HIERARCHICAL_SIGMOID_THRESHOLD_PATH = RESULTS_DIR / "winner_climate_regime_season_sigmoid_threshold_curve.csv"
TIME_SERIES_HIERARCHICAL_ISOTONIC_THRESHOLD_PATH = RESULTS_DIR / "winner_climate_regime_season_isotonic_threshold_curve.csv"

RAW_FIGURE_PATH = FIGURES_DIR / "fig_33_winner_calibration_raw.png"
SIGMOID_FIGURE_PATH = FIGURES_DIR / "fig_34_winner_calibration_sigmoid.png"
COMPARISON_FIGURE_PATH = FIGURES_DIR / "fig_35_winner_calibration_comparison.png"
ISOTONIC_FIGURE_PATH = FIGURES_DIR / "fig_43_winner_calibration_isotonic.png"
SEASON_SEGMENTED_FIGURE_PATH = FIGURES_DIR / "fig_50_winner_calibration_segmented_season.png"
CLIMATE_SEGMENTED_FIGURE_PATH = FIGURES_DIR / "fig_51_winner_calibration_segmented_climate_regime.png"
SEGMENTED_COMPARISON_FIGURE_PATH = FIGURES_DIR / "fig_52_winner_calibration_segmented_comparison.png"
HIERARCHICAL_SEGMENTED_FIGURE_PATH = FIGURES_DIR / "fig_53_winner_calibration_segmented_hierarchical.png"

METHOD_CONFIG: dict[str, dict[str, Any]] = {
    "validation_block_raw": {
        "label": "Validation-block raw",
        "scheme": "validation_block",
        "family": "raw",
        "segment_strategy": "global",
        "curve_path": VALIDATION_RAW_CURVE_PATH,
        "threshold_path": VALIDATION_RAW_THRESHOLD_PATH,
        "color": "#9a3412",
    },
    "validation_block_sigmoid": {
        "label": "Validation-block sigmoid",
        "scheme": "validation_block",
        "family": "sigmoid",
        "segment_strategy": "global",
        "curve_path": VALIDATION_SIGMOID_CURVE_PATH,
        "threshold_path": VALIDATION_SIGMOID_THRESHOLD_PATH,
        "color": "#0f766e",
    },
    "validation_block_isotonic": {
        "label": "Validation-block isotonic",
        "scheme": "validation_block",
        "family": "isotonic",
        "segment_strategy": "global",
        "curve_path": VALIDATION_ISOTONIC_CURVE_PATH,
        "threshold_path": VALIDATION_ISOTONIC_THRESHOLD_PATH,
        "color": "#15803d",
    },
    "validation_block_season_sigmoid": {
        "label": "Validation-block sigmoid by season",
        "scheme": "validation_block",
        "family": "sigmoid",
        "segment_strategy": "season",
        "curve_path": VALIDATION_SEASON_SIGMOID_CURVE_PATH,
        "threshold_path": VALIDATION_SEASON_SIGMOID_THRESHOLD_PATH,
        "color": "#0d9488",
    },
    "validation_block_season_isotonic": {
        "label": "Validation-block isotonic by season",
        "scheme": "validation_block",
        "family": "isotonic",
        "segment_strategy": "season",
        "curve_path": VALIDATION_SEASON_ISOTONIC_CURVE_PATH,
        "threshold_path": VALIDATION_SEASON_ISOTONIC_THRESHOLD_PATH,
        "color": "#22c55e",
    },
    "validation_block_climate_regime_sigmoid": {
        "label": "Validation-block sigmoid by climate regime",
        "scheme": "validation_block",
        "family": "sigmoid",
        "segment_strategy": "climate_regime",
        "curve_path": VALIDATION_CLIMATE_SIGMOID_CURVE_PATH,
        "threshold_path": VALIDATION_CLIMATE_SIGMOID_THRESHOLD_PATH,
        "color": "#0891b2",
    },
    "validation_block_climate_regime_isotonic": {
        "label": "Validation-block isotonic by climate regime",
        "scheme": "validation_block",
        "family": "isotonic",
        "segment_strategy": "climate_regime",
        "curve_path": VALIDATION_CLIMATE_ISOTONIC_CURVE_PATH,
        "threshold_path": VALIDATION_CLIMATE_ISOTONIC_THRESHOLD_PATH,
        "color": "#16a34a",
    },
    "validation_block_climate_regime_season_sigmoid": {
        "label": "Validation-block sigmoid by climate regime and season",
        "scheme": "validation_block",
        "family": "sigmoid",
        "segment_strategy": "climate_regime_season_hierarchical",
        "curve_path": VALIDATION_HIERARCHICAL_SIGMOID_CURVE_PATH,
        "threshold_path": VALIDATION_HIERARCHICAL_SIGMOID_THRESHOLD_PATH,
        "color": "#0ea5e9",
    },
    "validation_block_climate_regime_season_isotonic": {
        "label": "Validation-block isotonic by climate regime and season",
        "scheme": "validation_block",
        "family": "isotonic",
        "segment_strategy": "climate_regime_season_hierarchical",
        "curve_path": VALIDATION_HIERARCHICAL_ISOTONIC_CURVE_PATH,
        "threshold_path": VALIDATION_HIERARCHICAL_ISOTONIC_THRESHOLD_PATH,
        "color": "#2563eb",
    },
    "time_series_raw": {
        "label": "Time-aware raw refit",
        "scheme": "time_series_oof",
        "family": "raw",
        "segment_strategy": "global",
        "curve_path": RAW_CURVE_PATH,
        "threshold_path": RAW_THRESHOLD_PATH,
        "color": "#d97706",
    },
    "time_series_sigmoid": {
        "label": "Time-aware sigmoid",
        "scheme": "time_series_oof",
        "family": "sigmoid",
        "segment_strategy": "global",
        "curve_path": SIGMOID_CURVE_PATH,
        "threshold_path": SIGMOID_THRESHOLD_PATH,
        "color": "#12798a",
    },
    "time_series_isotonic": {
        "label": "Time-aware isotonic",
        "scheme": "time_series_oof",
        "family": "isotonic",
        "segment_strategy": "global",
        "curve_path": ISOTONIC_CURVE_PATH,
        "threshold_path": ISOTONIC_THRESHOLD_PATH,
        "color": "#2563eb",
    },
    "time_series_season_sigmoid": {
        "label": "Time-aware sigmoid by season",
        "scheme": "time_series_oof",
        "family": "sigmoid",
        "segment_strategy": "season",
        "curve_path": TIME_SERIES_SEASON_SIGMOID_CURVE_PATH,
        "threshold_path": TIME_SERIES_SEASON_SIGMOID_THRESHOLD_PATH,
        "color": "#0f766e",
    },
    "time_series_season_isotonic": {
        "label": "Time-aware isotonic by season",
        "scheme": "time_series_oof",
        "family": "isotonic",
        "segment_strategy": "season",
        "curve_path": TIME_SERIES_SEASON_ISOTONIC_CURVE_PATH,
        "threshold_path": TIME_SERIES_SEASON_ISOTONIC_THRESHOLD_PATH,
        "color": "#15803d",
    },
    "time_series_climate_regime_sigmoid": {
        "label": "Time-aware sigmoid by climate regime",
        "scheme": "time_series_oof",
        "family": "sigmoid",
        "segment_strategy": "climate_regime",
        "curve_path": TIME_SERIES_CLIMATE_SIGMOID_CURVE_PATH,
        "threshold_path": TIME_SERIES_CLIMATE_SIGMOID_THRESHOLD_PATH,
        "color": "#0369a1",
    },
    "time_series_climate_regime_isotonic": {
        "label": "Time-aware isotonic by climate regime",
        "scheme": "time_series_oof",
        "family": "isotonic",
        "segment_strategy": "climate_regime",
        "curve_path": TIME_SERIES_CLIMATE_ISOTONIC_CURVE_PATH,
        "threshold_path": TIME_SERIES_CLIMATE_ISOTONIC_THRESHOLD_PATH,
        "color": "#1d4ed8",
    },
    "time_series_climate_regime_season_sigmoid": {
        "label": "Time-aware sigmoid by climate regime and season",
        "scheme": "time_series_oof",
        "family": "sigmoid",
        "segment_strategy": "climate_regime_season_hierarchical",
        "curve_path": TIME_SERIES_HIERARCHICAL_SIGMOID_CURVE_PATH,
        "threshold_path": TIME_SERIES_HIERARCHICAL_SIGMOID_THRESHOLD_PATH,
        "color": "#0284c7",
    },
    "time_series_climate_regime_season_isotonic": {
        "label": "Time-aware isotonic by climate regime and season",
        "scheme": "time_series_oof",
        "family": "isotonic",
        "segment_strategy": "climate_regime_season_hierarchical",
        "curve_path": TIME_SERIES_HIERARCHICAL_ISOTONIC_CURVE_PATH,
        "threshold_path": TIME_SERIES_HIERARCHICAL_ISOTONIC_THRESHOLD_PATH,
        "color": "#1e40af",
    },
}


def fit_sigmoid_calibrator(valid_proba: np.ndarray, y_valid: pd.Series) -> LogisticRegression:
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator.fit(np.asarray(valid_proba).reshape(-1, 1), np.asarray(y_valid).astype(int))
    return calibrator


def apply_sigmoid_calibrator(calibrator: LogisticRegression, proba: np.ndarray) -> np.ndarray:
    calibrated = calibrator.predict_proba(np.asarray(proba).reshape(-1, 1))[:, 1]
    return np.clip(calibrated, 1e-6, 1 - 1e-6)


def fit_isotonic_calibrator(valid_proba: np.ndarray, y_valid: pd.Series) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    calibrator.fit(np.asarray(valid_proba).astype(float), np.asarray(y_valid).astype(int))
    return calibrator


def apply_isotonic_calibrator(calibrator: IsotonicRegression, proba: np.ndarray) -> np.ndarray:
    calibrated = calibrator.predict(np.asarray(proba).astype(float))
    return np.clip(np.asarray(calibrated, dtype=float), 1e-6, 1 - 1e-6)


def fit_calibrator(family: str, proba: np.ndarray, y_true: pd.Series) -> Any:
    if family == "sigmoid":
        return fit_sigmoid_calibrator(proba, y_true)
    if family == "isotonic":
        return fit_isotonic_calibrator(proba, y_true)
    raise ValueError(f"Unsupported calibration family: {family}")


def apply_calibrator(family: str, calibrator: Any, proba: np.ndarray) -> np.ndarray:
    if family == "sigmoid":
        return apply_sigmoid_calibrator(calibrator, proba)
    if family == "isotonic":
        return apply_isotonic_calibrator(calibrator, proba)
    raise ValueError(f"Unsupported calibration family: {family}")


def load_location_to_climate_regime() -> dict[str, str]:
    metadata_path = enrich_metadata_with_rainfall_zone()
    metadata = pd.read_csv(metadata_path)
    if "rainfall_zone" not in metadata.columns:
        raise ValueError("Rainfall-zone metadata is required for climate-regime calibration.")
    return metadata.set_index("location")["rainfall_zone"].astype(str).to_dict()


def build_segment_labels(frame: pd.DataFrame, location_to_climate_regime: dict[str, str]) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)

    if "month" in frame.columns:
        months = pd.to_numeric(frame["month"], errors="coerce")
    elif "date" in frame.columns:
        months = pd.to_datetime(frame["date"], errors="coerce").dt.month
    else:
        raise ValueError("The winner frames must retain month or date for seasonal calibration.")

    result["season"] = (
        months.fillna(-1)
        .astype(int)
        .map(lambda value: month_to_season(int(value)) if int(value) in range(1, 13) else "Unknown")
        .astype(str)
    )

    if "location" not in frame.columns:
        raise ValueError("The winner frames must retain location for climate-regime calibration.")
    result["climate_regime"] = frame["location"].astype(str).map(location_to_climate_regime).fillna("Unknown")
    result["climate_regime_season"] = result["climate_regime"].astype(str) + " | " + result["season"].astype(str)
    return result.reset_index(drop=True)


def segment_is_trainable(y_true: pd.Series) -> bool:
    positives = int(np.asarray(y_true).astype(int).sum())
    negatives = int(len(y_true) - positives)
    return len(y_true) >= SEGMENT_MIN_SUPPORT and positives >= SEGMENT_MIN_CLASS_SUPPORT and negatives >= SEGMENT_MIN_CLASS_SUPPORT


def fit_segmented_calibrator(
    family: str,
    proba: np.ndarray,
    y_true: pd.Series,
    segments: pd.Series,
    segment_strategy: str,
    min_support: int = SEGMENT_MIN_SUPPORT,
    min_class_support: int = SEGMENT_MIN_CLASS_SUPPORT,
) -> dict[str, Any]:
    proba_array = np.asarray(proba).astype(float)
    y_series = pd.Series(np.asarray(y_true).astype(int)).reset_index(drop=True)
    segment_series = pd.Series(segments).fillna("Unknown").astype(str).reset_index(drop=True)

    global_calibrator = fit_calibrator(family, proba_array, y_series)
    segment_calibrators: dict[str, Any] = {}
    support_rows: list[dict[str, Any]] = []

    for segment_value in segment_series.drop_duplicates().tolist():
        mask = (segment_series == segment_value).to_numpy()
        y_segment = y_series.loc[mask].reset_index(drop=True)
        positives = int(y_segment.sum())
        negatives = int(len(y_segment) - positives)
        use_segment_calibrator = (
            segment_is_trainable(y_segment)
            and len(y_segment) >= min_support
            and positives >= min_class_support
            and negatives >= min_class_support
        )
        if use_segment_calibrator:
            segment_calibrators[str(segment_value)] = fit_calibrator(family, proba_array[mask], y_segment)
        support_rows.append(
            {
                "segment_strategy": segment_strategy,
                "segment_value": str(segment_value),
                "support": int(len(y_segment)),
                "positive_support": positives,
                "negative_support": negatives,
                "event_rate": float(y_segment.mean()),
                "min_support_required": int(min_support),
                "min_class_support_required": int(min_class_support),
                "use_segment_calibrator": bool(use_segment_calibrator),
            }
        )

    return {
        "family": family,
        "segment_strategy": segment_strategy,
        "global_calibrator": global_calibrator,
        "segment_calibrators": segment_calibrators,
        "support_frame": pd.DataFrame(support_rows),
    }


def apply_segmented_calibrator(bundle: dict[str, Any], proba: np.ndarray, segments: pd.Series) -> np.ndarray:
    proba_array = np.asarray(proba).astype(float)
    segment_series = pd.Series(segments).fillna("Unknown").astype(str).reset_index(drop=True)
    calibrated = np.empty(len(proba_array), dtype=float)

    for segment_value in segment_series.drop_duplicates().tolist():
        mask = (segment_series == segment_value).to_numpy()
        calibrator = bundle["segment_calibrators"].get(str(segment_value), bundle["global_calibrator"])
        calibrated[mask] = apply_calibrator(bundle["family"], calibrator, proba_array[mask])

    return np.clip(calibrated, 1e-6, 1 - 1e-6)


def fit_hierarchical_calibrator(
    family: str,
    proba: np.ndarray,
    y_true: pd.Series,
    primary_segments: pd.Series,
    secondary_segments: pd.Series,
    segment_strategy: str,
    primary_role: str,
    secondary_role: str,
    min_support: int = SEGMENT_MIN_SUPPORT,
    min_class_support: int = SEGMENT_MIN_CLASS_SUPPORT,
) -> dict[str, Any]:
    proba_array = np.asarray(proba).astype(float)
    y_series = pd.Series(np.asarray(y_true).astype(int)).reset_index(drop=True)
    primary_series = pd.Series(primary_segments).fillna("Unknown").astype(str).reset_index(drop=True)
    secondary_series = pd.Series(secondary_segments).fillna("Unknown").astype(str).reset_index(drop=True)

    global_calibrator = fit_calibrator(family, proba_array, y_series)
    primary_calibrators: dict[str, Any] = {}
    secondary_calibrators: dict[str, Any] = {}
    support_rows: list[dict[str, Any]] = []

    for role, segment_series, store in [
        (primary_role, primary_series, primary_calibrators),
        (secondary_role, secondary_series, secondary_calibrators),
    ]:
        for segment_value in segment_series.drop_duplicates().tolist():
            mask = (segment_series == segment_value).to_numpy()
            y_segment = y_series.loc[mask].reset_index(drop=True)
            positives = int(y_segment.sum())
            negatives = int(len(y_segment) - positives)
            use_segment_calibrator = (
                len(y_segment) >= min_support and positives >= min_class_support and negatives >= min_class_support
            )
            if use_segment_calibrator:
                store[str(segment_value)] = fit_calibrator(family, proba_array[mask], y_segment)
            support_rows.append(
                {
                    "segment_strategy": segment_strategy,
                    "segment_role": role,
                    "segment_value": str(segment_value),
                    "support": int(len(y_segment)),
                    "positive_support": positives,
                    "negative_support": negatives,
                    "event_rate": float(y_segment.mean()),
                    "min_support_required": int(min_support),
                    "min_class_support_required": int(min_class_support),
                    "use_segment_calibrator": bool(use_segment_calibrator),
                }
            )

    return {
        "family": family,
        "segment_strategy": segment_strategy,
        "global_calibrator": global_calibrator,
        "primary_role": primary_role,
        "secondary_role": secondary_role,
        "primary_calibrators": primary_calibrators,
        "secondary_calibrators": secondary_calibrators,
        "support_frame": pd.DataFrame(support_rows),
    }


def apply_hierarchical_calibrator(
    bundle: dict[str, Any],
    proba: np.ndarray,
    primary_segments: pd.Series,
    secondary_segments: pd.Series,
) -> np.ndarray:
    proba_array = np.asarray(proba).astype(float)
    primary_series = pd.Series(primary_segments).fillna("Unknown").astype(str).reset_index(drop=True)
    secondary_series = pd.Series(secondary_segments).fillna("Unknown").astype(str).reset_index(drop=True)
    calibrated = np.empty(len(proba_array), dtype=float)

    for index in range(len(proba_array)):
        primary_key = str(primary_series.iloc[index])
        secondary_key = str(secondary_series.iloc[index])
        calibrator = bundle["primary_calibrators"].get(primary_key)
        if calibrator is None:
            calibrator = bundle["secondary_calibrators"].get(secondary_key, bundle["global_calibrator"])
        calibrated[index] = apply_calibrator(bundle["family"], calibrator, np.asarray([proba_array[index]], dtype=float))[0]

    return np.clip(calibrated, 1e-6, 1 - 1e-6)


def segmented_result_fields(bundle: dict[str, Any]) -> dict[str, Any]:
    support_frame = bundle["support_frame"]
    return {
        "segment_value_count": int(len(support_frame)),
        "segment_calibrator_count": int(support_frame["use_segment_calibrator"].sum()),
        "segment_fallback_count": int((~support_frame["use_segment_calibrator"]).sum()),
        "segment_min_support": int(SEGMENT_MIN_SUPPORT),
        "segment_min_class_support": int(SEGMENT_MIN_CLASS_SUPPORT),
    }


def hierarchical_result_fields(bundle: dict[str, Any]) -> dict[str, Any]:
    support_frame = bundle["support_frame"]
    primary_mask = support_frame["segment_role"] == bundle["primary_role"]
    secondary_mask = support_frame["segment_role"] == bundle["secondary_role"]
    return {
        "primary_segment_role": str(bundle["primary_role"]),
        "secondary_segment_role": str(bundle["secondary_role"]),
        "primary_segment_value_count": int(primary_mask.sum()),
        "primary_segment_calibrator_count": int(support_frame.loc[primary_mask, "use_segment_calibrator"].sum()),
        "secondary_segment_value_count": int(secondary_mask.sum()),
        "secondary_segment_calibrator_count": int(support_frame.loc[secondary_mask, "use_segment_calibrator"].sum()),
        "segment_min_support": int(SEGMENT_MIN_SUPPORT),
        "segment_min_class_support": int(SEGMENT_MIN_CLASS_SUPPORT),
    }


def build_threshold_frame(y_true: pd.Series, proba: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        metrics = score_predictions(y_true, proba, threshold=threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "f1": float(metrics["f1"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    best_threshold = float(frame.loc[frame["f1"].idxmax(), "threshold"])
    return best_threshold, frame


def build_reliability_frame(
    y_true: pd.Series,
    proba: np.ndarray,
    bins: int = RELIABILITY_BINS,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true).astype(int),
            "proba": np.asarray(proba).astype(float),
        }
    )
    quantiles = min(bins, max(frame["proba"].nunique(), 1))
    if quantiles == 1:
        frame["bin"] = "all"
    else:
        frame["bin"] = pd.qcut(frame["proba"], q=quantiles, duplicates="drop")

    grouped = (
        frame.groupby("bin", dropna=False, observed=False)
        .agg(
            mean_predicted_probability=("proba", "mean"),
            observed_frequency=("y_true", "mean"),
            support=("y_true", "size"),
        )
        .reset_index()
    )
    grouped["bin"] = grouped["bin"].astype(str)
    return grouped


def summarize_reliability_curve(curve_df: pd.DataFrame) -> tuple[float, float]:
    gaps = np.abs(curve_df["mean_predicted_probability"] - curve_df["observed_frequency"])
    supports = curve_df["support"].astype(float)
    weighted_gap = float(np.average(gaps, weights=supports))
    max_gap = float(gaps.max())
    return weighted_gap, max_gap


def plot_reliability_curve(
    curve_df: pd.DataFrame,
    title: str,
    label: str,
    output_path: Path,
    line_color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(
        curve_df["mean_predicted_probability"],
        curve_df["observed_frequency"],
        marker="o",
        linewidth=2.0,
        color=line_color,
        label=label,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="#7a7a7a", label="Perfect calibration")
    ax.set_title(title)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed rain frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_comparison(
    curve_specs: list[tuple[pd.DataFrame, str, str]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for curve_df, label, color in curve_specs:
        ax.plot(
            curve_df["mean_predicted_probability"],
            curve_df["observed_frequency"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="#7a7a7a", label="Perfect calibration")
    ax.set_title("Winner Model Calibration Comparison")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed rain frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_time_series_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    params: dict[str, float | int],
    n_splits: int = DEFAULT_OOF_SPLITS,
) -> tuple[np.ndarray, pd.Series, pd.DataFrame, pd.DataFrame, int]:
    ordered_dates = pd.to_datetime(dates).reset_index(drop=True)
    unique_dates = pd.Index(pd.Series(ordered_dates).drop_duplicates().sort_values())
    split_count = min(n_splits, max(len(unique_dates) - 1, 1))
    if split_count < 2:
        raise ValueError("At least three unique dates are required for time-series out-of-fold calibration.")

    splitter = TimeSeriesSplit(n_splits=split_count)
    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []

    for fold, (train_date_idx, valid_date_idx) in enumerate(splitter.split(unique_dates), start=1):
        train_dates = unique_dates[train_date_idx]
        valid_dates = unique_dates[valid_date_idx]
        train_mask = ordered_dates.isin(train_dates).to_numpy()
        valid_mask = ordered_dates.isin(valid_dates).to_numpy()
        train_idx = np.flatnonzero(train_mask)
        valid_idx = np.flatnonzero(valid_mask)

        X_train_fold = X.iloc[train_idx].copy()
        X_valid_fold = X.iloc[valid_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()
        y_valid_fold = y.iloc[valid_idx].copy()

        fold_model = fit_model_by_name("CatBoost", X_train_fold, y_train_fold, params=params)
        fold_proba = predict_proba_for_model("CatBoost", fold_model, X_valid_fold)
        oof_rows.append(
            pd.DataFrame(
                {
                    "row_index": valid_idx,
                    "proba": fold_proba,
                    "y_true": y_valid_fold.to_numpy(),
                }
            )
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_start": str(pd.Timestamp(train_dates.min()).date()),
                "train_end": str(pd.Timestamp(train_dates.max()).date()),
                "valid_start": str(pd.Timestamp(valid_dates.min()).date()),
                "valid_end": str(pd.Timestamp(valid_dates.max()).date()),
                "train_support": int(len(train_idx)),
                "valid_support": int(len(valid_idx)),
                "train_event_rate": float(y_train_fold.mean()),
                "valid_event_rate": float(y_valid_fold.mean()),
            }
        )

    oof_frame = pd.concat(oof_rows, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    fold_frame = pd.DataFrame(fold_rows)
    return (
        oof_frame["proba"].to_numpy(),
        pd.Series(oof_frame["y_true"].astype(int)),
        oof_frame,
        fold_frame,
        split_count,
    )


def evaluate_method(
    method_key: str,
    threshold_y: pd.Series,
    threshold_proba: np.ndarray,
    test_y: pd.Series,
    test_proba: np.ndarray,
    extra_fields: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    config = METHOD_CONFIG[method_key]
    threshold_value, threshold_frame = build_threshold_frame(threshold_y, threshold_proba)
    threshold_frame.to_csv(config["threshold_path"], index=False)

    selection_curve_df = build_reliability_frame(threshold_y, threshold_proba)
    selection_reliability_mae, selection_reliability_max_gap = summarize_reliability_curve(selection_curve_df)
    curve_df = build_reliability_frame(test_y, test_proba)
    curve_df.to_csv(config["curve_path"], index=False)
    reliability_mae, reliability_max_gap = summarize_reliability_curve(curve_df)
    metrics = score_predictions(test_y, test_proba, threshold=threshold_value)

    result = {
        "method_key": method_key,
        "method_label": config["label"],
        "training_scheme": config["scheme"],
        "calibration_family": config["family"],
        "segment_strategy": config.get("segment_strategy", "global"),
        "threshold_selection_support": int(len(threshold_y)),
        "selection_event_rate": float(threshold_y.mean()),
        "selection_brier": float(brier_score_loss(threshold_y, threshold_proba)),
        "selection_log_loss": float(log_loss(threshold_y, threshold_proba)),
        "selection_reliability_mae": selection_reliability_mae,
        "selection_reliability_max_gap": selection_reliability_max_gap,
        "validation_threshold": float(threshold_value),
        "test_support": int(len(test_y)),
        "test_event_rate": float(test_y.mean()),
        "test_roc_auc": float(metrics["roc_auc"]),
        "test_f1": float(metrics["f1"]),
        "test_precision": float(metrics["precision"]),
        "test_recall": float(metrics["recall"]),
        "test_brier": float(brier_score_loss(test_y, test_proba)),
        "test_log_loss": float(log_loss(test_y, test_proba)),
        "test_reliability_mae": reliability_mae,
        "test_reliability_max_gap": reliability_max_gap,
        "curve_path": str(config["curve_path"]),
        "threshold_path": str(config["threshold_path"]),
    }
    if extra_fields:
        result.update(extra_fields)
    return result, curve_df


def choose_best_method(
    method_results: dict[str, dict[str, Any]],
    scheme: str,
    segment_strategy: str | None = None,
) -> str:
    eligible = [
        key
        for key, result in method_results.items()
        if result["training_scheme"] == scheme and result["calibration_family"] != "raw"
    ]
    if segment_strategy is not None:
        eligible = [key for key in eligible if method_results[key].get("segment_strategy", "global") == segment_strategy]
    if not eligible:
        raise ValueError(f"No eligible calibration methods found for scheme={scheme}, segment_strategy={segment_strategy}.")
    return min(
        eligible,
        key=lambda key: (
            method_results[key]["selection_brier"],
            method_results[key]["selection_log_loss"],
            method_results[key]["selection_reliability_mae"],
        ),
    )


def compare_methods(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_method_key": candidate["method_key"],
        "candidate_method_label": candidate["method_label"],
        "baseline_method_key": baseline["method_key"],
        "baseline_method_label": baseline["method_label"],
        "selection_brier_delta": float(candidate["selection_brier"] - baseline["selection_brier"]),
        "selection_log_loss_delta": float(candidate["selection_log_loss"] - baseline["selection_log_loss"]),
        "selection_reliability_mae_delta": float(candidate["selection_reliability_mae"] - baseline["selection_reliability_mae"]),
        "test_brier_delta": float(candidate["test_brier"] - baseline["test_brier"]),
        "test_log_loss_delta": float(candidate["test_log_loss"] - baseline["test_log_loss"]),
        "test_reliability_mae_delta": float(candidate["test_reliability_mae"] - baseline["test_reliability_mae"]),
    }

def run_experiment() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("rain_prediction_comparisons")
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.start_run(run_name="winner_model_calibration")
    
    winner_variant, selected_feature_set_name, train_df, valid_df, test_df, feature_sets, features = prepare_final_winner_frames(
        keep_date=True
    )
    best_selection = load_best_hybrid_selection()
    params = best_selection["params"]
    
    mlflow.log_params(params)
    mlflow.log_param("feature_count", len(features))
    
    location_to_climate_regime = load_location_to_climate_regime()

    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df["rain_tomorrow"].astype(int)
    y_valid = valid_df["rain_tomorrow"].astype(int)
    y_test = test_df["rain_tomorrow"].astype(int)
    valid_segments = build_segment_labels(valid_df, location_to_climate_regime)
    test_segments = build_segment_labels(test_df, location_to_climate_regime)

    validation_model = fit_model_by_name("CatBoost", X_train, y_train, params=params)
    valid_proba_raw = predict_proba_for_model("CatBoost", validation_model, X_valid)
    test_proba_validation_raw = predict_proba_for_model("CatBoost", validation_model, X_test)

    validation_sigmoid = fit_sigmoid_calibrator(valid_proba_raw, y_valid)
    valid_proba_sigmoid = apply_sigmoid_calibrator(validation_sigmoid, valid_proba_raw)
    test_proba_validation_sigmoid = apply_sigmoid_calibrator(validation_sigmoid, test_proba_validation_raw)

    validation_isotonic = fit_isotonic_calibrator(valid_proba_raw, y_valid)
    valid_proba_isotonic = apply_isotonic_calibrator(validation_isotonic, valid_proba_raw)
    test_proba_validation_isotonic = apply_isotonic_calibrator(validation_isotonic, test_proba_validation_raw)

    validation_season_sigmoid = fit_segmented_calibrator(
        "sigmoid",
        valid_proba_raw,
        y_valid,
        valid_segments["season"],
        segment_strategy="season",
    )
    valid_proba_validation_season_sigmoid = apply_segmented_calibrator(
        validation_season_sigmoid,
        valid_proba_raw,
        valid_segments["season"],
    )
    test_proba_validation_season_sigmoid = apply_segmented_calibrator(
        validation_season_sigmoid,
        test_proba_validation_raw,
        test_segments["season"],
    )

    validation_season_isotonic = fit_segmented_calibrator(
        "isotonic",
        valid_proba_raw,
        y_valid,
        valid_segments["season"],
        segment_strategy="season",
    )
    valid_proba_validation_season_isotonic = apply_segmented_calibrator(
        validation_season_isotonic,
        valid_proba_raw,
        valid_segments["season"],
    )
    test_proba_validation_season_isotonic = apply_segmented_calibrator(
        validation_season_isotonic,
        test_proba_validation_raw,
        test_segments["season"],
    )

    validation_climate_sigmoid = fit_segmented_calibrator(
        "sigmoid",
        valid_proba_raw,
        y_valid,
        valid_segments["climate_regime"],
        segment_strategy="climate_regime",
    )
    valid_proba_validation_climate_sigmoid = apply_segmented_calibrator(
        validation_climate_sigmoid,
        valid_proba_raw,
        valid_segments["climate_regime"],
    )
    test_proba_validation_climate_sigmoid = apply_segmented_calibrator(
        validation_climate_sigmoid,
        test_proba_validation_raw,
        test_segments["climate_regime"],
    )

    validation_climate_isotonic = fit_segmented_calibrator(
        "isotonic",
        valid_proba_raw,
        y_valid,
        valid_segments["climate_regime"],
        segment_strategy="climate_regime",
    )
    valid_proba_validation_climate_isotonic = apply_segmented_calibrator(
        validation_climate_isotonic,
        valid_proba_raw,
        valid_segments["climate_regime"],
    )
    test_proba_validation_climate_isotonic = apply_segmented_calibrator(
        validation_climate_isotonic,
        test_proba_validation_raw,
        test_segments["climate_regime"],
    )

    validation_hierarchical_sigmoid = fit_hierarchical_calibrator(
        "sigmoid",
        valid_proba_raw,
        y_valid,
        valid_segments["climate_regime_season"],
        valid_segments["climate_regime"],
        segment_strategy="climate_regime_season_hierarchical",
        primary_role="climate_regime_season",
        secondary_role="climate_regime",
    )
    valid_proba_validation_hierarchical_sigmoid = apply_hierarchical_calibrator(
        validation_hierarchical_sigmoid,
        valid_proba_raw,
        valid_segments["climate_regime_season"],
        valid_segments["climate_regime"],
    )
    test_proba_validation_hierarchical_sigmoid = apply_hierarchical_calibrator(
        validation_hierarchical_sigmoid,
        test_proba_validation_raw,
        test_segments["climate_regime_season"],
        test_segments["climate_regime"],
    )

    validation_hierarchical_isotonic = fit_hierarchical_calibrator(
        "isotonic",
        valid_proba_raw,
        y_valid,
        valid_segments["climate_regime_season"],
        valid_segments["climate_regime"],
        segment_strategy="climate_regime_season_hierarchical",
        primary_role="climate_regime_season",
        secondary_role="climate_regime",
    )
    valid_proba_validation_hierarchical_isotonic = apply_hierarchical_calibrator(
        validation_hierarchical_isotonic,
        valid_proba_raw,
        valid_segments["climate_regime_season"],
        valid_segments["climate_regime"],
    )
    test_proba_validation_hierarchical_isotonic = apply_hierarchical_calibrator(
        validation_hierarchical_isotonic,
        test_proba_validation_raw,
        test_segments["climate_regime_season"],
        test_segments["climate_regime"],
    )

    combined = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)
    X_combined = combined[features].copy()
    y_combined = combined["rain_tomorrow"].astype(int)
    combined_dates = pd.to_datetime(combined["date"])
    combined_segments = build_segment_labels(combined, location_to_climate_regime)

    oof_raw_proba, y_oof, oof_frame, fold_frame, actual_oof_splits = build_time_series_oof_predictions(
        X_combined,
        y_combined,
        combined_dates,
        params=params,
        n_splits=DEFAULT_OOF_SPLITS,
    )
    fold_frame.to_csv(OOF_FOLDS_PATH, index=False)
    oof_segments = combined_segments.iloc[oof_frame["row_index"].to_numpy()].reset_index(drop=True)

    time_sigmoid = fit_sigmoid_calibrator(oof_raw_proba, y_oof)
    oof_sigmoid_proba = apply_sigmoid_calibrator(time_sigmoid, oof_raw_proba)

    time_isotonic = fit_isotonic_calibrator(oof_raw_proba, y_oof)
    oof_isotonic_proba = apply_isotonic_calibrator(time_isotonic, oof_raw_proba)

    time_season_sigmoid = fit_segmented_calibrator(
        "sigmoid",
        oof_raw_proba,
        y_oof,
        oof_segments["season"],
        segment_strategy="season",
    )
    oof_season_sigmoid_proba = apply_segmented_calibrator(time_season_sigmoid, oof_raw_proba, oof_segments["season"])

    time_season_isotonic = fit_segmented_calibrator(
        "isotonic",
        oof_raw_proba,
        y_oof,
        oof_segments["season"],
        segment_strategy="season",
    )
    oof_season_isotonic_proba = apply_segmented_calibrator(time_season_isotonic, oof_raw_proba, oof_segments["season"])

    time_climate_sigmoid = fit_segmented_calibrator(
        "sigmoid",
        oof_raw_proba,
        y_oof,
        oof_segments["climate_regime"],
        segment_strategy="climate_regime",
    )
    oof_climate_sigmoid_proba = apply_segmented_calibrator(
        time_climate_sigmoid,
        oof_raw_proba,
        oof_segments["climate_regime"],
    )

    time_climate_isotonic = fit_segmented_calibrator(
        "isotonic",
        oof_raw_proba,
        y_oof,
        oof_segments["climate_regime"],
        segment_strategy="climate_regime",
    )
    oof_climate_isotonic_proba = apply_segmented_calibrator(
        time_climate_isotonic,
        oof_raw_proba,
        oof_segments["climate_regime"],
    )

    time_hierarchical_sigmoid = fit_hierarchical_calibrator(
        "sigmoid",
        oof_raw_proba,
        y_oof,
        oof_segments["climate_regime_season"],
        oof_segments["climate_regime"],
        segment_strategy="climate_regime_season_hierarchical",
        primary_role="climate_regime_season",
        secondary_role="climate_regime",
    )
    oof_hierarchical_sigmoid_proba = apply_hierarchical_calibrator(
        time_hierarchical_sigmoid,
        oof_raw_proba,
        oof_segments["climate_regime_season"],
        oof_segments["climate_regime"],
    )

    time_hierarchical_isotonic = fit_hierarchical_calibrator(
        "isotonic",
        oof_raw_proba,
        y_oof,
        oof_segments["climate_regime_season"],
        oof_segments["climate_regime"],
        segment_strategy="climate_regime_season_hierarchical",
        primary_role="climate_regime_season",
        secondary_role="climate_regime",
    )
    oof_hierarchical_isotonic_proba = apply_hierarchical_calibrator(
        time_hierarchical_isotonic,
        oof_raw_proba,
        oof_segments["climate_regime_season"],
        oof_segments["climate_regime"],
    )

    final_model = fit_model_by_name("CatBoost", X_combined, y_combined, params=params)
    test_proba_raw = predict_proba_for_model("CatBoost", final_model, X_test)
    test_proba_sigmoid = apply_sigmoid_calibrator(time_sigmoid, test_proba_raw)
    test_proba_isotonic = apply_isotonic_calibrator(time_isotonic, test_proba_raw)
    test_proba_season_sigmoid = apply_segmented_calibrator(time_season_sigmoid, test_proba_raw, test_segments["season"])
    test_proba_season_isotonic = apply_segmented_calibrator(time_season_isotonic, test_proba_raw, test_segments["season"])
    test_proba_climate_sigmoid = apply_segmented_calibrator(
        time_climate_sigmoid,
        test_proba_raw,
        test_segments["climate_regime"],
    )
    test_proba_climate_isotonic = apply_segmented_calibrator(
        time_climate_isotonic,
        test_proba_raw,
        test_segments["climate_regime"],
    )
    test_proba_hierarchical_sigmoid = apply_hierarchical_calibrator(
        time_hierarchical_sigmoid,
        test_proba_raw,
        test_segments["climate_regime_season"],
        test_segments["climate_regime"],
    )
    test_proba_hierarchical_isotonic = apply_hierarchical_calibrator(
        time_hierarchical_isotonic,
        test_proba_raw,
        test_segments["climate_regime_season"],
        test_segments["climate_regime"],
    )

    method_results: dict[str, dict[str, Any]] = {}
    curve_frames: dict[str, pd.DataFrame] = {}
    segmented_support_frames: list[pd.DataFrame] = []

    segmented_bundles = {
        "validation_block_season_sigmoid": validation_season_sigmoid,
        "validation_block_season_isotonic": validation_season_isotonic,
        "validation_block_climate_regime_sigmoid": validation_climate_sigmoid,
        "validation_block_climate_regime_isotonic": validation_climate_isotonic,
        "validation_block_climate_regime_season_sigmoid": validation_hierarchical_sigmoid,
        "validation_block_climate_regime_season_isotonic": validation_hierarchical_isotonic,
        "time_series_season_sigmoid": time_season_sigmoid,
        "time_series_season_isotonic": time_season_isotonic,
        "time_series_climate_regime_sigmoid": time_climate_sigmoid,
        "time_series_climate_regime_isotonic": time_climate_isotonic,
        "time_series_climate_regime_season_sigmoid": time_hierarchical_sigmoid,
        "time_series_climate_regime_season_isotonic": time_hierarchical_isotonic,
    }

    for method_key, bundle in segmented_bundles.items():
        support_frame = bundle["support_frame"].copy()
        support_frame.insert(0, "method_key", method_key)
        support_frame.insert(1, "method_label", METHOD_CONFIG[method_key]["label"])
        support_frame.insert(2, "training_scheme", METHOD_CONFIG[method_key]["scheme"])
        support_frame.insert(3, "calibration_family", METHOD_CONFIG[method_key]["family"])
        segmented_support_frames.append(support_frame)

    if segmented_support_frames:
        pd.concat(segmented_support_frames, ignore_index=True).to_csv(SEGMENT_SUPPORT_PATH, index=False)

    evaluation_specs = [
        ("validation_block_raw", y_valid, valid_proba_raw, y_test, test_proba_validation_raw, None),
        ("validation_block_sigmoid", y_valid, valid_proba_sigmoid, y_test, test_proba_validation_sigmoid, None),
        ("validation_block_isotonic", y_valid, valid_proba_isotonic, y_test, test_proba_validation_isotonic, None),
        (
            "validation_block_season_sigmoid",
            y_valid,
            valid_proba_validation_season_sigmoid,
            y_test,
            test_proba_validation_season_sigmoid,
            segmented_result_fields(validation_season_sigmoid),
        ),
        (
            "validation_block_season_isotonic",
            y_valid,
            valid_proba_validation_season_isotonic,
            y_test,
            test_proba_validation_season_isotonic,
            segmented_result_fields(validation_season_isotonic),
        ),
        (
            "validation_block_climate_regime_sigmoid",
            y_valid,
            valid_proba_validation_climate_sigmoid,
            y_test,
            test_proba_validation_climate_sigmoid,
            segmented_result_fields(validation_climate_sigmoid),
        ),
        (
            "validation_block_climate_regime_isotonic",
            y_valid,
            valid_proba_validation_climate_isotonic,
            y_test,
            test_proba_validation_climate_isotonic,
            segmented_result_fields(validation_climate_isotonic),
        ),
        (
            "validation_block_climate_regime_season_sigmoid",
            y_valid,
            valid_proba_validation_hierarchical_sigmoid,
            y_test,
            test_proba_validation_hierarchical_sigmoid,
            hierarchical_result_fields(validation_hierarchical_sigmoid),
        ),
        (
            "validation_block_climate_regime_season_isotonic",
            y_valid,
            valid_proba_validation_hierarchical_isotonic,
            y_test,
            test_proba_validation_hierarchical_isotonic,
            hierarchical_result_fields(validation_hierarchical_isotonic),
        ),
        ("time_series_raw", y_oof, oof_raw_proba, y_test, test_proba_raw, None),
        ("time_series_sigmoid", y_oof, oof_sigmoid_proba, y_test, test_proba_sigmoid, None),
        ("time_series_isotonic", y_oof, oof_isotonic_proba, y_test, test_proba_isotonic, None),
        (
            "time_series_season_sigmoid",
            y_oof,
            oof_season_sigmoid_proba,
            y_test,
            test_proba_season_sigmoid,
            segmented_result_fields(time_season_sigmoid),
        ),
        (
            "time_series_season_isotonic",
            y_oof,
            oof_season_isotonic_proba,
            y_test,
            test_proba_season_isotonic,
            segmented_result_fields(time_season_isotonic),
        ),
        (
            "time_series_climate_regime_sigmoid",
            y_oof,
            oof_climate_sigmoid_proba,
            y_test,
            test_proba_climate_sigmoid,
            segmented_result_fields(time_climate_sigmoid),
        ),
        (
            "time_series_climate_regime_isotonic",
            y_oof,
            oof_climate_isotonic_proba,
            y_test,
            test_proba_climate_isotonic,
            segmented_result_fields(time_climate_isotonic),
        ),
        (
            "time_series_climate_regime_season_sigmoid",
            y_oof,
            oof_hierarchical_sigmoid_proba,
            y_test,
            test_proba_hierarchical_sigmoid,
            hierarchical_result_fields(time_hierarchical_sigmoid),
        ),
        (
            "time_series_climate_regime_season_isotonic",
            y_oof,
            oof_hierarchical_isotonic_proba,
            y_test,
            test_proba_hierarchical_isotonic,
            hierarchical_result_fields(time_hierarchical_isotonic),
        ),
    ]

    for method_key, threshold_y, threshold_proba, test_y, test_proba, extra_fields in evaluation_specs:
        result, curve_df = evaluate_method(method_key, threshold_y, threshold_proba, test_y, test_proba, extra_fields=extra_fields)
        method_results[method_key] = result
        curve_frames[method_key] = curve_df

    best_validation_key = choose_best_method(method_results, scheme="validation_block")
    best_time_series_key = choose_best_method(method_results, scheme="time_series_oof")
    best_global_time_series_key = choose_best_method(method_results, scheme="time_series_oof", segment_strategy="global")
    best_time_series_season_key = choose_best_method(method_results, scheme="time_series_oof", segment_strategy="season")
    best_time_series_climate_key = choose_best_method(
        method_results,
        scheme="time_series_oof",
        segment_strategy="climate_regime",
    )
    best_time_series_hierarchical_key = choose_best_method(
        method_results,
        scheme="time_series_oof",
        segment_strategy="climate_regime_season_hierarchical",
    )
    best_overall_key = min(
        [key for key in method_results if method_results[key]["calibration_family"] != "raw"],
        key=lambda key: (
            method_results[key]["selection_brier"],
            method_results[key]["selection_log_loss"],
            method_results[key]["selection_reliability_mae"],
        ),
    )

    plot_reliability_curve(
        curve_frames["time_series_raw"],
        title="Winner Model Reliability Check",
        label="Winner model (raw refit)",
        output_path=RAW_FIGURE_PATH,
        line_color=METHOD_CONFIG["time_series_raw"]["color"],
    )
    plot_reliability_curve(
        curve_frames["time_series_sigmoid"],
        title="Winner Model Reliability After Time-Aware Sigmoid Calibration",
        label="Winner model (time-aware sigmoid)",
        output_path=SIGMOID_FIGURE_PATH,
        line_color=METHOD_CONFIG["time_series_sigmoid"]["color"],
    )
    plot_reliability_curve(
        curve_frames["time_series_isotonic"],
        title="Winner Model Reliability After Time-Aware Isotonic Calibration",
        label="Winner model (time-aware isotonic)",
        output_path=ISOTONIC_FIGURE_PATH,
        line_color=METHOD_CONFIG["time_series_isotonic"]["color"],
    )
    plot_reliability_curve(
        curve_frames[best_time_series_season_key],
        title=f"Winner Model Reliability After {METHOD_CONFIG[best_time_series_season_key]['label']}",
        label=METHOD_CONFIG[best_time_series_season_key]["label"],
        output_path=SEASON_SEGMENTED_FIGURE_PATH,
        line_color=METHOD_CONFIG[best_time_series_season_key]["color"],
    )
    plot_reliability_curve(
        curve_frames[best_time_series_climate_key],
        title=f"Winner Model Reliability After {METHOD_CONFIG[best_time_series_climate_key]['label']}",
        label=METHOD_CONFIG[best_time_series_climate_key]["label"],
        output_path=CLIMATE_SEGMENTED_FIGURE_PATH,
        line_color=METHOD_CONFIG[best_time_series_climate_key]["color"],
    )
    plot_reliability_curve(
        curve_frames[best_time_series_hierarchical_key],
        title=f"Winner Model Reliability After {METHOD_CONFIG[best_time_series_hierarchical_key]['label']}",
        label=METHOD_CONFIG[best_time_series_hierarchical_key]["label"],
        output_path=HIERARCHICAL_SEGMENTED_FIGURE_PATH,
        line_color=METHOD_CONFIG[best_time_series_hierarchical_key]["color"],
    )
    plot_reliability_comparison(
        [
            (
                curve_frames["time_series_raw"],
                METHOD_CONFIG["time_series_raw"]["label"],
                METHOD_CONFIG["time_series_raw"]["color"],
            ),
            (
                curve_frames["time_series_sigmoid"],
                METHOD_CONFIG["time_series_sigmoid"]["label"],
                METHOD_CONFIG["time_series_sigmoid"]["color"],
            ),
            (
                curve_frames["time_series_isotonic"],
                METHOD_CONFIG["time_series_isotonic"]["label"],
                METHOD_CONFIG["time_series_isotonic"]["color"],
            ),
        ],
        COMPARISON_FIGURE_PATH,
    )
    plot_reliability_comparison(
        [
            (
                curve_frames["time_series_raw"],
                METHOD_CONFIG["time_series_raw"]["label"],
                METHOD_CONFIG["time_series_raw"]["color"],
            ),
            (
                curve_frames[best_global_time_series_key],
                METHOD_CONFIG[best_global_time_series_key]["label"],
                METHOD_CONFIG[best_global_time_series_key]["color"],
            ),
            (
                curve_frames[best_time_series_season_key],
                METHOD_CONFIG[best_time_series_season_key]["label"],
                METHOD_CONFIG[best_time_series_season_key]["color"],
            ),
            (
                curve_frames[best_time_series_climate_key],
                METHOD_CONFIG[best_time_series_climate_key]["label"],
                METHOD_CONFIG[best_time_series_climate_key]["color"],
            ),
            (
                curve_frames[best_time_series_hierarchical_key],
                METHOD_CONFIG[best_time_series_hierarchical_key]["label"],
                METHOD_CONFIG[best_time_series_hierarchical_key]["color"],
            ),
        ],
        SEGMENTED_COMPARISON_FIGURE_PATH,
    )

    methods_frame = pd.DataFrame(method_results.values()).sort_values(
        ["selection_brier", "selection_log_loss", "selection_reliability_mae", "test_brier"],
        ascending=[True, True, True, True],
    )
    methods_frame.to_csv(METHODS_PATH, index=False)

    for method_key, result in method_results.items():
        mlflow.log_metrics(
            {
                f"{method_key}__test_roc_auc": result["test_roc_auc"],
                f"{method_key}__test_f1": result["test_f1"],
                f"{method_key}__test_brier": result["test_brier"],
                f"{method_key}__test_log_loss": result["test_log_loss"],
                f"{method_key}__test_reliability_mae": result["test_reliability_mae"],
            }
        )
        
    hierarchical_promotion = (
        method_results[best_time_series_hierarchical_key]["test_brier"] <= method_results[best_time_series_climate_key]["test_brier"]
        and method_results[best_time_series_hierarchical_key]["test_log_loss"] <= method_results[best_time_series_climate_key]["test_log_loss"]
        and method_results[best_time_series_hierarchical_key]["test_reliability_mae"] <= method_results[best_time_series_climate_key]["test_reliability_mae"]
    )
    recommended_key = best_time_series_hierarchical_key if hierarchical_promotion else best_time_series_climate_key
    
    mlflow.set_tag("recommended_method_key", recommended_key)
    mlflow.set_tag("recommended_method_label", method_results[recommended_key]["method_label"])
    
    recommended_figure = {
        "time_series_sigmoid": SIGMOID_FIGURE_PATH,
        "time_series_isotonic": ISOTONIC_FIGURE_PATH,
        best_time_series_season_key: SEASON_SEGMENTED_FIGURE_PATH,
        best_time_series_climate_key: CLIMATE_SEGMENTED_FIGURE_PATH,
        best_time_series_hierarchical_key: HIERARCHICAL_SEGMENTED_FIGURE_PATH,
    }.get(recommended_key, SEGMENTED_COMPARISON_FIGURE_PATH)

    season_assessment = compare_methods(method_results[best_time_series_season_key], method_results[best_global_time_series_key])
    climate_assessment = compare_methods(
        method_results[best_time_series_climate_key],
        method_results[best_global_time_series_key],
    )
    hierarchical_assessment = compare_methods(
        method_results[best_time_series_hierarchical_key],
        method_results[best_global_time_series_key],
    )
    hierarchical_vs_climate_assessment = compare_methods(
        method_results[best_time_series_hierarchical_key],
        method_results[best_time_series_climate_key],
    )

    summary = {
        "winner_variant": winner_variant,
        "feature_set_name": selected_feature_set_name,
        "feature_count": int(len(features)),
        "validation_support": int(len(y_valid)),
        "oof_calibration_support": int(len(y_oof)),
        "test_support": int(len(y_test)),
        "time_series_oof_splits": int(actual_oof_splits),
        "selection_basis": "non_test_calibration_support",
        "event_rates": {
            "train": float(y_train.mean()),
            "validation": float(y_valid.mean()),
            "train_valid_combined": float(y_combined.mean()),
            "test": float(y_test.mean()),
        },
        "params": params,
        "best_validation_method_key": best_validation_key,
        "best_validation_method_label": method_results[best_validation_key]["method_label"],
        "best_time_series_method_key": recommended_key,
        "best_time_series_method_label": method_results[recommended_key]["method_label"],
        "best_time_series_selection_key": best_time_series_key,
        "best_time_series_selection_label": method_results[best_time_series_key]["method_label"],
        "best_global_time_series_method_key": best_global_time_series_key,
        "best_global_time_series_method_label": method_results[best_global_time_series_key]["method_label"],
        "best_time_series_season_method_key": best_time_series_season_key,
        "best_time_series_season_method_label": method_results[best_time_series_season_key]["method_label"],
        "best_time_series_climate_method_key": best_time_series_climate_key,
        "best_time_series_climate_method_label": method_results[best_time_series_climate_key]["method_label"],
        "best_time_series_hierarchical_method_key": best_time_series_hierarchical_key,
        "best_time_series_hierarchical_method_label": method_results[best_time_series_hierarchical_key]["method_label"],
        "best_overall_method_key": best_overall_key,
        "best_overall_method_label": method_results[best_overall_key]["method_label"],
        "uncalibrated": method_results["time_series_raw"],
        "sigmoid_calibrated": method_results["time_series_sigmoid"],
        "isotonic_calibrated": method_results["time_series_isotonic"],
        "season_segmented_calibrated": method_results[best_time_series_season_key],
        "climate_regime_segmented_calibrated": method_results[best_time_series_climate_key],
        "hierarchical_segmented_calibrated": method_results[best_time_series_hierarchical_key],
        "recommended_calibrated": method_results[recommended_key],
        "validation_sigmoid_calibrated": method_results["validation_block_sigmoid"],
        "validation_isotonic_calibrated": method_results["validation_block_isotonic"],
        "segmented_assessment": {
            "baseline_global_time_series": method_results[best_global_time_series_key],
            "season_vs_global": season_assessment,
            "climate_regime_vs_global": climate_assessment,
            "hierarchical_vs_global": hierarchical_assessment,
            "hierarchical_vs_climate_regime": hierarchical_vs_climate_assessment,
            "hierarchical_promotion_gate_passed": bool(hierarchical_promotion),
            "hierarchical_promotion_gate": "promote_only_if_holdout_brier_log_loss_and_reliability_mae_do_not_regress_vs_best_climate_regime_method",
        },
        "validation_block_methods": {
            "raw": method_results["validation_block_raw"],
            "sigmoid": method_results["validation_block_sigmoid"],
            "isotonic": method_results["validation_block_isotonic"],
        },
        "validation_block_segmented_methods": {
            "season_sigmoid": method_results["validation_block_season_sigmoid"],
            "season_isotonic": method_results["validation_block_season_isotonic"],
            "climate_regime_sigmoid": method_results["validation_block_climate_regime_sigmoid"],
            "climate_regime_isotonic": method_results["validation_block_climate_regime_isotonic"],
            "climate_regime_season_sigmoid": method_results["validation_block_climate_regime_season_sigmoid"],
            "climate_regime_season_isotonic": method_results["validation_block_climate_regime_season_isotonic"],
        },
        "time_series_oof_methods": {
            "raw": method_results["time_series_raw"],
            "sigmoid": method_results["time_series_sigmoid"],
            "isotonic": method_results["time_series_isotonic"],
        },
        "time_series_oof_segmented_methods": {
            "season_sigmoid": method_results["time_series_season_sigmoid"],
            "season_isotonic": method_results["time_series_season_isotonic"],
            "climate_regime_sigmoid": method_results["time_series_climate_regime_sigmoid"],
            "climate_regime_isotonic": method_results["time_series_climate_regime_isotonic"],
            "climate_regime_season_sigmoid": method_results["time_series_climate_regime_season_sigmoid"],
            "climate_regime_season_isotonic": method_results["time_series_climate_regime_season_isotonic"],
        },
        "methods": method_results,
        "leaderboard_path": str(METHODS_PATH),
        "oof_folds_path": str(OOF_FOLDS_PATH),
        "segmented_support_path": str(SEGMENT_SUPPORT_PATH),
        "curve_paths": {
            "raw_curve_csv": str(RAW_CURVE_PATH),
            "sigmoid_curve_csv": str(SIGMOID_CURVE_PATH),
            "isotonic_curve_csv": str(ISOTONIC_CURVE_PATH),
            "validation_raw_curve_csv": str(VALIDATION_RAW_CURVE_PATH),
            "validation_sigmoid_curve_csv": str(VALIDATION_SIGMOID_CURVE_PATH),
            "validation_isotonic_curve_csv": str(VALIDATION_ISOTONIC_CURVE_PATH),
            "validation_season_sigmoid_curve_csv": str(VALIDATION_SEASON_SIGMOID_CURVE_PATH),
            "validation_season_isotonic_curve_csv": str(VALIDATION_SEASON_ISOTONIC_CURVE_PATH),
            "validation_climate_regime_sigmoid_curve_csv": str(VALIDATION_CLIMATE_SIGMOID_CURVE_PATH),
            "validation_climate_regime_isotonic_curve_csv": str(VALIDATION_CLIMATE_ISOTONIC_CURVE_PATH),
            "validation_climate_regime_season_sigmoid_curve_csv": str(VALIDATION_HIERARCHICAL_SIGMOID_CURVE_PATH),
            "validation_climate_regime_season_isotonic_curve_csv": str(VALIDATION_HIERARCHICAL_ISOTONIC_CURVE_PATH),
            "time_series_season_sigmoid_curve_csv": str(TIME_SERIES_SEASON_SIGMOID_CURVE_PATH),
            "time_series_season_isotonic_curve_csv": str(TIME_SERIES_SEASON_ISOTONIC_CURVE_PATH),
            "time_series_climate_regime_sigmoid_curve_csv": str(TIME_SERIES_CLIMATE_SIGMOID_CURVE_PATH),
            "time_series_climate_regime_isotonic_curve_csv": str(TIME_SERIES_CLIMATE_ISOTONIC_CURVE_PATH),
            "time_series_climate_regime_season_sigmoid_curve_csv": str(TIME_SERIES_HIERARCHICAL_SIGMOID_CURVE_PATH),
            "time_series_climate_regime_season_isotonic_curve_csv": str(TIME_SERIES_HIERARCHICAL_ISOTONIC_CURVE_PATH),
            "raw_figure": str(RAW_FIGURE_PATH),
            "sigmoid_figure": str(SIGMOID_FIGURE_PATH),
            "isotonic_figure": str(ISOTONIC_FIGURE_PATH),
            "season_segmented_figure": str(SEASON_SEGMENTED_FIGURE_PATH),
            "climate_regime_segmented_figure": str(CLIMATE_SEGMENTED_FIGURE_PATH),
            "hierarchical_segmented_figure": str(HIERARCHICAL_SEGMENTED_FIGURE_PATH),
            "recommended_figure": str(recommended_figure),
            "comparison_figure": str(SEGMENTED_COMPARISON_FIGURE_PATH),
            "global_comparison_figure": str(COMPARISON_FIGURE_PATH),
        },
        "threshold_paths": {
            "raw_threshold_curve_csv": str(RAW_THRESHOLD_PATH),
            "sigmoid_threshold_curve_csv": str(SIGMOID_THRESHOLD_PATH),
            "isotonic_threshold_curve_csv": str(ISOTONIC_THRESHOLD_PATH),
            "validation_raw_threshold_curve_csv": str(VALIDATION_RAW_THRESHOLD_PATH),
            "validation_sigmoid_threshold_curve_csv": str(VALIDATION_SIGMOID_THRESHOLD_PATH),
            "validation_isotonic_threshold_curve_csv": str(VALIDATION_ISOTONIC_THRESHOLD_PATH),
            "validation_season_sigmoid_threshold_curve_csv": str(VALIDATION_SEASON_SIGMOID_THRESHOLD_PATH),
            "validation_season_isotonic_threshold_curve_csv": str(VALIDATION_SEASON_ISOTONIC_THRESHOLD_PATH),
            "validation_climate_regime_sigmoid_threshold_curve_csv": str(VALIDATION_CLIMATE_SIGMOID_THRESHOLD_PATH),
            "validation_climate_regime_isotonic_threshold_curve_csv": str(VALIDATION_CLIMATE_ISOTONIC_THRESHOLD_PATH),
            "validation_climate_regime_season_sigmoid_threshold_curve_csv": str(VALIDATION_HIERARCHICAL_SIGMOID_THRESHOLD_PATH),
            "validation_climate_regime_season_isotonic_threshold_curve_csv": str(VALIDATION_HIERARCHICAL_ISOTONIC_THRESHOLD_PATH),
            "time_series_season_sigmoid_threshold_curve_csv": str(TIME_SERIES_SEASON_SIGMOID_THRESHOLD_PATH),
            "time_series_season_isotonic_threshold_curve_csv": str(TIME_SERIES_SEASON_ISOTONIC_THRESHOLD_PATH),
            "time_series_climate_regime_sigmoid_threshold_curve_csv": str(TIME_SERIES_CLIMATE_SIGMOID_THRESHOLD_PATH),
            "time_series_climate_regime_isotonic_threshold_curve_csv": str(TIME_SERIES_CLIMATE_ISOTONIC_THRESHOLD_PATH),
            "time_series_climate_regime_season_sigmoid_threshold_curve_csv": str(TIME_SERIES_HIERARCHICAL_SIGMOID_THRESHOLD_PATH),
            "time_series_climate_regime_season_isotonic_threshold_curve_csv": str(TIME_SERIES_HIERARCHICAL_ISOTONIC_THRESHOLD_PATH),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    mlflow.log_artifact(str(SUMMARY_PATH))
    mlflow.log_artifact(str(METHODS_PATH))
    mlflow.end_run()
    
    return summary


def main() -> None:
    summary = run_experiment()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

