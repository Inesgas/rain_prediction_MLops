from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_object_dtype, is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.models.experiments.daily_zonal_baseline.experiment import THRESHOLDS
from src.models.experiments.hybrid_imputation_breakthrough.experiment import tune_threshold_from_validation
from src.models.ines_feature_modeling import TARGET, score_predictions
from src.models.ines_modeling_core import derive_accuracy_from_summary_metrics
from src.utils.validation import BEST_FEATURE_SET_NAME, prepare_standard_split_frames


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "models" / "dense_neural_network_benchmark"
RESULTS_PATH = RESULTS_DIR / "dense_nn_candidate_results.csv"
SUMMARY_PATH = RESULTS_DIR / "dense_nn_summary.json"
HISTORY_PATH = RESULTS_DIR / "dense_nn_training_history.csv"
THRESHOLD_CURVE_PATH = RESULTS_DIR / "dense_nn_threshold_curve.csv"
NOTES_PATH = RESULTS_DIR / "notes.md"
WINNER_SUMMARY_PATH = PROJECT_ROOT / "models" / "final_winner_package" / "robustness_summary.json"

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
LEARNING_CURVE_FIGURE = FIGURES_DIR / "fig_44_dense_nn_learning_curve.png"
THRESHOLD_FIGURE = FIGURES_DIR / "fig_45_dense_nn_threshold_curve.png"
COMPARISON_FIGURE = FIGURES_DIR / "fig_46_dense_nn_vs_winner_comparison.png"

MAX_EPOCHS = 60
PATIENCE = 10
MIN_IMPROVEMENT = 1e-4
RANDOM_STATE = 42

CANDIDATE_CONFIGS: list[dict[str, Any]] = [
    {
        "candidate": "dense_small_relu",
        "model_label": "Dense neural network",
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "alpha": 0.0001,
        "learning_rate_init": 0.0010,
        "batch_size": 512,
    },
    {
        "candidate": "dense_small_tanh",
        "model_label": "Dense neural network",
        "hidden_layer_sizes": (128, 64),
        "activation": "tanh",
        "alpha": 0.0005,
        "learning_rate_init": 0.0008,
        "batch_size": 512,
    },
    {
        "candidate": "dense_medium_relu",
        "model_label": "Dense neural network",
        "hidden_layer_sizes": (192, 96, 48),
        "activation": "relu",
        "alpha": 0.0005,
        "learning_rate_init": 0.0008,
        "batch_size": 512,
    },
    {
        "candidate": "dense_tanh_regularized",
        "model_label": "Dense neural network",
        "hidden_layer_sizes": (128, 64, 32),
        "activation": "tanh",
        "alpha": 0.0010,
        "learning_rate_init": 0.0008,
        "batch_size": 512,
    },
    {
        "candidate": "dense_deep_relu_regularized",
        "model_label": "Dense neural network",
        "hidden_layer_sizes": (256, 128, 64),
        "activation": "relu",
        "alpha": 0.0010,
        "learning_rate_init": 0.0006,
        "batch_size": 512,
    },
    {
        "candidate": "dense_wide_tanh",
        "model_label": "Dense neural network",
        "hidden_layer_sizes": (192, 128, 64),
        "activation": "tanh",
        "alpha": 0.0008,
        "learning_rate_init": 0.0007,
        "batch_size": 512,
    },
    {
        "candidate": "dense_balanced_relu",
        "model_label": "Dense neural network",
        "hidden_layer_sizes": (160, 96, 48),
        "activation": "relu",
        "alpha": 0.0008,
        "learning_rate_init": 0.0007,
        "batch_size": 256,
    },
]


def candidate_display_label(candidate: str) -> str:
    labels = {
        "dense_small_relu": "Dense neural net, 128-64 ReLU",
        "dense_small_tanh": "Dense neural net, 128-64 tanh",
        "dense_medium_relu": "Dense neural net, 192-96-48 ReLU",
        "dense_tanh_regularized": "Dense neural net, 128-64-32 tanh",
        "dense_deep_relu_regularized": "Dense neural net, 256-128-64 ReLU",
        "dense_wide_tanh": "Dense neural net, 192-128-64 tanh",
        "dense_balanced_relu": "Dense neural net, 160-96-48 ReLU",
    }
    return labels.get(candidate, candidate.replace("_", " ").title())


def write_notes() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    text = """# Dense Neural-Network Benchmark

## Goal

Test whether a dense neural network can outperform the final CatBoost winner when both models use the same
chronologically prepared winner feature space.

## Method

- Use the exact hybrid-plus-core winner features from the retained baseline branch.
- Keep the same chronological train / validation / test structure used elsewhere in the project.
- Select the neural-network architecture on the validation block only.
- Tune the classification threshold on validation probabilities only.
- Refit the selected architecture on train + validation for the same externally chosen number of epochs.

## Important Limitation

This benchmark uses scikit-learn's multilayer perceptron rather than TensorFlow or PyTorch because no dedicated
deep-learning runtime is installed in the current environment. The benchmark is still a real dense neural network,
but it remains a lightweight dense benchmark rather than a full custom deep-learning stack.
"""
    NOTES_PATH.write_text(text, encoding="utf-8")


def _is_categorical(series: pd.Series) -> bool:
    return (
        is_object_dtype(series)
        or is_string_dtype(series)
        or isinstance(series.dtype, CategoricalDtype)
    )


def build_dense_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [column for column in X.columns if _is_categorical(X[column])]
    numeric_cols = [column for column in X.columns if column not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        sparse_threshold=0.0,
    )


def make_mlp(config: dict[str, Any]) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=tuple(config["hidden_layer_sizes"]),
        activation=str(config["activation"]),
        alpha=float(config["alpha"]),
        learning_rate_init=float(config["learning_rate_init"]),
        batch_size=int(config["batch_size"]),
        solver="adam",
        shuffle=True,
        warm_start=True,
        max_iter=1,
        random_state=RANDOM_STATE,
    )


def threshold_frame(proba: np.ndarray, y_true: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        scores = score_predictions(y_true, proba, threshold=float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "roc_auc": float(scores["roc_auc"]),
                "f1": float(scores["f1"]),
                "precision": float(scores["precision"]),
                "recall": float(scores["recall"]),
            }
        )
    return pd.DataFrame(rows)


def accuracy_from_proba(y_true: pd.Series | np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> float:
    y_array = np.asarray(y_true, dtype=int)
    preds = (proba >= threshold).astype(int)
    return float((preds == y_array).mean())


def resolve_winner_metrics(winner_summary: dict[str, Any]) -> dict[str, float]:
    if "holdout_metrics" in winner_summary:
        holdout = winner_summary["holdout_metrics"]
        return {
            "test_accuracy": float(holdout["accuracy"]),
            "test_roc_auc": float(holdout["roc_auc"]),
            "test_f1": float(holdout["f1"]),
            "test_precision": float(holdout["precision"]),
            "test_recall": float(holdout["recall"]),
        }

    metrics = dict(winner_summary["uncalibrated"])
    metrics["test_accuracy"] = float(
        metrics.get(
            "test_accuracy",
            derive_accuracy_from_summary_metrics(
                support=metrics["test_support"],
                event_rate=metrics["test_event_rate"],
                precision=metrics["test_precision"],
                recall=metrics["test_recall"],
            ),
        )
    )
    return metrics


def fit_with_chronological_early_stopping(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_valid: np.ndarray,
    y_valid: pd.Series,
    config: dict[str, Any],
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> dict[str, Any]:
    model = make_mlp(config)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    history_rows: list[dict[str, float | int | str]] = []
    best_model: MLPClassifier | None = None
    best_epoch = 0
    best_valid_log_loss = float("inf")
    patience_counter = 0

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for epoch in range(1, max_epochs + 1):
        model.fit(X_train, y_train, sample_weight=sample_weight)
        train_proba = model.predict_proba(X_train)[:, 1]
        valid_proba = model.predict_proba(X_valid)[:, 1]
        train_loss = float(log_loss(y_train, train_proba, labels=[0, 1]))
        valid_loss = float(log_loss(y_valid, valid_proba, labels=[0, 1]))
        train_accuracy = accuracy_from_proba(y_train, train_proba)
        valid_accuracy = accuracy_from_proba(y_valid, valid_proba)
        valid_auc = float(roc_auc_score(y_valid, valid_proba))
        history_rows.append(
            {
                "candidate": str(config["candidate"]),
                "epoch": epoch,
                "train_log_loss": train_loss,
                "train_accuracy_at_0_5": train_accuracy,
                "validation_log_loss": valid_loss,
                "validation_accuracy_at_0_5": valid_accuracy,
                "validation_roc_auc": valid_auc,
            }
        )

        if valid_loss < best_valid_log_loss - MIN_IMPROVEMENT:
            best_valid_log_loss = valid_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model is None:
        best_model = copy.deepcopy(model)
        best_epoch = int(history_rows[-1]["epoch"])

    train_proba = best_model.predict_proba(X_train)[:, 1]
    valid_proba = best_model.predict_proba(X_valid)[:, 1]
    best_threshold, validation_threshold_metrics = tune_threshold_from_validation(valid_proba, y_valid)
    train_metrics = score_predictions(y_train, train_proba, threshold=best_threshold)
    validation_metrics = score_predictions(y_valid, valid_proba, threshold=best_threshold)

    return {
        "model": best_model,
        "best_epoch": best_epoch,
        "stop_epoch": int(history_rows[-1]["epoch"]),
        "best_validation_log_loss": best_valid_log_loss,
        "train_metrics": train_metrics,
        "validation_proba": valid_proba,
        "validation_threshold": float(best_threshold),
        "validation_threshold_metrics": validation_threshold_metrics,
        "validation_metrics": validation_metrics,
        "validation_brier": float(brier_score_loss(y_valid, valid_proba)),
        "validation_log_loss": float(log_loss(y_valid, valid_proba, labels=[0, 1])),
        "history": pd.DataFrame(history_rows),
    }


def refit_for_fixed_epochs(
    X_train: np.ndarray,
    y_train: pd.Series,
    config: dict[str, Any],
    epochs: int,
) -> MLPClassifier:
    model = make_mlp(config)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for _ in range(int(epochs)):
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def save_learning_curve(history: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["epoch"], history["train_log_loss"], label="Train log loss", linewidth=2)
    axes[0].plot(history["epoch"], history["validation_log_loss"], label="Validation log loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Log loss")
    axes[0].set_title("Dense network training stability")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(history["epoch"], history["validation_roc_auc"], color="#cc5500", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation ROC-AUC")
    axes[1].set_title("Validation discrimination by epoch")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(LEARNING_CURVE_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_threshold_curve(frame: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    selected_threshold = float(frame.loc[frame["f1"].idxmax(), "threshold"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(frame["threshold"], frame["f1"], label="F1", linewidth=2)
    ax.plot(frame["threshold"], frame["precision"], label="Precision", linewidth=2)
    ax.plot(frame["threshold"], frame["recall"], label="Recall", linewidth=2)
    ax.axvline(
        selected_threshold,
        color="#222222",
        linestyle="--",
        linewidth=1.8,
        label=f"Selected threshold = {selected_threshold:.2f}",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Dense network validation threshold sweep")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(THRESHOLD_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(dense_metrics: dict[str, float], winner_metrics: dict[str, float]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metric_keys = ["test_accuracy", "test_roc_auc", "test_f1", "test_precision", "test_recall"]
    metric_labels = ["Accuracy", "ROC-AUC", "F1", "Precision", "Recall"]
    dense_values = [float(dense_metrics[key]) for key in metric_keys]
    winner_values = [float(winner_metrics[key]) for key in metric_keys]
    positions = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.bar(positions - width / 2, dense_values, width=width, label="Dense neural net", color="#d97824")
    ax.bar(positions + width / 2, winner_values, width=width, label="Final CatBoost winner", color="#2f6db2")
    ax.set_xticks(positions, metric_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Dense neural network versus final CatBoost winner")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(COMPARISON_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment() -> dict[str, Any]:
    write_notes()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df, feature_sets = prepare_standard_split_frames(keep_date=True)
    features = list(feature_sets[BEST_FEATURE_SET_NAME])
    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df[TARGET].astype(int).copy()
    y_valid = valid_df[TARGET].astype(int).copy()
    y_test = test_df[TARGET].astype(int).copy()
    train_preprocessor = build_dense_preprocessor(X_train)
    X_train_ready = train_preprocessor.fit_transform(X_train).astype(np.float32, copy=False)
    X_valid_ready = train_preprocessor.transform(X_valid).astype(np.float32, copy=False)
    transformed_feature_count = int(train_preprocessor.get_feature_names_out().shape[0])

    train_valid_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
    X_train_valid = train_valid_df[features].copy()
    y_train_valid = train_valid_df[TARGET].astype(int).copy()
    final_preprocessor = build_dense_preprocessor(X_train_valid)
    X_train_valid_ready = final_preprocessor.fit_transform(X_train_valid).astype(np.float32, copy=False)
    X_test_ready = final_preprocessor.transform(X_test).astype(np.float32, copy=False)
    final_transformed_feature_count = int(final_preprocessor.get_feature_names_out().shape[0])

    results_rows: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    candidate_thresholds: dict[str, pd.DataFrame] = {}

    with WINNER_SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        winner_summary = json.load(handle)
    winner_metrics = resolve_winner_metrics(winner_summary)

    for config in CANDIDATE_CONFIGS:
        fit_result = fit_with_chronological_early_stopping(X_train_ready, y_train, X_valid_ready, y_valid, config)
        history = fit_result["history"].copy()
        histories.append(history)

        threshold_results = threshold_frame(fit_result["validation_proba"], y_valid)
        candidate_thresholds[str(config["candidate"])] = threshold_results

        final_model = refit_for_fixed_epochs(
            X_train_valid_ready,
            y_train_valid,
            config,
            epochs=int(fit_result["best_epoch"]),
        )
        test_proba = final_model.predict_proba(X_test_ready)[:, 1]
        threshold = float(fit_result["validation_threshold"])
        test_metrics = score_predictions(y_test, test_proba, threshold=threshold)

        results_rows.append(
            {
                "candidate": str(config["candidate"]),
                "candidate_label": candidate_display_label(str(config["candidate"])),
                "model": str(config["model_label"]),
                "feature_count": len(features),
                "transformed_feature_count": final_transformed_feature_count,
                "hidden_layer_sizes": "-".join(str(value) for value in config["hidden_layer_sizes"]),
                "activation": str(config["activation"]),
                "alpha": float(config["alpha"]),
                "learning_rate_init": float(config["learning_rate_init"]),
                "batch_size": int(config["batch_size"]),
                "best_epoch": int(fit_result["best_epoch"]),
                "stop_epoch": int(fit_result["stop_epoch"]),
                "best_validation_log_loss": float(fit_result["best_validation_log_loss"]),
                "validation_threshold": threshold,
                "train_accuracy": float(fit_result["train_metrics"]["accuracy"]),
                "validation_accuracy": float(fit_result["validation_metrics"]["accuracy"]),
                "validation_roc_auc": float(fit_result["validation_metrics"]["roc_auc"]),
                "validation_f1": float(fit_result["validation_metrics"]["f1"]),
                "validation_precision": float(fit_result["validation_metrics"]["precision"]),
                "validation_recall": float(fit_result["validation_metrics"]["recall"]),
                "validation_brier": float(fit_result["validation_brier"]),
                "validation_log_loss": float(fit_result["validation_log_loss"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_roc_auc": float(test_metrics["roc_auc"]),
                "test_f1": float(test_metrics["f1"]),
                "test_precision": float(test_metrics["precision"]),
                "test_recall": float(test_metrics["recall"]),
                "test_brier": float(brier_score_loss(y_test, test_proba)),
                "test_log_loss": float(log_loss(y_test, test_proba, labels=[0, 1])),
                "winner_test_roc_auc_gap": float(test_metrics["roc_auc"]) - float(winner_metrics["test_roc_auc"]),
                "winner_test_f1_gap": float(test_metrics["f1"]) - float(winner_metrics["test_f1"]),
                "winner_test_precision_gap": float(test_metrics["precision"]) - float(winner_metrics["test_precision"]),
                "winner_test_recall_gap": float(test_metrics["recall"]) - float(winner_metrics["test_recall"]),
            }
        )

    results = pd.DataFrame(results_rows).sort_values(
        ["validation_f1", "validation_roc_auc", "validation_precision", "validation_recall", "candidate"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    results.to_csv(RESULTS_PATH, index=False)

    history_frame = pd.concat(histories, axis=0, ignore_index=True)
    history_frame.to_csv(HISTORY_PATH, index=False)

    best_result = results.iloc[0].to_dict()
    best_candidate = str(best_result["candidate"])
    best_threshold_frame = candidate_thresholds[best_candidate]
    best_threshold_frame.to_csv(THRESHOLD_CURVE_PATH, index=False)

    best_history = history_frame.loc[history_frame["candidate"] == best_candidate].reset_index(drop=True)
    save_learning_curve(best_history)
    save_threshold_curve(best_threshold_frame)
    save_comparison_figure(best_result, winner_metrics)

    summary = {
        "benchmark_name": "dense_neural_network_benchmark",
        "feature_set_name": BEST_FEATURE_SET_NAME,
        "feature_count": len(features),
        "train_support": int(len(train_df)),
        "validation_support": int(len(valid_df)),
        "test_support": int(len(test_df)),
        "selection_basis": "validation_first_chronological_split",
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "best_result": best_result,
        "winner_comparison": {
            "winner_variant": str(winner_summary.get("winner_variant") or winner_summary.get("candidate", "baseline_locked_raw")),
            "winner_test_accuracy": float(winner_metrics["test_accuracy"]),
            "winner_test_roc_auc": float(winner_metrics["test_roc_auc"]),
            "winner_test_f1": float(winner_metrics["test_f1"]),
            "winner_test_precision": float(winner_metrics["test_precision"]),
            "winner_test_recall": float(winner_metrics["test_recall"]),
            "dense_minus_winner_roc_auc": float(best_result["winner_test_roc_auc_gap"]),
            "dense_minus_winner_f1": float(best_result["winner_test_f1_gap"]),
            "dense_minus_winner_precision": float(best_result["winner_test_precision_gap"]),
            "dense_minus_winner_recall": float(best_result["winner_test_recall_gap"]),
        },
        "figure_paths": {
            "learning_curve": str(LEARNING_CURVE_FIGURE),
            "threshold_curve": str(THRESHOLD_FIGURE),
            "winner_comparison": str(COMPARISON_FIGURE),
        },
        "results_path": str(RESULTS_PATH),
        "history_path": str(HISTORY_PATH),
        "threshold_path": str(THRESHOLD_CURVE_PATH),
        "notes_path": str(NOTES_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = run_experiment()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

