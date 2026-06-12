from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch import nn

from src.models.experiments.hybrid_imputation_breakthrough.experiment import tune_threshold_from_validation
from src.models.ines_feature_modeling import score_predictions
from src.models.ines_modeling_core import derive_accuracy_from_summary_metrics
from src.utils.validation import BEST_FEATURE_SET_NAME
from src.models.experiments.lstm_sequence_benchmark.experiment import (
    DEVICE,
    accuracy_from_proba,
    build_sequence_splits,
    make_loader,
    predict_probabilities,
    prepare_partition_frames,
    set_seeds,
    threshold_frame,
    transform_partitions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "models" / "attention_sequence_benchmark"
RESULTS_PATH = RESULTS_DIR / "attention_candidate_results.csv"
SUMMARY_PATH = RESULTS_DIR / "attention_summary.json"
HISTORY_PATH = RESULTS_DIR / "attention_training_history.csv"
THRESHOLD_CURVE_PATH = RESULTS_DIR / "attention_threshold_curve.csv"
NOTES_PATH = RESULTS_DIR / "notes.md"
WINNER_SUMMARY_PATH = PROJECT_ROOT / "models" / "final_winner_package" / "robustness_summary.json"

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
LEARNING_CURVE_FIGURE = FIGURES_DIR / "fig_57_attention_learning_curve.png"
THRESHOLD_FIGURE = FIGURES_DIR / "fig_58_attention_threshold_curve.png"
COMPARISON_FIGURE = FIGURES_DIR / "fig_59_attention_vs_winner_comparison.png"

MAX_EPOCHS = 16
PATIENCE = 5
MIN_IMPROVEMENT = 1e-4

CANDIDATE_CONFIGS: list[dict[str, Any]] = [
    {
        "candidate": "attention_context_14_light",
        "model_label": "Temporal attention",
        "lookback": 14,
        "model_dim": 64,
        "num_heads": 4,
        "num_layers": 1,
        "ff_hidden": 128,
        "dropout": 0.10,
        "learning_rate": 0.0008,
        "batch_size": 768,
        "weight_decay": 0.0001,
    },
    {
        "candidate": "attention_context_14_wide",
        "model_label": "Temporal attention",
        "lookback": 14,
        "model_dim": 96,
        "num_heads": 4,
        "num_layers": 1,
        "ff_hidden": 192,
        "dropout": 0.10,
        "learning_rate": 0.0007,
        "batch_size": 512,
        "weight_decay": 0.0001,
    },
    {
        "candidate": "attention_context_21_balanced",
        "model_label": "Temporal attention",
        "lookback": 21,
        "model_dim": 64,
        "num_heads": 4,
        "num_layers": 1,
        "ff_hidden": 128,
        "dropout": 0.10,
        "learning_rate": 0.0007,
        "batch_size": 768,
        "weight_decay": 0.0001,
    },
    {
        "candidate": "attention_context_21_deep",
        "model_label": "Temporal attention",
        "lookback": 21,
        "model_dim": 96,
        "num_heads": 4,
        "num_layers": 2,
        "ff_hidden": 192,
        "dropout": 0.15,
        "learning_rate": 0.0006,
        "batch_size": 512,
        "weight_decay": 0.0002,
    },
    {
        "candidate": "attention_context_28_balanced",
        "model_label": "Temporal attention",
        "lookback": 28,
        "model_dim": 64,
        "num_heads": 4,
        "num_layers": 1,
        "ff_hidden": 128,
        "dropout": 0.10,
        "learning_rate": 0.0006,
        "batch_size": 768,
        "weight_decay": 0.0001,
    },
]


def candidate_display_label(candidate: str) -> str:
    labels = {
        "attention_context_14_light": "Temporal attention, 14-day light",
        "attention_context_14_wide": "Temporal attention, 14-day wide",
        "attention_context_21_balanced": "Temporal attention, 21-day balanced",
        "attention_context_21_deep": "Temporal attention, 21-day deeper encoder",
        "attention_context_28_balanced": "Temporal attention, 28-day balanced",
    }
    return labels.get(candidate, candidate.replace("_", " ").title())


def write_notes() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    text = """# Temporal Attention Benchmark

## Goal

Test whether a lightweight temporal-attention model can outperform the final CatBoost winner when both models are
built from the same hybrid-plus-core winner feature space.

## Sequence Design

- Each training example is a rolling window of past daily rows from the same location.
- The target remains `rain_tomorrow` for the final day in the window.
- Validation and test windows may use earlier historical rows from prior partitions because that information would be
  available in real forecasting at prediction time.

## Important Constraint

The benchmark uses a CPU-only lightweight temporal-attention encoder because the project environment does not include
a GPU runtime. The benchmark is therefore a real temporal deep-learning model, but still sized for reproducibility in
the current repository while keeping the exact same validation-first chronological selection rule used elsewhere in the
project.
"""
    NOTES_PATH.write_text(text, encoding="utf-8")


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


class TemporalAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_hidden: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Tanh(),
            nn.Linear(model_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(model_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        seq_len = int(inputs.shape[1])
        encoded = self.input_projection(inputs)
        encoded = encoded + self.position_embedding[:, :seq_len]
        encoded = self.encoder(encoded)
        attention_logits = self.attention_pool(encoded).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(encoded * attention_weights.unsqueeze(-1), dim=1)
        pooled = self.dropout(pooled)
        return self.head(pooled).squeeze(-1)


def make_model(config: dict[str, Any], input_size: int) -> TemporalAttentionClassifier:
    return TemporalAttentionClassifier(
        input_size=input_size,
        model_dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        ff_hidden=int(config["ff_hidden"]),
        dropout=float(config["dropout"]),
        max_len=int(config["lookback"]),
    )


def make_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def fit_with_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    set_seeds()
    model = make_model(config, input_size=int(X_train.shape[-1])).to(DEVICE)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(MAX_EPOCHS, 1),
        eta_min=max(float(config["learning_rate"]) * 0.05, 1e-5),
    )
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = make_loader(X_train, y_train, batch_size=int(config["batch_size"]), shuffle=True)

    history_rows: list[dict[str, float | int | str]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_valid_log_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        total_examples = 0
        total_correct = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size = int(batch_y.shape[0])
            total_train_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += int((preds == batch_y).sum().item())

        train_loss = total_train_loss / max(total_examples, 1)
        train_accuracy = total_correct / max(total_examples, 1)
        valid_proba = predict_probabilities(model, X_valid, batch_size=int(config["batch_size"]))
        valid_loss = float(log_loss(y_valid.astype(int), valid_proba, labels=[0, 1]))
        valid_accuracy = accuracy_from_proba(y_valid, valid_proba)
        valid_auc = float(roc_auc_score(y_valid.astype(int), valid_proba))
        history_rows.append(
            {
                "candidate": str(config["candidate"]),
                "epoch": epoch,
                "train_log_loss": train_loss,
                "train_accuracy_at_0_5": float(train_accuracy),
                "validation_log_loss": valid_loss,
                "validation_accuracy_at_0_5": valid_accuracy,
                "validation_roc_auc": valid_auc,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        scheduler.step()

        if valid_loss < best_valid_log_loss - MIN_IMPROVEMENT:
            best_valid_log_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = int(history_rows[-1]["epoch"])

    model.load_state_dict(best_state)
    train_proba = predict_probabilities(model, X_train, batch_size=int(config["batch_size"]))
    valid_proba = predict_probabilities(model, X_valid, batch_size=int(config["batch_size"]))
    best_threshold, validation_threshold_metrics = tune_threshold_from_validation(valid_proba, pd.Series(y_valid.astype(int)))
    train_metrics = score_predictions(pd.Series(y_train.astype(int)), train_proba, threshold=best_threshold)
    validation_metrics = score_predictions(pd.Series(y_valid.astype(int)), valid_proba, threshold=best_threshold)

    return {
        "state_dict": best_state,
        "best_epoch": int(best_epoch),
        "stop_epoch": int(history_rows[-1]["epoch"]),
        "best_validation_log_loss": float(best_valid_log_loss),
        "train_metrics": train_metrics,
        "validation_proba": valid_proba,
        "validation_threshold": float(best_threshold),
        "validation_threshold_metrics": validation_threshold_metrics,
        "validation_metrics": validation_metrics,
        "validation_brier": float(brier_score_loss(y_valid.astype(int), valid_proba)),
        "validation_log_loss": float(log_loss(y_valid.astype(int), valid_proba, labels=[0, 1])),
        "history": pd.DataFrame(history_rows),
    }


def fit_fixed_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
    epochs: int,
) -> nn.Module:
    set_seeds()
    model = make_model(config, input_size=int(X_train.shape[-1])).to(DEVICE)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(epochs), 1),
        eta_min=max(float(config["learning_rate"]) * 0.05, 1e-5),
    )
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = make_loader(X_train, y_train, batch_size=int(config["batch_size"]), shuffle=True)

    for _ in range(int(epochs)):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

    return model


def save_learning_curve(history: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["epoch"], history["train_log_loss"], label="Train log loss", linewidth=2)
    axes[0].plot(history["epoch"], history["validation_log_loss"], label="Validation log loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Log loss")
    axes[0].set_title("Attention model training stability")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(history["epoch"], history["validation_roc_auc"], color="#aa3377", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation ROC-AUC")
    axes[1].set_title("Attention validation discrimination by epoch")
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
    ax.set_title("Attention validation threshold sweep")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(THRESHOLD_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(attention_metrics: dict[str, float], winner_metrics: dict[str, float]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metric_keys = ["test_accuracy", "test_roc_auc", "test_f1", "test_precision", "test_recall"]
    metric_labels = ["Accuracy", "ROC-AUC", "F1", "Precision", "Recall"]
    attention_values = [float(attention_metrics[key]) for key in metric_keys]
    winner_values = [float(winner_metrics[key]) for key in metric_keys]
    positions = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.bar(positions - width / 2, attention_values, width=width, label="Temporal attention", color="#b5478f")
    ax.bar(positions + width / 2, winner_values, width=width, label="Final CatBoost winner", color="#2f6db2")
    ax.set_xticks(positions, metric_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Temporal attention versus final CatBoost winner")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(COMPARISON_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment() -> dict[str, Any]:
    write_notes()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df, features = prepare_partition_frames()

    train_preprocessor, initial_vectors, initial_metadata = transform_partitions(
        train_df, valid_df, test_df, features, fit_on="train"
    )
    initial_transformed_feature_count = int(train_preprocessor.get_feature_names_out().shape[0])
    final_preprocessor, final_vectors, final_metadata = transform_partitions(
        train_df, valid_df, test_df, features, fit_on="train_valid"
    )
    final_transformed_feature_count = int(final_preprocessor.get_feature_names_out().shape[0])

    with WINNER_SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        winner_summary = json.load(handle)
    winner_metrics = resolve_winner_metrics(winner_summary)

    results_rows: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    candidate_thresholds: dict[str, pd.DataFrame] = {}
    initial_sequence_cache: dict[int, dict[str, Any]] = {}
    final_sequence_cache: dict[int, dict[str, Any]] = {}

    for config in CANDIDATE_CONFIGS:
        lookback = int(config["lookback"])
        if lookback not in initial_sequence_cache:
            initial_sequence_cache[lookback] = build_sequence_splits(
                metadata=initial_metadata,
                vectors=initial_vectors,
                lookback=lookback,
            )
        if lookback not in final_sequence_cache:
            final_sequence_cache[lookback] = build_sequence_splits(
                metadata=final_metadata,
                vectors=final_vectors,
                lookback=lookback,
                split_map={"train": "train_valid", "valid": "train_valid", "test": "test"},
            )

        initial_sequences = initial_sequence_cache[lookback]
        X_train, y_train = initial_sequences["train"]
        X_valid, y_valid = initial_sequences["valid"]

        fit_result = fit_with_early_stopping(X_train, y_train, X_valid, y_valid, config)
        history = fit_result["history"].copy()
        histories.append(history)
        threshold_results = threshold_frame(fit_result["validation_proba"], y_valid)
        candidate_thresholds[str(config["candidate"])] = threshold_results

        final_sequences = final_sequence_cache[lookback]
        X_train_valid, y_train_valid = final_sequences["train_valid"]
        X_test, y_test = final_sequences["test"]
        final_model = fit_fixed_epochs(X_train_valid, y_train_valid, config, epochs=int(fit_result["best_epoch"]))
        test_proba = predict_probabilities(final_model, X_test, batch_size=int(config["batch_size"]))
        threshold = float(fit_result["validation_threshold"])
        test_metrics = score_predictions(pd.Series(y_test.astype(int)), test_proba, threshold=threshold)

        results_rows.append(
            {
                "candidate": str(config["candidate"]),
                "candidate_label": candidate_display_label(str(config["candidate"])),
                "model": str(config["model_label"]),
                "feature_count": len(features),
                "transformed_feature_count": final_transformed_feature_count,
                "lookback": lookback,
                "model_dim": int(config["model_dim"]),
                "num_heads": int(config["num_heads"]),
                "num_layers": int(config["num_layers"]),
                "ff_hidden": int(config["ff_hidden"]),
                "dropout": float(config["dropout"]),
                "learning_rate": float(config["learning_rate"]),
                "batch_size": int(config["batch_size"]),
                "weight_decay": float(config.get("weight_decay", 0.0)),
                "sequence_train_support": int(initial_sequences["counts"]["train"]),
                "sequence_validation_support": int(initial_sequences["counts"]["valid"]),
                "sequence_test_support": int(final_sequences["counts"]["test"]),
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
                "test_brier": float(brier_score_loss(y_test.astype(int), test_proba)),
                "test_log_loss": float(log_loss(y_test.astype(int), test_proba, labels=[0, 1])),
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
        "benchmark_name": "attention_sequence_benchmark",
        "feature_set_name": BEST_FEATURE_SET_NAME,
        "feature_count": len(features),
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
            "attention_minus_winner_roc_auc": float(best_result["winner_test_roc_auc_gap"]),
            "attention_minus_winner_f1": float(best_result["winner_test_f1_gap"]),
            "attention_minus_winner_precision": float(best_result["winner_test_precision_gap"]),
            "attention_minus_winner_recall": float(best_result["winner_test_recall_gap"]),
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

