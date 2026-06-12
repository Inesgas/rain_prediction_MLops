from __future__ import annotations

import random
from copy import deepcopy
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cpu")
MAX_SEQUENCE_EPOCHS = 8
SEQUENCE_PATIENCE = 3
SEQUENCE_RANDOM_STATE = 42

_BUILD_TABULAR_PREPROCESSOR: Callable | None = None
_TO_BINARY_TARGET: Callable | None = None
_FIND_BEST_THRESHOLD: Callable | None = None


def configure_sequence_modeling(
    build_tabular_preprocessor_fn: Callable,
    to_binary_target_fn: Callable,
    find_best_threshold_fn: Callable,
) -> None:
    global _BUILD_TABULAR_PREPROCESSOR, _TO_BINARY_TARGET, _FIND_BEST_THRESHOLD
    _BUILD_TABULAR_PREPROCESSOR = build_tabular_preprocessor_fn
    _TO_BINARY_TARGET = to_binary_target_fn
    _FIND_BEST_THRESHOLD = find_best_threshold_fn


def _require_configured(name: str, func: Callable | None) -> Callable:
    if func is None:
        raise RuntimeError(
            f"{name} is not configured. Call configure_sequence_modeling(...) first."
        )
    return func


def set_sequence_seeds() -> None:
    random.seed(SEQUENCE_RANDOM_STATE)
    np.random.seed(SEQUENCE_RANDOM_STATE)
    torch.manual_seed(SEQUENCE_RANDOM_STATE)


def prepare_sequence_partitions(train_df, valid_df, test_df, features):
    build_tabular_preprocessor = _require_configured(
        "build_tabular_preprocessor", _BUILD_TABULAR_PREPROCESSOR
    )

    keep_cols = list(dict.fromkeys(["location", "date", "rain_tomorrow"] + list(features)))
    train_part = train_df[keep_cols].copy()
    valid_part = valid_df[keep_cols].copy()
    test_part = test_df[keep_cols].copy()

    preprocessor = build_tabular_preprocessor(train_part[features], scale_numeric=True)
    preprocessor.fit(train_part[features])

    combined = pd.concat([train_part, valid_part, test_part], axis=0, ignore_index=True).copy()
    combined["split"] = ["train"] * len(train_part) + ["valid"] * len(valid_part) + ["test"] * len(test_part)
    combined["source_index"] = np.arange(len(combined))
    vectors = preprocessor.transform(combined[features]).astype(np.float32, copy=False)
    return combined, vectors


def build_sequence_splits(metadata: pd.DataFrame, vectors: np.ndarray, lookback: int):
    to_binary_target = _require_configured("to_binary_target", _TO_BINARY_TARGET)

    frame = metadata.sort_values(["location", "date", "source_index"]).reset_index(drop=True)
    sequence_rows = {}
    target_rows = {}

    for _, group in frame.groupby("location", sort=False):
        if len(group) < lookback:
            continue
        source_indices = group["source_index"].to_numpy()
        targets = to_binary_target(group["rain_tomorrow"]).to_numpy(dtype=np.float32)
        split_values = group["split"].tolist()
        for position in range(lookback - 1, len(group)):
            split_name = str(split_values[position])
            sequence = vectors[source_indices[position - lookback + 1 : position + 1]]
            sequence_rows.setdefault(split_name, []).append(sequence)
            target_rows.setdefault(split_name, []).append(float(targets[position]))

    result = {}
    for split_name, sequences in sequence_rows.items():
        result[split_name] = (
            np.stack(sequences).astype(np.float32, copy=False),
            np.asarray(target_rows[split_name], dtype=np.float32),
        )
    return result


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def predict_sequence_probabilities(model: nn.Module, X: np.ndarray, batch_size: int) -> np.ndarray:
    loader = make_loader(X, np.zeros(len(X), dtype=np.float32), batch_size=batch_size, shuffle=False)
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch_X, _ in loader:
            logits = model(batch_X.to(DEVICE))
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def fit_sequence_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    learning_rate: float,
    batch_size: int,
    weight_decay: float = 0.0,
):
    find_best_threshold = _require_configured("find_best_threshold", _FIND_BEST_THRESHOLD)

    set_sequence_seeds()
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(MAX_SEQUENCE_EPOCHS, 1),
        eta_min=max(float(learning_rate) * 0.05, 1e-5),
    )
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = make_loader(X_train, y_train, batch_size=int(batch_size), shuffle=True)

    best_state = None
    best_valid_loss = float("inf")
    patience_counter = 0
    history_rows = []

    for epoch in range(1, MAX_SEQUENCE_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * int(batch_y.shape[0])
            total_examples += int(batch_y.shape[0])

        valid_proba = predict_sequence_probabilities(model, X_valid, batch_size=int(batch_size))
        valid_loss = float(log_loss(y_valid.astype(int), valid_proba, labels=[0, 1]))
        valid_auc = float(roc_auc_score(y_valid.astype(int), valid_proba))
        history_rows.append(
            {
                "epoch": epoch,
                "train_log_loss": total_loss / max(total_examples, 1),
                "validation_log_loss": valid_loss,
                "validation_roc_auc": valid_auc,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        scheduler.step()

        if valid_loss < best_valid_loss - 1e-4:
            best_valid_loss = valid_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= SEQUENCE_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    valid_proba = predict_sequence_probabilities(model, X_valid, batch_size=int(batch_size))
    best_threshold, threshold_frame = find_best_threshold(pd.Series(y_valid.astype(int)), valid_proba)
    return model, pd.DataFrame(history_rows), valid_proba, best_threshold, threshold_frame


class LSTMSequenceClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden_state, _) = self.lstm(inputs)
        return self.head(self.dropout(hidden_state[-1])).squeeze(-1)


class GRUSequenceClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden_state = self.gru(inputs)
        return self.head(self.dropout(hidden_state[-1])).squeeze(-1)


class TemporalAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        model_dim: int,
        num_heads: int,
        ff_hidden: int,
        num_layers: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, model_dim)
        self.position_embedding = nn.Embedding(max_len, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(model_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0)
        encoded = self.input_projection(inputs) + self.position_embedding(positions)
        encoded = self.encoder(encoded)
        pooled = self.dropout(encoded.mean(dim=1))
        return self.head(pooled).squeeze(-1)


class TemporalCNNClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        projection_dim: int,
        channels: tuple[int, ...],
        kernel_size: int,
        dilation_base: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, projection_dim)
        layers = []
        in_channels = projection_dim
        for index, out_channels in enumerate(channels):
            dilation = int(dilation_base) ** index
            padding = (kernel_size - 1) * dilation // 2
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(channels[-1] * 2, channels[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(inputs)
        encoded = self.encoder(projected.transpose(1, 2))
        pooled = torch.cat(
            [self.avg_pool(encoded).squeeze(-1), self.max_pool(encoded).squeeze(-1)],
            dim=1,
        )
        return self.head(pooled).squeeze(-1)


__all__ = [
    "DEVICE",
    "configure_sequence_modeling",
    "prepare_sequence_partitions",
    "build_sequence_splits",
    "make_loader",
    "predict_sequence_probabilities",
    "fit_sequence_model",
    "LSTMSequenceClassifier",
    "GRUSequenceClassifier",
    "TemporalAttentionClassifier",
    "TemporalCNNClassifier",
]

