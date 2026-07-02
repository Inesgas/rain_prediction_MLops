from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring import drift_report


def _make_frame(rng, n=200, humidity_mean=50.0, pressure_mean=1015.0, extra_col=False):
    data = {
        "humidity_3pm": rng.normal(humidity_mean, 5, n),
        "humidity_9am": rng.normal(humidity_mean + 10, 5, n),
        "pressure_9am": rng.normal(pressure_mean, 3, n),
    }
    if extra_col:
        data["not_monitored_col"] = rng.normal(0, 1, n)
    return pd.DataFrame(data)


def test_safe_timestamp_matches_expected_format():
    timestamp = drift_report.safe_timestamp()
    assert len(timestamp) == 16
    assert timestamp.endswith("Z")
    assert timestamp[8] == "T"


def test_build_snapshot_only_uses_columns_present_in_both_frames():
    rng = np.random.default_rng(42)
    reference = _make_frame(rng, extra_col=True)
    current = _make_frame(rng)

    snapshot = drift_report.build_snapshot(reference, current)
    summary = drift_report.extract_summary(snapshot)

    assert set(summary["per_column"].keys()) == {
        "humidity_3pm", "humidity_9am", "pressure_9am",
    }


def test_extract_summary_no_drift_for_identical_distributions():
    rng = np.random.default_rng(42)
    reference = _make_frame(rng)
    current = _make_frame(rng)

    snapshot = drift_report.build_snapshot(reference, current)
    summary = drift_report.extract_summary(snapshot)

    assert summary["dataset_drift"] is False
    assert summary["number_of_drifted_columns"] == 0


def test_extract_summary_flags_dataset_drift_for_shifted_distributions():
    rng = np.random.default_rng(42)
    reference = _make_frame(rng, humidity_mean=50.0, pressure_mean=1015.0)
    current = _make_frame(rng, humidity_mean=90.0, pressure_mean=950.0)

    snapshot = drift_report.build_snapshot(reference, current)
    summary = drift_report.extract_summary(snapshot)

    assert summary["dataset_drift"] is True
    assert summary["share_of_drifted_columns"] == 1.0


def test_load_reference_reads_configured_csv(tmp_path, monkeypatch):
    reference_path = tmp_path / "reference_dataset.csv"
    pd.DataFrame({"humidity_3pm": [40, 41]}).to_csv(reference_path, index=False)
    monkeypatch.setattr(drift_report, "REFERENCE_DATASET_PATH", reference_path)

    result = drift_report.load_reference()

    assert list(result["humidity_3pm"]) == [40, 41]


def test_load_current_window_filters_by_days_back(tmp_path, monkeypatch):
    dataset_path = tmp_path / "rain_model_dataset_aligned.csv"
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    pd.DataFrame({"date": dates, "humidity_3pm": range(30)}).to_csv(dataset_path, index=False)
    monkeypatch.setattr(drift_report, "RAIN_MODEL_DATASET_ALIGNED", dataset_path)

    result = drift_report.load_current_window(days_back=5)

    assert result["date"].min() > dates.max() - pd.Timedelta(days=5)
    assert len(result) == 5