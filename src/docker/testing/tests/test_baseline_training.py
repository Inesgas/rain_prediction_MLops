from __future__ import annotations

import pandas as pd

from src.models.train_baseline import train_baseline


def test_train_baseline_evaluates_chronological_split(tmp_path) -> None:
    rows = []
    for index in range(30):
        rows.append(
            {
                "date": f"2020-01-{index + 1:02d}",
                "humidity_3pm": 40 + index,
                "rain_today": "Yes" if index % 3 == 0 else "No",
                "rain_tomorrow": "Yes" if index % 4 in (0, 1) else "No",
            }
        )
    dataset_path = tmp_path / "baseline_dataset.csv"
    features_path = tmp_path / "features.txt"
    pd.DataFrame(rows).to_csv(dataset_path, index=False)
    features_path.write_text("humidity_3pm\nrain_today\n", encoding="utf-8")

    result = train_baseline(dataset_path=dataset_path, features_path=features_path, threshold=0.5)

    assert result["model_name"] == "logistic_regression_baseline"
    assert result["feature_count"] == 2
    assert result["rows"] == {"train": 24, "test": 6}
    assert set(result["metrics"]) == {"roc_auc", "accuracy", "f1", "precision", "recall"}
