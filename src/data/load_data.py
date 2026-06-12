from pathlib import Path

import pandas as pd

from src.config.paths import ALIGNED_TOP25_FEATURES, RAIN_MODEL_DATASET_ALIGNED, RAW_BASE


def load_raw_weather(path: Path = RAW_BASE) -> pd.DataFrame:
    return pd.read_csv(path)


def load_aligned_modeling_data(path: Path = RAIN_MODEL_DATASET_ALIGNED) -> pd.DataFrame:
    return pd.read_csv(path)


def load_aligned_feature_names(path: Path = ALIGNED_TOP25_FEATURES) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
