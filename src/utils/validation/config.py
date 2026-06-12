from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models.ines_feature_modeling import BEST_CATBOOST_PARAMS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HYBRID_SUMMARY_PATH = (
    PROJECT_ROOT / "models" / "hybrid_imputation_breakthrough" / "hybrid_regime_imputer_summary.json"
)
BEST_FEATURE_SET_NAME = "hybrid_regime_keep_location_plus_core"
CORE_FEATURE_COLUMNS = [
    "temp_range",
    "humidity_temp_3pm_interaction",
    "pressure_humidity_3pm_ratio",
    "cloud_humidity_3pm_interaction",
    "moisture_stability_3pm",
    "humidity_rising_fast",
    "warming_day",
]


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    use_spatial_fill: bool = True
    use_regime_fill: bool = True
    add_missing_indicators: bool = True
    add_core_features: bool = True
    drop_location: bool = False


def load_best_hybrid_selection() -> dict[str, Any]:
    if HYBRID_SUMMARY_PATH.exists():
        summary = json.loads(HYBRID_SUMMARY_PATH.read_text(encoding="utf-8"))
        best = summary.get("best_result", {})
        return {
            "summary": summary,
            "feature_set": best.get("feature_set", BEST_FEATURE_SET_NAME),
            "params": json.loads(best["params"]) if "params" in best else dict(BEST_CATBOOST_PARAMS),
            "threshold": float(best.get("validation_threshold", 0.5)),
        }

    return {
        "summary": {},
        "feature_set": BEST_FEATURE_SET_NAME,
        "params": dict(BEST_CATBOOST_PARAMS),
        "threshold": 0.5,
    }

