from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
REFERENCES_DIR = BASE_DIR / "references"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
MODELS_DIR = BASE_DIR / "models"

RAW_BASE = RAW_DIR / "weatherAUS.csv"
RAIN_MODEL_DATASET_ALIGNED = PREPROCESSED_DIR / "rain_model_dataset_aligned.csv"
ALIGNED_TOP25_FEATURES = PREPROCESSED_DIR / "ines_selected_top25_features_aligned.txt"

FINAL_WINNER_DIR = MODELS_DIR / "final_winner"
FINAL_WINNER_MODEL_ARTIFACT = FINAL_WINNER_DIR / "winner_model.joblib"
FINAL_WINNER_METADATA_PATH = FINAL_WINNER_DIR / "metadata.json"
FINAL_WINNER_SAMPLE_INPUT_PATH = FINAL_WINNER_DIR / "sample_input.json"
FINAL_WINNER_CONFIG_PATH = FINAL_WINNER_DIR / "model_config.json"

RANDOM_STATE = 42
TARGET_COLUMN = "rain_tomorrow"
DATE_COLUMN = "date"
