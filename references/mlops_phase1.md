# MLOps Phase 1

Phase 1 provides the foundations, reproducible environment, prepared data, baseline model evidence, tests, container assets, and inference API.

## Checklist

| Requirement | Status | Evidence |
| --- | --- | --- |
| Define project objectives and key metrics | Complete | `README.md`, `models/final_winner/metadata.json`, `references/phase1_baseline_metrics.json` |
| Set up a reproducible development environment | Complete | `requirements.txt`, `src/docker/docker-compose-dev.yml`, service Dockerfiles under `src/docker/` |
| Collect and preprocess data | Complete | `data/raw/`, `data/preprocessed/`, `references/`, `src/features/` |
| Build and evaluate a baseline ML model, implement unit test | Complete | `src/models/train_baseline.py`, `src/docker/testing/tests/test_baseline_training.py` |
| Implement a basic inference API | Complete | `src/models/api.py`, `src/models/inference.py`, `src/docker/prediction/` |

## Baseline Model

| Item | Value |
| --- | --- |
| Model | Logistic Regression |
| Purpose | Phase 1 baseline evaluation |
| Feature set | Aligned top-25 feature list |
| Target | `rain_tomorrow` |
| Threshold | 0.58 |
| Metrics file | `references/phase1_baseline_metrics.json` |

## Served Model

| Item | Value |
| --- | --- |
| Model | Final hybrid CatBoost |
| Purpose | Inference API model |
| Artifact | `models/final_winner/winner_model.joblib` |
| Feature count | 68 |
| Target | `rain_tomorrow` |
| Threshold | 0.58 |

## API Contract

| Method | Endpoint | Response |
| --- | --- | --- |
| GET | `/health` | Service status, API version, and loaded model name |
| GET | `/model-info` | Model metadata, metrics, threshold, and required features |
| POST | `/predict` | Predicted label, rain probability, no-rain probability, threshold, and feature count |
