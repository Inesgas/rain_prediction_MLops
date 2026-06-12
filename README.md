# Rain Prediction MLOps

MLOps project for serving the final CatBoost rain prediction winner for Australian weather stations.

The repository follows a microservice-oriented structure with configuration, Docker services, setup scripts, references, reports, notebooks, data, and model artifacts separated clearly.

## Repository Tree

```text
rain_prediction_mlops/
|-- .github/
|   `-- workflows/
|-- data/
|   |-- cleaned/
|   |-- preprocessed/
|   |-- raw/
|   `-- sample/
|-- models/
|   `-- final_winner/
|-- notebooks/
|-- references/
|-- reports/
|   `-- figures/
|-- src/
|   |-- config/
|   |-- data/
|   |-- docker/
|   |   |-- data-download-prep/
|   |   |-- database/
|   |   |-- frontend/
|   |   |-- gateway/
|   |   |-- initialization/
|   |   |-- prediction/
|   |   |-- scoring/
|   |   |-- testing/
|   |   |-- training/
|   |   `-- users/
|   |-- features/
|   |-- models/
|   |   `-- experiments/
|   |-- script/
|   `-- utils/
`-- requirements.txt
```

## Main Components

| Area | Content |
| --- | --- |
| `src/config/` | Shared paths and project constants |
| `src/docker/prediction/` | Prediction API service |
| `src/docker/training/` | Winner model training service |
| `src/docker/frontend/` | Streamlit service dashboard |
| `src/docker/testing/` | API and inference validation service |
| `src/script/` | Project automation scripts |
| `src/models/` | Winner training, inference, API logic, model utilities, and experiment modules |
| `data/raw/` | Original weather dataset |
| `data/preprocessed/` | Modeling tables, selected feature lists, feature rankings, and location metadata |
| `data/sample/` | Sample payload for the winner API |
| `references/` | Climate grids, station reference data, and MLOps notes |
| `models/final_winner/` | Single served model artifact, metadata, model configuration, and sample payload |

## Artifact Storage

Large datasets and the served model artifact are tracked with DVC. The configured remote is DagsHub: `https://dagshub.com/Inesgas/rain_prediction_MLops.dvc`.

## Served Model

| Item | Value |
| --- | --- |
| Model | Final hybrid CatBoost |
| Artifact | `models/final_winner/winner_model.joblib` |
| Feature count | 68 |
| Target | `rain_tomorrow` |
| Threshold | 0.58 |
| Holdout ROC-AUC | 0.9016 |
| Holdout F1-score | 0.6853 |
| Holdout precision | 0.6302 |
| Holdout recall | 0.7510 |

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Service status, API version, and loaded model name |
| GET | `/model-info` | Model metadata, metrics, threshold, and required features |
| POST | `/predict` | Rain/no-rain prediction with class probabilities |
