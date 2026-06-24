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

## Experiment Tracking (MLflow + DagsHub)

Model training runs are tracked with MLflow. 
The configured MLflow tracking server is hosted on DagsHub, using the same project as the DVC remote: 
`https://dagshub.com/Inesgas/rain_prediction_MLops.mlflow`.

Each run of `src/models/train_winner.py` logs:

| Logged item | Examples |
| ----------- | -------- |
| Parameters  | CatBoost hyperparameters, `threshold`, `feature_set_name`, `test_start_date`, `test_end_date` |
| Metrics     |                           `roc_auc`, `accuracy`, `f1`, `precision`, `recall`, `train_rows`, `test_rows` |
| Artifacts   | Trained CatBoost model    (via `mlflow.catboost.log_model`) |

### Local setup

Install the MLflow client (already included in `requirements.txt` as `mlflow-skinny`, kept lightweight to avoid pulling in MLflow's full `pandas<3` dependency chain):

```bash
pip install -r requirements.txt```

Copy the example environment file and fill in your DagsHub credentials:

cp .env.example .env

# .env.example
MLFLOW_TRACKING_URI=https://dagshub.com/Inesgas/rain_prediction_MLops.mlflow
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
API_USERS=<username1:role1,username2:role2,...>

Export the variables before training:

```bash
export $(grep -v '^#' .env | xargs)

Then run the training script as usual:

```bash
python -m src.models.train_winner


Each run will print a DagsHub link to the experiment and the specific run, e.g.:

    View run final_hybrid_catboost at: https://dagshub.com/Inesgas/rain_prediction_MLops.mlflow/#/experiments/0/runs/<run_id>
    View experiment at: https://dagshub.com/Inesgas/rain_prediction_MLops.mlflow/#/experiments/0

Security note: .env is git-ignored and must never be committed. Only .env.example (with placeholder values) is tracked in the repository.

