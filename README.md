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
| `src/docker/airflow/` | Local Airflow service for DVC-backed data and model versioning |
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

## Team Integration Notes

Andrey's FastAPI and Nginx work is the official API and gateway layer for this project. The integration work kept that decision and connected the rest of the local MLOps stack around it.

| Area | Project role |
| --- | --- |
| `src/prediction_api/main.py` | Official FastAPI application used by Docker, Airflow checks, CI contract tests, monitoring, and Kubernetes. |
| `nginx/nginx.conf` | Security gateway for authentication, forwarded users, rate limiting, and protected API access. |
| `docker-compose.yml` | Local stack that connects FastAPI, Nginx, Prometheus, Grafana, Airflow, and prediction traffic. |
| `tests/test_prediction_api_contract.py` | Lightweight contract test for Airflow and CI. It complements the running-stack Nginx/API tests instead of replacing them. |

The files connected to Andrey's work were touched only to make the API reachable from orchestration, monitoring, Docker Compose, and Kubernetes. The API behavior and Nginx security design were not replaced.

## Reuse Notes

The project has two local deployment paths.

| Path | Purpose | Main File |
| --- | --- | --- |
| Docker Compose stack | Local development for FastAPI, Nginx, Prometheus, Grafana, and Airflow | `docker-compose.yml` |
| Kubernetes stack | Production-style local deployment with scalable FastAPI and separated Airflow services | `kubernetes/kustomization.yaml` |

The work is local-first. The Airflow DAGs update local files, local DVC metadata, and local reports. They do not push to GitHub or DagsHub.

### Local Credentials

| Service | Deployment | URL | Username | Password | Notes |
| --- | --- | --- | --- | --- | --- |
| Airflow | Docker Compose root stack | `http://localhost:8080` | `admin` | `airflow` | Defined in `docker-compose.yml`. |
| Airflow | Legacy dev compose | `http://localhost:8080` | `airflow` | `airflow` | Defined in `src/docker/docker-compose-dev.yml`. |
| Airflow | Kubernetes | port-forward to `http://localhost:18080` | `admin` | `rain-airflow-admin-change-me` | Defined in `kubernetes/airflow-secret.yaml`; placeholder for local reuse. |
| Grafana | Docker Compose | `http://localhost:3000` | `admin` | `admin` | Defined in `docker-compose.yml`. |
| Prometheus | Docker Compose | `http://localhost:9090` | none | none | Local monitoring only. |
| Nginx gateway | Docker Compose | `https://localhost` | `andrey`, `ines`, `gunter`, `admin` | stored as hashes | Password hashes are in `nginx/.htpasswd`; plaintext passwords cannot be recovered from the file. |

The Kubernetes Airflow password is different from Docker Compose on purpose. Kubernetes uses `kubernetes/airflow-secret.yaml` because the production-style deployment separates Airflow webserver, scheduler, workers, Postgres, and Redis. The value `rain-airflow-admin-change-me` is only the local workspace placeholder; shared or production environments use their own managed secret.

For a fresh Kubernetes deployment, change the password in `kubernetes/airflow-secret.yaml` before applying the manifests. For an already running cluster, reset it from an Airflow pod:

```powershell
kubectl exec -n rain-prediction deployment/rain-prediction-airflow-webserver -- airflow users reset-password -u admin -p "<new-password>"
```

### Kubernetes Reuse

The Kubernetes deployment includes:

- FastAPI deployment with 3 baseline replicas and HPA up to 6 replicas
- Airflow webserver deployment with 2 replicas
- Airflow scheduler deployment with 2 replicas
- Airflow Celery worker deployment with HPA from 2 to 3 replicas for local Docker Desktop
- Airflow Postgres StatefulSet for metadata
- Airflow Redis StatefulSet for Celery broker
- Airflow migration Job
- PersistentVolumeClaims for Airflow project files, logs, Postgres data, and Redis data
- PodDisruptionBudgets for FastAPI and Airflow components

Build the local images before applying Kubernetes:

```powershell
docker build -f docker/prediction-api/api.Dockerfile -t rain_prediction_mlops-fastapi:latest .
docker build -f docker/airflow/airflow.Dockerfile -t rain_prediction_mlops-airflow:latest .
docker tag rain_prediction_mlops-airflow:latest rain_prediction_mlops-airflow:production-dvc
```

The Airflow Kubernetes image includes the DVC workspace metadata required for local `dvc add` commands. The private `.dvc/config.local`, DVC cache, and DVC temporary files are excluded from the image. Inside Kubernetes, the shared Airflow project PVC is configured with `core.no_scm = true` because the PVC is not a Git checkout and the project rule is local-only versioning.

The Airflow worker is sized for the model-training task. Celery worker concurrency is set to `1`, each worker requests `1Gi` memory and can use up to `3Gi`, and worker scale-down is stabilized for 30 minutes. This prevents long training tasks from being killed by aggressive HPA scale-down or by the previous `1Gi` memory limit.

Apply and validate:

```powershell
kubectl apply -k .\kubernetes
kubectl get deploy,statefulset,job,hpa,pdb,svc,pvc -n rain-prediction
kubectl get pods -n rain-prediction
kubectl top pods -n rain-prediction
```

Expected healthy status:

| Resource | Expected |
| --- | --- |
| `rain-prediction-api` | `3/3` |
| `rain-prediction-airflow-webserver` | `2/2` |
| `rain-prediction-airflow-scheduler` | `2/2` |
| `rain-prediction-airflow-worker` | at least `2/2`, can scale to `3/3` |
| `airflow-postgres` | `1/1` |
| `airflow-redis` | `1/1` |
| `airflow-migrate` | `Complete` |

Test the services through temporary port-forwards:

```powershell
kubectl port-forward svc/rain-prediction-api 18502:8502 -n rain-prediction
Invoke-WebRequest -UseBasicParsing http://localhost:18502/health
```

```powershell
kubectl port-forward svc/rain-prediction-airflow 18080:8080 -n rain-prediction
Invoke-WebRequest -UseBasicParsing http://localhost:18080/health
```

Stop each port-forward with `Ctrl+C` after testing.

For Airflow DVC validation inside Kubernetes:

```powershell
kubectl exec -n rain-prediction deployment/rain-prediction-airflow-worker -- bash -c "cd /opt/airflow/project && dvc status"
kubectl exec -n rain-prediction deployment/rain-prediction-airflow-worker -- bash -c "cd /opt/airflow/project && export PYTHONPATH=/opt/airflow/project:$PYTHONPATH && python -m src.versioning.dvc_versioning dvc-add --target data/raw/weatherAUS.csv"
```

### Docker Compose Reuse

The root Docker Compose stack is useful for local monitoring and gateway testing:

```powershell
docker compose up -d --build
docker compose ps
```

Main URLs:

| Service | URL |
| --- | --- |
| Nginx gateway | `https://localhost` |
| Airflow | `http://localhost:8080` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` |

FastAPI can also be scaled behind Nginx:

```powershell
docker compose up -d --scale fastapi=3
```

The Docker Compose stack also starts `prediction-traffic`, a local traffic generator that loads all supported locations from FastAPI and sends one valid batch prediction for every location once per minute. This keeps Prometheus and Grafana populated so the dashboards show data immediately when opened.

The Grafana `Predictions by City` panel uses `sum(rain_predictions_total) by (location)`, so all supported location labels are available in the dashboard instead of only a top-10 subset.

Check it with:

```powershell
docker compose logs -f prediction-traffic
```

Stop only the automatic prediction traffic with:

```powershell
docker compose stop prediction-traffic
```

## Local Airflow Versioning

The `data_model_versioning` DAG orchestrates local-only data and model versioning with DVC:

1. Extract or validate `data/raw/weatherAUS.csv`.
2. Update the local DVC pointer for the raw dataset.
3. Verify required DVC-tracked input files already exist locally.
4. Write an input version manifest to `reports/versioning/`.
5. Train the final winner model.
6. Update the local DVC pointer for `models/final_winner/winner_model.joblib`.
7. Write output and DVC status manifests to `reports/versioning/`.

It does not push to GitHub or DagsHub. To run Airflow locally:

```powershell
docker compose up -d --build airflow
```

Access Airflow at `http://localhost:8080` with username `admin` and password `airflow`, then trigger `data_model_versioning`.

The `daily_weather_ingestion` DAG runs on a daily schedule and only updates the
local raw dataset plus DVC metadata. It does not retrain the served model.

For new local data, place a WeatherAUS-format CSV or ZIP somewhere local and set:

```powershell
$env:WEATHER_AUS_SOURCE = "data/incoming/weatherAUS_new.csv"
$env:WEATHER_AUS_EXTRACT_MODE = "upsert"
docker compose -f src/docker/docker-compose-dev.yml up --build airflow
```

`upsert` inserts new `Date` + `Location` rows and applies corrections for matching rows. Use `append` to reject corrections, or `replace` when the source is a full refreshed raw dataset.

For online extraction, use either a direct CSV/ZIP URL:

```powershell
$env:WEATHER_AUS_SOURCE_URL = "https://example.com/weatherAUS.csv"
$env:WEATHER_AUS_SOURCE_SHA256 = "<optional expected sha256>"
$env:WEATHER_AUS_EXTRACT_MODE = "upsert"
docker compose -f src/docker/docker-compose-dev.yml up --build airflow
```

Or use Kaggle with credentials:

```powershell
$env:KAGGLE_USERNAME = "<your username>"
$env:KAGGLE_KEY = "<your api key>"
$env:WEATHER_AUS_KAGGLE_DATASET = "jsphyg/weather-dataset-rattle-package"
$env:WEATHER_AUS_KAGGLE_FILE = "weatherAUS.csv"
$env:WEATHER_AUS_EXTRACT_MODE = "replace"
docker compose -f src/docker/docker-compose-dev.yml up --build airflow
```

For daily automated extraction without Kaggle, use the Open-Meteo provider:

```powershell
$env:WEATHER_AUS_DAILY_PROVIDER = "open-meteo"
$env:WEATHER_AUS_EXTRACT_MODE = "upsert"
$env:WEATHER_AUS_DAILY_DAYS_BACK = "7"
$env:WEATHER_AUS_DAILY_END_LAG_DAYS = "1"
docker compose -f src/docker/docker-compose-dev.yml up --build airflow
```

The daily provider fetches recent weather by station coordinates, converts it to
the WeatherAUS column schema, and backfills `RainTomorrow` from the next day's
rainfall when that next day is available.

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