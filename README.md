# Rain Prediction MLOps

<p align="center">
  <b>Production-style MLOps for Australian rainfall prediction</b><br>
  FastAPI · Nginx · Airflow · MLflow · DVC · Docker Compose · Kubernetes
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-API-009688">
  <img alt="MLflow" src="https://img.shields.io/badge/MLflow-Tracking-0194E2">
  <img alt="DVC" src="https://img.shields.io/badge/DVC-Versioning-13ADC7">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-Compose-2496ED">
  <img alt="Kubernetes" src="https://img.shields.io/badge/Kubernetes-Orchestration-326CE5">
</p>

***

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data Science Foundation](#data-science-foundation)
- [Production MLOps Contribution](#production-mlops-contribution)
- [Repository Layout](#repository-layout)
- [Served Model](#served-model)
- [API](#api)
- [Quick Start](#quick-start)
- [Docker Compose](#docker-compose)
- [Kubernetes](#kubernetes)
- [Airflow Pipelines](#airflow-pipelines)
- [MLflow and DVC](#mlflow-and-dvc)
- [Data Ingestion](#data-ingestion)
- [Local Credentials](#local-credentials)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)
- [Project Notes](#project-notes)

***

## Overview

This repository documents a complete rainfall prediction project that moved from a data science model into a production-style MLOps system.
The data science part answers the modeling question: **given Australian weather observations, can we predict whether it will rain tomorrow?**
The MLOps part answers the production question: **can the model, data, training process, monitoring, and deployment be reproduced and operated as a system instead of remaining as a notebook result?**

The current project serves a final **hybrid CatBoost winner model** through a FastAPI-based prediction service and connects training, versioning, tracking, orchestration, monitoring, and Kubernetes deployment into one local-first production workflow.
This README is written as the project report: it explains what was built, what each stage contributes, and what was validated.

### What this project covers

- A data science workflow for weather cleaning, feature engineering, chronological splitting, model training, and evaluation
- A final CatBoost model package with a stable 68-feature prediction contract
- Data and model artifact versioning with **DVC** and DagsHub remote storage
- Experiment and model metadata logging with **MLflow**
- Automated training, versioning, and drift-monitoring workflows with **Airflow**
- Local integration with **Docker Compose**
- Production-style orchestration with **Kubernetes**, PVC-backed Airflow state, HPA/PDB resources, and service separation
- API, gateway, and dashboard integration points where the FastAPI, Nginx, Prometheus, and Grafana parts can be extended by the team owners

***

## Architecture

The project architecture has two connected layers.
The first layer is the **data science layer**, where raw weather observations are cleaned, enriched, split chronologically, trained, and evaluated.
The second layer is the **MLOps production layer**, where the same data/model workflow is versioned, scheduled, monitored, containerized, and deployed.

```text
Weather observations
        |
        v
Data cleaning + feature engineering
        |
        v
Chronological train / validation / test split
        |
        v
Hybrid CatBoost winner model
        |
        +---------------------> FastAPI prediction service
        |
        +---------------------> MLflow run metadata
        |
        +---------------------> DVC model and data versions
        |
        v
Airflow automated workflows
        |
        +---------------------> data/model versioning
        +---------------------> retraining and artifact refresh
        +---------------------> Evidently drift reports
        |
        v
Docker Compose and Kubernetes runtime
```

### Deployment modes

| Mode | Purpose | Main Entry Point |
|------|---------|------------------|
| Docker Compose | Local development and integrated service testing | `docker-compose.yml` |
| Kubernetes | Production-style local deployment with scaling and service separation | `kubernetes/kustomization.yaml` |

***

## Data Science Foundation

The data science stage transformed the WeatherAUS rainfall dataset into a supervised binary classification problem.
The target is `rain_tomorrow`, and the model uses daily weather, location, climate, lag, and engineered meteorological signals to estimate the probability of rain on the following day.

### Data preparation and feature engineering

The modeling dataset was not used as a raw table directly.
It was prepared as a feature contract that can be reused by training, API inference, Airflow automation, and Kubernetes deployment.

What was prepared:

- Weather records were standardized around date, location, numeric meteorological columns, categorical location columns, and the `rain_tomorrow` target.
- Missing values were handled with explicit hybrid imputation indicators such as `rainfall_missing_hybrid`, `sunshine_missing_hybrid`, `cloud_3pm_missing_hybrid`, and pressure/humidity missingness flags.
- Date features were expanded into `month`, `day`, `year`, and cyclic yearly signals.
- Location context was enriched with latitude, longitude, elevation, and rainfall-zone indicators.
- Wind direction was converted into numerical direction vectors so that circular direction values could be learned by the model.
- Short-term weather dynamics were added with features such as temperature difference, humidity difference, pressure difference, previous-day rainfall, previous-day maximum temperature, and 24-hour changes.
- Dew point and stability-style features were added to represent moisture and atmospheric behavior.

### Modeling strategy

The final model was selected as a **hybrid CatBoost classifier** because it handles mixed numeric/categorical tabular data well and gave the best balance between ranking quality and rain-event recall.
The final contract contains **68 input features**, including categorical features such as `location`, `humidity_9am_bin`, `pressure_9am_bin`, and `temp_9am_bin`.

The split strategy was chronological instead of random.
This is important for weather data because random splitting can leak future seasonal patterns into training.
The workflow uses older observations for training and the newest observations for final testing.
The latest trained artifact reports:

| Item | Value |
|------|-------|
| Training rows | 113,488 |
| Test rows | 28,297 |
| Test period | 2015-12-04 to 2026-07-12 |
| Decision threshold | 0.58 |

### Data science outputs

The data science layer produces the assets that the MLOps layer operates:

| Output | Purpose |
|--------|---------|
| `data/raw/weatherAUS.csv` | DVC-tracked raw/weather source used by training and Airflow retraining |
| `data/preprocessed/rain_model_dataset_aligned.csv` | DVC-tracked aligned modeling dataset used by drift comparison |
| `models/final_winner/winner_model.joblib` | DVC-tracked production model artifact |
| `models/final_winner/metadata.json` | Model contract, metrics, feature list, threshold, and training metadata |
| `models/final_winner/sample_input.json` | Valid example payload aligned with the 68-feature contract |
| `data/monitoring/reference_dataset.csv` | Training reference data written by training for drift comparison |

***

## Production MLOps Contribution

The MLOps contribution turns the data science result into an operating workflow.
Instead of only storing a model file, the project now has a reproducible path for extracting data, retraining, versioning artifacts, logging model metadata, checking drift, and running the system in Docker and Kubernetes.

The production layer focused on three responsibilities:

| Area | What was implemented |
|------|----------------------|
| DVC | Large data and model artifacts are tracked with DVC pointers and synced to the DagsHub DVC remote. |
| Airflow | Four DAGs coordinate ingestion, retraining, versioning, MLflow logging, and drift monitoring. |
| Kubernetes | Airflow, FastAPI integration, Postgres, Redis, Pushgateway, PVCs, HPAs, and PDBs are deployed through `kubernetes/kustomization.yaml`. |

Final validation confirmed:

| Check | Result |
|-------|--------|
| DVC remote status for committed raw/model pointers | Clean and synced |
| Docker Airflow health | Scheduler and metadatabase healthy |
| Docker Airflow DAG imports | No import errors |
| Kubernetes manifest dry-run | Passed server-side dry-run |
| Kubernetes pods | Airflow, Postgres, Redis, Pushgateway, and API pods running |
| Kubernetes PVCs | Airflow project/logs, Postgres, and Redis PVCs bound |
| Kubernetes Airflow DAG imports | No import errors |
| In-cluster service reachability | Airflow can reach FastAPI and Pushgateway |

***

## Repository Layout

```text
rain_prediction_mlops/
├── .github/
│   └── workflows/
├── data/
│   ├── cleaned/
│   ├── preprocessed/
│   ├── raw/
│   └── sample/
├── models/
│   └── final_winner/
├── notebooks/
├── references/
├── reports/
│   └── figures/
├── src/
│   ├── config/
│   ├── data/
│   ├── docker/
│   │   ├── data-download-prep/
│   │   ├── database/
│   │   ├── frontend/
│   │   ├── gateway/
│   │   ├── initialization/
│   │   ├── prediction/
│   │   ├── scoring/
│   │   ├── testing/
│   │   ├── training/
│   │   └── users/
│   ├── features/
│   ├── models/
│   │   └── experiments/
│   ├── script/
│   └── utils/
└── requirements.txt
```

### Key directories

| Path | Purpose |
|------|---------|
| `src/config/` | Shared paths and project constants |
| `src/docker/prediction/` | Prediction API service |
| `src/docker/training/` | Winner model training service |
| `src/docker/airflow/` | Airflow service for versioning and orchestration |
| `src/docker/frontend/` | Streamlit dashboard service |
| `src/docker/testing/` | API and inference validation |
| `src/models/` | Training, inference, API logic, utilities, experiments |
| `src/script/` | Project automation scripts |
| `data/raw/` | Original weather dataset |
| `data/preprocessed/` | Modeling tables and feature-related artifacts |
| `data/sample/` | Sample payloads for the API |
| `models/final_winner/` | Final served model artifact and metadata |
| `references/` | Climate references, station data, and notes |
| `src/monitoring/` | Evidently drift report generation |
| `data/monitoring/` | DVC-tracked training reference dataset for drift comparisons |

***

## Served Model

The repository serves a single final winner model for rainfall prediction.
This artifact is the current production candidate used by the API, Airflow validation, DVC versioning, and Kubernetes deployment.

| Item | Value |
|------|-------|
| Model | Final hybrid CatBoost |
| Artifact | `models/final_winner/winner_model.joblib` |
| Feature count | 68 |
| Target | `rain_tomorrow` |
| Threshold | 0.58 |
| Holdout ROC-AUC | 0.8945 |
| Holdout F1-score | 0.6839 |
| Holdout precision | 0.6386 |
| Holdout recall | 0.7361 |
| Training rows | 113,488 |
| Test rows | 28,297 |
| Test period | 2015-12-04 to 2026-07-12 |

The threshold is fixed in the metadata so the API, Airflow retraining checks, and monitoring reports use the same decision boundary.
The final model metadata also stores the feature list, categorical feature list, numeric fill values, CatBoost parameters, sample input, and artifact paths.

***

## API

### Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Service status, API version, loaded model |
| GET | `/model-info` | Model metadata, threshold, metrics, required features |
| POST | `/predict` | Rain / no-rain prediction with class probabilities |

### Example workflow

1. Start the local stack.
2. Check service health.
3. Send a prediction payload.

```bash
# VM Bash
curl http://localhost:8502/health
```

***

## Quick Start

### 1. Clone the repository

```bash
# VM Bash
git clone <your-repository-url>
cd rain_prediction_mlops
```

### 2. Create and activate a virtual environment

```bash
# VM Bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
# VM Bash
pip install -r requirements.txt
```

### 4. Create `.env`

```bash
# VM Bash
cp .env.example .env
```

Example values:

```env
MLFLOW_TRACKING_URI=https://dagshub.com/Inesgas/rain_prediction_MLops.mlflow
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
API_USERS=<username1:role1,username2:role2,...>
```

### 5. Load environment variables safely

```bash
# VM Bash
set -a
source .env
set +a
```

> Do **not** use `export $(grep -v '^#' .env | xargs)` because unquoted values may break parsing.

***

## Docker Compose

Docker Compose is the fastest way to run the full local stack for development, testing, monitoring, and API access.

### Start the stack

```bash
# VM Bash
docker compose up -d --build
docker compose ps
```

### Main local URLs

| Service | URL |
|---------|-----|
| Nginx gateway | `https://localhost` |
| Airflow | `http://localhost:8080` |
| MLflow | `http://localhost:5000` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` |

### Scale FastAPI behind Nginx

```bash
# VM Bash
docker compose up -d --scale fastapi=3
```

### Prediction traffic generator

The stack includes `prediction-traffic`, which periodically sends valid predictions so monitoring dashboards display data immediately.

```bash
# VM Bash
docker compose logs -f prediction-traffic
```

```bash
# VM Bash
docker compose stop prediction-traffic
```

***

## Kubernetes

Kubernetes is the production-style deployment layer for the project.
The Kubernetes work focused on making Airflow and the model-serving stack run as separated services with persistent state, predictable scheduling, autoscaling hooks, and repeatable manifests.

The validation environment used **Docker Desktop Kubernetes**, not K3s.
The manifests are still portable to K3s, Minikube, or a VM cluster, but non-Docker-Desktop clusters must receive the locally built images through either image loading or a registry.

### Implemented resources

| Resource | Role in the project |
|----------|---------------------|
| `rain-prediction-api` | FastAPI model-serving deployment integrated into the Kubernetes stack |
| `rain-prediction-airflow-webserver` | Airflow UI/API layer with 3 replicas |
| `rain-prediction-airflow-scheduler` | Single stable scheduler replica to avoid duplicate scheduler races |
| `rain-prediction-airflow-worker` | Celery workers with HPA support |
| `airflow-postgres` | StatefulSet for Airflow metadata database |
| `airflow-redis` | StatefulSet for Celery broker |
| `airflow-migrate-inesgas-20260713` | Airflow database migration and admin-user setup job |
| `pushgateway` | Batch metric target for the drift-monitoring DAG |
| PVCs | Persistent project workspace, Airflow logs, Postgres data, and Redis data |
| HPAs | API and Airflow worker autoscaling rules |
| PDBs | Availability protection for API, webserver, scheduler, and workers |

### Persistence design

Kubernetes uses persistent volumes for state that must survive pod replacement:

| PVC | Purpose |
|-----|---------|
| `airflow-project` | Shared project workspace used by Airflow tasks |
| `airflow-logs` | Airflow task and scheduler logs |
| `postgres-data-airflow-postgres-0` | Airflow metadata database storage |
| `redis-data-airflow-redis-0` | Redis broker persistence |

The Airflow deployments also refresh code from the image into the PVC-backed workspace at pod startup.
This prevents stale PVC contents from hiding new DAG support code after a rollout, while still avoiding accidental overwrite of trained data and model outputs.

### Portability design

The Kubernetes manifests use local image names for the Airflow, FastAPI, and model-fetcher containers.
For a fresh VM or another local Kubernetes cluster, the important portability requirements are:

- DVC-tracked raw data and model artifacts must exist locally before image build or must be available through the DVC remote.
- The `dagshub-credentials` secret must exist in the `rain-prediction` namespace so the model-fetcher init container can pull the model artifact.
- Non-Docker-Desktop clusters must use image loading or registry-published image tags.
- The Airflow image tag used by the manifests is `rain_prediction_mlops-airflow:inesgas-airflow-20260713`.

### Validated Kubernetes state

The Kubernetes layer was checked after the Airflow/DVC fixes.
The server-side manifest dry-run passed, pods were ready, PVCs were bound, and the missing DagsHub secret was created.

Validated runtime state:

| Check | Result |
|-------|--------|
| Manifest validation | `kubectl apply -k kubernetes --dry-run=server` passed |
| Airflow scheduler | `1/1` ready |
| Airflow webserver | `3/3` ready |
| Airflow workers | `2/2` ready, HPA configured for 2-3 |
| FastAPI | `3/3` ready |
| Postgres | `1/1` ready |
| Redis | `1/1` ready |
| Pushgateway | `1/1` ready |
| PVCs | Airflow, Postgres, and Redis PVCs bound |
| DAG imports | No Kubernetes Airflow import errors |
| In-cluster service checks | Airflow reached FastAPI `/health` and Pushgateway `/-/healthy` |

***

## Airflow Pipelines

Airflow is the orchestration layer for the production workflow.
It connects data ingestion, retraining, DVC versioning, MLflow logging, and drift monitoring into scheduled and repeatable DAGs.

The Airflow configuration was changed from a manual-only setup into an automated workflow.
The DAGs are unpaused, scheduled, and visible in both Docker Airflow and Kubernetes Airflow.

### Implemented DAGs

| DAG | Production role | Status |
|-----|-----------------|--------|
| `daily_weather_ingestion` | Daily ingestion entry point for WeatherAUS-style incoming data | Loaded and unpaused |
| `end_to_end_mlops_pipeline` | Full extract, version, train, log, validate workflow | Loaded, unpaused, and scheduled |
| `data_model_versioning` | DVC-aware data/model versioning and MLflow model logging | Loaded and unpaused |
| `drift_monitoring` | Evidently reference-vs-current drift check with Pushgateway metrics | Loaded and unpaused |

### `end_to_end_mlops_pipeline`

The end-to-end DAG is the main production storyline.
It starts from weather data extraction and finishes with refreshed model artifacts and validation outputs.

What the DAG does:

1. Extracts or updates `data/raw/weatherAUS.csv`.
2. Applies append/upsert logic using `Date` and `Location` so incoming rows can update existing weather observations without blindly duplicating records.
3. Versions the raw data pointer with DVC.
4. Trains the final winner model using the current raw dataset.
5. Rebuilds the aligned feature set and chronological train/test split.
6. Writes the final model, metadata, sample input, and monitoring reference dataset.
7. Versions the model artifact with DVC.
8. Logs model metadata and metrics to MLflow.
9. Validates that the expected artifacts exist for serving and monitoring.

### `data_model_versioning`

This DAG focuses on the versioning and audit trail around the training pipeline.
It records the state of the input data, the output model, and the model metadata so that a training run can be traced after it finishes.

Pipeline steps:

1. Extract or validate `data/raw/weatherAUS.csv`
2. Update the local DVC pointer for the raw dataset
3. Verify required DVC-tracked input files exist locally
4. Write an input version manifest to `reports/versioning/`
5. Train the final winner model
6. Update the local DVC pointer for `models/final_winner/winner_model.joblib`
7. Write the output version manifest
8. Log parameters, metrics, metadata artifacts, and model artifacts to MLflow
9. Write the DVC status manifest

### `drift_monitoring`

This DAG runs an Evidently data drift check between the training reference dataset (`data/monitoring/reference_dataset.csv`, written by `train_winner.py` on every training run) and a rolling window of `data/preprocessed/rain_model_dataset_aligned.csv`.
The monitoring window was set to cover a full seasonal cycle because short windows can falsely report complete drift when they compare one season against a year-round reference.

Pipeline steps:

1. Verify the DVC-tracked reference dataset is present locally
2. Run `src.monitoring.drift_report` over the configured window and log the HTML report to MLflow
3. Write `reports/monitoring/drift_<timestamp>.html` and `drift_<timestamp>_summary.json`
4. Push summary drift metrics to Pushgateway for dashboard/alert integration

### Airflow validation

The Airflow layer was validated in both Docker Compose and Kubernetes.

| Validation | Result |
|------------|--------|
| Docker Airflow health | Scheduler and metadatabase healthy |
| Docker Airflow DAG imports | No import errors |
| Docker Airflow DAG list | Four expected DAGs loaded and unpaused |
| Docker Airflow DVC check | Raw/model DVC targets up to date inside the Airflow runtime |
| Docker Airflow MLflow check | Model logging dry-run found the model artifact, metadata, config, and sample input |
| Kubernetes Airflow DAG imports | No import errors |
| Kubernetes Airflow DAG list | Four expected DAGs loaded and unpaused |
| Kubernetes Airflow DVC check | Raw/model DVC targets up to date inside the scheduler runtime |
| Kubernetes service check | Airflow reached FastAPI `/health` and Pushgateway `/-/healthy` |

***

## MLflow and DVC

### DVC

DVC is used because the project contains large assets that should not be stored directly in Git.
Git stores the `.dvc` pointer files, while DVC stores the actual raw data and model objects in the configured DagsHub remote.

Configured remote:

```text
https://dagshub.com/Inesgas/rain_prediction_MLops.dvc
```

Tracked production assets:

| DVC target | Purpose |
|------------|---------|
| `data/raw/weatherAUS.csv.dvc` | Raw weather dataset used as the training source |
| `data/preprocessed/rain_model_dataset_aligned.csv.dvc` | Aligned feature table used for monitoring comparison |
| `data/monitoring/reference_dataset.csv.dvc` | Drift reference dataset produced by training |
| `models/final_winner/winner_model.joblib.dvc` | Final served model artifact |

The latest committed raw data pointer and winner model pointer were pushed to the remote before merge.
This means a fresh machine can resolve the committed raw/model artifacts from DagsHub.
The local training workflow can also produce a newer `data/monitoring/reference_dataset.csv`; that file should only receive a new DVC pointer after the object has been successfully uploaded to the remote.

### MLflow

MLflow records model metadata, metrics, parameters, and artifacts.
The project supports two tracking targets:

- Local Docker MLflow for Airflow-triggered development and demonstration runs
- DagsHub MLflow for hosted experiment tracking

The Airflow integration was adjusted so local Docker Airflow uses the Docker MLflow service by default.
This avoids accidental DagsHub `403` failures when a teammate does not have hosted MLflow credentials configured.

| `AIRFLOW_MLFLOW_TRACKING_URI` | Airflow-triggered runs appear in | Manual training runs appear in |
|-------------------------------|----------------------------------|--------------------------------|
| `http://mlflow:5000` (default) | Local MLflow UI | DagsHub |
| `https://dagshub.com/Inesgas/rain_prediction_MLops.mlflow` | DagsHub Experiments | DagsHub |

MLflow dry-run validation was executed in Docker Airflow and Kubernetes Airflow.
Both runtimes found the model artifact, metadata, config, and sample input and produced a valid logging payload.

***

## Data Ingestion

The ingestion layer was designed so new WeatherAUS-format data can enter the same path as the original training data.
Incoming rows are not treated as a separate prediction-only file.
They are merged into the raw dataset and then included in the next Airflow-driven training cycle.

### Supported incoming data modes

The extractor supports several sources:

- A local CSV or ZIP file placed in the project
- A direct online CSV or ZIP URL
- A Kaggle WeatherAUS-style dataset source
- Open-Meteo daily ingestion for recent observations

The important production behavior is the same across sources:

1. The extractor normalizes the incoming data into the expected weather schema.
2. The update is merged into `data/raw/weatherAUS.csv`.
3. Append/upsert logic uses `Date` and `Location` so repeated observations update the existing row instead of creating uncontrolled duplicates.
4. Airflow versions the raw dataset pointer with DVC.
5. Training reloads the full raw dataset and rebuilds the modeling features.

### How new data reaches training and test

After new rows are extracted, they follow the normal training path.
The training module reloads `data/raw/weatherAUS.csv`, rebuilds features, and creates chronological splits.

The split behavior is:

| Split | Meaning |
|-------|---------|
| Training portion | Older observations used to learn model parameters |
| Validation portion | Newest part inside the training portion used during model selection/calibration |
| Test portion | Newest observations reserved for final evaluation |

Because the split is chronological, newly extracted rows generally enter the newest side of the dataset.
They can become part of the test window first, and as more future data arrives, the split boundary moves and older new rows can move into training.

Each successful training run writes:

| Artifact | Purpose |
|----------|---------|
| `models/final_winner/winner_model.joblib` | Updated winner model |
| `models/final_winner/metadata.json` | Metrics, threshold, features, split dates, and model parameters |
| `models/final_winner/sample_input.json` | Valid example input for API/testing |
| `data/monitoring/reference_dataset.csv` | Drift reference dataset for Evidently |

***

## Local Credentials

> Replace placeholder credentials before shared or production use.

| Service | Deployment | URL | Username | Password | Notes |
|---------|------------|-----|----------|----------|-------|
| Airflow | Docker Compose root stack | `http://localhost:8080` | `admin` | `airflow` | Defined in `docker-compose.yml` |
| Airflow | Legacy dev compose | `http://localhost:8080` | `airflow` | `airflow` | Defined in `src/docker/docker-compose-dev.yml` |
| Airflow | Kubernetes | `http://localhost:18080` via port-forward | `admin` | `airflow` | Placeholder in `kubernetes/airflow-secret.yaml` |
| MLflow | Docker Compose root stack | `http://localhost:5000` | none | none | Local tracking UI |
| Grafana | Docker Compose | `http://localhost:3000` | `admin` | `admin` | Defined in compose |
| Prometheus | Docker Compose | `http://localhost:9090` | none | none | Local monitoring only |
| Nginx gateway | Docker Compose | `https://localhost` | `andrey`, `ines`, `gunter`, `admin` | stored as hashes | See `nginx/.htpasswd` |

### Reset Kubernetes Airflow password

```bash
# VM Bash
kubectl exec -n rain-prediction deployment/rain-prediction-airflow-webserver -- \
airflow users reset-password -u admin -p "<new-password>"
```

***

## Troubleshooting

### Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ResolutionImpossible` for `mlflow-skinny` | Conflicting version pins | Keep only the pin matching the Dockerized MLflow version |
| `docker compose build airflow` fails with the same dependency conflict | Duplicate `mlflow-skinny` pin in Airflow requirements | Remove the conflicting duplicate pin |
| `dvc pull` or `dvc status` reports missing files | DVC remote credentials not configured | Configure DagsHub credentials in `.dvc/config.local` |
| MLflow connection errors to DagsHub | Missing `https://` scheme | Use the full tracking URI with `https://` |
| Airflow `PermissionError` on mounted paths | UID mismatch between host and container | Set `AIRFLOW_UID=$(id -u)` and recreate containers |
| `AirflowTimetableInvalid` for schedule | Stale exported environment variable overrides `.env` | `unset MODEL_VERSIONING_SCHEDULE` before startup |
| Airflow run not visible on DagsHub | Airflow still points to local MLflow | Set `AIRFLOW_MLFLOW_TRACKING_URI` explicitly |
| `ModuleNotFoundError: No module named 'evidently'` in `run_drift_report` | `evidently`/`plotly` missing from the Airflow image's requirements file | Add `evidently==0.7.21` and `plotly==5.24.1` to `docker/airflow/airflow_requirements.txt` and `src/docker/airflow/airflow_requirements.txt`, then rebuild the Airflow image |
| Airflow image build fails with `ResolutionImpossible` on `cryptography` | `evidently` needs `cryptography>=43.0.1`, which conflicts with the Airflow 2.10.5 constraints file (`cryptography==42.0.8`) | Install `evidently`/`plotly` in a separate `pip install` step without `--constraint`, see `docker/airflow/airflow.Dockerfile` |
| `drift_monitoring` reports `dataset_drift: true` on every run, 100% of columns | Comparison window too short (e.g. 14 days) captures one season only, while the reference dataset spans a full year | Use `--days-back 365` (the default) so the window covers a full seasonal cycle |


### Helpful diagnostics

```bash
# VM Bash
docker exec airflow id
docker exec airflow env | grep -E 'MLFLOW|MODEL_VERSIONING'
```

### DVC remote authentication example

```bash
# VM Bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <dagshub-username>
dvc remote modify origin --local password <dagshub-token>
```

***

## Security Notes

- `.env` is git-ignored and must never be committed
- Only `.env.example` with placeholder values should be tracked
- Replace local default credentials before shared use
- Store DVC credentials in `.dvc/config.local`, not in versioned files
- Plaintext passwords cannot be recovered from `nginx/.htpasswd`

***

## Project Notes

### Local-first workflow

This project is intentionally local-first.
Airflow DAGs update local files, local DVC metadata, local reports, and the configured MLflow target, but they do **not** push Git commits or DVC objects automatically to GitHub or DagsHub.

### Team integration

Andrey's FastAPI and Nginx implementation remains the official API and gateway layer.
The integration work extends orchestration, monitoring, Docker Compose, and Kubernetes support without replacing the API behavior or the gateway security design.

| Area | Role |
|------|------|
| `src/prediction_api/main.py` | Official FastAPI application used by Docker, Airflow, CI, monitoring, and Kubernetes |
| `nginx/nginx.conf` | Security gateway for authentication, forwarded users, rate limiting, and protected API access |
| `docker-compose.yml` | Integrates FastAPI, Nginx, Prometheus, Grafana, Airflow, and prediction traffic |
| `tests/test_prediction_api_contract.py` | Lightweight contract test for Airflow and CI |
| `src/monitoring/drift_report.py` | Evidently data drift report generation, invoked by the `drift_monitoring` Airflow DAG |

***

## Suggested Next Improvements
- Add real CI status badges from GitHub Actions
- Add an architecture diagram in `docs/`
- Add example request and response payloads for `/predict`
- Add a `Makefile` for common workflows
- Add a `Contributing` section for onboarding new collaborators
- Fail the `drift_monitoring` DAG task when `dataset_drift` is `true`, instead of only logging it
- Point `drift_monitoring`'s current-window comparison at a continuously updated data source instead of the static `rain_model_dataset_aligned.csv`
