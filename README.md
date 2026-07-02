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

This repository provides an end-to-end **MLOps workflow** for Australian weather rainfall prediction.
It serves the final **CatBoost winner model** through a FastAPI-based prediction service and connects training, versioning, tracking, orchestration, monitoring, and deployment into one local-first project.

### What this project covers

- Model training and evaluation
- Batch and API-based prediction serving
- Data and model versioning with **DVC**
- Experiment tracking with **MLflow**
- API routing and protection through **Nginx**
- Workflow orchestration with **Airflow**
- Monitoring via **Prometheus** and **Grafana**, plus data drift monitoring with **evidently**
- Local development with **Docker Compose**
- Production-style deployment with **Kubernetes**

***

## Architecture

The repository is organized as a microservice-oriented MLOps stack.
The main interaction path is shown below.

```text
Weather data -> preprocessing -> model training -> model artifact
      |               |                |                |
      v               v                v                v
     DVC         feature tables     MLflow        FastAPI service
                                                      |
                                                      v
                                                   Nginx gateway
                                                      |
                                                      v
                                          Monitoring / clients / tests
                                                      ^
                                                      |
                                    Evidently drift reports (reference vs. current data)
```

### Deployment modes

| Mode | Purpose | Main Entry Point |
|------|---------|------------------|
| Docker Compose | Local development and integrated service testing | `docker-compose.yml` |
| Kubernetes | Production-style local deployment with scaling and service separation | `kubernetes/kustomization.yaml` |

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

| Item | Value |
|------|-------|
| Model | Final hybrid CatBoost |
| Artifact | `models/final_winner/winner_model.joblib` |
| Feature count | 68 |
| Target | `rain_tomorrow` |
| Threshold | 0.58 |
| Holdout ROC-AUC | 0.9016 |
| Holdout F1-score | 0.6853 |
| Holdout precision | 0.6302 |
| Holdout recall | 0.7510 |

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

Kubernetes provides a more production-style setup with replicated services and autoscaling.

### Included resources

- FastAPI deployment with 3 baseline replicas and HPA up to 6 replicas
- Airflow webserver deployment with 2 replicas
- Airflow scheduler deployment with 2 replicas
- Airflow Celery workers with autoscaling
- Postgres StatefulSet for Airflow metadata
- Redis StatefulSet for Celery broker
- Airflow migration job
- PersistentVolumeClaims
- PodDisruptionBudgets

### Build images first

```bash
# VM Bash
docker build -f docker/prediction-api/api.Dockerfile -t rain_prediction_mlops-fastapi:latest .
docker build -f docker/airflow/airflow.Dockerfile -t rain_prediction_mlops-airflow:latest .
docker tag rain_prediction_mlops-airflow:latest rain_prediction_mlops-airflow:production-dvc
```

### Apply manifests

```bash
# VM Bash
kubectl apply -k ./kubernetes
kubectl get deploy,statefulset,job,hpa,pdb,svc,pvc -n rain-prediction
kubectl get pods -n rain-prediction
kubectl top pods -n rain-prediction
```

### Expected health status

| Resource | Expected |
|----------|----------|
| `rain-prediction-api` | `3/3` |
| `rain-prediction-airflow-webserver` | `2/2` |
| `rain-prediction-airflow-scheduler` | `2/2` |
| `rain-prediction-airflow-worker` | at least `2/2`, can scale to `3/3` |
| `airflow-postgres` | `1/1` |
| `airflow-redis` | `1/1` |
| `airflow-migrate` | `Complete` |

### Port-forward checks

```bash
# VM Bash
kubectl port-forward svc/rain-prediction-api 18502:8502 -n rain-prediction
curl http://localhost:18502/health
```

```bash
# VM Bash
kubectl port-forward svc/rain-prediction-airflow 18080:8080 -n rain-prediction
curl http://localhost:18080/health
```

***

## Airflow Pipelines

### `data_model_versioning`

This DAG performs local-only data and model versioning with DVC.

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

Default schedule:

```text
MODEL_VERSIONING_SCHEDULE=0 */6 * * *
```

This runs every 6 hours.


### `drift_monitoring`

This DAG runs an Evidently data drift check between the training reference dataset (`data/monitoring/reference_dataset.csv`, written by `train_winner.py` on every training run) and a rolling window of `data/preprocessed/rain_model_dataset_aligned.csv`.

Pipeline steps:

1. Verify the DVC-tracked reference dataset is present locally
2. Run `src.monitoring.drift_report` over the configured window and log the HTML report to MLflow
3. Write `reports/monitoring/drift_<timestamp>.html` and `drift_<timestamp>_summary.json`

Default schedule:

```text
DRIFT_MONITORING_SCHEDULE=0 6 * * *
```


### Start Airflow with local MLflow

```bash
# VM Bash
docker compose up -d --build mlflow airflow
```

### Access

| Service | URL | Username | Password |
|---------|-----|----------|----------|
| Airflow | `http://localhost:8080` | `admin` | `airflow` |
| MLflow | `http://localhost:5000` | none | none |

***

## MLflow and DVC

### DVC

Large datasets and the served model artifact are tracked with DVC.

Configured remote:

```text
https://dagshub.com/Inesgas/rain_prediction_MLops.dvc
```

### MLflow

Model training runs are tracked with MLflow and can target either local MLflow or DagsHub.

Configured hosted tracking server:

```text
https://dagshub.com/Inesgas/rain_prediction_MLops.mlflow
```

### Where Airflow runs appear

| `AIRFLOW_MLFLOW_TRACKING_URI` | Airflow-triggered runs appear in | Manual training runs appear in |
|-------------------------------|----------------------------------|--------------------------------|
| `http://mlflow:5000` (default) | Local MLflow UI | DagsHub |
| `https://dagshub.com/Inesgas/rain_prediction_MLops.mlflow` | DagsHub Experiments | DagsHub |

### Check the active target

```bash
# VM Bash
grep AIRFLOW_MLFLOW_TRACKING_URI .env
docker exec airflow env | grep MLFLOW_TRACKING_URI
```

### Point Airflow to DagsHub

```bash
# VM Bash
docker compose down
docker compose up -d --build --force-recreate mlflow airflow
```

### Manual training

```bash
# VM Bash
python -m src.models.train_winner
```

***

## Data Ingestion

The project supports multiple ingestion modes for WeatherAUS-format data.

### Local CSV or ZIP

```powershell
# lokale PowerShell
$env:WEATHER_AUS_SOURCE="data/incoming/weatherAUS_new.csv"
$env:WEATHER_AUS_EXTRACT_MODE="upsert"
docker compose -f src/docker/docker-compose-dev.yml up --build airflow
```

### Direct online CSV or ZIP

```powershell
# lokale PowerShell
$env:WEATHER_AUS_SOURCE_URL="https://example.com/weatherAUS.csv"
$env:WEATHER_AUS_SOURCE_SHA256="<optional expected sha256>"
$env:WEATHER_AUS_EXTRACT_MODE="upsert"
docker compose -f src/docker/docker-compose-dev.yml up --build airflow
```

### Kaggle source

```powershell
# lokale PowerShell
$env:KAGGLE_USERNAME="<your username>"
$env:KAGGLE_KEY="<your api key>"
$env:WEATHER_AUS_KAGGLE_DATASET="jsphyg/weather-dataset-rattle-package"
$env:WEATHER_AUS_KAGGLE_FILE="weatherAUS.csv"
$env:WEATHER_AUS_EXTRACT_MODE="replace"
docker compose -f src/docker/docker-compose-dev.yml up --build airflow
```

### Open-Meteo daily ingestion

```powershell
# lokale PowerShell
$env:WEATHER_AUS_DAILY_PROVIDER="open-meteo"
$env:WEATHER_AUS_EXTRACT_MODE="upsert"
$env:WEATHER_AUS_DAILY_DAYS_BACK="7"
$env:WEATHER_AUS_DAILY_END_LAG_DAYS="1"
docker compose -f src/docker/docker-compose-dev.yml up --build airflow
```

***

## Local Credentials

> Replace placeholder credentials before shared or production use.

| Service | Deployment | URL | Username | Password | Notes |
|---------|------------|-----|----------|----------|-------|
| Airflow | Docker Compose root stack | `http://localhost:8080` | `admin` | `airflow` | Defined in `docker-compose.yml` |
| Airflow | Legacy dev compose | `http://localhost:8080` | `airflow` | `airflow` | Defined in `src/docker/docker-compose-dev.yml` |
| Airflow | Kubernetes | `http://localhost:18080` via port-forward | `admin` | `rain-airflow-admin-change-me` | Placeholder in `kubernetes/airflow-secret.yaml` |
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
