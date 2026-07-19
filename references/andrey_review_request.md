# Review Request: Automated Model Rollout and Demo Stability

Hi Andrey,

I followed your automated model rollout note and added the missing pieces on this branch:

`inesgas/automated-model-rollout`

## What changed

- Airflow now has a DVC push step after the trained winner model is tracked.
- Airflow now triggers a Kubernetes rollout restart for `deployment/rain-prediction-api` after the model artifact is pushed.
- The restart uses the Kubernetes Python client from inside Airflow.
- Kubernetes RBAC is scoped to allow Airflow to patch only the `rain-prediction-api` deployment.
- The FastAPI/model-fetcher/shared-volume architecture was kept unchanged.

## Runtime fixes added after testing

- The model-fetcher script now uses `/bin/sh`, which is available in `python:3.10-slim`.
- The model-fetcher Dockerfile normalizes Windows line endings before running the script.
- Airflow init now removes stale seeded DVC files before pulling from DagsHub, so the PVC starts from the remote artifact state.
- Open-Meteo extraction now continues when only a small number of stations fail. It still fails if all locations fail or no rows are produced.

## Latest PR refresh

The latest pushed branch includes the fixes from the local Kubernetes demo debugging session:

- `docker/model-fetcher/fetch_model.sh`
- `docker/model-fetcher/model-fetcher.Dockerfile`
- `kubernetes/airflow-deployment.yaml`
- `kubernetes/airflow-scheduler-deployment.yaml`
- `kubernetes/airflow-worker-deployment.yaml`
- `kubernetes/airflow-migrate-job.yaml`
- `src/data/extract_open_meteo_daily.py`

These changes are intentionally on this review branch, not on `main`.

## What I verified locally

- Airflow has 4 active DAGs with no import errors:
  - `daily_weather_ingestion`
  - `data_model_versioning`
  - `drift_monitoring`
  - `end_to_end_mlops_pipeline`
- Kubernetes FastAPI pod is running and `/metrics` is reachable.
- Prometheus is scraping the current FastAPI pod successfully.
- MLflow now has a run and a registered model after logging from the current Kubernetes Airflow worker.
- The extractor was tested inside Airflow and continued successfully when one Open-Meteo location failed.

Could you please review whether this matches the rollout behavior you expected, especially the model-fetcher and API restart flow?
