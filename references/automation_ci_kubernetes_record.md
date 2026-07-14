# Automation, CI, and Kubernetes Record

Date: 20 June 2026  
Repository: rain_prediction_mlops  
Scope: local workspace only

## Scheduled Data Decision

True streaming is not needed for the training dataset.

The project target is `RainTomorrow`. That label is only confirmed after the following day has happened. A live stream cannot create a correct training label at the moment the weather row arrives.

The project now uses API-based scheduled ingestion:

- Open-Meteo API extraction
- Airflow scheduling
- local upsert into `data/raw/weatherAUS.csv`
- local DVC versioning
- a freshness report after ingestion

This gives automated new-data ingestion without adding Kafka, Spark Streaming, or another streaming platform outside the agreed training methods.

The local freshness check on 18 June 2026 found the current raw file ending at 2017-06-25. That confirms the file in the workspace is still the historical dataset until the Airflow API ingestion DAG is run with network access.

## Airflow Coverage

The project Airflow flow now covers:

1. online weather extraction from API
2. local raw data upsert
3. local DVC tracking for raw data
4. local artifact verification
5. data freshness check
6. input version snapshot
7. winner model training
8. local DVC tracking for the trained model
9. output version snapshot
10. official FastAPI contract validation
11. local DVC status report

The end-to-end DAG is:

`airflow/dags/end_to_end_mlops_dag.py`

The daily ingestion DAG is:

`airflow/dags/daily_weather_ingestion_dag.py`

Freshness reports are written by:

`src/data/check_data_freshness.py`

Reports are stored under:

`reports/versioning/`

No DagsHub push is performed by these DAGs.

## CI Pipeline

The GitHub Actions workflow is:

`.github/workflows/ci.yml`

The CI pipeline has two jobs.

`python-contract` checks:

- Python syntax for Airflow DAGs
- Python syntax for extraction and versioning modules
- Python syntax for the official FastAPI app
- the FastAPI contract test

The contract test skips model-loading assertions when the DVC model artifact is not present in CI.

`docker-build` checks:

- Docker Compose configuration
- FastAPI Docker image build
- Airflow Docker image build

The workflow does not push images.

## Kubernetes Scalability

Kubernetes manifests were added under:

`kubernetes/`

The FastAPI API is defined as:

- `kubernetes/fastapi-deployment.yaml`
- `kubernetes/fastapi-service.yaml`
- `kubernetes/fastapi-hpa.yaml`

The API starts with three replicas and can autoscale to six replicas based on CPU utilization.

Airflow was upgraded from a single demo-style pod to a production-style separated deployment:

- `kubernetes/airflow-postgres.yaml`
- `kubernetes/airflow-redis.yaml`
- `kubernetes/airflow-migrate-job.yaml`
- `kubernetes/airflow-deployment.yaml`
- `kubernetes/airflow-scheduler-deployment.yaml`
- `kubernetes/airflow-worker-deployment.yaml`
- `kubernetes/airflow-worker-hpa.yaml`
- `kubernetes/airflow-pdb.yaml`
- `kubernetes/airflow-service.yaml`
- `kubernetes/airflow-pvc.yaml`
- `kubernetes/airflow-configmap.yaml`
- `kubernetes/airflow-secret.yaml`

The Airflow image includes the DAGs, project source code, data pointers, model files, references, tests, and the DVC workspace metadata needed for local DVC commands. The private `.dvc/config.local`, DVC cache, and DVC temporary files are excluded from the image.

Kubernetes seeds those files into a project PVC so the webserver, schedulers, and workers share the same project workspace. The Airflow init containers also keep the shared DVC config in local-only mode with `core.no_scm = true`, because the Kubernetes PVC is not a Git checkout and the project rule is not to push to GitHub or DagsHub.

The Airflow worker settings were adjusted after validation found two real runtime issues:

| Issue | Fix |
|---|---|
| A worker pod running `train_winner_model` was removed during HPA scale-down | Worker HPA scale-down stabilization was set to 30 minutes, with a one-pod scale-down policy. |
| The model-training process was OOMKilled at the old `1Gi` memory limit | Worker memory was raised to `1Gi` request and `3Gi` limit. |
| Multiple heavy tasks could run in the same worker pod | Celery worker concurrency was set to `1`. |

After these changes, the exact training command completed successfully inside the Kubernetes Airflow worker and wrote `winner_model.joblib`, `metadata.json`, and `sample_input.json`.

Current Airflow Kubernetes layout:

| Component | Replicas or Status |
|---|---|
| Airflow webserver | 2 replicas |
| Airflow scheduler | 2 replicas |
| Airflow Celery worker | 2 minimum, 3 maximum through HPA |
| Postgres metadata database | 1 StatefulSet pod |
| Redis Celery broker | 1 StatefulSet pod |
| Airflow migration job | Complete |
| Airflow logs | PVC |
| Airflow project workspace | PVC |

The Kubernetes Airflow login is:

| Username | Password |
|---|---|
| `admin` | `airflow` |

This password is a local placeholder stored in `kubernetes/airflow-secret.yaml`. Shared environments use a team-managed secret instead of this local placeholder.

The Kubernetes bundle is listed in:

`kubernetes/kustomization.yaml`

API security hardening and MLflow tracking are left for the teammates responsible for those parts.

## Andrey Integration Notes

Andrey's FastAPI and Nginx files were touched only where orchestration and deployment needed to connect to them.

| Area | What changed | Why it was needed |
|---|---|---|
| FastAPI | Treated `src/prediction_api/main.py` as the official API target for Airflow, CI, Docker, and Kubernetes. | The project needed one API contract instead of two competing API entry points. |
| Nginx | Kept Basic Auth, forwarded user handling, and rate limiting; aligned upstream routing with Docker service discovery. | Scaled FastAPI containers need stable service-name routing from Nginx. |
| API tests | Added a local FastAPI contract test. | Airflow and CI need a fast test that can run without the full HTTPS Nginx stack. |
| Monitoring | Connected Prometheus and Grafana to the FastAPI metrics endpoint. | The monitoring dashboard depends on metrics emitted by the official API. |
| Docker/Kubernetes | Built and deployed the official FastAPI app rather than the older API draft. | The deployment needed to reflect the API chosen by the team. |

The integration work did not replace Andrey's API or security design. It made those components usable from the automated pipeline and local deployment stack.

## Local Verification

Python syntax checks passed for the Airflow DAGs, extraction modules, versioning module, official FastAPI module, and FastAPI contract test.

Docker Compose configuration validation passed.

Docker Compose monitoring was updated with a `prediction-traffic` service. This local service waits for FastAPI to become healthy, loads all supported locations from `/locations`, and sends one valid batch prediction for every location once per minute on the internal Docker network. Prometheus confirmed `54` distinct `rain_predictions_total` location labels, so Grafana dashboards show data immediately when opened.

The Grafana `Predictions by City` query was changed from a top-10 query to `sum(rain_predictions_total) by (location)` so all supported locations can appear in the dashboard data.

Kubernetes validation passed locally with Docker Desktop Kubernetes.

Final Kubernetes status:

| Resource | Status |
|---|---|
| `deployment/rain-prediction-api` | `3/3` |
| `deployment/rain-prediction-airflow-webserver` | `2/2` |
| `deployment/rain-prediction-airflow-scheduler` | `2/2` |
| `deployment/rain-prediction-airflow-worker` | `2/2` baseline, HPA to `3` |
| `statefulset/airflow-postgres` | `1/1` |
| `statefulset/airflow-redis` | `1/1` |
| `job/airflow-migrate` | `Complete` |
| FastAPI HPA | min `3`, max `6` |
| Airflow worker HPA | min `2`, max `3` |

Service health checks passed through Kubernetes port-forward:

| Service | Result |
|---|---|
| FastAPI `/health` | healthy, model loaded, 68 features |
| Airflow `/health` | metadatabase healthy, scheduler healthy |

Airflow DVC validation passed inside the Kubernetes worker:

| Check | Result |
|---|---|
| `.dvc/config` exists in `/opt/airflow/project` | Passed |
| `dvc status` recognizes the workspace | Passed |
| Exact failed task command, `python -m src.versioning.dvc_versioning dvc-add --target data/raw/weatherAUS.csv` | Passed |

The final manifest dry run also passed:

`kubectl apply -k .\kubernetes --dry-run=client`
