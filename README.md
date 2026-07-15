# Rain Prediction MLOps

<p align="center">
  <b>Production-style MLOps for Australian rainfall prediction</b><br>
  FastAPI | Airflow | MLflow | DVC | Container Images | Kubernetes
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-API-009688">
  <img alt="MLflow" src="https://img.shields.io/badge/MLflow-Tracking-0194E2">
  <img alt="DVC" src="https://img.shields.io/badge/DVC-Versioning-13ADC7">
  <img alt="Container Images" src="https://img.shields.io/badge/Images-Packaging-2496ED">
  <img alt="Kubernetes" src="https://img.shields.io/badge/Kubernetes-Deployment-326CE5">
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
- [Environment and Reproducibility Context](#environment-and-reproducibility-context)
- [Local Integration Context](#local-integration-context)
- [Kubernetes](#kubernetes)
- [Airflow Pipelines](#airflow-pipelines)
- [DVC Artifact Versioning](#dvc-artifact-versioning)
- [Data Ingestion](#data-ingestion)
- [Access and Security Context](#access-and-security-context)
- [Resolved Integration Issues](#resolved-integration-issues)
- [Security Handling](#security-handling)
- [Project Notes](#project-notes)

***

## Overview

This repository presents a complete rainfall prediction project that starts as a data science problem and finishes as a production-style MLOps workflow.
The data science question is direct: **given Australian daily weather observations, can the project predict whether it will rain tomorrow?**
The production question is larger: **can the model, data, training process, artifact history, monitoring, and deployment be reproduced and operated as one system?**

The final system serves a **hybrid CatBoost rainfall classifier** through a FastAPI prediction service.
Around that model, the project connects DVC artifact versioning, MLflow tracking, Airflow automation, Evidently drift reporting, containerized services, and Kubernetes deployment.
The result is no longer only a trained notebook model: it is a reproducible workflow with a defined feature contract, scheduled retraining path, versioned data/model artifacts, runtime validation checks, and production-style service separation.

This README is the project report.
It records the work behind the model, the production architecture, the repository structure, the serving contract, and the validation evidence used to judge readiness.

### What this project covers

| Area | Final project content |
|------|-----------------------|
| Data science | Weather cleaning, feature engineering, chronological splitting, model comparison, final CatBoost training, and evaluation |
| Model artifact | DVC-tracked winner model with a 68-feature contract, metadata, sample input, threshold, and monitoring reference data |
| Artifact history | DVC pointers for raw data, processed data, monitoring reference data, and the final served model |
| Experiment tracking | MLflow logging payloads for model metrics, parameters, tags, and artifacts |
| Orchestration | Airflow DAGs for daily ingestion, end-to-end retraining, model/data versioning, and drift monitoring |
| Monitoring | Evidently drift reports, Pushgateway handoff, prediction traffic, and metrics integration points |
| Serving | FastAPI model service with health, prediction, batch prediction, metadata, feature, and metrics endpoints |
| Deployment | Kubernetes manifests for Airflow, FastAPI, Postgres, Redis, Pushgateway, PVCs, HPA/PDB resources, and services |
| Local integration | Docker Compose remains as a same-machine integration context for checking the service stack together |

***

## Architecture

The architecture is organized as a pipeline that moves from weather observations to a served, monitored, versioned model.
The first layer is the **data science foundation**, where the raw WeatherAUS-style observations are cleaned, enriched, split chronologically, trained, and evaluated.
The second layer is the **production MLOps layer**, where the data/model workflow is automated, versioned, tracked, monitored, packaged, and deployed.

The same artifact contract connects both layers.
Training writes the model, metadata, sample input, and reference dataset.
The API reads that contract for inference.
Airflow uses it for retraining and validation.
DVC versions it.
MLflow records it.
Evidently compares live/current data against it.
Kubernetes runs the services that operate it.

```text
Weather data sources
        |
        v
Raw WeatherAUS-style dataset
        |
        +-----------------------------+
        |                             |
        v                             v
Training data path             Production/inference data path
        |                             |
        v                             v
Cleaning + feature engineering  FastAPI prediction requests
        |
        v
Chronological train / validation / test split
        |
        v
Hybrid CatBoost winner model + metadata contract
        |
        +---------------------> FastAPI prediction service
        |
        +---------------------> MLflow metrics, params, tags, artifacts
        |
        +---------------------> DVC data/model artifact versions
        |
        v
Airflow automated production workflows
        |
        +---------------------> daily data extraction and merge
        +---------------------> end-to-end retraining and artifact refresh
        +---------------------> data/model versioning snapshots
        +---------------------> Evidently drift reports
        |
        v
Kubernetes production-style runtime
```

### Runtime roles

| Runtime area | Purpose | Main Entry Point |
|--------------|---------|------------------|
| Kubernetes | Production-style runtime with separated services, persistent Airflow state, service discovery, scaling hooks, and availability policies | `kubernetes/kustomization.yaml` |
| Airflow | Scheduled automation layer for ingestion, retraining, versioning, tracking handoff, API checks, and drift monitoring | `airflow/dags/` |
| DVC + DagsHub | Versioned external artifact history for large data/model files that should not live directly in Git | `.dvc/`, `*.dvc` pointers |
| MLflow | Experiment and model metadata tracking for training runs and logging dry-runs | `src/versioning/mlflow_tracking.py` |
| Evidently | Drift report generation from reference/current datasets and monitoring metrics handoff | `src/monitoring/drift_report.py` |
| Local integration context | Same-machine service validation where the repository stack needs multiple services started together | `docker-compose.yml` |

### Service flow

| Step | What happens | Main files |
|------|--------------|------------|
| 1. Data enters | Historical or incoming weather rows are normalized into the project schema and merged by `Date` + `Location` | `src/data/extract_weather_data.py`, `src/data/extract_open_meteo_daily.py` |
| 2. Data is prepared | Feature engineering rebuilds the model-ready table with weather, date, location, lag, missingness, and climate context | `src/features/`, `src/models/train_winner.py` |
| 3. Model is trained | The winner training path produces the CatBoost artifact, metadata, sample input, and monitoring reference dataset | `src/models/train_winner.py`, `models/final_winner/` |
| 4. Artifacts are tracked | DVC records the raw data, processed data, model, and monitoring reference pointers | `*.dvc`, `src/versioning/dvc_versioning.py` |
| 5. Runs are logged | MLflow receives metrics, parameters, tags, and artifact references | `src/versioning/mlflow_tracking.py` |
| 6. Service is validated | FastAPI loads the model contract and exposes health, prediction, metadata, and metrics endpoints | `src/prediction_api/main.py` |
| 7. Drift is checked | Evidently compares reference/current data and can push summary metrics to Pushgateway | `src/monitoring/drift_report.py` |
| 8. Runtime is deployed | Kubernetes runs the Airflow and serving stack with PVCs, services, HPAs, and PDBs | `kubernetes/` |

***

## Data Science Foundation

The data science stage transformed Australian weather observations into a supervised binary classification problem.
The target is `rain_tomorrow`.
Each row represents a daily station observation, and the model estimates the probability that the next day will be rainy for that location.

The important design choice was to make the modeling output reusable by production code.
The project therefore does not stop at a notebook score.
It produces a stable feature list, metadata file, decision threshold, sample input, categorical feature list, numeric fill values, and monitoring reference dataset.
Those outputs are the contract used later by FastAPI, Airflow, DVC, MLflow, Evidently, and Kubernetes.

### Data preparation and feature engineering

The raw table was reshaped into a production feature contract.
The preparation work included schema normalization, missing-value handling, temporal features, location enrichment, meteorological transformations, lag features, and categorical handling.

| Preparation area | What was built |
|------------------|----------------|
| Schema alignment | Weather rows are standardized around `Date`, `Location`, weather measurements, categorical weather fields, and the `rain_tomorrow` target |
| Missingness handling | Hybrid missingness flags preserve information from absent rainfall, sunshine, cloud, pressure, humidity, evaporation, and temperature values |
| Calendar features | Date is expanded into month/day/year and cyclic seasonal signals |
| Geographic context | Location is enriched with latitude, longitude, elevation, region/climate context, and rainfall-zone indicators |
| Wind encoding | Wind direction is converted into numerical direction vectors so circular direction values can be learned correctly |
| Weather dynamics | Temperature, humidity, pressure, rainfall, and previous-day changes are added to capture short-term atmospheric movement |
| Moisture/stability features | Dew point and stability-style variables represent humidity and atmospheric behavior beyond raw columns |
| API contract support | A valid `sample_input.json` is stored with the model so inference tests and demos use the same feature expectations as training |

### Modeling strategy

Several modeling directions were explored in the project, including baseline modeling, feature-aligned tabular modeling, location-aware refinements, missingness-aware experiments, rolling robustness validation, and sequence/deep-learning benchmarks.
The final served model is a **hybrid CatBoost classifier** because it fits the project data shape well: mixed numeric/categorical tabular weather features, missingness indicators, location effects, and nonlinear interactions.

The final contract contains **68 input features**.
It includes categorical features such as `location`, `humidity_9am_bin`, `pressure_9am_bin`, and `temp_9am_bin`, while the rest of the contract is numeric weather, calendar, location, lag, and engineered meteorological context.

The split strategy was chronological instead of random.
This is important for weather data because random splitting can leak future seasonal patterns into training.
The workflow uses older observations for training and keeps the newest observations for final testing, so the model is evaluated closer to how it would behave on future data.

The latest served artifact reports:

| Item | Value |
|------|-------|
| Training rows | 113,488 |
| Test rows | 28,297 |
| Test period | 2015-12-04 to 2026-07-12 |
| Decision threshold | 0.58 |
| ROC-AUC | 0.8945 |
| F1-score | 0.6839 |
| Precision | 0.6386 |
| Recall | 0.7361 |

The selected threshold prioritizes practical rain-event detection rather than only optimizing a default probability cutoff.
That threshold is stored in metadata and reused by serving and validation code.

### Data science outputs

The data science layer produces the assets that the production layer operates.
These files are treated as model-system artifacts, not temporary notebook outputs.

| Output | Purpose |
|--------|---------|
| `data/raw/weatherAUS.csv` | DVC-tracked raw/weather source used by training and Airflow retraining |
| `data/preprocessed/rain_model_dataset.csv` | Prepared modeling dataset from the feature pipeline |
| `data/preprocessed/rain_model_dataset_aligned.csv` | DVC-tracked aligned modeling dataset used by drift comparison and contract checks |
| `models/final_winner/winner_model.joblib` | DVC-tracked production model artifact |
| `models/final_winner/metadata.json` | Model contract, metrics, feature list, threshold, and training metadata |
| `models/final_winner/sample_input.json` | Valid example payload aligned with the 68-feature contract |
| `data/monitoring/reference_dataset.csv` | Training reference data written by training for drift comparison |
| `reports/versioning/` | Airflow/DVC manifests that record extraction, input, output, and status snapshots |

***

## Production MLOps Contribution

The production MLOps contribution turns the data science result into an operating workflow.
The model is not treated as a standalone file.
It is part of a system that extracts and merges data, retrains on schedule, versions artifacts, logs run metadata, validates serving behavior, checks drift, and runs through Kubernetes.

The production layer focused on three responsibilities:

| Area | What was implemented |
|------|----------------------|
| DVC | Large data/model files are represented by Git-tracked DVC pointers and stored outside Git through the configured DagsHub remote |
| Airflow | Four DAGs coordinate daily ingestion, end-to-end retraining, DVC versioning, MLflow handoff, API validation, and drift monitoring |
| Kubernetes | Airflow, FastAPI, Postgres, Redis, Pushgateway, PVCs, services, HPAs, and PDBs are represented through `kubernetes/kustomization.yaml` |

The work was not only a packaging step around a trained model.
The project was reorganized so that each production action leaves an auditable trace:

| Process area | Work behind the final state |
|--------------|-----------------------------|
| Raw data control | Incoming rows are normalized, validated, merged by `Date` and `Location`, and recorded through extraction manifests. |
| Artifact versioning | Raw data, model artifacts, feature metadata, sample input, and DVC status are recorded before and after training. |
| Automated orchestration | Airflow tasks were split into extraction, DVC pointer updates, local artifact checks, freshness checks, training, output snapshots, tracking handoff, API contract validation, and status recording. |
| Runtime separation | Kubernetes separates Airflow webserver, scheduler, workers, Postgres, Redis, Pushgateway, and service endpoints instead of running everything as one local process. |
| Portability | Container images, Airflow settings, Kubernetes manifests, DVC targets, and model artifact paths stay aligned so the workflow behaves consistently across machines and clusters. |
| Safety of automation | DAGs update local artifacts and reports, but remote DVC/Git publishing remains a deliberate merge-time action. |

### Implemented production workflow

| Workflow stage | Production behavior |
|----------------|---------------------|
| Daily extraction | Incoming weather data is normalized into the WeatherAUS-style schema and merged into `data/raw/weatherAUS.csv` |
| Upsert logic | Repeated station-date rows are updated by `Date` + `Location` instead of blindly duplicated |
| DVC input snapshot | Raw data and input manifests are checked before training |
| Training | The winner training module reloads the full raw dataset, rebuilds features, retrains the CatBoost model, and rewrites the served artifact set |
| MLflow handoff | Model metadata, metrics, parameters, tags, and artifacts are prepared for MLflow logging |
| DVC output snapshot | Model outputs and DVC status are recorded after training |
| API contract validation | The serving layer is checked against the model artifact and sample input contract |
| Drift monitoring | Evidently compares reference/current data and can expose summary metrics through Pushgateway |
| Kubernetes runtime | Airflow and service components run as separated Kubernetes workloads with persistent state and scaling/availability resources |

Final validation confirmed:

| Check | Result |
|-------|--------|
| DVC remote status for committed raw/model pointers | Clean and synced |
| Local Airflow health | Scheduler and metadatabase healthy |
| Local Airflow DAG imports | No import errors |
| Airflow schedules | E2E training, versioning, and drift-monitoring schedules are configured |
| Kubernetes manifest dry-run | Passed server-side dry-run |
| Kubernetes pods | Airflow, Postgres, Redis, Pushgateway, and API pods running |
| Kubernetes PVCs | Airflow project/logs, Postgres, and Redis PVCs bound |
| Kubernetes Airflow DAG imports | No import errors |
| In-cluster service reachability | Airflow can reach FastAPI and Pushgateway |
| Focused tests | Airflow, dependency pins, MLflow tracking, and drift-report tests pass together |

***

## Repository Layout

```text
rain_prediction_mlops/
|-- .github/
|   `-- workflows/               CI checks for Python contracts and container builds
|-- airflow/
|   `-- dags/                    Scheduled ingestion, retraining, versioning, and drift workflows
|-- data/
|   |-- cleaned/                 Local cleaned-data workspace
|   |-- incoming/                Local incoming-data landing area
|   |-- monitoring/              Drift reference/current data and prediction logs
|   |-- preprocessed/            Modeling datasets and aligned feature tables
|   |-- processed/               Processed reference/location metadata
|   |-- raw/                     DVC-tracked source weather data
|   `-- sample/                  API/demo sample payloads
|-- deployment/                  Monitoring deployment assets used by the local stack
|-- docker/
|   |-- airflow/                 Airflow image and dependency pins
|   |-- frontend/                Streamlit/frontend service material
|   |-- gateway/                 Gateway image material
|   |-- model-fetcher/           Model artifact fetcher image
|   |-- prediction-api/          Official FastAPI image
|   |-- testing/                 Containerized test helpers
|   |-- traffic-generator/       Synthetic prediction traffic for monitoring
|   `-- training/                Training image material
|-- kubernetes/                  Production-style Kubernetes manifests and kustomization
|-- models/
|   `-- final_winner/            Served model, metadata, and sample input
|-- nginx/                       Gateway configuration and local certificate placeholders
|-- notebooks/                   Data science exploration and modeling notebooks
|-- references/                  Climate/location references and validation records
|-- reports/
|   |-- figures/
|   `-- versioning/              Airflow/DVC manifest output area
|-- Slides/                      Final presentation deck artifacts
|-- src/
|   |-- config/                  Shared paths and constants
|   |-- data/                    Extraction, merge, and freshness modules
|   |-- features/                Feature engineering pipeline
|   |-- models/                  Training, inference, and experiment code
|   |-- monitoring/              Evidently drift reporting
|   |-- prediction_api/          FastAPI application
|   |-- script/                  Smoke-test and automation scripts
|   |-- utils/                   Shared helpers and validation utilities
|   `-- versioning/              DVC and MLflow integration code
|-- tests/                       Contract, orchestration, dependency, MLflow, and drift tests
|-- docker-compose.yml           Local integration context
`-- requirements.txt
```

### Key directories

| Path | Purpose |
|------|---------|
| `airflow/dags/` | Production DAG definitions for ingestion, retraining, versioning, and drift monitoring |
| `src/data/` | Data extraction, Open-Meteo ingestion, raw-data merge, and freshness validation |
| `src/features/` | Feature-building logic used by training and model preparation |
| `src/models/` | Winner training code, inference support, and modeling experiments |
| `src/prediction_api/` | FastAPI app that loads the final model and exposes prediction/metadata/metrics endpoints |
| `src/versioning/` | DVC snapshot logic and MLflow logging payload construction |
| `src/monitoring/` | Evidently drift report generation |
| `docker/airflow/` | Airflow Dockerfile and dependency pins aligned with drift-monitoring requirements |
| `docker/prediction-api/` | Container image definition for the official FastAPI service |
| `docker/model-fetcher/` | Kubernetes init-container image for retrieving DVC model artifacts |
| `kubernetes/` | Kustomize entry point plus deployments, services, PVCs, HPAs, PDBs, and stateful services |
| `data/raw/` | DVC-tracked source weather data used by training and Airflow |
| `data/preprocessed/` | Prepared feature tables and aligned modeling datasets |
| `data/monitoring/` | Reference/current data used for Evidently drift comparison |
| `models/final_winner/` | Final served model artifact, metadata, and sample input |
| `references/` | Climate references, station metadata, validation notes, and integration records |
| `reports/versioning/` | Extraction, freshness, input, output, and DVC-status manifests emitted by automation |
| `tests/` | Focused tests for Airflow automation, dependency pins, MLflow payloads, drift reporting, and API contract behavior |

***

## Served Model

The repository serves one final winner model for rainfall prediction.
This model is the production candidate used by the API, Airflow validation, DVC versioning, MLflow logging, Evidently monitoring, and Kubernetes deployment.
The served artifact is intentionally packaged with metadata and a sample input so every runtime reads the same contract.

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

### Served artifact package

| File | Role |
|------|------|
| `models/final_winner/winner_model.joblib` | Serialized CatBoost model used by FastAPI and validation checks |
| `models/final_winner/metadata.json` | Feature order, categorical features, numeric fill values, threshold, model parameters, metrics, split dates, and artifact paths |
| `models/final_winner/sample_input.json` | Valid example request body aligned with the model contract |
| `data/monitoring/reference_dataset.csv` | Reference dataset used by Evidently drift reports |

The threshold is fixed in metadata so the API, Airflow retraining checks, and monitoring reports use the same decision boundary.
The model metadata also stores the feature list, categorical feature list, numeric fill values, CatBoost parameters, sample input path, and artifact paths.

### Prediction contract

The prediction contract is stable around the 68-feature order stored in metadata.
FastAPI accepts a compact business-facing request with key weather fields, fills or derives the remaining model features from the saved contract where appropriate, and returns the binary rain decision with confidence.

| Contract element | Production use |
|------------------|----------------|
| Feature order | Keeps training, inference, batch prediction, tests, and monitoring aligned |
| Categorical feature list | Ensures CatBoost receives categorical columns consistently |
| Numeric fill values | Allows inference to handle absent optional fields without changing the model schema |
| Threshold | Converts model probability into the final `rain_tomorrow` decision |
| Sample input | Provides a known-good payload for API checks, Airflow validation, and presentation demos |

***

## API

The API layer is the serving layer for the trained model.
It exposes the model health, model metadata, and prediction behavior expected by the rest of the stack.
This report records how the MLOps workflow depends on the running API service.

### Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Public service health and model-loaded status |
| GET | `/locations` | Public list of supported WeatherAUS locations |
| POST | `/predict` | Authenticated single-row rain prediction with confidence |
| POST | `/predict/batch` | Authenticated batch prediction with prediction-log writing |
| GET | `/model/info` | Admin model metadata, feature count, and artifact path |
| GET | `/model/features` | Admin feature contract and numeric fill values |
| GET | `/metrics` | Prometheus metrics exposed by the API service |

### API role in the MLOps system

| Integration point | Role |
|-------------------|------|
| Local integration context | Runs the API service alongside the existing local services for same-machine checks |
| Kubernetes | Runs the API as a 3-replica deployment with HPA support |
| Airflow | Uses the API health endpoint as part of the production validation path |
| DVC/model artifacts | The API loads the DVC-tracked winner model and metadata contract |
| Monitoring | Prediction traffic and API metrics feed the monitoring layer |

***

## Environment and Reproducibility Context

The project was structured to be reproducible across a local workstation, a VM, and a local Kubernetes cluster.
Python dependency files, Dockerfiles, DVC pointers, Airflow DAGs, and Kubernetes manifests work together so that the same model artifacts and workflow can be inspected outside the original notebook environment.

The reproducibility design includes:

| Component | Reproducibility role |
|-----------|----------------------|
| `.env.example` | Documents the environment variables expected by local services without storing real secrets |
| `requirements.txt` | Captures shared Python dependencies for data science and validation code |
| Dockerfiles | Package API, Airflow, model-fetcher, gateway, frontend, training, and testing runtimes |
| DVC pointers | Keep raw data and model artifacts reproducible without committing large binary files to Git |
| Airflow DAGs | Preserve the production order of ingestion, training, versioning, logging, and monitoring |
| Kubernetes manifests | Describe the production-style runtime topology and persistent state |

The latest validation work confirmed that the committed raw dataset pointer and winner model pointer are present in the DagsHub DVC remote.
This keeps the merged Git state consistent with the external artifact store.

***

## Local Integration Context

Docker Compose exists in the repository as a local integration context.
It is useful for same-machine checks when the full service stack needs to be started together, but it is not the deployment target for the production-style architecture.
The deployment/runtime target described in this report is Kubernetes.

### Integrated services

| Service | URL |
|---------|-----|
| Nginx gateway | `https://localhost` |
| Airflow | `http://localhost:8080` |
| MLflow | `http://localhost:5000` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` |

### Local integration role in the project

| Area | What the local stack demonstrates |
|------|----------------------------------|
| FastAPI serving | The model API runs with the mounted model artifact and health checks |
| Airflow automation | Airflow runs the same DAGs used in Kubernetes validation |
| MLflow tracking | Local model logging is available without relying on hosted credentials |
| Pushgateway | Batch drift metrics have a local metrics target |
| Prediction traffic | Synthetic prediction traffic keeps monitoring dashboards populated |
| Gateway/dashboard integration | Nginx, Prometheus, and Grafana are integrated as gateway and monitoring services |

During the final review, the local Airflow stack, FastAPI, MLflow, Pushgateway, Grafana, node-exporter, and prediction traffic were running.
Nginx and Prometheus still had local mount cleanup items during final review, so this report keeps the detailed readiness discussion focused on Airflow, DVC, Kubernetes, and the data flow.

***

## Kubernetes

Kubernetes is the production-style deployment layer for the project.
The Kubernetes work focused on making Airflow and the model-serving stack run as separated services with persistent state, predictable scheduling, autoscaling hooks, and repeatable manifests.

The validation environment used **Docker Desktop Kubernetes**, not K3s.
The manifests are still portable to K3s, Minikube, or a VM cluster, with non-Docker-Desktop clusters using either image loading or registry-published images.

### Kubernetes implementation process

The Kubernetes work separated the parts that need different lifecycle behavior in a cluster.
Airflow was split into a webserver deployment, one scheduler deployment, and worker deployment so UI/API traffic, scheduling, and task execution can scale or restart independently.
Postgres and Redis were added as stateful backing services for Airflow metadata and Celery task coordination.
Pushgateway was added to the kustomization after the drift DAG already depended on it, closing the gap between the Airflow monitoring code and the Kubernetes runtime.

The cluster design also had to handle the fact that Airflow writes runtime files while Kubernetes pods are replaceable.
For that reason the manifests use PVCs for the shared project workspace, task logs, Postgres data, and Redis data.
The Airflow pods copy project code from the image seed into the PVC-backed workspace at startup, then set DVC into local no-SCM mode inside the pod workspace.
That combination keeps DAG code fresh after image rebuilds while preserving runtime data and avoiding Git operations from inside the cluster.

Scheduling was defined through Kubernetes environment variables and kept consistent with local defaults:

| Schedule variable | Kubernetes value | Workflow |
|-------------------|------------------|----------|
| `MLOPS_E2E_SCHEDULE` | `0 6 * * *` | End-to-end extraction, training, versioning, logging, and validation |
| `MODEL_VERSIONING_SCHEDULE` | `0 4 * * *` | Data/model versioning and manifests |
| `DRIFT_MONITORING_SCHEDULE` | `0 7 * * *` | Drift report and Pushgateway metric handoff |

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

The default `kubernetes/kustomization.yaml` applies this API, Airflow, state, scaling, and Pushgateway bundle.
Nginx, Prometheus, and Grafana manifests exist in the repository as integration material, but they are not part of the default kustomization.
This keeps the Kubernetes deployment boundary clear: the production-style runtime validated for this report is FastAPI, Airflow, Postgres, Redis, Pushgateway, PVCs, HPA/PDB resources, and the DVC/model artifact path.

### Persistence design

Kubernetes uses persistent volumes for state that survives pod replacement:

| PVC | Purpose |
|-----|---------|
| `airflow-project` | Shared project workspace used by Airflow tasks |
| `airflow-logs` | Airflow task and scheduler logs |
| `postgres-data-airflow-postgres-0` | Airflow metadata database storage |
| `redis-data-airflow-redis-0` | Redis broker persistence |

The Airflow deployments also refresh code from the image into the PVC-backed workspace at pod startup.
This prevents stale PVC contents from hiding new DAG support code after a rollout, while still avoiding accidental overwrite of trained data and model outputs.

The Airflow project and log PVCs use `ReadWriteOnce`, which matched the Docker Desktop Kubernetes validation environment.
For a multi-node K3s, Minikube, or VM cluster, the same design should be backed by storage that can satisfy the selected scheduling pattern: shared read-write storage for pods that may land on different nodes, or node placement rules that keep the Airflow pods using those PVCs on the same node.

### Portability design

The Kubernetes manifests use local image names for the Airflow, FastAPI, and model-fetcher containers.
The portability assumptions recorded during validation were:

- DVC-tracked raw data and model artifacts are available through the DVC remote.
- The `dagshub-credentials` secret exists in the `rain-prediction` namespace so the model-fetcher init container can pull the model artifact.
- Non-Docker-Desktop clusters need image loading or registry-published image tags because the manifests reference local image names.
- The Airflow image tag used by the manifests is `rain_prediction_mlops-airflow:inesgas-airflow-20260713`.

### Validated Kubernetes state

The Kubernetes layer was checked after the Airflow/DVC fixes.
The server-side manifest dry-run passed, pods were ready, PVCs were bound, and the DagsHub secret was present after validation.

Validated runtime state:

| Check | Result |
|-------|--------|
| Manifest validation | Server-side kustomize dry-run passed |
| Airflow scheduler | `1/1` ready |
| Airflow webserver | `3/3` ready |
| Airflow workers | `2/2` ready, HPA configured for 2-3 |
| FastAPI | HPA configured for 3-4 replicas |
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
The DAGs are unpaused, scheduled, and visible in Kubernetes Airflow.
The same DAG set was also checked in the local Airflow stack as a parity check.

### Airflow automation design

The Airflow work turned separate local scripts into a production sequence with clear task order, task outputs, and failure points.
The DAGs do not only call the training script.
They prepare the raw dataset, update DVC pointers, verify local artifacts, record manifests, train the model, record output versions, check API compatibility, and leave a DVC status snapshot for review.

The schedule design is daily rather than every few hours because the target label is `rain_tomorrow`.
The project needs enough time for the next-day label to become meaningful before retraining.
The freshness check records the latest available `Date`, the observed lag, the allowed lag, and the reason daily automation is the correct level for this dataset.

The DAGs are also deliberately local-first.
They can update local `.dvc` pointer files and reports inside Docker or Kubernetes, but they do not push to GitHub or DagsHub from inside an Airflow run.
This keeps automation reproducible while leaving remote publishing tied to review and merge.

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

Workflow behavior:

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

Workflow behavior:

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

Workflow behavior:

1. Verify the DVC-tracked reference dataset is present locally
2. Run `src.monitoring.drift_report` over the configured window and log the HTML report to MLflow
3. Write `reports/monitoring/drift_<timestamp>.html` and `drift_<timestamp>_summary.json`
4. Push summary drift metrics to Pushgateway for dashboard/alert integration

### Airflow validation

The Airflow layer was validated with Kubernetes as the production-style runtime target.
The local stack was also checked as a development parity signal because the repository still includes a same-machine integration environment.

| Validation | Result |
|------------|--------|
| Local Airflow health | Scheduler and metadatabase healthy |
| Local Airflow DAG imports | No import errors |
| Local Airflow DAG list | Four expected DAGs loaded and unpaused |
| Local Airflow DVC check | Raw/model DVC targets up to date inside the Airflow runtime |
| Local Airflow MLflow check | Model logging dry-run found the model artifact, metadata, config, and sample input |
| Kubernetes Airflow DAG imports | No import errors |
| Kubernetes Airflow DAG list | Four expected DAGs loaded and unpaused |
| Kubernetes Airflow DVC check | Raw/model DVC targets up to date inside the scheduler runtime |
| Kubernetes service check | Airflow reached FastAPI `/health` and Pushgateway `/-/healthy` |

***

## DVC Artifact Versioning

### DVC

DVC is used because the project contains large assets that are kept outside Git.
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
The local training workflow can also produce a newer `data/monitoring/reference_dataset.csv`.
During the latest validation, a newer local reference dataset was produced but its remote upload did not complete, so the committed pointer intentionally remained on the last remote-synced object.

### Versioning process

The DVC process was built around traceability rather than only file storage.
The versioning module reads every `.dvc` pointer, extracts the tracked path, hash, hash type, and size, and writes that state into JSON manifests under `reports/versioning/`.
The same manifest also records the Git context, Airflow run context, DVC remote, model metadata, model configuration summary, sample input path, and model artifact path.

Two snapshots are recorded around training:

| Snapshot | What it captures |
|----------|------------------|
| Input snapshot | Raw data pointer, existing model metadata/config, local artifact presence, Git context, DVC remote, and Airflow run metadata |
| Output snapshot | Updated model artifact, updated metadata, updated sample input, DVC pointer state, and tracking handoff state |

The workflow also writes a DVC status manifest after each automated run.
That status file is useful because it records whether the local runtime changed artifacts without silently assuming those changes are already available in the remote store.
During the final DVC cleanup, the raw dataset and winner model object were pushed successfully to DagsHub before merge.
The newer local monitoring reference dataset was left uncommitted because its remote upload did not complete, preserving a repository state that can still be pulled cleanly on another machine.

### MLflow integration context

MLflow records model metadata, metrics, parameters, and artifacts.
This section is included because the Airflow and training flow connect to a tracking target.
The project supports two tracking targets:

- Local MLflow for Airflow-triggered development and demonstration runs
- DagsHub MLflow for hosted experiment tracking

The Airflow integration was adjusted so the local Airflow stack uses the local MLflow service by default.
This avoids accidental DagsHub `403` failures when hosted MLflow credentials are not configured.

In Kubernetes, Airflow uses `file:///opt/airflow/project/mlruns` in the default config map.
That stores Kubernetes-triggered tracking output in the PVC-backed Airflow project workspace instead of requiring a separate MLflow server deployment in the default kustomization.

| `AIRFLOW_MLFLOW_TRACKING_URI` | Airflow-triggered runs appear in | Manual training runs appear in |
|-------------------------------|----------------------------------|--------------------------------|
| `http://mlflow:5000` (default) | Local MLflow UI | DagsHub |
| `https://dagshub.com/Inesgas/rain_prediction_MLops.mlflow` | DagsHub Experiments | DagsHub |

MLflow dry-run validation was executed in both local Airflow and Kubernetes Airflow.
Both runtimes found the model artifact, metadata, config, and sample input and produced a valid logging payload.

***

## Data Ingestion

The ingestion layer was designed so new WeatherAUS-format data can enter the same path as the original training data.
Incoming rows are not treated as a separate prediction-only file.
They are merged into the raw dataset and then included in the next Airflow-driven training cycle.

### Extraction and merge process

The ingestion code supports existing WeatherAUS files, compressed files, online CSV/ZIP sources, Kaggle-style sources, and recent Open-Meteo daily observations.
Regardless of source, the data is converted back into the raw WeatherAUS-style training table before training sees it.
The extraction step validates row counts, writes the target file atomically, and records an extraction manifest with the action, mode, source record, target record, validation result, and merge summary.

The upsert mode is important for production use.
It compares incoming rows against existing rows by the `Date` and `Location` key.
Rows with new keys are inserted, rows with repeated keys replace the older version, unchanged overlaps are counted, and duplicate incoming keys are reported in the manifest.
This makes repeated daily ingestion safe because rerunning a DAG does not blindly duplicate the same station-date observation.

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

## Access and Security Context

| Service | Deployment | URL | Username | Password | Notes |
|---------|------------|-----|----------|----------|-------|
| Airflow | Local stack | `http://localhost:8080` | `admin` | `airflow` | Defined in `docker-compose.yml` |
| Airflow | Legacy dev compose | `http://localhost:8080` | `airflow` | `airflow` | Defined in `src/docker/docker-compose-dev.yml` |
| Airflow | Kubernetes | `http://localhost:18080` via port-forward | `admin` | `airflow` | Placeholder in `kubernetes/airflow-secret.yaml` |
| MLflow | Local stack | `http://localhost:5000` | none | none | Local tracking UI |
| Grafana | Local stack | `http://localhost:3000` | `admin` | `admin` | Local monitoring UI |
| Prometheus | Local stack | `http://localhost:9090` | none | none | Local monitoring only |
| Nginx gateway | Local stack | `https://localhost` | local `.htpasswd` users | stored as hashes | Local gateway file, not committed |

The project separates committed configuration from local secrets.
`.env`, `.dvc/config.local`, TLS certificates, and gateway password hashes are local runtime material and are not part of the committed model or orchestration report.

***

## Resolved Integration Issues

Several integration problems were found and resolved during the production-readiness work.
This section records the work behind the final state.

| Area | Issue found | Resolution |
|------|-------------|------------|
| Airflow scheduling | End-to-end automation and drift monitoring were not fully scheduled in the earlier configuration | Airflow schedule variables were aligned so ingestion, E2E training/versioning, and drift monitoring run as automated DAGs |
| Airflow dependencies | Drift monitoring required Evidently, Plotly, and Prometheus client support in the Airflow image | Airflow dependency pins and Dockerfile installation order were aligned with Airflow 2.10.5 constraints |
| MLflow target | Local Airflow could point to hosted DagsHub MLflow and fail without credentials | Local Airflow now uses local MLflow by default through `AIRFLOW_MLFLOW_TRACKING_URI` |
| Kubernetes scheduler | Multiple Airflow schedulers caused duplicate serialized DAG writes and instability | Kubernetes Airflow scheduler was reduced to one stable replica |
| Kubernetes PVC source refresh | PVC-backed Airflow pods could keep stale project code after image rebuilds | Airflow init containers now refresh source code from the image into the PVC-backed workspace |
| Kubernetes Pushgateway | Drift DAG expected Pushgateway, but the Kubernetes stack did not include it earlier | Pushgateway deployment and service are included in `kubernetes/kustomization.yaml` |
| Kubernetes credentials | The model-fetcher init container needed DagsHub credentials | `dagshub-credentials` was created in the `rain-prediction` namespace during validation |
| DVC artifact consistency | Updated raw data/model pointers would break other machines if objects were not uploaded | The new raw dataset and winner model objects were pushed to the DagsHub DVC remote before merge |
| Drift reference artifact | A newer local reference dataset existed, but its DVC upload timed out | Its pointer was intentionally not committed, preserving a pullable merged state |

***

## Security Handling

| Security area | Project handling |
|---------------|------------------|
| Environment variables | `.env` is git-ignored; `.env.example` records placeholder structure only |
| DVC credentials | Local authentication material lives in `.dvc/config.local`, outside version control |
| Gateway passwords | Nginx password material is represented by hashes in `nginx/.htpasswd` |
| Shared defaults | Local demonstration credentials remain visible in compose/manifests and are separate from private secrets |
| Password recovery | Plaintext source values are not recoverable from `nginx/.htpasswd` hashes |

***

