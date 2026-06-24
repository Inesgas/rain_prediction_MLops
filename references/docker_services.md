# Pipeline Assets

| Path | Content |
| --- | --- |
| `src/docker/prediction/prediction.Dockerfile` | Prediction API image |
| `src/docker/training/training.Dockerfile` | Winner training image |
| `src/docker/airflow/airflow.Dockerfile` | Local Airflow scheduler/webserver image for DVC-backed versioning |
| `src/docker/frontend/frontend.Dockerfile` | Streamlit service dashboard image |
| `src/docker/docker-compose-dev.yml` | Local service composition for prediction, training, Airflow, and dashboard services |
| `src/docker/prediction/prediction_requirements.txt` | Prediction API dependencies |
| `src/docker/training/training_requirements.txt` | Training dependencies |
| `src/docker/airflow/airflow_requirements.txt` | Airflow DAG runtime dependencies for local DVC versioning |
| `src/docker/frontend/frontend_requirements.txt` | Streamlit dashboard dependencies |
| `src/docker/testing/` | Testing service assets |
| `.github/workflows/` | Continuous integration workflow assets |
