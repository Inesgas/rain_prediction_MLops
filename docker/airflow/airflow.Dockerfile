FROM apache/airflow:2.10.5-python3.12

ARG AIRFLOW_VERSION=2.10.5
ARG PYTHON_VERSION=3.12

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY docker/airflow/airflow_requirements.txt /tmp/airflow_requirements.txt
RUN python -m pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
    -r /tmp/airflow_requirements.txt

COPY --chown=airflow:root airflow/dags /opt/airflow/dags
COPY --chown=airflow:root src /opt/airflow/project-seed/src
COPY --chown=airflow:root data /opt/airflow/project-seed/data
COPY --chown=airflow:root models /opt/airflow/project-seed/models
COPY --chown=airflow:root references /opt/airflow/project-seed/references
COPY --chown=airflow:root tests /opt/airflow/project-seed/tests
COPY --chown=airflow:root .dvc /opt/airflow/project-seed/.dvc
COPY --chown=airflow:root .dvcignore .gitignore /opt/airflow/project-seed/

ENV AIRFLOW_PROJECT_DIR=/opt/airflow/project
