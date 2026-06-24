FROM apache/airflow:2.10.5-python3.12

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY src/docker/airflow/airflow_requirements.txt /tmp/airflow_requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/airflow_requirements.txt
