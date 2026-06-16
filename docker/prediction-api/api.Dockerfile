# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Copy and install specific API dependencies
COPY docker/prediction-api/api_requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# 2. Install testing dependencies (for demo and development)
RUN python -m pip install pytest pytest-cov httpx requests

# 3. Copy source code and real model artifacts into the container
COPY src/ /app/src/
COPY models/ /app/models/
COPY tests/ /app/tests/

# 4. Ensure Python can discover packages inside /app
ENV PYTHONPATH="/app"

EXPOSE 8502

# Start Uvicorn directly using the proper module path relative to WORKDIR
CMD ["uvicorn", "src.prediction_api.main:app", "--host", "0.0.0.0", "--port", "8502"]