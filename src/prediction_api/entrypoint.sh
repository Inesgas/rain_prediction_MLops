#!/bin/bash
set -e

echo "Pulling latest model from DVC remote..."
dvc pull models/final_winner/winner_model.joblib.dvc

echo "Starting FastAPI..."
exec uvicorn main:app --host 0.0.0.0 --port 8000