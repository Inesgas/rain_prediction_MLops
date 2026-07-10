#!/bin/bash
set -e

echo "Restoring model metadata into mounted volume..."
mkdir -p /repo/models
cp -r /repo/models_source/* /repo/models/

echo "Configuring DVC remote authentication..."
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "$DVC_REMOTE_USER"
dvc remote modify origin --local password "$DVC_REMOTE_PASSWORD"

echo "Pulling latest model from DagsHub..."
dvc pull models/final_winner/winner_model.joblib.dvc

echo "Model fetch complete."
ls -la /repo/models/final_winner/
