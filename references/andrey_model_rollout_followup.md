# Automated Model Rollout Follow-Up

This branch follows the open items from the automated model rollout note.

## What was added

- `src/versioning/dvc_versioning.py` now has a `dvc-push-model` command. After Airflow tracks the freshly trained `models/final_winner/winner_model.joblib` artifact with DVC, the DAG can push the DVC object to the configured DagsHub remote.
- `airflow/dags/data_model_versioning_dag.py` now runs `push_model_artifact_to_dagshub` after `version_model_artifact`, then runs `restart_api_deployment` near the end of the successful pipeline.
- `src/versioning/kubernetes_rollout.py` restarts `deployment/rain-prediction-api` from inside Kubernetes by patching the deployment pod-template annotation, equivalent to `kubectl rollout restart`.
- `kubernetes/airflow-api-rollout-rbac.yaml` grants the Airflow ServiceAccount only the permissions needed to patch the `rain-prediction-api` deployment.
- The Airflow image requirements now include the Kubernetes Python client, so the Airflow worker can run the rollout task without adding `kubectl`.

## What was intentionally left unchanged

- The FastAPI deployment, model-fetcher init container, shared model volume, and FastAPI startup loading logic were not changed.
- The API image still does not need to be rebuilt after training. A pod restart is enough because the init container pulls the latest DVC-tracked model from DagsHub and FastAPI loads the model on startup.

## Expected runtime flow

1. Airflow trains the winner model.
2. Airflow runs `dvc add` for the model artifact.
3. Airflow pushes the DVC-tracked model artifact to DagsHub.
4. Airflow restarts `deployment/rain-prediction-api`.
5. New FastAPI pods start.
6. The existing model-fetcher init container pulls the latest model artifact from DagsHub into the shared model volume.
7. FastAPI loads the refreshed model from disk on startup.

## Validation already done

- Python compile check passed for the changed versioning, rollout, DAG, and test files.
- Focused automation tests passed: `14 passed`.
- `kubectl kustomize kubernetes --load-restrictor LoadRestrictionsNone` renders the new RBAC resources and includes the scoped `rain-prediction-api` deployment permission.

## Runtime check still needed after redeploy

Rebuild and redeploy the Airflow image, then trigger `data_model_versioning` in Kubernetes Airflow and confirm:

- the DVC push task succeeds,
- the API deployment receives a rollout restart,
- the restarted API pods complete the model-fetcher init container,
- predictions are served from the refreshed model artifact.
