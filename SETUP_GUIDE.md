# Onboarding Guide — Setting Up the Project After `git pull`

This project stores no secrets in Git (by design). After cloning/pulling the
repo, you need to generate a few local files and Kubernetes secrets yourself
before the cluster will work. This guide walks through exactly that.

## Prerequisites

- Docker Desktop running, with WSL2 integration enabled (Windows) or Docker
  running natively (Linux/Mac)
- `kubectl`, `k3d`, and `kustomize` installed
- A DagsHub account with **collaborator access** to the project repo
  (`https://dagshub.com/Inesgas/rain_prediction_MLops`) — ask a teammate to
  invite you if you don't have it yet

## Step 1 — Clone the repo

```bash
git clone https://github.com/Inesgas/rain_prediction_MLops.git
cd rain_prediction_MLops
```

## Step 2 — Get a DagsHub access token

1. Log in at [dagshub.com](https://dagshub.com)
2. Profile icon (top right) → Settings → Tokens → **Generate New Token**
3. Copy the token immediately — it's shown only once

## Step 3 — Create the DagsHub credentials secret file

```bash
cat > kubernetes/fastapi-secret.yaml << 'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: dagshub-credentials
  namespace: rain-prediction
type: Opaque
stringData:
  DVC_REMOTE_USER: "<your-dagshub-username>"
  DVC_REMOTE_PASSWORD: "<your-dagshub-token>"
EOF
```

Open the file and replace both placeholders with your real DagsHub username
and token. This file is `.gitignore`d — it will never be committed.

## Step 4 — Generate a self-signed TLS certificate for Nginx

```bash
mkdir -p nginx/certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/certs/nginx.key \
  -out nginx/certs/nginx.crt \
  -subj "/CN=rain-prediction-api"
```

## Step 5 — Set up Nginx Basic Auth users

```bash
sudo apt-get update && sudo apt-get install -y apache2-utils

htpasswd -c nginx/.htpasswd andrey   # -c creates the file (first user only)
htpasswd nginx/.htpasswd ines
htpasswd nginx/.htpasswd gunter
htpasswd nginx/.htpasswd admin
```

Pick any passwords you like — you'll use them to log into the API, Grafana,
Prometheus, and Airflow (via Nginx). Note: **`admin`** is the only user with
admin rights inside the FastAPI app; the rest are regular users.

## Step 6 — Create the k3d cluster

```bash
k3d cluster create rain-prediction-cluster \
  --port "443:443@loadbalancer" \
  --port "80:80@loadbalancer" \
  --port "8081:8080@loadbalancer" \
  --k3s-arg "--disable=traefik@server:0"
```

## Step 7 — Build all Docker images

```bash
docker build -t rain_prediction_mlops-model-fetcher:latest -f docker/model-fetcher/model-fetcher.Dockerfile .
docker build -t rain_prediction_mlops-fastapi:latest -f docker/prediction-api/api.Dockerfile .
docker build -t rain_prediction_mlops-nginx:latest -f docker/gateway/nginx.Dockerfile .
docker build -t rain_prediction_mlops-airflow:latest -f docker/airflow/airflow.Dockerfile .
docker tag rain_prediction_mlops-airflow:latest rain_prediction_mlops-airflow:inesgas-airflow-20260713
```

MLflow uses the official image, but importing the multi-arch manifest into
k3d can fail. Pull the AMD64-only build explicitly and re-tag it:

```bash
docker pull --platform linux/amd64 ghcr.io/mlflow/mlflow:v2.17.2
docker tag ghcr.io/mlflow/mlflow:v2.17.2 mlflow-local:v2.17.2
```

(`kubernetes/mlflow-deployment.yaml` already references `mlflow-local:v2.17.2`.)

## Step 8 — Import all images into the cluster

```bash
k3d image import \
  rain_prediction_mlops-model-fetcher:latest \
  rain_prediction_mlops-fastapi:latest \
  rain_prediction_mlops-nginx:latest \
  rain_prediction_mlops-airflow:latest \
  rain_prediction_mlops-airflow:inesgas-airflow-20260713 \
  mlflow-local:v2.17.2 \
  -c rain-prediction-cluster
```

## Step 9 — Deploy everything

```bash
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/fastapi-secret.yaml

kubectl create secret generic nginx-tls-and-auth \
  --namespace rain-prediction \
  --from-file=nginx.crt=nginx/certs/nginx.crt \
  --from-file=nginx.key=nginx/certs/nginx.key \
  --from-file=.htpasswd=nginx/.htpasswd

kustomize build kubernetes/ --load-restrictor LoadRestrictionsNone | kubectl apply -f -
```

## Step 10 — Verify

```bash
kubectl get pods -n rain-prediction
```

Wait until all pods show `Running` or `Completed`. This takes a few minutes
on first boot (images need to pull, DVC data needs to sync, DB migrations
need to run).

Then check in the browser (accept the self-signed certificate warning):

| Service    | URL                              |
|------------|-----------------------------------|
| FastAPI    | https://localhost/docs           |
| Grafana    | https://localhost/grafana/       |
| Prometheus | https://localhost/prometheus/    |
| MLflow     | https://localhost/mlflow/        |
| Airflow    | https://localhost/airflow/ (login: `admin` / `airflow`) |

## Troubleshooting

- **`dvc pull` fails / `Init:Error` on Airflow pods** — double-check your
  DagsHub token in `kubernetes/fastapi-secret.yaml` and that you're a
  collaborator on the DagsHub repo.
- **Image pulls from `ghcr.io` time out** — this is a known flaky Docker
  Desktop/WSL2 network issue; just retry the command.
- **`kustomize build` errors about paths outside the kustomization root** —
  make sure you're using the standalone `kustomize` binary with
  `--load-restrictor LoadRestrictionsNone`, not `kubectl apply -k` (which
  doesn't support that flag in recent versions).
