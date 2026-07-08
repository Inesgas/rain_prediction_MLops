# System Control: Prometheus & Grafana Monitoring Architecture

This document describes the automated infrastructure monitoring, alerting framework, and visualization layout for the Rain Prediction MLOps system. 

The entire architecture runs strictly containerized inside Docker.

---

## 1. Prometheus Architecture & Security

Prometheus acts as the central time-series database that pulls (scrapes) performance metrics every 15 seconds.

### Automated Password-Free Scrapes
To ensure GitHub compliance, Prometheus authenticates against the secure HTTPS Nginx Gateway without using cleartext credentials. It reads user credentials natively from the shared, encrypted `.htpasswd` container volume link.

### Target Jobs
1. **`fastapi-gateway`**: Collects real-time ML application metrics and HTTP response states routed through the Nginx gateway at `nginx-gateway:443/metrics`.
2. **`node-exporter`**: Collects raw OS and hardware metrics (CPU, Memory, Disk) directly from the virtual machine at `node-exporter:9100`.

---

## 2. Infrastructure Alerting Rules

Prometheus continuously evaluates threshold conditions specified inside `deployment/prometheus/rules/alert_rules.yml`.

"It monitors the overall connectivity and server availability."

### Configured Alert: `RainApiServerDown`
* **Expression:** `up{job="fastapi-gateway"} == 0`
* **Trigger Window (`for`):** 10 seconds.
* **Severity:** `critical`
* **Behavior:** If the Nginx proxy or all 3 underlying FastAPI container replicas fail to respond to a scrape for over 10 seconds, Prometheus triggers a critical alert. Status can be tracked internally at `https://<VM-IP>/prometheus/alerts` and the alert states are seamlessly forwarded to Grafana for unified visualization."

---

## 3. Grafana Visualizations (Provisioning)

Grafana dashboards are fully automated. Upon initial container boot, Grafana reads files inside `deployment/grafana/provisioning/` to auto-configure its default data source (Prometheus) and import all charts seamlessly under the **"MLOps"** directory.

The infrastructure features three standalone dashboards:

### Dashboard 1: API Performance
* **Purpose:** Monitors core web service reliability.
* **Key Panels:** Live API Request Rate, P95 Request Latency, and HTTP 5xx Server Error Rate Gauges.

### Dashboard 2: Infrastructure Overview
* **Purpose:** Monitors host hardware constraints to prevent hardware-induced system crashes.
* **Key Panels:** Host CPU Usage, Host RAM Availability, and Root System `/` Disk Space saturation tracking.

### Dashboard 3: Model Performance & Drift
* **Purpose:** Dedicated MLOps tracking for the predictive capabilities of the model.
* **Key Panels:** Live Regression Loss (RMSE / MAE counters), Goodness-of-Fit trends ($R^2$ Score), and real-time **Data Drift Status** timelines to catch feature changes early.

---

## 4. System Control Commands

### Start or Restart the System
```bash
docker compose down --remove-orphans && docker compose up --build -d
```

### Inspect Monitoring Operational Logs
```bash
docker compose logs -f prometheus
docker compose logs -f grafana
```
