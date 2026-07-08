# Rain Prediction MLOps Dashboard

## Overview

This Grafana dashboard monitors production metrics for the Rain Prediction FastAPI. It displays real-time data on predictions, model confidence, data drift, and system health.

## Dashboard Panels

### 1. Predictions per Location
- **Type:** Bar Chart
- **Metric:** `sum(rain_predictions_total) by (location)`
- **Description:** Shows the total number of predictions per location. Helps understand which cities are most frequently requested.
- **Unit:** Count

### 2. Predictions over Time
- **Type:** Time Series
- **Metric:** `sum(rate(rain_predictions_total[1h])) by (location)`
- **Description:** Shows prediction throughput per location over time. Detects traffic patterns and load spikes.
- **Unit:** Requests per second (cps)

### 3. Prediction Confidence (95th Percentile)
- **Type:** Time Series
- **Metric:** `histogram_quantile(0.95, sum(rate(rain_prediction_confidence_bucket[5m])) by (le))`
- **Description:** Displays the 95th percentile of prediction confidence. High values (90%+) indicate a reliable model.
- **Unit:** Percent (0-1)

### 4. Confidence Distribution
- **Type:** Histogram
- **Metric:** `sum(rate(rain_prediction_confidence_bucket[5m])) by (le)`
- **Description:** Distributes confidence values into buckets (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99). Shows how confident the model is for different predictions.
- **Unit:** Count per bucket

### 5. Input Humidity per Location (Data Drift)
- **Type:** Time Series
- **Metric:** `avg(rain_input_humidity) by (location)`
- **Description:** Monitors incoming humidity per location. Detects data drift – sudden changes indicate altered input data.
- **Unit:** Percent (0-100%)

### 6. Input Pressure per Location
- **Type:** Time Series
- **Metric:** `avg(rain_input_pressure) by (location)`
- **Description:** Monitors incoming atmospheric pressure per location. Helps detect anomalies in input data.
- **Unit:** hPa

### 7. HTTP Requests per Endpoint
- **Type:** Bar Chart
- **Metric:** `sum(rate(http_requests_total[1h])) by (handler)`
- **Description:** Shows distribution of HTTP requests across different API endpoints. Identifies which endpoints are most heavily used.
- **Unit:** Requests per second (cps)

### 8. HTTP Error Rate (5xx)
- **Type:** Time Series
- **Metric:** `sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))`
- **Description:** Shows the proportion of server errors (HTTP 5xx) among all requests. Should ideally be 0%.
- **Unit:** Percent (0-1)

### 9. Model Load Time
- **Type:** Stat
- **Metric:** `rain_model_load_time_seconds`
- **Description:** Displays the time taken to load the model during startup.
- **Unit:** Seconds
- **Thresholds:** Green <5s, Yellow 5-10s, Red >10s

### 10. Prediction Errors
- **Type:** Stat
- **Metric:** `sum(rain_prediction_errors_total)`
- **Description:** Counts failed predictions (e.g., model not loaded, internal errors). Should ideally be 0.
- **Unit:** Count
- **Thresholds:** Green = 0, Red >0

## Useful PromQL Queries

```promql
# Total predictions per location
sum(rain_predictions_total) by (location)

# Prediction throughput per location (last hour)
sum(rate(rain_predictions_total[1h])) by (location)

# 95% Confidence percentile
histogram_quantile(0.95, sum(rate(rain_prediction_confidence_bucket[5m])) by (le))

# Data drift - humidity
avg(rain_input_humidity) by (location)

# Error rate (5xx)
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))

# Model load time
rain_model_load_time_seconds

# Prediction errors
sum(rain_prediction_errors_total)