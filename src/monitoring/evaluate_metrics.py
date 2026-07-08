import os
import datetime
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================================
# PATH CONFIGURATION (Matches your project structure)
# ========================================================
BASE_DIR = "/opt/airflow/project/data/monitoring"
PRED_DIR = os.path.join(BASE_DIR, "predictions")
ACT_DIR = os.path.join(BASE_DIR, "actuals")
PUSHGATEWAY_URL = "http://pushgateway:9091"

def evaluate_daily_metrics():
    # 1. Calculate yesterday's date string (e.g., 2026-07-07)
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    logger.info(f"Starting model evaluation for date: {yesterday_str}")

    # 2. Define targeted file paths
    pred_file = os.path.join(PRED_DIR, f"predictions_{yesterday_str}.csv")
    act_file = os.path.join(ACT_DIR, f"actuals_{yesterday_str}.csv")

    # 3. Check for file existence before loading
    if not os.path.exists(pred_file):
        logger.error(f"Missing prediction log file: {pred_file}")
        return
    if not os.path.exists(act_file):
        logger.error(f"Missing actual targets log file: {act_file}")
        return

    try:
        # 4. Load CSV datasets into Pandas DataFrames
        df_pred = pd.read_csv(pred_file)
        df_act = pd.read_csv(act_file)

        if df_pred.empty or df_act.empty:
            logger.warning(f"One of the validation datasets is empty for {yesterday_str}")
            return

        # 5. Feature Alignment (Ensure prediction and actual arrays match perfectly)
        # Note: In a production setup, consider matching rows by a unique timestamp/ID.
        # For this prototype, we extract the sequential values directly from your sample format.
        y_pred = df_pred["predicted_rain"].values
        y_true = df_act["actual_rain"].values

        if len(y_pred) != len(y_true):
            logger.error(f"Mismatched row counts! Predictions: {len(y_pred)}, Actuals: {len(y_true)}")
            return

        # 6. Statistical Computations (Expected by Grafana Panels)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        logger.info(f"Calculated Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # 7. Register and assign metrics to Prometheus Gauges
        registry = CollectorRegistry()
        
        # Names must strictly match the 'expr' definitions inside your Grafana JSON
        gauge_rmse = Gauge('model_rmse_score', 'Root Mean Squared Error of the model', registry=registry)
        gauge_mae = Gauge('model_mae_score', 'Mean Absolute Error of the model', registry=registry)
        gauge_r2 = Gauge('model_r2_score', 'R2 Score of the model', registry=registry)

        gauge_rmse.set(rmse)
        gauge_mae.set(mae)
        gauge_r2.set(r2)

        # 8. Push Metrics to Prometheus via Pushgateway
        push_to_gateway(PUSHGATEWAY_URL, job='model_evaluation', registry=registry)
        logger.info(f"Successfully pushed metrics to Pushgateway at {PUSHGATEWAY_URL}")

    except Exception as e:
        logger.error(f"Batch evaluation pipeline failed: {str(e)}")

if __name__ == "__main__":
    evaluate_daily_metrics()
