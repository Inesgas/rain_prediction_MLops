import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Rain Prediction Service", layout="wide")
st.title("Rain Prediction Service")

health_response = None
model_response = None

try:
    health_response = requests.get(f"{API_URL}/health", timeout=5).json()
except requests.RequestException as exc:
    st.error(f"API health check failed: {exc}")

if health_response:
    st.subheader("Service Status")
    st.json(health_response)

try:
    model_response = requests.get(f"{API_URL}/model-info", timeout=5).json()
except requests.RequestException as exc:
    st.error(f"Model metadata request failed: {exc}")

if model_response:
    st.subheader("Model Metadata")
    summary = {
        "model_name": model_response.get("model_name"),
        "model_role": model_response.get("model_role"),
        "feature_count": model_response.get("feature_count"),
        "threshold": model_response.get("threshold"),
        "metrics": model_response.get("metrics"),
    }
    st.json(summary)
