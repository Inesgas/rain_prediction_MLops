from __future__ import annotations

import json
import math
from datetime import date, timedelta
from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_DIR = PROJECT_ROOT / "presentation"
ASSET_DIR = PRESENTATION_DIR / "assets"
DATA_SCIENCE_ASSET_DIR = ASSET_DIR / "data_science"

METADATA_PATH = PROJECT_ROOT / "models" / "final_winner" / "metadata.json"
MODEL_CONFIG_PATH = PROJECT_ROOT / "models" / "final_winner" / "model_config.json"
SAMPLE_INPUT_PATH = PROJECT_ROOT / "models" / "final_winner" / "sample_input.json"
WINNER_MODEL_PATH = PROJECT_ROOT / "models" / "final_winner" / "winner_model.joblib"
LOCATIONS_METADATA_PATHS = [
    PROJECT_ROOT / "data" / "preprocessed" / "locations_metadata.csv",
    PROJECT_ROOT / "data" / "processed" / "daily_zonal_locations_metadata.csv",
]
ARCHITECTURE_IMAGE_PATH = ASSET_DIR / "rain_mlops_architecture.png"

CLASS_IMBALANCE_PATH = DATA_SCIENCE_ASSET_DIR / "class_imbalance.png"
SEASONAL_PATTERNS_PATH = DATA_SCIENCE_ASSET_DIR / "seasonal_patterns.png"
CHRONOLOGICAL_SPLIT_PATH = DATA_SCIENCE_ASSET_DIR / "chronological_split.png"
MODEL_COMPARISON_PATH = DATA_SCIENCE_ASSET_DIR / "model_comparison.png"
THRESHOLD_CURVE_PATH = DATA_SCIENCE_ASSET_DIR / "threshold_curve.png"
KOPPEN_HEATMAP_PATH = DATA_SCIENCE_ASSET_DIR / "koppen_heatmap.png"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


METADATA = read_json(METADATA_PATH)
MODEL_CONFIG = read_json(MODEL_CONFIG_PATH)
METRICS = METADATA.get("metrics", {}) or MODEL_CONFIG.get("test_metrics", {})
FEATURES = METADATA.get("features") or MODEL_CONFIG.get("features", [])
FEATURE_COUNT = int(METADATA.get("feature_count") or MODEL_CONFIG.get("feature_count") or len(FEATURES) or 68)
THRESHOLD = float(METADATA.get("threshold") or MODEL_CONFIG.get("threshold") or 0.58)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def fmt_pct(value: object, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def fmt_date(value: date) -> str:
    return value.strftime("%d %b %Y")


@st.cache_data
def load_sample_input() -> dict:
    return read_json(SAMPLE_INPUT_PATH)


@st.cache_data
def load_locations() -> pd.DataFrame:
    for path in LOCATIONS_METADATA_PATHS:
        if path.exists():
            frame = pd.read_csv(path)
            break
    else:
        return pd.DataFrame(columns=["location", "lat", "lon", "elevation", "rainfall_zone"])

    keep = [col for col in ["location", "lat", "lon", "elevation", "rainfall_zone"] if col in frame.columns]
    frame = frame[keep].copy()
    if "rainfall_zone" not in frame.columns:
        frame["rainfall_zone"] = "Unknown"
    if "elevation" not in frame.columns:
        frame["elevation"] = None
    return (
        frame.dropna(subset=["location", "lat", "lon"])
        .drop_duplicates(subset=["location"])
        .sort_values("location")
        .reset_index(drop=True)
    )


@st.cache_resource
def load_prediction_runtime() -> tuple[dict | None, object | None, str | None]:
    try:
        import joblib
        from catboost import Pool
    except Exception as exc:  # pragma: no cover - local presentation dependency
        return None, None, f"Prediction runtime is missing: {exc}"

    if not WINNER_MODEL_PATH.exists():
        return None, None, "Winner model artifact is missing."

    try:
        artifact = joblib.load(WINNER_MODEL_PATH)
    except Exception as exc:  # pragma: no cover - local artifact dependency
        return None, None, f"Winner model could not be loaded: {exc}"

    if isinstance(artifact, dict):
        model = artifact.get("model")
        features = artifact.get("features") or FEATURES
        categorical = artifact.get("categorical_features") or METADATA.get("categorical_features", ["location"])
        fill_values = artifact.get("numeric_fill_values") or METADATA.get("numeric_fill_values", {})
        threshold = float(artifact.get("threshold") or THRESHOLD)
    else:
        model = artifact
        features = list(getattr(model, "feature_names_", FEATURES))
        categorical = METADATA.get("categorical_features", ["location"])
        fill_values = METADATA.get("numeric_fill_values", {})
        threshold = THRESHOLD

    if model is None or not features:
        return None, None, "Winner model contract is incomplete."

    return (
        {
            "model": model,
            "features": features,
            "categorical": categorical,
            "fill_values": fill_values,
            "threshold": threshold,
        },
        Pool,
        None,
    )


def inject_theme() -> None:
    html(
        """
        <style>
        :root {
          --ink:#173848;
          --ink-2:#21323a;
          --muted:#5b6e76;
          --line:#d8e2e3;
          --surface:#ffffff;
          --soft:#f7f4ee;
          --blue:#1f76d2;
          --teal:#008c95;
          --green:#278455;
          --gold:#c99522;
          --red:#c9473a;
          --violet:#6557b6;
        }
        html, body, [class*="css"] {
          font-family: "Aptos", "Segoe UI", "Trebuchet MS", sans-serif;
          color: var(--ink-2);
        }
        .stApp {
          background:
            radial-gradient(circle at top left, rgba(31,118,210,.10), transparent 25%),
            radial-gradient(circle at top right, rgba(184,106,59,.12), transparent 24%),
            linear-gradient(180deg, #f7f4ee 0%, #eef5f5 42%, #fbfcfd 100%);
          overflow-x: hidden;
        }
        .block-container {
          max-width: 1160px;
          padding-top: .9rem;
          padding-left: 1.4rem;
          padding-right: 1.4rem;
          padding-bottom: 2.2rem;
        }
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stDeployButton"],
        div[data-testid="stAppDeployButton"],
        div[data-testid="stMainMenu"],
        a[data-testid="stDeployButton"],
        div[data-testid="stStatusWidget"],
        div[data-testid="stSidebarCollapsedControl"],
        header[data-testid="stHeader"] {
          display:none !important;
        }
        section[data-testid="stSidebar"] {
          background: linear-gradient(180deg, rgba(23,56,72,.99) 0%, rgba(34,79,84,.98) 100%);
          border-right: 1px solid rgba(255,255,255,.10);
        }
        section[data-testid="stSidebar"] * {
          color: #f8faf7;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
          color: rgba(248,250,247,.82);
          font-size:.86rem;
          line-height:1.38;
        }
        section[data-testid="stSidebar"] label {
          color:#f8faf7;
          font-weight:700;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
          border-radius: 10px;
          padding: .34rem .46rem;
          margin: .08rem 0;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
          background: rgba(255,255,255,.15);
          border: 1px solid rgba(255,255,255,.20);
        }
        h1, h2, h3, h4 {
          color: var(--ink);
          letter-spacing: 0;
        }
        p, li, label, div[data-testid="stMarkdownContainer"] {
          color: var(--muted);
          line-height: 1.55;
        }
        div[data-testid="stButton"] button {
          min-height: 2.45rem;
          border-radius: 999px;
          border: 1px solid var(--ink);
          background: var(--ink);
          color: white;
          font-weight: 800;
        }
        div[data-testid="stButton"] button:hover {
          border-color: var(--blue);
          background: var(--blue);
          color: white;
        }
        div[data-testid="stButton"] button:disabled {
          border-color: #cfdadd;
          background: #edf3f4;
          color: #8a9aa1;
        }
        .hero {
          position: relative;
          overflow: hidden;
          border-radius: 8px;
          padding: 1.1rem 1.35rem;
          margin: .1rem 0 .85rem;
          background:
            linear-gradient(135deg, rgba(23,56,72,.98) 0%, rgba(31,118,210,.84) 55%, rgba(184,106,59,.88) 100%);
          box-shadow: 0 26px 60px rgba(23,56,72,.18);
        }
        .hero:before {
          content:"";
          position:absolute;
          right:-90px;
          top:-145px;
          width:340px;
          height:340px;
          border-radius:50%;
          background: radial-gradient(circle, rgba(255,255,255,.24), transparent 64%);
        }
        .hero h1 {
          position: relative;
          color: white;
          margin: .22rem 0 .62rem;
          font-size: 2.35rem;
          line-height: 1.02;
          max-width: 860px;
        }
        .hero p {
          position: relative;
          color: rgba(255,255,255,.90);
          font-size: 1rem;
          margin: 0;
          max-width: 930px;
        }
        .compact-head {
          border-left:6px solid var(--accent);
          padding:.12rem 0 .12rem .72rem;
          margin:.45rem 0 .65rem;
        }
        .compact-head .kicker {
          color:var(--accent);
        }
        .compact-head h2 {
          margin:.04rem 0 .18rem;
          font-size:1.55rem;
          line-height:1.12;
        }
        .compact-head p {
          margin:0;
          font-size:.93rem;
          max-width:940px;
        }
        .kicker {
          position: relative;
          color:#9df0e7;
          font-size:.76rem;
          text-transform:uppercase;
          letter-spacing:.12em;
          font-weight:900;
        }
        .chip-row {
          display:flex;
          flex-wrap:wrap;
          gap:.5rem;
          margin:.8rem 0 1rem;
        }
        .tool-chip {
          display:inline-flex;
          align-items:center;
          gap:.45rem;
          border:1px solid var(--line);
          background:rgba(255,255,255,.92);
          border-radius:999px;
          padding:.42rem .62rem;
          box-shadow:0 10px 24px rgba(23,56,72,.06);
          color:var(--ink);
          font-weight:800;
          font-size:.85rem;
        }
        .tool-chip span {
          width:1.72rem;
          height:1.72rem;
          display:inline-flex;
          align-items:center;
          justify-content:center;
          border-radius:50%;
          background:var(--c);
          color:white;
          font-size:.66rem;
          font-weight:900;
        }
        .metric-card, .story-card, .stage-card, .demo-card, .prediction-card {
          border:1px solid var(--line);
          background:rgba(255,255,255,.96);
          border-radius:8px;
          box-shadow:0 16px 36px rgba(23,56,72,.08);
        }
        .metric-card {
          padding: .95rem 1rem;
          min-height: 116px;
        }
        .metric-label {
          color:var(--muted);
          font-size:.76rem;
          text-transform:uppercase;
          font-weight:850;
        }
        .metric-value {
          color:var(--ink);
          font-size:1.62rem;
          line-height:1.04;
          margin:.25rem 0;
          font-weight:900;
        }
        .metric-note {
          color:var(--muted);
          font-size:.88rem;
        }
        .section-head {
          border-left:6px solid var(--accent);
          padding:.25rem 0 .25rem .9rem;
          margin:.8rem 0 1rem;
        }
        .section-head h2 {
          margin:.1rem 0 .25rem;
          font-size:2.2rem;
          line-height:1.08;
        }
        .section-head p {
          margin:0;
          max-width:1040px;
        }
        .story-card {
          padding:1.05rem 1.12rem;
          min-height: 158px;
        }
        .story-card h3 {
          margin:.05rem 0 .55rem;
          color:var(--ink);
          font-size:1.18rem;
        }
        .story-card ul {
          margin:.2rem 0 0 1.05rem;
          padding:0;
        }
        .story-card li {
          margin:.35rem 0;
        }
        .prediction-card {
          padding:1rem;
        }
        .result-panel {
          border-radius:8px;
          padding:1.1rem 1.2rem;
          color:white;
          min-height:178px;
          background: linear-gradient(135deg, var(--green), var(--teal));
        }
        .result-panel.rain {
          background: linear-gradient(135deg, var(--blue), var(--teal));
        }
        .result-panel .eyebrow {
          color:rgba(255,255,255,.80);
          font-size:.78rem;
          text-transform:uppercase;
          letter-spacing:.08em;
          font-weight:850;
        }
        .result-panel .answer {
          color:white;
          font-size:2.15rem;
          line-height:1.03;
          margin:.25rem 0;
          font-weight:950;
        }
        .result-panel .copy {
          color:rgba(255,255,255,.92);
          font-weight:760;
        }
        .mini-grid {
          display:grid;
          grid-template-columns:repeat(3,minmax(0,1fr));
          gap:.55rem;
          margin-top:.9rem;
        }
        .mini-grid div {
          border:1px solid rgba(255,255,255,.35);
          border-radius:8px;
          padding:.55rem .6rem;
          color:white;
        }
        .mini-grid span {
          display:block;
          color:rgba(255,255,255,.76);
          text-transform:uppercase;
          font-size:.7rem;
          font-weight:850;
        }
        .stage-grid {
          display:grid;
          grid-template-columns:repeat(auto-fit,minmax(185px,1fr));
          gap:.72rem;
          margin:1rem 0;
        }
        .stage-card {
          border-top:5px solid var(--c);
          padding:.9rem;
          min-height:150px;
        }
        .stage-card .step {
          color:var(--c);
          font-size:.78rem;
          text-transform:uppercase;
          font-weight:900;
        }
        .stage-card b {
          display:block;
          color:var(--ink);
          margin:.25rem 0;
          font-size:1.05rem;
        }
        .stage-card span {
          color:var(--muted);
          font-size:.9rem;
          line-height:1.42;
        }
        .asset-frame {
          background: white;
          border:1px solid var(--line);
          border-radius:8px;
          padding:.75rem;
          box-shadow:0 18px 42px rgba(23,56,72,.08);
        }
        .caption {
          color:var(--muted);
          font-size:.86rem;
          margin:.35rem 0 .85rem;
        }
        .demo-grid {
          display:grid;
          grid-template-columns:repeat(3,minmax(0,1fr));
          gap:.75rem;
        }
        .demo-card {
          padding:1rem;
          min-height:150px;
          border-left:5px solid var(--c);
        }
        .demo-card b {
          color:var(--ink);
          display:block;
          margin-bottom:.35rem;
          font-size:1.08rem;
        }
        .demo-card span {
          color:var(--muted);
          line-height:1.42;
        }
        .handoff {
          border:1px solid rgba(31,118,210,.22);
          background:#eef7fb;
          border-radius:8px;
          padding:1rem 1.1rem;
          color:var(--ink);
          font-weight:760;
          margin-top:1rem;
        }
        .side-progress {
          height:.52rem;
          border-radius:999px;
          overflow:hidden;
          background:rgba(255,255,255,.16);
          margin:.85rem 0 .55rem;
        }
        .side-progress span {
          display:block;
          height:100%;
          width:var(--progress);
          background:linear-gradient(90deg,#9df0e7,#f4c95d);
        }
        .side-caption {
          color:rgba(248,250,247,.78);
          font-size:.8rem;
          margin:.15rem 0 .7rem;
        }
        @media(max-width:980px) {
          .hero h1 { font-size:1.9rem; }
          .demo-grid { grid-template-columns:1fr; }
          .mini-grid { grid-template-columns:1fr; }
        }
        </style>
        """
    )


def tool_chips(items: Iterable[tuple[str, str, str]]) -> None:
    chips = ["<div class='chip-row'>"]
    for name, mark, color in items:
        chips.append(
            f"<div class='tool-chip'><span style='--c:{escape(color)}'>{escape(mark)}</span>{escape(name)}</div>"
        )
    chips.append("</div>")
    html("".join(chips))


def metric_cards(items: list[tuple[str, str, str]], columns: int = 4) -> None:
    cols = st.columns(columns, gap="medium")
    for index, (label, value, note) in enumerate(items):
        with cols[index % columns]:
            html(
                f"""
                <div class="metric-card">
                  <div class="metric-label">{escape(label)}</div>
                  <div class="metric-value">{escape(value)}</div>
                  <div class="metric-note">{escape(note)}</div>
                </div>
                """
            )


def section_header(kicker: str, title: str, body: str, color: str = "#1f76d2") -> None:
    html(
        f"""
        <div class="section-head" style="--accent:{escape(color)}">
          <div class="kicker" style="color:{escape(color)}">{escape(kicker)}</div>
          <h2>{escape(title)}</h2>
          <p>{escape(body)}</p>
        </div>
        """
    )


def compact_header(kicker: str, title: str, body: str, color: str = "#1f76d2") -> None:
    html(
        f"""
        <div class="compact-head" style="--accent:{escape(color)}">
          <div class="kicker">{escape(kicker)}</div>
          <h2>{escape(title)}</h2>
          <p>{escape(body)}</p>
        </div>
        """
    )


def story_card(title: str, items: list[str]) -> None:
    body = "".join(f"<li>{escape(item)}</li>" for item in items)
    html(
        f"""
        <div class="story-card">
          <h3>{escape(title)}</h3>
          <ul>{body}</ul>
        </div>
        """
    )


def hero() -> None:
    html(
        """
        <div class="hero">
          <div class="kicker">Rain Prediction MLOps</div>
          <h1>From weather data science to an operating ML system</h1>
          <p>
            A rain-tomorrow classifier becomes
            a versioned, scheduled, monitored, Kubernetes-ready workflow with an inspectable
            prediction example.
          </p>
        </div>
        """
    )
    tool_chips(
        [
            ("Python", "Py", "#3776ab"),
            ("Pandas", "pd", "#150458"),
            ("CatBoost", "CB", "#c99700"),
            ("DVC", "DVC", "#945dd6"),
            ("Airflow", "AF", "#017cee"),
            ("MLflow", "ML", "#0194e2"),
            ("Evidently", "EV", "#e31b23"),
            ("FastAPI", "FA", "#009688"),
            ("Docker", "Do", "#2496ed"),
            ("Kubernetes", "K8s", "#326ce5"),
            ("GitHub Actions", "GA", "#2088ff"),
        ]
    )


def sample_observation_date(sample: dict) -> date:
    try:
        return date(int(sample.get("year", 2015)), int(sample.get("month", 12)), int(sample.get("day", 4)))
    except Exception:
        return date(2015, 12, 4)


def dewpoint(temp_c: float, humidity: float) -> float:
    humidity = max(1.0, min(100.0, humidity))
    a = 17.27
    b = 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + math.log(humidity / 100.0)
    return (b * alpha) / (a - alpha)


def bin_value(value: float, cuts: tuple[float, float, float]) -> str:
    if value < cuts[0]:
        return "0"
    if value < cuts[1]:
        return "1"
    if value < cuts[2]:
        return "2"
    return "3"


def update_payload(
    sample: dict,
    location_row: pd.Series,
    observation_date: date,
    min_temp: float,
    max_temp: float,
    rainfall: float,
    humidity_9am: float,
    humidity_3pm: float,
    pressure_9am: float,
    pressure_3pm: float,
    wind_gust_speed: float,
    cloud_3pm: float,
    rain_today: str,
) -> dict:
    payload = dict(sample)
    temp_9am = min_temp + (max_temp - min_temp) * 0.55
    temp_3pm = min_temp + (max_temp - min_temp) * 0.82
    day_of_year = observation_date.timetuple().tm_yday
    dp_9am = dewpoint(temp_9am, humidity_9am)
    dp_3pm = dewpoint(temp_3pm, humidity_3pm)

    payload.update(
        {
            "location": str(location_row["location"]),
            "lat": float(location_row["lat"]),
            "lon": float(location_row["lon"]),
            "elevation": float(location_row["elevation"]) if pd.notna(location_row.get("elevation")) else 0.0,
            "month": observation_date.month,
            "day": observation_date.day,
            "year": observation_date.year,
            "min_temp": min_temp,
            "max_temp": max_temp,
            "temp_9am": temp_9am,
            "temp_3pm": temp_3pm,
            "rainfall": rainfall,
            "rain_today": 1 if rain_today == "Yes" else 0,
            "humidity_9am": humidity_9am,
            "humidity_3pm": humidity_3pm,
            "pressure_9am": pressure_9am,
            "pressure_3pm": pressure_3pm,
            "cloud_3pm": cloud_3pm,
            "wind_gust_speed": wind_gust_speed,
            "temp_range": max_temp - min_temp,
            "temp_day_diff": temp_3pm - temp_9am,
            "humidity_day_diff": humidity_3pm - humidity_9am,
            "pressure_day_diff": pressure_3pm - pressure_9am,
            "dewpoint_9am": dp_9am,
            "dewpoint_spread_9am": temp_9am - dp_9am,
            "dewpoint_3pm": dp_3pm,
            "dewpoint_spread_3pm": temp_3pm - dp_3pm,
            "year_cycle_sin": math.sin(2 * math.pi * day_of_year / 365.25),
            "year_cycle_cos": math.cos(2 * math.pi * day_of_year / 365.25),
            "humidity_temp_3pm_interaction": humidity_3pm * temp_3pm / 100.0,
            "pressure_humidity_3pm_ratio": pressure_3pm / max(humidity_3pm + 1.0, 1.0),
            "cloud_humidity_3pm_interaction": cloud_3pm * humidity_3pm,
            "moisture_stability_3pm": (pressure_3pm - 950.0) * (humidity_3pm / 15.0),
            "humidity_rising_fast": 1 if humidity_3pm - humidity_9am >= 15 else 0,
            "warming_day": 1 if temp_3pm > temp_9am else 0,
            "humidity_9am_bin": bin_value(humidity_9am, (50, 65, 80)),
            "pressure_9am_bin": bin_value(pressure_9am, (1005, 1015, 1025)),
            "temp_9am_bin": bin_value(temp_9am, (10, 18, 26)),
            "rainfall_missing_hybrid": 0,
            "cloud_3pm_missing_hybrid": 0,
            "pressure_3pm_missing_hybrid": 0,
            "humidity_3pm_missing_hybrid": 0,
        }
    )

    for zone_col in [
        "rainfall_zone_Summer",
        "rainfall_zone_Summer dominant",
        "rainfall_zone_Uniform",
        "rainfall_zone_Winter",
        "rainfall_zone_Winter dominant",
    ]:
        payload[zone_col] = 0
    zone = str(location_row.get("rainfall_zone", ""))
    matching_col = f"rainfall_zone_{zone}"
    if matching_col in payload:
        payload[matching_col] = 1

    return payload


def predict_many(payloads: list[dict]) -> tuple[list[float] | None, float, str | None]:
    runtime, pool_class, error = load_prediction_runtime()
    if error or runtime is None or pool_class is None:
        return None, THRESHOLD, error

    features = runtime["features"]
    categorical = runtime["categorical"]
    fill_values = runtime["fill_values"]
    threshold = float(runtime["threshold"])
    rows = []
    for payload in payloads:
        row = {}
        for feature in features:
            if feature in payload:
                row[feature] = payload[feature]
            elif feature in categorical:
                row[feature] = str(payload.get("location", "Unknown"))
            else:
                row[feature] = fill_values.get(feature, 0.0)
        rows.append(row)

    frame = pd.DataFrame(rows)[features]
    try:
        pool = pool_class(frame, cat_features=categorical)
        probabilities = runtime["model"].predict_proba(pool)[:, 1].tolist()
    except Exception as exc:  # pragma: no cover - local artifact dependency
        return None, threshold, f"Prediction failed: {exc}"
    return [float(value) for value in probabilities], threshold, None


def build_prediction_scenario(
    selected_location: str,
    observation_date: date,
    min_temp: float,
    max_temp: float,
    rainfall: float,
    humidity_9am: float,
    humidity_3pm: float,
    pressure_9am: float,
    pressure_3pm: float,
    wind_gust_speed: float,
    cloud_3pm: float,
    rain_today: str,
) -> tuple[pd.DataFrame, dict, float, str | None]:
    sample = load_sample_input()
    locations = load_locations()
    if locations.empty or not sample:
        return pd.DataFrame(), {}, THRESHOLD, "Location metadata or sample input is missing."

    payloads = [
        update_payload(
            sample,
            row,
            observation_date,
            min_temp,
            max_temp,
            rainfall,
            humidity_9am,
            humidity_3pm,
            pressure_9am,
            pressure_3pm,
            wind_gust_speed,
            cloud_3pm,
            rain_today,
        )
        for _, row in locations.iterrows()
    ]
    probabilities, threshold, error = predict_many(payloads)
    scenario = locations.copy()
    if probabilities is None:
        scenario["probability"] = None
        selected_payload = next((item for item in payloads if item["location"] == selected_location), payloads[0])
        return scenario, selected_payload, threshold, error

    scenario["probability"] = probabilities
    scenario["prediction"] = (scenario["probability"] >= threshold).astype(int)
    selected_payload = next((item for item in payloads if item["location"] == selected_location), payloads[0])
    return scenario, selected_payload, threshold, None


def prediction_map(frame: pd.DataFrame, selected_location: str, threshold: float, error: str | None) -> None:
    if frame.empty:
        st.info("Station metadata is not available.")
        return

    selected = frame.loc[frame["location"] == selected_location]
    selected_lat = float(selected.iloc[0]["lat"]) if not selected.empty else -25.0
    selected_lon = float(selected.iloc[0]["lon"]) if not selected.empty else 133.0

    if frame["probability"].notna().any():
        color = frame["probability"] * 100.0
        hover = [
            f"{row.location}<br>Rain probability: {row.probability * 100:.1f}%<br>Zone: {row.rainfall_zone}"
            for row in frame.itertuples()
        ]
        colorbar_title = "Rain probability"
        colorscale = [[0, "#278455"], [0.55, "#c99522"], [1, "#1f76d2"]]
    else:
        zones = {name: idx for idx, name in enumerate(sorted(frame["rainfall_zone"].astype(str).unique()))}
        color = frame["rainfall_zone"].astype(str).map(zones)
        hover = [f"{row.location}<br>Zone: {row.rainfall_zone}" for row in frame.itertuples()]
        colorbar_title = "Rainfall zone"
        colorscale = "Viridis"

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lat=frame["lat"],
            lon=frame["lon"],
            text=hover,
            hoverinfo="text",
            mode="markers",
            marker={
                "size": frame["location"].eq(selected_location).map({True: 18, False: 9}),
                "color": color,
                "colorscale": colorscale,
                "cmin": 0,
                "cmax": 100 if frame["probability"].notna().any() else None,
                "line": {"width": frame["location"].eq(selected_location).map({True: 2.8, False: 0.8}), "color": "#173848"},
                "colorbar": {"title": colorbar_title, "thickness": 12},
                "opacity": 0.9,
            },
            name="Weather stations",
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lat=[selected_lat],
            lon=[selected_lon],
            text=[selected_location],
            mode="markers+text",
            textposition="top center",
            marker={"size": 24, "symbol": "star", "color": "#c9473a", "line": {"color": "white", "width": 2}},
            name="Selected station",
            hoverinfo="skip",
        )
    )
    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#edf4ef",
        showocean=True,
        oceancolor="#dceef6",
        showcountries=True,
        countrycolor="rgba(23,56,72,.25)",
        lonaxis_range=[110, 156],
        lataxis_range=[-45, -9],
    )
    fig.update_layout(
        height=420,
        margin={"l": 0, "r": 0, "t": 22, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    if error:
        st.caption(error)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    html(f"<div class='caption'>Map shows all supported WeatherAUS station locations. The selected station is highlighted.</div>")


def prediction_result_panel(
    selected_location: str,
    observation_date: date,
    threshold: float,
    scenario: pd.DataFrame,
    error: str | None,
) -> None:
    prediction_date = observation_date + timedelta(days=1)
    selected = scenario.loc[scenario["location"] == selected_location] if not scenario.empty else pd.DataFrame()
    if error or selected.empty or pd.isna(selected.iloc[0].get("probability")):
        html(
            f"""
            <div class="result-panel">
              <div class="eyebrow">Prediction date</div>
              <div class="answer">{escape(fmt_date(prediction_date))}</div>
              <div class="copy">The live model runtime is needed to score the selected station.</div>
              <div class="mini-grid">
                <div><span>Station</span>{escape(selected_location)}</div>
                <div><span>Observed on</span>{escape(fmt_date(observation_date))}</div>
                <div><span>Threshold</span>{threshold * 100:.0f}%</div>
              </div>
            </div>
            """
        )
        return

    probability = float(selected.iloc[0]["probability"])
    is_rain = probability >= threshold
    label = "Rain likely" if is_rain else "No rain likely"
    css_class = "rain" if is_rain else ""
    html(
        f"""
        <div class="result-panel {css_class}">
          <div class="eyebrow">Prediction for {escape(selected_location)}</div>
          <div class="answer">{escape(label)}</div>
          <div class="copy">Exact prediction date: {escape(fmt_date(prediction_date))}</div>
          <div class="mini-grid">
            <div><span>Rain probability</span>{probability * 100:.1f}%</div>
            <div><span>Decision threshold</span>{threshold * 100:.0f}%</div>
            <div><span>Observed on</span>{escape(fmt_date(observation_date))}</div>
          </div>
        </div>
        """
    )


def prediction_studio() -> None:
    sample = load_sample_input()
    locations = load_locations()
    if sample == {} or locations.empty:
        st.warning("The model sample input or station metadata is missing.")
        return

    location_names = locations["location"].tolist()
    default_location = sample.get("location", "Hobart")
    default_index = location_names.index(default_location) if default_location in location_names else 0
    default_date = sample_observation_date(sample)

    compact_header(
        "Live prediction",
        "Exact next-day forecast example",
        "The scenario combines station context, weather observations, model probability, decision threshold, and geographic coverage in one view.",
        "#009688",
    )
    controls, result_col, map_col = st.columns([0.25, 0.25, 0.50], gap="large")
    with controls:
        selected_location = st.selectbox("Station", location_names, index=default_index)
        observation_date = st.date_input("Observation date", value=default_date)
        if not isinstance(observation_date, date):
            observation_date = default_date
        min_temp = st.slider("Minimum temperature (°C)", -5.0, 35.0, float(sample.get("min_temp", 10.1)), 0.1)
        max_temp = st.slider("Maximum temperature (°C)", -2.0, 48.0, float(sample.get("max_temp", 20.1)), 0.1)
        rainfall = st.slider("Rainfall today (mm)", 0.0, 80.0, float(sample.get("rainfall", 0.0)), 0.2)
        humidity_3pm = st.slider("Humidity at 3pm (%)", 0.0, 100.0, float(sample.get("humidity_3pm", 61.0)), 1.0)
        pressure_3pm = st.slider("Pressure at 3pm (hPa)", 950.0, 1045.0, float(sample.get("pressure_3pm", 1022.0)), 0.1)
        wind_gust_speed = st.slider("Wind gust speed (km/h)", 0.0, 115.0, float(sample.get("wind_gust_speed", 41.0)), 1.0)
        rain_today = st.radio("Rain today", ["No", "Yes"], horizontal=True, index=int(float(sample.get("rain_today", 0)) > 0))

    pressure_9am = float(sample.get("pressure_9am", pressure_3pm + 1.4))
    humidity_9am = float(sample.get("humidity_9am", 53.0))
    cloud_3pm = float(sample.get("cloud_3pm", 6.0))
    scenario, _, threshold, error = build_prediction_scenario(
        selected_location,
        observation_date,
        min_temp,
        max_temp,
        rainfall,
        humidity_9am,
        humidity_3pm,
        pressure_9am,
        pressure_3pm,
        wind_gust_speed,
        cloud_3pm,
        rain_today,
    )

    with result_col:
        prediction_result_panel(selected_location, observation_date, threshold, scenario, error)
    with map_col:
        prediction_map(scenario, selected_location, threshold, error)


def image_with_caption(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), width="stretch")
        html(f"<div class='caption'>{escape(caption)}</div>")


def stage_cards(items: list[tuple[str, str, str, str]]) -> None:
    blocks = ["<div class='stage-grid'>"]
    for step, title, text, color in items:
        blocks.append(
            f"<div class='stage-card' style='--c:{escape(color)}'>"
            f"<div class='step'>{escape(step)}</div>"
            f"<b>{escape(title)}</b>"
            f"<span>{escape(text)}</span>"
            "</div>"
        )
    blocks.append("</div>")
    html("".join(blocks))


def data_science_page() -> None:
    hero()
    prediction_date = sample_observation_date(load_sample_input()) + timedelta(days=1)
    prediction_studio()

    st.divider()
    metric_cards(
        [
            ("Winner model", "CatBoost", "hybrid tabular weather classifier"),
            ("Feature contract", str(FEATURE_COUNT), "stored with metadata and sample input"),
            ("ROC AUC", fmt_pct(METRICS.get("roc_auc")), "chronological holdout evaluation"),
            ("Try-out date", fmt_date(prediction_date), "exact next-day prediction example"),
        ]
    )

    st.divider()
    left, right = st.columns([0.42, 0.58], gap="large")
    with left:
        story_card(
            "Data science foundation",
            [
                "WeatherAUS station observations become a binary rain-tomorrow classification problem.",
                "Feature engineering adds location, season, missingness memory, lag signals, wind vectors, and moisture features.",
                "The split is chronological, so the final test window behaves more like future data.",
                "The model package keeps the feature order, fill values, threshold, metadata, and sample input together.",
            ],
        )
    with right:
        image_with_caption(CHRONOLOGICAL_SPLIT_PATH, "Time-aware split used before the production workflow.")
        image_with_caption(THRESHOLD_CURVE_PATH, "The saved threshold is part of the deployed model contract.")

    st.divider()
    fig1, fig2, fig3 = st.columns(3, gap="medium")
    with fig1:
        image_with_caption(CLASS_IMBALANCE_PATH, "Rain/no-rain imbalance explains why recall matters.")
    with fig2:
        image_with_caption(SEASONAL_PATTERNS_PATH, "Seasonality is visible before modeling.")
    with fig3:
        image_with_caption(KOPPEN_HEATMAP_PATH, "Climate-zone evidence keeps geography in the story.")


def architecture_page() -> None:
    section_header(
        "MLOps architecture",
        "A rain model operating system",
        "The diagram separates monitoring, training, and deployment so the project stack is readable at presentation scale.",
        "#326ce5",
    )
    tool_chips(
        [
            ("DVC", "DVC", "#945dd6"),
            ("Airflow", "AF", "#017cee"),
            ("MLflow", "ML", "#0194e2"),
            ("Evidently", "EV", "#e31b23"),
            ("Pushgateway", "PG", "#f39c12"),
            ("FastAPI", "FA", "#009688"),
            ("Docker", "Do", "#2496ed"),
            ("Kubernetes", "K8s", "#326ce5"),
        ]
    )
    if ARCHITECTURE_IMAGE_PATH.exists():
        st.image(str(ARCHITECTURE_IMAGE_PATH), width="stretch")
    else:
        st.warning("Architecture image is missing.")
    stage_cards(
        [
            ("Data", "Source to raw table", "Weather rows are extracted and merged by Date and Location.", "#1f76d2"),
            ("Train", "CatBoost package", "Training writes the model, metadata, sample input, and reference dataset.", "#278455"),
            ("Track", "DVC + MLflow", "Artifacts and model metadata stay inspectable outside notebooks.", "#945dd6"),
            ("Monitor", "Evidently", "Reference/current drift checks feed monitoring signals.", "#e31b23"),
            ("Deploy", "Kubernetes", "Airflow and FastAPI run with services, PVCs, HPAs, and PDBs.", "#326ce5"),
        ]
    )


def structure_page() -> None:
    section_header(
        "Project structure",
        "Production implementation structure",
        "The project is organized as data flow, artifact flow, automation flow, monitoring flow, and runtime flow.",
        "#278455",
    )
    rows = [
        ("01", "Data preparation", "Extract, normalize, validate, and upsert incoming weather rows into the raw WeatherAUS-style dataset.", "#1f76d2"),
        ("02", "Feature contract", "Build the 68-feature CatBoost contract with location, missingness, lag, wind, pressure, humidity, and climate context.", "#278455"),
        ("03", "Model artifact", "Write winner_model.joblib, metadata.json, sample_input.json, and reference_dataset.csv as one operating package.", "#c99522"),
        ("04", "DVC lineage", "Track raw data, model, prepared tables, and monitoring reference files through DVC pointers and status records.", "#945dd6"),
        ("05", "Airflow automation", "Schedule ingestion, versioning, end-to-end retraining, MLflow handoff, API validation, and drift monitoring.", "#017cee"),
        ("06", "Kubernetes runtime", "Run Airflow, FastAPI, Postgres, Redis, Pushgateway, PVCs, HPAs, PDBs, and services from kustomization.", "#326ce5"),
    ]
    stage_cards(rows)
    st.divider()
    c1, c2 = st.columns([0.48, 0.52], gap="large")
    with c1:
        image_with_caption(MODEL_COMPARISON_PATH, "Model comparison evidence from the data science base.")
    with c2:
        story_card(
            "Operational evidence",
            [
                "Airflow contains scheduled workflows for ingestion, versioning, training, validation, and drift monitoring.",
                "New data enters the same training/test path through the raw dataset.",
                "DVC and MLflow give traceability for artifacts and model metadata.",
                "Evidently and Pushgateway connect batch monitoring to the runtime.",
                "Kubernetes is the deployment target; local integration is only a same-machine support context.",
            ],
        )


def ci_page() -> None:
    section_header(
        "CI pipeline",
        "The merge-time quality gate",
        "The pipeline checks the contracts that keep the production workflow from silently drifting.",
        "#2088ff",
    )
    metric_cards(
        [
            ("Orchestration", "DAG compile", "Airflow files must load before merge"),
            ("Dependencies", "Pinned", "Evidently, Plotly, MLflow, and Airflow stay compatible"),
            ("Contracts", "API + drift", "Prediction and monitoring expectations are tested"),
            ("Images", "Build check", "FastAPI and Airflow images remain reproducible"),
        ]
    )
    stage_cards(
        [
            ("Step 1", "Repository context", "Checkout, Python setup, and dependency resolution for the workflow checks.", "#1f76d2"),
            ("Step 2", "Module compile", "DAGs, extraction, versioning, monitoring, and API contract files must import cleanly.", "#017cee"),
            ("Step 3", "Focused tests", "Airflow automation, dependency, MLflow, drift, and prediction API checks protect core contracts.", "#278455"),
            ("Step 4", "Image validation", "Local integration config and FastAPI/Airflow image builds are validated.", "#326ce5"),
            ("Step 5", "Merge readiness", "The branch is evaluated through reproducible checks before integration.", "#6557b6"),
        ]
    )


def live_demo_page() -> None:
    section_header(
        "Live system evidence",
        "Production components visible in the running project",
        "The operating system is represented by quality gates, orchestration, data lineage, tracking, monitoring, and Kubernetes runtime resources.",
        "#c99522",
    )
    items = [
        ("GitHub Actions", "Compile checks, focused tests, dependency pins, and image build validation.", "#2088ff"),
        ("Airflow", "Scheduled DAGs for ingestion, versioning, end-to-end training, validation, and drift monitoring.", "#017cee"),
        ("DVC", "Raw/model pointers, reproducible lineage, and remote-backed artifact state.", "#945dd6"),
        ("MLflow", "Model metrics, parameters, metadata, and artifact logging path.", "#0194e2"),
        ("Evidently", "Reference/current drift reports with Pushgateway metric handoff.", "#e31b23"),
        ("Kubernetes", "Pods, services, PVCs, HPAs, PDBs, and model API runtime resources.", "#326ce5"),
    ]
    cards = ["<div class='demo-grid'>"]
    for title, body, color in items:
        cards.append(
            f"<div class='demo-card' style='--c:{escape(color)}'>"
            f"<b>{escape(title)}</b>"
            f"<span>{escape(body)}</span>"
            "</div>"
        )
    cards.append("</div>")
    html("".join(cards))
    html(
        """
        <div class="handoff">
          System summary: the model is no longer only a notebook result. It is a reproducible production workflow with lineage,
          scheduling, monitoring, CI, and a Kubernetes runtime shape.
        </div>
        """
    )


PAGES = [
    ("Data Science + Prediction", data_science_page),
    ("MLOps Architecture", architecture_page),
    ("Structure", structure_page),
    ("CI Pipeline", ci_page),
    ("Live System", live_demo_page),
]


def sidebar_navigation() -> tuple[str, int]:
    labels = [label for label, _ in PAGES]
    if "active_page" not in st.session_state:
        st.session_state.active_page = labels[0]
    if st.session_state.active_page not in labels:
        st.session_state.active_page = labels[0]

    with st.sidebar:
        st.markdown("### Rain Prediction MLOps")
        st.markdown("Production story for the rain prediction project.")
        selected = st.radio("Sections", labels, index=labels.index(st.session_state.active_page), label_visibility="collapsed")
        st.session_state.active_page = selected
        current_index = labels.index(st.session_state.active_page)
        html(
            f"""
            <div class="side-progress"><span style="--progress:{((current_index + 1) / len(labels)) * 100:.0f}%"></span></div>
            <div class="side-caption">Slide {current_index + 1} of {len(labels)}</div>
            """
        )
        prev_col, next_col = st.columns(2)
        with prev_col:
            if st.button("Previous", disabled=current_index == 0, width="stretch", key="sidebar_previous"):
                st.session_state.active_page = labels[current_index - 1]
                st.rerun()
        with next_col:
            if st.button("Next", disabled=current_index == len(labels) - 1, width="stretch", key="sidebar_next"):
                st.session_state.active_page = labels[current_index + 1]
                st.rerun()
        st.divider()
        st.metric("Winner", "CatBoost")
        st.metric("Features", FEATURE_COUNT)
        st.metric("ROC AUC", fmt_pct(METRICS.get("roc_auc")))
    return st.session_state.active_page, labels.index(st.session_state.active_page)


def deck_controls(index: int, position: str) -> None:
    labels = [label for label, _ in PAGES]
    prev_col, mid_col, next_col = st.columns([1.1, 4.0, 1.1])
    with prev_col:
        if st.button("Previous", disabled=index == 0, width="stretch", key=f"{position}_previous"):
            st.session_state.active_page = labels[index - 1]
            st.rerun()
    with mid_col:
        st.progress((index + 1) / len(labels))
    with next_col:
        if st.button("Next", disabled=index == len(labels) - 1, width="stretch", key=f"{position}_next"):
            st.session_state.active_page = labels[index + 1]
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Rain Prediction MLOps", layout="wide", initial_sidebar_state="expanded")
    inject_theme()
    selected, index = sidebar_navigation()
    _, renderer = PAGES[index]
    renderer()


if __name__ == "__main__":
    main()
