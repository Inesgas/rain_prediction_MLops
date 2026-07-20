from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.paths import BASE_DIR


DEFAULT_METADATA_PATH = BASE_DIR / "data" / "preprocessed" / "locations_metadata.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "incoming" / "open_meteo_weatherAUS_daily.csv"
DEFAULT_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
RAIN_THRESHOLD_MM = 1.0

DAILY_VARIABLES = [
    "temperature_2m_min",
    "temperature_2m_max",
    "precipitation_sum",
    "et0_fao_evapotranspiration_sum",
    "sunshine_duration",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
]
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
]
WEATHER_AUS_COLUMNS = [
    "Date",
    "Location",
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustDir",
    "WindGustSpeed",
    "WindDir9am",
    "WindDir3pm",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Cloud9am",
    "Cloud3pm",
    "Temp9am",
    "Temp3pm",
    "RainToday",
    "RainTomorrow",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def default_date_window(days_back: int, end_lag_days: int) -> tuple[date, date]:
    end_date = date.today() - timedelta(days=end_lag_days)
    start_date = end_date - timedelta(days=max(days_back - 1, 0))
    return start_date, end_date


def degrees_to_compass(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    directions = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    idx = int((float(value) + 11.25) // 22.5) % 16
    return directions[idx]


def cloud_percent_to_oktas(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(max(0, min(8, round(float(value) / 12.5))))


def rain_label(rainfall_mm: Any) -> str | None:
    if rainfall_mm is None or pd.isna(rainfall_mm):
        return None
    return "Yes" if float(rainfall_mm) > RAIN_THRESHOLD_MM else "No"


def request_open_meteo(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    archive_url: str,
    max_retries: int,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(DAILY_VARIABLES),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    url = f"{archive_url}?{urllib.parse.urlencode(params)}"
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "rain-prediction-mlops-airflow/1.0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_delay_seconds * attempt)

    raise RuntimeError(f"Open-Meteo request failed after {max_retries} attempt(s): {last_error}")


def hourly_value(hourly: dict[str, list[Any]], variable: str, day: str, hour: int) -> Any:
    timestamps = hourly.get("time", [])
    target_prefix = f"{day}T{hour:02d}:"
    for idx, timestamp in enumerate(timestamps):
        if str(timestamp).startswith(target_prefix):
            values = hourly.get(variable, [])
            return values[idx] if idx < len(values) else None
    return None


def daily_value(daily: dict[str, list[Any]], variable: str, idx: int) -> Any:
    values = daily.get(variable, [])
    return values[idx] if idx < len(values) else None


def build_location_frame(
    location: str,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    archive_url: str,
    max_retries: int,
    retry_delay_seconds: float,
) -> pd.DataFrame:
    payload = request_open_meteo(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        archive_url=archive_url,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )
    daily = payload.get("daily", {})
    hourly = payload.get("hourly", {})
    rows: list[dict[str, Any]] = []

    for idx, day in enumerate(daily.get("time", [])):
        rainfall = daily_value(daily, "precipitation_sum", idx)
        rows.append(
            {
                "Date": day,
                "Location": location,
                "MinTemp": daily_value(daily, "temperature_2m_min", idx),
                "MaxTemp": daily_value(daily, "temperature_2m_max", idx),
                "Rainfall": rainfall,
                "Evaporation": daily_value(daily, "et0_fao_evapotranspiration_sum", idx),
                "Sunshine": (
                    daily_value(daily, "sunshine_duration", idx) / 3600.0
                    if daily_value(daily, "sunshine_duration", idx) is not None
                    else None
                ),
                "WindGustDir": degrees_to_compass(daily_value(daily, "wind_direction_10m_dominant", idx)),
                "WindGustSpeed": daily_value(daily, "wind_gusts_10m_max", idx),
                "WindDir9am": degrees_to_compass(hourly_value(hourly, "wind_direction_10m", day, 9)),
                "WindDir3pm": degrees_to_compass(hourly_value(hourly, "wind_direction_10m", day, 15)),
                "WindSpeed9am": hourly_value(hourly, "wind_speed_10m", day, 9),
                "WindSpeed3pm": hourly_value(hourly, "wind_speed_10m", day, 15),
                "Humidity9am": hourly_value(hourly, "relative_humidity_2m", day, 9),
                "Humidity3pm": hourly_value(hourly, "relative_humidity_2m", day, 15),
                "Pressure9am": hourly_value(hourly, "pressure_msl", day, 9),
                "Pressure3pm": hourly_value(hourly, "pressure_msl", day, 15),
                "Cloud9am": cloud_percent_to_oktas(hourly_value(hourly, "cloud_cover", day, 9)),
                "Cloud3pm": cloud_percent_to_oktas(hourly_value(hourly, "cloud_cover", day, 15)),
                "Temp9am": hourly_value(hourly, "temperature_2m", day, 9),
                "Temp3pm": hourly_value(hourly, "temperature_2m", day, 15),
                "RainToday": rain_label(rainfall),
                "RainTomorrow": None,
            }
        )

    frame = pd.DataFrame(rows, columns=WEATHER_AUS_COLUMNS)
    if not frame.empty:
        next_rain = frame["Rainfall"].shift(-1)
        frame["RainTomorrow"] = next_rain.map(rain_label)
    return frame


def load_locations(metadata_path: Path, requested_locations: list[str] | None) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Location metadata not found: {metadata_path}")
    locations = pd.read_csv(metadata_path)
    required = {"location", "lat", "lon"}
    missing = required - set(locations.columns)
    if missing:
        formatted = ", ".join(sorted(missing))
        raise ValueError(f"Location metadata is missing required columns: {formatted}")

    locations = locations[["location", "lat", "lon"]].dropna().drop_duplicates(subset=["location"])
    if requested_locations:
        keep = set(requested_locations)
        locations = locations[locations["location"].isin(keep)]
        missing_requested = keep - set(locations["location"])
        if missing_requested:
            formatted = ", ".join(sorted(missing_requested))
            raise ValueError(f"Requested locations not found in metadata: {formatted}")
    return locations.sort_values("location").reset_index(drop=True)


def fetch_open_meteo_daily(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    start_date: date | None = None,
    end_date: date | None = None,
    days_back: int = 7,
    end_lag_days: int = 1,
    archive_url: str = DEFAULT_ARCHIVE_URL,
    requested_locations: list[str] | None = None,
    max_retries: int | None = None,
    retry_delay_seconds: float | None = None,
    request_delay_seconds: float | None = None,
) -> Path:
    max_retries = max_retries if max_retries is not None else int(os.environ.get("OPEN_METEO_MAX_RETRIES", "3"))
    retry_delay_seconds = (
        retry_delay_seconds
        if retry_delay_seconds is not None
        else float(os.environ.get("OPEN_METEO_RETRY_DELAY_SECONDS", "2.0"))
    )
    request_delay_seconds = (
        request_delay_seconds
        if request_delay_seconds is not None
        else float(os.environ.get("OPEN_METEO_REQUEST_DELAY_SECONDS", "0.25"))
    )

    if start_date is None or end_date is None:
        default_start, default_end = default_date_window(days_back=days_back, end_lag_days=end_lag_days)
        start_date = start_date or default_start
        end_date = end_date or default_end

    locations = load_locations(metadata_path, requested_locations)
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    for row in locations.itertuples(index=False):
        try:
            frames.append(
                build_location_frame(
                    location=row.location,
                    latitude=float(row.lat),
                    longitude=float(row.lon),
                    start_date=start_date,
                    end_date=end_date,
                    archive_url=archive_url,
                    max_retries=max_retries,
                    retry_delay_seconds=retry_delay_seconds,
                )
            )
            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)
        except Exception as exc:
            failures.append({"location": str(row.location), "error": str(exc)})

    if not frames:
        if failures:
            details = "; ".join(f"{item['location']}: {item['error']}" for item in failures[:5])
            raise RuntimeError(f"Open-Meteo extraction failed for all {len(failures)} location(s): {details}")
        raise RuntimeError("Open-Meteo extraction produced no rows.")
    if failures:
        details = "; ".join(f"{item['location']}: {item['error']}" for item in failures[:5])
        print(
            "WARNING: Open-Meteo extraction skipped "
            f"{len(failures)} of {len(locations)} location(s): {details}"
        )

    output = pd.concat(frames, ignore_index=True)
    output = output.sort_values(["Location", "Date"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(
        "Fetched Open-Meteo daily data "
        f"for {len(locations)} locations, {start_date.isoformat()} to {end_date.isoformat()}: "
        f"{output_path}"
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch daily Australian weather rows from Open-Meteo.")
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("WEATHER_AUS_OPEN_METEO_OUTPUT", DEFAULT_OUTPUT_PATH)))
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--start-date", default=os.environ.get("WEATHER_AUS_DAILY_START_DATE"))
    parser.add_argument("--end-date", default=os.environ.get("WEATHER_AUS_DAILY_END_DATE"))
    parser.add_argument("--days-back", type=int, default=int(os.environ.get("WEATHER_AUS_DAILY_DAYS_BACK", "7")))
    parser.add_argument("--end-lag-days", type=int, default=int(os.environ.get("WEATHER_AUS_DAILY_END_LAG_DAYS", "1")))
    parser.add_argument("--archive-url", default=os.environ.get("OPEN_METEO_ARCHIVE_URL", DEFAULT_ARCHIVE_URL))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("OPEN_METEO_MAX_RETRIES", "3")))
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=float(os.environ.get("OPEN_METEO_RETRY_DELAY_SECONDS", "2.0")),
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=float(os.environ.get("OPEN_METEO_REQUEST_DELAY_SECONDS", "0.25")),
    )
    parser.add_argument("--location", action="append", dest="locations", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fetch_open_meteo_daily(
        output_path=args.output,
        metadata_path=args.metadata,
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        days_back=args.days_back,
        end_lag_days=args.end_lag_days,
        archive_url=args.archive_url,
        requested_locations=args.locations,
        max_retries=args.max_retries,
        retry_delay_seconds=args.retry_delay_seconds,
        request_delay_seconds=args.request_delay_seconds,
    )


if __name__ == "__main__":
    main()
