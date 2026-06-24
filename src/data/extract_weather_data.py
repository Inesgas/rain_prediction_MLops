from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.paths import BASE_DIR, RAW_BASE


DEFAULT_REPORT_DIR = BASE_DIR / "reports" / "versioning"
DEFAULT_SOURCE_ENV = "WEATHER_AUS_SOURCE"
DEFAULT_SOURCE_URL_ENV = "WEATHER_AUS_SOURCE_URL"
DEFAULT_MODE_ENV = "WEATHER_AUS_EXTRACT_MODE"
DEFAULT_SOURCE_SHA256_ENV = "WEATHER_AUS_SOURCE_SHA256"
DEFAULT_KAGGLE_DATASET_ENV = "WEATHER_AUS_KAGGLE_DATASET"
DEFAULT_KAGGLE_FILE_ENV = "WEATHER_AUS_KAGGLE_FILE"
DEFAULT_DAILY_PROVIDER_ENV = "WEATHER_AUS_DAILY_PROVIDER"
DEFAULT_SOURCE_MIN_ROWS_ENV = "WEATHER_AUS_SOURCE_MIN_ROWS"
DEFAULT_DOWNLOAD_DIR = BASE_DIR / "data" / "incoming" / "downloads"
KEY_COLUMNS = ["Date", "Location"]
EXPECTED_COLUMNS = [
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


def safe_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": relative(path),
        "exists": path.exists(),
    }
    if path.exists():
        record.update(
            {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "modified_at_utc": datetime.fromtimestamp(
                    path.stat().st_mtime,
                    tz=timezone.utc,
                ).replace(microsecond=0).isoformat(),
            }
        )
    return record


def resolve_source(source: str | None) -> Path | None:
    source = source or os.environ.get(DEFAULT_SOURCE_ENV)
    if not source:
        return None
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def resolve_env_value(value: str | None, env_key: str) -> str | None:
    value = value or os.environ.get(env_key)
    if value is None or not value.strip():
        return None
    return value.strip()


def filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    return name or "weatheraus_download"


def download_url_source(url: str, download_dir: Path, expected_sha256: str | None) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    target = download_dir / filename_from_url(url)
    temp_target = target.with_suffix(f"{target.suffix}.tmp")

    with urllib.request.urlopen(url, timeout=120) as response, temp_target.open("wb") as file_obj:
        shutil.copyfileobj(response, file_obj)

    if expected_sha256:
        actual_sha256 = sha256_file(temp_target)
        if actual_sha256.lower() != expected_sha256.lower():
            temp_target.unlink(missing_ok=True)
            raise ValueError(
                f"Downloaded source hash mismatch. Expected {expected_sha256}, got {actual_sha256}."
            )

    temp_target.replace(target)
    return target


def download_kaggle_source(dataset: str, filename: str, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-f",
        filename,
        "-p",
        str(download_dir),
        "--unzip",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Kaggle download failed. Confirm the kaggle package is installed and "
            "KAGGLE_USERNAME/KAGGLE_KEY or kaggle.json credentials are configured.\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )

    preferred = download_dir / filename
    if preferred.exists():
        return preferred
    return find_csv_in_directory(download_dir)


def validate_weather_csv(path: Path, min_rows: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Weather data file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.reader(file_obj)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Weather data file is empty: {path}") from exc
        row_count = sum(1 for _ in reader)

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in header]
    if missing_columns:
        formatted = ", ".join(missing_columns)
        raise ValueError(f"Weather data is missing required columns: {formatted}")
    if row_count < min_rows:
        raise ValueError(f"Weather data has {row_count} rows; expected at least {min_rows}.")

    return {
        "path": relative(path),
        "rows": row_count,
        "columns": len(header),
        "required_columns_present": True,
    }


def find_csv_in_directory(source_dir: Path) -> Path:
    preferred = source_dir / RAW_BASE.name
    if preferred.exists():
        return preferred
    candidates = sorted(source_dir.glob("*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in source directory: {source_dir}")
    names = ", ".join(candidate.name for candidate in candidates)
    raise ValueError(f"Multiple CSV files found in {source_dir}; expected {RAW_BASE.name}. Found: {names}")


def extract_csv_from_zip(source_zip: Path) -> Path:
    with zipfile.ZipFile(source_zip) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        preferred = [name for name in members if Path(name).name == RAW_BASE.name]
        if preferred:
            member = preferred[0]
        elif len(members) == 1:
            member = members[0]
        elif not members:
            raise FileNotFoundError(f"No CSV files found in ZIP archive: {source_zip}")
        else:
            names = ", ".join(Path(name).name for name in members)
            raise ValueError(f"Multiple CSV files found in ZIP archive; expected {RAW_BASE.name}. Found: {names}")

        temp_dir = Path(tempfile.mkdtemp(prefix="weatheraus_extract_"))
        extracted = temp_dir / RAW_BASE.name
        with archive.open(member) as source_obj, extracted.open("wb") as target_obj:
            shutil.copyfileobj(source_obj, target_obj)
    return extracted


def materialize_source_csv(source: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Configured data source does not exist: {source}")
    if source.is_dir():
        return find_csv_in_directory(source)
    if source.suffix.lower() == ".zip":
        return extract_csv_from_zip(source)
    if source.suffix.lower() == ".csv":
        return source
    raise ValueError(f"Unsupported data source type: {source}. Use a CSV, ZIP, or directory.")


def copy_if_changed(source_csv: Path, target_csv: Path) -> str:
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    if target_csv.exists() and sha256_file(source_csv) == sha256_file(target_csv):
        return "unchanged"

    temp_target = target_csv.with_suffix(f"{target_csv.suffix}.tmp")
    shutil.copyfile(source_csv, temp_target)
    temp_target.replace(target_csv)
    return "copied"


def sort_weather_rows(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["__date_sort"] = pd.to_datetime(result["Date"], errors="coerce")
    result = result.sort_values(["Location", "__date_sort", "Date"], kind="mergesort")
    return result.drop(columns=["__date_sort"]).reset_index(drop=True)


def write_csv_atomic(df: pd.DataFrame, target_csv: Path) -> None:
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_csv.with_suffix(f"{target_csv.suffix}.tmp")
    df.to_csv(temp_target, index=False)
    temp_target.replace(target_csv)


def compare_common_rows(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    common_keys: set[tuple[Any, ...]],
) -> int:
    if not common_keys:
        return 0

    existing_common = existing.set_index(KEY_COLUMNS, drop=False)
    incoming_common = incoming.set_index(KEY_COLUMNS, drop=False)
    compare_columns = [column for column in incoming.columns if column in existing.columns]
    changed = 0

    for key in common_keys:
        left = existing_common.loc[key, compare_columns].astype(str).fillna("")
        right = incoming_common.loc[key, compare_columns].astype(str).fillna("")
        if not left.equals(right):
            changed += 1
    return changed


def merge_incremental_data(source_csv: Path, target_csv: Path, mode: str) -> dict[str, Any]:
    incoming = pd.read_csv(source_csv)
    incoming_before_dedup = len(incoming)
    incoming = incoming.drop_duplicates(subset=KEY_COLUMNS, keep="last").reset_index(drop=True)

    if not target_csv.exists():
        output = sort_weather_rows(incoming)
        write_csv_atomic(output, target_csv)
        return {
            "mode": mode,
            "existing_rows": 0,
            "incoming_rows": int(incoming_before_dedup),
            "incoming_duplicate_key_rows": int(incoming_before_dedup - len(incoming)),
            "inserted_rows": int(len(incoming)),
            "updated_rows": 0,
            "unchanged_overlap_rows": 0,
            "output_rows": int(len(output)),
        }

    existing = pd.read_csv(target_csv)
    existing_keys = set(existing[KEY_COLUMNS].itertuples(index=False, name=None))
    incoming_keys = set(incoming[KEY_COLUMNS].itertuples(index=False, name=None))
    common_keys = existing_keys & incoming_keys
    new_keys = incoming_keys - existing_keys
    changed_overlap = compare_common_rows(existing, incoming, common_keys)

    if mode == "append" and changed_overlap:
        raise ValueError(
            f"Incoming data changes {changed_overlap} existing Date/Location rows. "
            "Use WEATHER_AUS_EXTRACT_MODE=upsert if those corrections should replace local rows."
        )

    if mode == "append":
        new_rows = incoming[incoming[KEY_COLUMNS].apply(tuple, axis=1).isin(new_keys)]
        output = pd.concat([existing, new_rows], ignore_index=True)
        updated_rows = 0
    elif mode == "upsert":
        incoming_key_series = incoming[KEY_COLUMNS].apply(tuple, axis=1)
        existing_key_series = existing[KEY_COLUMNS].apply(tuple, axis=1)
        existing_without_incoming = existing[~existing_key_series.isin(incoming_keys)]
        output = pd.concat([existing_without_incoming, incoming], ignore_index=True)
        updated_rows = changed_overlap
        new_keys = set(incoming_key_series) - existing_keys
    else:
        raise ValueError("mode must be 'append' or 'upsert'")

    output = sort_weather_rows(output)
    write_csv_atomic(output, target_csv)
    return {
        "mode": mode,
        "existing_rows": int(len(existing)),
        "incoming_rows": int(incoming_before_dedup),
        "incoming_duplicate_key_rows": int(incoming_before_dedup - len(incoming)),
        "inserted_rows": int(len(new_keys)),
        "updated_rows": int(updated_rows),
        "unchanged_overlap_rows": int(len(common_keys) - changed_overlap),
        "output_rows": int(len(output)),
    }


def write_manifest(report_dir: Path, manifest: dict[str, Any], run_id: str | None) -> Path:
    safe_run_id = (run_id or "manual").replace("/", "_").replace(":", "_")
    filename = f"{safe_timestamp()}-extraction-{safe_run_id}.json"
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / filename
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except PermissionError:
        fallback_dir = Path(tempfile.gettempdir()) / "rain_prediction_mlops" / "versioning"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        path = fallback_dir / filename
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Warning: report directory is not writable; used temporary path: {path}")
        return path

    print(f"Wrote extraction manifest: {relative(path)}")
    return path


def extract_weather_data(
    source: Path | None,
    target: Path,
    report_dir: Path,
    min_rows: int,
    source_min_rows: int,
    run_id: str | None,
    mode: str,
    online_source_url: str | None = None,
    kaggle_dataset: str | None = None,
) -> Path:
    source_csv: Path | None = None
    action = "validated_existing"
    merge_summary: dict[str, Any] | None = None

    if source is not None:
        source_csv = materialize_source_csv(source)
        validate_weather_csv(source_csv, min_rows=source_min_rows)
        if mode == "replace":
            action = copy_if_changed(source_csv, target)
        else:
            merge_summary = merge_incremental_data(source_csv, target, mode=mode)
            action = "merged_incremental"
    elif not target.exists():
        raise FileNotFoundError(
            f"No local raw dataset found at {target}. "
            f"Set {DEFAULT_SOURCE_ENV} to a local CSV, ZIP, or directory."
        )

    validation = validate_weather_csv(target, min_rows=min_rows)
    manifest = {
        "schema_version": "1.0",
        "project": "rain_prediction_mlops",
        "stage": "raw_weather_extraction",
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "action": action,
        "mode": mode,
        "online_source_url": online_source_url,
        "kaggle_dataset": kaggle_dataset,
        "source": file_record(source_csv) if source_csv is not None else None,
        "target": file_record(target),
        "merge_summary": merge_summary,
        "validation": validation,
        "local_only": True,
        "remote_push_performed": False,
    }
    write_manifest(report_dir, manifest, run_id=run_id)
    print(f"Raw weather data {action}: {relative(target)}")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract and validate the local WeatherAUS raw dataset.")
    parser.add_argument("--source", default=None, help=f"Local CSV, ZIP, or directory. Defaults to ${DEFAULT_SOURCE_ENV}.")
    parser.add_argument("--source-url", default=None, help=f"Direct CSV or ZIP URL. Defaults to ${DEFAULT_SOURCE_URL_ENV}.")
    parser.add_argument("--source-sha256", default=None, help=f"Optional source hash. Defaults to ${DEFAULT_SOURCE_SHA256_ENV}.")
    parser.add_argument("--kaggle-dataset", default=None, help=f"Kaggle dataset slug. Defaults to ${DEFAULT_KAGGLE_DATASET_ENV}.")
    parser.add_argument("--kaggle-file", default=None, help=f"Kaggle file name. Defaults to ${DEFAULT_KAGGLE_FILE_ENV} or weatherAUS.csv.")
    parser.add_argument(
        "--daily-provider",
        default=os.environ.get(DEFAULT_DAILY_PROVIDER_ENV),
        choices=["open-meteo"],
        help=f"Fetch a daily online provider before local upsert. Defaults to ${DEFAULT_DAILY_PROVIDER_ENV}.",
    )
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument(
        "--mode",
        choices=["append", "upsert", "replace"],
        default=os.environ.get(DEFAULT_MODE_ENV, "upsert"),
        help="How to combine a configured source with the existing raw file.",
    )
    parser.add_argument("--target", type=Path, default=RAW_BASE)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--source-min-rows", type=int, default=int(os.environ.get(DEFAULT_SOURCE_MIN_ROWS_ENV, "1")))
    parser.add_argument("--run-id", default=os.environ.get("AIRFLOW_CTX_RUN_ID"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = resolve_source(args.source)
    source_url = resolve_env_value(args.source_url, DEFAULT_SOURCE_URL_ENV)
    source_sha256 = resolve_env_value(args.source_sha256, DEFAULT_SOURCE_SHA256_ENV)
    kaggle_dataset = resolve_env_value(args.kaggle_dataset, DEFAULT_KAGGLE_DATASET_ENV)
    kaggle_file = resolve_env_value(args.kaggle_file, DEFAULT_KAGGLE_FILE_ENV) or RAW_BASE.name

    if source is None and source_url:
        source = download_url_source(source_url, args.download_dir, expected_sha256=source_sha256)
    elif source is None and kaggle_dataset:
        source = download_kaggle_source(kaggle_dataset, kaggle_file, args.download_dir)
    elif source is None and args.daily_provider == "open-meteo":
        from src.data.extract_open_meteo_daily import fetch_open_meteo_daily

        source = fetch_open_meteo_daily()

    extract_weather_data(
        source=source,
        target=args.target,
        report_dir=args.report_dir,
        min_rows=args.min_rows,
        source_min_rows=args.source_min_rows,
        run_id=args.run_id,
        mode=args.mode,
        online_source_url=source_url,
        kaggle_dataset=kaggle_dataset,
    )


if __name__ == "__main__":
    main()
