from __future__ import annotations

import argparse
import json
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.paths import BASE_DIR, RAW_BASE


DEFAULT_REPORT_DIR = BASE_DIR / "reports" / "versioning"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_latest_date(path: Path) -> date:
    if not path.exists():
        raise FileNotFoundError(f"Raw weather data file not found: {path}")

    frame = pd.read_csv(path, usecols=["Date"])
    if frame.empty:
        raise ValueError(f"Raw weather data file is empty: {path}")

    parsed_dates = pd.to_datetime(frame["Date"], errors="coerce").dropna()
    if parsed_dates.empty:
        raise ValueError(f"No valid Date values found in: {path}")

    return parsed_dates.max().date()


def build_freshness_report(path: Path, max_lag_days: int) -> dict[str, Any]:
    latest_date = load_latest_date(path)
    expected_latest = date.today() - timedelta(days=max_lag_days)
    lag_days = (date.today() - latest_date).days
    is_fresh = latest_date >= expected_latest

    return {
        "schema_version": "1.0",
        "project": "rain_prediction_mlops",
        "stage": "data_freshness_check",
        "created_at_utc": utc_now(),
        "data_path": relative(path),
        "latest_data_date": latest_date.isoformat(),
        "expected_latest_date": expected_latest.isoformat(),
        "max_lag_days": max_lag_days,
        "observed_lag_days": lag_days,
        "fresh": is_fresh,
        "realtime_streaming_required": False,
        "reason": (
            "The training label RainTomorrow is only complete after the next day. "
            "Daily API ingestion with a label lag is the correct automation level."
        ),
    }


def write_report(report: dict[str, Any], report_dir: Path, run_id: str | None) -> Path:
    safe_run_id = (run_id or "manual").replace("/", "_").replace(":", "_")
    filename = f"{safe_timestamp()}-freshness-{safe_run_id}.json"
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / filename
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except PermissionError:
        fallback_dir = Path(tempfile.gettempdir()) / "rain_prediction_mlops" / "versioning"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        path = fallback_dir / filename
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Warning: report directory is not writable; used temporary path: {path}")
        return path

    print(f"Wrote data freshness report: {relative(path)}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether the raw weather dataset is fresh enough for training.")
    parser.add_argument("--target", type=Path, default=RAW_BASE)
    parser.add_argument("--max-lag-days", type=int, default=2)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--warn-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_freshness_report(path=args.target, max_lag_days=args.max_lag_days)
    write_report(report, report_dir=args.report_dir, run_id=args.run_id)

    message = (
        f"Latest data date: {report['latest_data_date']} "
        f"(observed lag: {report['observed_lag_days']} day(s), "
        f"allowed lag: {report['max_lag_days']} day(s))."
    )
    print(message)

    if not report["fresh"] and not args.warn_only:
        raise SystemExit("Raw weather data is outside the expected freshness window.")


if __name__ == "__main__":
    main()
