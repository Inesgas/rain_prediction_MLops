from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "versioning"
DEFAULT_MODEL_ARTIFACT = PROJECT_ROOT / "models" / "final_winner" / "winner_model.joblib"
DEFAULT_MODEL_METADATA = PROJECT_ROOT / "models" / "final_winner" / "metadata.json"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "models" / "final_winner" / "model_config.json"
DEFAULT_MODEL_SAMPLE = PROJECT_ROOT / "models" / "final_winner" / "sample_input.json"

DEFAULT_LOCAL_INPUTS = [
    "data/raw/weatherAUS.csv",
    "data/preprocessed/rain_model_dataset.csv",
    "data/preprocessed/rain_model_dataset_aligned.csv",
    "data/preprocessed/rain_model_dataset_feature_experiments.csv",
]


@dataclass(frozen=True)
class CommandResult:
    args: Sequence[str]
    returncode: int
    stdout: str
    stderr: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def relative(path: Path, root: Path = PROJECT_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def run_command(
    args: Sequence[str],
    cwd: Path = PROJECT_ROOT,
    check: bool = True,
    echo: bool = True,
) -> CommandResult:
    result = subprocess.run(
        list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if echo and result.stdout:
        print(result.stdout, end="")
    if echo and result.stderr:
        print(result.stderr, end="")

    command_result = CommandResult(
        args=args,
        returncode=result.returncode,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
    )
    if check and result.returncode != 0:
        quoted = " ".join(args)
        raise SystemExit(f"Command failed with exit code {result.returncode}: {quoted}")
    return command_result


def command_snapshot(args: Sequence[str], cwd: Path = PROJECT_ROOT) -> dict[str, Any]:
    try:
        result = run_command(args, cwd=cwd, check=False, echo=False)
    except FileNotFoundError as exc:
        return {
            "args": list(args),
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "args": list(args),
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, include_sha256: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": relative(path),
        "exists": path.exists(),
    }
    if path.exists():
        record["size_bytes"] = path.stat().st_size
        record["modified_at_utc"] = datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=timezone.utc,
        ).replace(microsecond=0).isoformat()
        if include_sha256:
            record["sha256"] = sha256_file(path)
    return record


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def coerce_scalar(value: str) -> Any:
    value = value.strip().strip("\"'")
    if value.isdigit():
        return int(value)
    return value


def parse_simple_dvc_yaml(text: str) -> dict[str, Any]:
    """Parse the simple DVC pointer shape used by this repository.

    PyYAML is preferred when available, but this fallback keeps snapshots usable
    in minimal environments where only the standard library is installed.
    """
    outs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            if current:
                outs.append(current)
            current = {}
            line = line[2:].strip()
        if current is not None and ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = coerce_scalar(value)
    if current:
        outs.append(current)
    return {"outs": outs}


def load_dvc_pointer(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]

        loaded = yaml.safe_load(text)
        return loaded or {}
    except Exception:
        return parse_simple_dvc_yaml(text)


def collect_dvc_outs(root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    outs: list[dict[str, Any]] = []
    for dvc_file in sorted(root.rglob("*.dvc")):
        if ".dvc" in dvc_file.parts:
            continue
        pointer = load_dvc_pointer(dvc_file)
        for out in pointer.get("outs", []):
            out_path = dvc_file.parent / str(out.get("path", ""))
            outs.append(
                {
                    "dvc_file": relative(dvc_file, root),
                    "path": relative(out_path, root),
                    "exists": out_path.exists(),
                    "actual_size_bytes": out_path.stat().st_size if out_path.exists() else None,
                    "dvc_hash": out.get("md5") or out.get("etag") or out.get("hash"),
                    "dvc_hash_type": out.get("hash", "md5"),
                    "dvc_size_bytes": out.get("size"),
                }
            )
    return outs


def read_dvc_remote(root: Path = PROJECT_ROOT) -> dict[str, str | None]:
    config_path = root / ".dvc" / "config"
    remote_name: str | None = None
    remote_url: str | None = None
    current_section = ""
    if not config_path.exists():
        return {"name": None, "url": None}

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line.strip("[]").strip("'\"")
            continue
        if line.startswith("remote ="):
            remote_name = line.split("=", 1)[1].strip()
            continue
        if current_section == f'remote "{remote_name}"' and line.startswith("url ="):
            remote_url = line.split("=", 1)[1].strip()
    return {"name": remote_name, "url": remote_url}


def git_context(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    commit = command_snapshot(["git", "rev-parse", "HEAD"], cwd=root)
    branch = command_snapshot(["git", "branch", "--show-current"], cwd=root)
    status = command_snapshot(["git", "status", "--short"], cwd=root)
    remote = command_snapshot(["git", "remote", "-v"], cwd=root)
    return {
        "commit": commit["stdout"] if commit["ok"] else None,
        "branch": branch["stdout"] if branch["ok"] else None,
        "is_dirty": bool(status["stdout"]),
        "status_short": status["stdout"].splitlines() if status["stdout"] else [],
        "remotes": remote["stdout"].splitlines() if remote["stdout"] else [],
        "commands": {
            "commit": commit,
            "branch": branch,
            "status": status,
            "remote": remote,
        },
    }


def airflow_context() -> dict[str, str]:
    keys = [
        "AIRFLOW_CTX_DAG_ID",
        "AIRFLOW_CTX_TASK_ID",
        "AIRFLOW_CTX_RUN_ID",
        "AIRFLOW_CTX_EXECUTION_DATE",
        "AIRFLOW_CTX_TRY_NUMBER",
    ]
    return {key: value for key in keys if (value := os.environ.get(key))}


def build_manifest(phase: str, run_id: str | None = None) -> dict[str, Any]:
    metadata = read_json(DEFAULT_MODEL_METADATA)
    config = read_json(DEFAULT_MODEL_CONFIG)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "project": "rain_prediction_mlops",
        "phase": phase,
        "run_id": run_id or os.environ.get("AIRFLOW_CTX_RUN_ID"),
        "created_at_utc": utc_now(),
        "airflow": airflow_context(),
        "git": git_context(),
        "dvc": {
            "remote": read_dvc_remote(),
            "tracked_outputs": collect_dvc_outs(),
        },
        "artifacts": {
            "model": file_record(DEFAULT_MODEL_ARTIFACT, include_sha256=False),
            "model_metadata": file_record(DEFAULT_MODEL_METADATA),
            "model_config": file_record(DEFAULT_MODEL_CONFIG),
            "sample_input": file_record(DEFAULT_MODEL_SAMPLE),
        },
        "model_metadata": metadata,
        "model_config_summary": {
            "model_name": config.get("model_name") if config else None,
            "feature_set_name": config.get("feature_set_name") if config else None,
            "feature_count": len(config.get("features", [])) if config else None,
            "threshold": config.get("threshold") if config else None,
        },
        "tracking_handoff": {
            "intended_consumer": "MLflow tracking task",
            "status": "ready_for_external_logging",
            "note": "This DAG records artifact versions only; experiment tracking can log this manifest as an MLflow artifact.",
        },
    }
    return manifest


def write_manifest(phase: str, run_id: str | None, output_dir: Path) -> Path:
    safe_run_id = (run_id or "manual").replace("/", "_").replace(":", "_")
    filename = f"{safe_timestamp()}-{phase}-{safe_run_id}.json"
    payload = json.dumps(build_manifest(phase=phase, run_id=run_id), indent=2)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        path.write_text(payload, encoding="utf-8")
    except PermissionError:
        fallback_dir = Path(tempfile.gettempdir()) / "rain_prediction_mlops" / "versioning"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        path = fallback_dir / filename
        path.write_text(payload, encoding="utf-8")
        print(f"Warning: versioning output directory is not writable; used temporary path: {path}")
        return path

    print(f"Wrote versioning manifest: {relative(path)}")
    return path


def verify_local_artifacts(targets: Iterable[str]) -> None:
    missing = [target for target in targets if not (PROJECT_ROOT / target).exists()]
    if missing:
        formatted = "\n".join(f"- {target}" for target in missing)
        raise SystemExit(
            "Missing local DVC-tracked artifact(s):\n"
            f"{formatted}\n"
            "This workflow is local-only and will not pull from DagsHub automatically."
        )
    print("All required local DVC-tracked artifacts are present.")


def dvc_add_path(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"DVC target does not exist: {path}")
    run_command(["dvc", "add", relative(path)])


def dvc_add_model(model_path: Path = DEFAULT_MODEL_ARTIFACT) -> None:
    dvc_add_path(model_path)


def dvc_status(output_dir: Path, run_id: str | None) -> Path:
    result = command_snapshot(["dvc", "status"], cwd=PROJECT_ROOT)
    safe_run_id = (run_id or "manual").replace("/", "_").replace(":", "_")
    filename = f"{safe_timestamp()}-dvc-status-{safe_run_id}.json"
    payload = json.dumps(result, indent=2)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        path.write_text(payload, encoding="utf-8")
    except PermissionError:
        fallback_dir = Path(tempfile.gettempdir()) / "rain_prediction_mlops" / "versioning"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        path = fallback_dir / filename
        path.write_text(payload, encoding="utf-8")
        print(f"Warning: DVC status output directory is not writable; used temporary path: {path}")
        return path

    print(f"Wrote DVC status: {relative(path)}")
    if not result["ok"]:
        print("Warning: dvc status failed; see the recorded status JSON for details.")
    return path


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", default=os.environ.get("AIRFLOW_CTX_RUN_ID"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DVC-backed data and model versioning utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = subparsers.add_parser("snapshot", help="Write a Git/DVC/model version manifest.")
    snapshot_parser.add_argument("--phase", required=True, choices=["inputs", "outputs", "manual"])
    add_common_args(snapshot_parser)

    verify_parser = subparsers.add_parser("verify-local", help="Check that local DVC-tracked inputs are present.")
    verify_parser.add_argument("--target", action="append", dest="targets", default=None)

    status_parser = subparsers.add_parser("dvc-status", help="Write DVC status output to reports/versioning.")
    add_common_args(status_parser)

    add_parser = subparsers.add_parser("dvc-add-model", help="Track the trained model artifact with DVC.")
    add_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_ARTIFACT)

    generic_add_parser = subparsers.add_parser("dvc-add", help="Track a local path with DVC.")
    generic_add_parser.add_argument("--target", type=Path, required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "snapshot":
        write_manifest(phase=args.phase, run_id=args.run_id, output_dir=args.output_dir)
    elif args.command == "verify-local":
        verify_local_artifacts(args.targets or DEFAULT_LOCAL_INPUTS)
    elif args.command == "dvc-status":
        dvc_status(output_dir=args.output_dir, run_id=args.run_id)
    elif args.command == "dvc-add-model":
        dvc_add_model(model_path=args.model_path)
    elif args.command == "dvc-add":
        dvc_add_path(path=args.target)
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
