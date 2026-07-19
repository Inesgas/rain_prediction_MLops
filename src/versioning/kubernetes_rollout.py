from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path


SERVICEACCOUNT_NAMESPACE = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def running_in_kubernetes() -> bool:
    return bool(os.environ.get("KUBERNETES_SERVICE_HOST"))


def default_namespace() -> str:
    if value := os.environ.get("KUBERNETES_NAMESPACE"):
        return value
    if SERVICEACCOUNT_NAMESPACE.exists():
        return SERVICEACCOUNT_NAMESPACE.read_text(encoding="utf-8").strip()
    return "rain-prediction"


def restart_deployment(namespace: str, deployment: str) -> str:
    try:
        from kubernetes import client, config
    except ImportError as exc:  # pragma: no cover - depends on Airflow image
        raise SystemExit(
            "The kubernetes Python package is required for API rollout restarts. "
            "Rebuild the Airflow image after installing kubernetes>=30,<32."
        ) from exc

    config.load_incluster_config()
    restarted_at = utc_now()
    body = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "kubectl.kubernetes.io/restartedAt": restarted_at,
                    }
                }
            }
        }
    }
    apps = client.AppsV1Api()
    apps.patch_namespaced_deployment(
        name=deployment,
        namespace=namespace,
        body=body,
    )
    return restarted_at


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Restart a Kubernetes deployment from an in-cluster Airflow task.")
    parser.add_argument("--namespace", default=default_namespace())
    parser.add_argument("--deployment", default=os.environ.get("API_DEPLOYMENT_NAME", "rain-prediction-api"))
    parser.add_argument(
        "--skip-outside-kubernetes",
        action="store_true",
        help="Exit successfully when the command is run outside a Kubernetes pod.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not running_in_kubernetes():
        message = "Kubernetes service environment was not detected."
        if args.skip_outside_kubernetes:
            print(f"{message} Skipping rollout restart.")
            return
        raise SystemExit(message)

    restarted_at = restart_deployment(namespace=args.namespace, deployment=args.deployment)
    print(f"Restarted deployment/{args.deployment} in namespace {args.namespace} at {restarted_at}.")


if __name__ == "__main__":
    main()
