"""Shared fixtures for K8s backend lifecycle tests."""

import base64
import sys
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.k8s_backend import K8sBackend  # noqa: E402


def k8s_backend(release_name: str = "alpha") -> K8sBackend:
    backend = K8sBackend(
        namespace="shared-ns",
        context=None,
        release_name=release_name,
        chart_dir="/chart",
    )
    # Lifecycle unit tests exercise deletion ordering without sleeping through
    # the production rollback quarantine window.
    backend._wait_for_secret_quarantine = lambda deadline: deadline
    backend._ensure_namespace = lambda: None
    backend._namespace_exists = lambda: True
    backend._release_operation_lock = nullcontext
    backend._assert_release_operation_lock = lambda: None
    return backend


def managed_secret_payload(
    name: str,
    release_name: str,
    data: dict[str, str],
) -> dict:
    encoded = {
        key: base64.b64encode(value.encode()).decode() for key, value in data.items()
    }
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": name,
            "uid": f"uid-{name}",
            "resourceVersion": f"rv-{name}",
            "labels": {
                "app.kubernetes.io/managed-by": "vllm-sr-cli",
                "app.kubernetes.io/instance": release_name,
                "semantic-router.vllm.ai/runtime-env-secret": "true",
            },
        },
        "immutable": True,
        "data": encoded,
    }
