"""Revisioned Kubernetes Secret planning for CLI-managed runtime credentials."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections.abc import Mapping, Set
from dataclasses import dataclass

LEGACY_ENV_SECRET_NAME = "vllm-sr-env-secrets"
ENV_SECRET_NAME_PREFIX = "vllm-sr-env"
ENV_SECRET_OWNER_LABEL = "vllm.ai/env-secret-owner"
ENV_SECRET_MANAGER_LABEL = "app.kubernetes.io/managed-by"
ENV_SECRET_MANAGER_VALUE = "vllm-sr-cli"
ENV_SECRET_REVISION_ANNOTATION = "vllm.ai/env-secret-revision"


@dataclass(frozen=True)
class EnvSecretPlan:
    """A new immutable Secret revision staged for one Helm release."""

    owner: str
    name: str
    manifest: str
    key_count: int
    keys: frozenset[str]


def env_secret_owner(namespace: str, release_name: str) -> str:
    """Return a non-secret stable owner id scoped to namespace and release."""

    identity = f"{namespace}\0{release_name}".encode()
    return hashlib.sha256(identity).hexdigest()[:16]


def build_env_secret_plan(
    *,
    namespace: str,
    release_name: str,
    env_vars: Mapping[str, str] | None,
    sensitive_names: Set[str],
) -> EnvSecretPlan | None:
    """Build a revisioned Secret manifest without mutating the cluster."""

    secret_data: dict[str, str] = {}
    for name, value in sorted((env_vars or {}).items()):
        if name not in sensitive_names or value == "":
            continue
        if not isinstance(value, str):
            raise ValueError(f"sensitive environment variable {name} must be a string")
        secret_data[name] = value
    if not secret_data:
        return None

    owner = env_secret_owner(namespace, release_name)
    revision = secrets.token_hex(16)
    name = f"{ENV_SECRET_NAME_PREFIX}-{owner}-{revision}"
    manifest = {
        "apiVersion": "v1",
        "immutable": True,
        "kind": "Secret",
        "metadata": {
            "labels": {
                ENV_SECRET_MANAGER_LABEL: ENV_SECRET_MANAGER_VALUE,
                ENV_SECRET_OWNER_LABEL: owner,
            },
            "name": name,
            "namespace": namespace,
        },
        "stringData": secret_data,
        "type": "Opaque",
    }
    return EnvSecretPlan(
        owner=owner,
        name=name,
        manifest=json.dumps(manifest, separators=(",", ":"), sort_keys=True),
        key_count=len(secret_data),
        keys=frozenset(secret_data),
    )


def is_managed_env_secret_name(name: str, owner: str) -> bool:
    """Return whether *name* is a valid revision name for *owner*."""

    prefix = re.escape(f"{ENV_SECRET_NAME_PREFIX}-{owner}-")
    return re.fullmatch(rf"{prefix}[0-9a-f]{{32}}", name) is not None


def referenced_secret_names(payload: str) -> set[str]:
    """Extract metadata-only envFrom names returned by kubectl jsonpath."""

    return {name for line in payload.splitlines() if (name := line.strip())}


def managed_env_secret_names(payload: str, owner: str) -> set[str]:
    """Validate metadata-only Secret names returned by kubectl jsonpath."""

    return {
        name
        for line in payload.splitlines()
        if (name := line.strip()) and is_managed_env_secret_name(name, owner)
    }
