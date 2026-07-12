"""Pure parsers for rollback-safe Kubernetes runtime Secret retention."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from typing import Any

import yaml

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]

# Keep this inventory aligned with core/v1 VolumeSource. Unknown source fields
# are unsafe for credential GC: a future Kubernetes source may itself contain a
# Secret reference, so silently ignoring it could delete a live generation.
_VOLUME_SOURCE_FIELDS = frozenset(
    {
        "awsElasticBlockStore",
        "azureDisk",
        "azureFile",
        "cephfs",
        "cinder",
        "configMap",
        "csi",
        "downwardAPI",
        "emptyDir",
        "ephemeral",
        "fc",
        "flexVolume",
        "flocker",
        "gcePersistentDisk",
        "gitRepo",
        "glusterfs",
        "hostPath",
        "image",
        "iscsi",
        "nfs",
        "persistentVolumeClaim",
        "photonPersistentDisk",
        "portworxVolume",
        "projected",
        "quobyte",
        "rbd",
        "scaleIO",
        "secret",
        "storageos",
        "vsphereVolume",
    }
)
_OPTIONAL_SECRET_REF_VOLUME_SOURCES = frozenset(
    {"cephfs", "cinder", "flexVolume", "iscsi", "rbd", "storageos"}
)
_PROJECTED_SOURCE_FIELDS = frozenset(
    {
        "clusterTrustBundle",
        "configMap",
        "downwardAPI",
        "podCertificate",
        "secret",
        "serviceAccountToken",
    }
)


def read_helm_retained_secret_refs(
    helm_base_cmd: list[str],
    release_name: str,
    namespace: str,
    max_revisions: int,
    runner: CommandRunner | None = None,
) -> set[str] | None:
    """Read Secret refs from retained Helm revisions without logging output."""
    run = runner or subprocess.run
    try:
        history = run(
            [
                *helm_base_cmd,
                "history",
                release_name,
                "--namespace",
                namespace,
                "--max",
                str(max_revisions),
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if history.returncode != 0:
        return None

    revisions = parse_helm_history_revisions(history.stdout)
    if revisions is None:
        return None

    refs: set[str] = set()
    for revision in revisions:
        revision_refs = _read_helm_revision_secret_refs(
            run,
            helm_base_cmd,
            release_name,
            namespace,
            revision,
        )
        if revision_refs is None:
            return None
        refs.update(revision_refs)
    return refs


def _read_helm_revision_secret_refs(
    runner: CommandRunner,
    helm_base_cmd: list[str],
    release_name: str,
    namespace: str,
    revision: int,
) -> set[str] | None:
    try:
        manifest = runner(
            [
                *helm_base_cmd,
                "get",
                "manifest",
                release_name,
                "--namespace",
                namespace,
                "--revision",
                str(revision),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if manifest.returncode != 0:
        return None
    return parse_helm_manifest_secret_refs(manifest.stdout)


def parse_helm_history_revisions(payload: str) -> list[int] | None:
    """Return retained Helm revision numbers, or ``None`` for unsafe input."""
    if not isinstance(payload, str):
        return None
    try:
        history = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(history, list):
        return None
    if not history:
        return None

    revisions: list[int] = []
    for entry in history:
        if not isinstance(entry, dict):
            return None
        revision = entry.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            return None
        revisions.append(revision)
    return revisions


def parse_helm_manifest_secret_refs(manifest: str) -> set[str] | None:
    """Collect Secret refs from Deployment pod specs in a Helm manifest."""
    if not isinstance(manifest, str):
        return None
    try:
        documents = list(yaml.safe_load_all(manifest))
    except yaml.YAMLError:
        return None

    refs: set[str] = set()
    found_deployment = False
    for document in documents:
        if document is None:
            continue
        if not isinstance(document, dict):
            return None
        kind = document.get("kind")
        if not isinstance(kind, str) or not kind:
            return None
        if kind != "Deployment":
            continue
        found_deployment = True
        try:
            refs.update(deployment_secret_refs(document))
        except ValueError:
            return None
    return refs if found_deployment else None


def deployment_secret_refs(deployment: dict[str, Any]) -> set[str]:
    """Collect Secret refs, rejecting structures that could hide a reference."""
    pod_spec = _pod_spec(deployment)
    refs = _container_secret_refs(pod_spec)
    refs.update(_volume_secret_refs(pod_spec))

    for item in _list_field(pod_spec, "imagePullSecrets"):
        item = _require_mapping(item, "imagePullSecrets item")
        refs.add(_required_name(item, "name", "imagePullSecrets item"))
    return refs


def _pod_spec(deployment: dict[str, Any]) -> dict[str, Any]:
    deployment = _require_mapping(deployment, "Deployment")
    spec = _required_mapping(deployment, "spec", "Deployment")
    template = _required_mapping(spec, "template", "Deployment spec")
    pod_spec = _required_mapping(template, "spec", "Deployment template")
    _list_field(pod_spec, "containers", required=True)
    return pod_spec


def _container_secret_refs(pod_spec: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for field in ("initContainers", "containers", "ephemeralContainers"):
        containers = _list_field(pod_spec, field, required=field == "containers")
        for container in containers:
            container = _require_mapping(container, f"{field} item")
            refs.update(_env_secret_refs(container))
    return refs


def _env_secret_refs(container: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for source in _list_field(container, "envFrom"):
        source = _require_mapping(source, "envFrom item")
        secret_ref = _optional_mapping(source, "secretRef", "envFrom item")
        if secret_ref is not None:
            refs.add(_required_name(secret_ref, "name", "secretRef"))

    for variable in _list_field(container, "env"):
        variable = _require_mapping(variable, "env item")
        value_from = _optional_mapping(variable, "valueFrom", "env item")
        if value_from is None:
            continue
        secret_ref = _optional_mapping(value_from, "secretKeyRef", "valueFrom")
        if secret_ref is not None:
            refs.add(_required_name(secret_ref, "name", "secretKeyRef"))
    return refs


def _volume_secret_refs(pod_spec: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for volume in _list_field(pod_spec, "volumes"):
        volume = _require_mapping(volume, "volumes item")
        volume_name = _required_name(volume, "name", "volumes item")
        source_name, source = _volume_source(volume, volume_name)

        if source_name == "secret":
            refs.add(_required_name(source, "secretName", "secret volume"))
        elif source_name == "azureFile":
            refs.add(_required_name(source, "secretName", "azureFile volume"))
        elif source_name == "projected":
            refs.update(_projected_secret_refs(source))
        elif source_name == "csi":
            refs.update(
                _optional_secret_ref(source, "nodePublishSecretRef", "csi volume")
            )
        elif source_name == "scaleIO":
            refs.add(_required_secret_ref(source, "secretRef", "scaleIO volume"))
        elif source_name in _OPTIONAL_SECRET_REF_VOLUME_SOURCES:
            refs.update(_optional_secret_ref(source, "secretRef", source_name))
    return refs


def _volume_source(
    volume: dict[str, Any],
    volume_name: str,
) -> tuple[str, dict[str, Any]]:
    fields = set(volume) - {"name"}
    unknown = fields - _VOLUME_SOURCE_FIELDS
    if unknown:
        raise ValueError(
            f"volume {volume_name} has unknown source fields: {sorted(unknown)}"
        )
    configured = fields & _VOLUME_SOURCE_FIELDS
    if len(configured) != 1:
        raise ValueError(f"volume {volume_name} must have exactly one source")
    source_name = next(iter(configured))
    return source_name, _require_mapping(
        volume[source_name],
        f"volume {volume_name}.{source_name}",
    )


def _projected_secret_refs(projected: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for source in _list_field(projected, "sources", required=True):
        source = _require_mapping(source, "projected sources item")
        fields = set(source)
        unknown = fields - _PROJECTED_SOURCE_FIELDS
        if unknown:
            raise ValueError(f"projected source has unknown fields: {sorted(unknown)}")
        configured = fields & _PROJECTED_SOURCE_FIELDS
        if len(configured) != 1:
            raise ValueError("projected source must have exactly one source")
        source_name = next(iter(configured))
        projection = _require_mapping(
            source[source_name],
            f"projected source.{source_name}",
        )
        if source_name == "secret":
            refs.add(_required_name(projection, "name", "projected secret"))
    return refs


def _optional_secret_ref(
    owner: dict[str, Any],
    field: str,
    owner_name: str,
) -> set[str]:
    secret_ref = _optional_mapping(owner, field, owner_name)
    if secret_ref is None:
        return set()
    return {_required_name(secret_ref, "name", f"{owner_name}.{field}")}


def _required_secret_ref(
    owner: dict[str, Any],
    field: str,
    owner_name: str,
) -> str:
    secret_ref = _required_mapping(owner, field, owner_name)
    return _required_name(secret_ref, "name", f"{owner_name}.{field}")


def _required_mapping(
    owner: dict[str, Any],
    field: str,
    owner_name: str,
) -> dict[str, Any]:
    value = _optional_mapping(owner, field, owner_name)
    if value is None:
        raise ValueError(f"{owner_name} is missing {field}")
    return value


def _optional_mapping(
    owner: dict[str, Any],
    field: str,
    owner_name: str,
) -> dict[str, Any] | None:
    if field not in owner:
        return None
    return _require_mapping(owner[field], f"{owner_name}.{field}")


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _list_field(
    owner: dict[str, Any],
    field: str,
    *,
    required: bool = False,
) -> list[Any]:
    if field not in owner:
        if required:
            raise ValueError(f"missing required list {field}")
        return []
    value = owner[field]
    if not isinstance(value, list) or (required and not value):
        raise ValueError(
            f"{field} must be a non-empty list"
            if required
            else f"{field} must be a list"
        )
    return value


def _required_name(owner: dict[str, Any], field: str, owner_name: str) -> str:
    value = owner.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner_name}.{field} must be a non-empty string")
    return value
