"""Translate a vLLM-SR config.yaml into Helm values overrides.

The user-facing config format is the router's ``config.yaml``.  For the
Kubernetes deployment path, this module converts the relevant sections into a
Helm ``values.yaml`` compatible dictionary that can be written to a temporary
file and passed to ``helm upgrade --install -f <file>``.
"""

from __future__ import annotations

import copy
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress

import yaml

from cli.control_plane_deployment import (
    control_plane_store_references,
    runtime_capabilities,
)
from cli.envoy_dispatch_contract import (
    validate_envoy_dispatch_contract,
    validate_networked_backend_dispatch,
)
from cli.models import UserConfig
from cli.recipe_package import literal_credential_paths
from cli.runtime_env_names import runtime_env_name_is_allowed
from cli.utils import get_logger, load_config

log = get_logger(__name__)

_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_ENV_REFERENCE_NAME = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_PURE_ENV_REFERENCE = re.compile(r"^\$\{[A-Z_][A-Z0-9_]*\}$")
_SENSITIVE_ENV_NAME = re.compile(
    r"(?:API_?KEY|ACCESS_?KEY|CLIENT_?SECRET|PASSWORD|SECRET|TOKEN|PRIVATE_?KEY)$",
    re.IGNORECASE,
)
_PROFILE_CONFIGMAP_CREDENTIAL_PATHS = (
    ("dependencies", "semanticCache", "redis", "password"),
    ("dependencies", "semanticCache", "milvus", "auth", "password"),
)


def translate_config_to_helm_values(
    config_file: str,
    *,
    config_document: dict[str, object] | None = None,
    source_config_file: str | None = None,
    image: str | None = None,
    pull_policy: str | None = None,
    enable_observability: bool = False,
    profile_values: dict | None = None,
    env_vars: dict[str, str] | None = None,
    env_secret_name: str | None = None,
    namespace: str | None = None,
    minimal: bool = False,
    readonly: bool = False,
) -> dict:
    """Build a Helm values dict from the user's ``config.yaml``.

    The returned dict can be serialized to YAML and passed via ``-f`` to
    ``helm upgrade --install``.
    """
    user_config = (
        copy.deepcopy(config_document)
        if config_document is not None
        else load_config(config_file)
    )
    _validate_canonical_config(user_config)
    _validate_dynamic_kubernetes_contract(user_config, profile_values, env_vars)
    _validate_profile_configmap_credentials(profile_values)
    values: dict = {}

    if image:
        repository, tag = _split_image_reference(image)
        values["image"] = {"repository": repository, "tag": tag}

    if pull_policy:
        policy_map = {
            "always": "Always",
            "ifnotpresent": "IfNotPresent",
            "never": "Never",
        }
        mapped = policy_map.get(pull_policy.lower(), pull_policy)
        values.setdefault("image", {})["pullPolicy"] = mapped

    _translate_observability(enable_observability, values)

    if profile_values:
        values = _deep_merge(profile_values, values)

    _apply_cli_deployment_overrides(
        values,
        namespace=namespace,
        minimal=minimal,
        readonly=readonly,
        image_overridden=image is not None,
    )

    # The user-selected canonical config is authoritative over chart defaults and
    # profile data. Bind it after merging so it remains one atomic document.
    _translate_config_section(user_config, values)
    _translate_env_vars(
        env_vars,
        values,
        secret_name=env_secret_name,
        config_file=source_config_file or config_file,
    )

    return values


def write_helm_values_file(values: dict, dest_dir: str) -> str:
    """Write *values* to a caller-owned private directory and return its path."""
    os.makedirs(dest_dir, exist_ok=True)
    values_path = os.path.join(dest_dir, "values-override.yaml")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(values_path, flags, 0o600)
    try:
        os.set_inheritable(fd, False)
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fd = -1
            yaml.safe_dump(values, fh, default_flow_style=False)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        with suppress(FileNotFoundError):
            os.unlink(values_path)
        raise
    log.debug(f"Wrote Helm values override to {values_path}")
    return values_path


@contextmanager
def temporary_helm_values_file(values: dict) -> Iterator[str]:
    """Yield one private Helm values file and remove it on every exit path."""

    with tempfile.TemporaryDirectory(prefix="vllm-sr-helm-") as dest_dir:
        yield write_helm_values_file(values, dest_dir)


def load_profile_values(profile: str | None, chart_dir: str) -> dict | None:
    """Load a named profile values file (``values-dev.yaml``, etc.)."""
    if not profile:
        return None
    if _PROFILE_NAME.fullmatch(profile) is None:
        raise ValueError(
            "Kubernetes profile names may contain letters, digits, _ and -"
        )
    profile_file = os.path.join(chart_dir, f"values-{profile}.yaml")
    if not os.path.exists(profile_file):
        raise FileNotFoundError(
            f"Kubernetes profile values file not found: {profile_file}"
        )
    return load_config(profile_file)


def _translate_config_section(user_config: dict, values: dict) -> None:
    """Bind one complete canonical config without Helm default-map coalescing."""

    # A CLI deployment is atomic: profile ``config`` is ignored by the chart once
    # configOverride is present, so do not persist stale or secret-bearing bytes in
    # the Helm release values either.
    values.pop("config", None)
    values["configOverride"] = copy.deepcopy(user_config)


def _validate_canonical_config(user_config: object) -> None:
    """Reject invalid or plaintext-secret config before any cluster mutation."""

    if not isinstance(user_config, dict) or not user_config:
        raise ValueError("Kubernetes Router config must be a non-empty mapping")
    config_bytes = yaml.safe_dump(user_config).encode("utf-8")
    credential_paths = literal_credential_paths(
        config_bytes,
        environment_name_is_allowed=runtime_env_name_is_allowed,
    )
    if credential_paths:
        raise ValueError(
            f"Kubernetes Router credential config.{credential_paths[0]} must use an "
            "uppercase, non-reserved environment reference such as ${CREDENTIAL_ENV}"
        )


def _validate_dynamic_kubernetes_contract(
    user_config: dict[str, object],
    profile_values: dict | None,
    env_vars: dict[str, str] | None,
) -> None:
    """Validate optional Management/runtime stores for Kubernetes."""

    capabilities = runtime_capabilities(user_config)
    if not capabilities.durable_management:
        return
    typed = UserConfig.model_validate(user_config)
    errors = validate_envoy_dispatch_contract(typed)
    if errors:
        first = errors[0]
        raise ValueError(f"{first.field}: {first.message}")
    validate_networked_backend_dispatch(typed, "Kubernetes")

    references = control_plane_store_references(user_config)
    secret_env_names = _profile_secret_reference_names(profile_values or {})
    for label, reference in (
        ("PostgreSQL", references.postgres),
        ("Valkey", references.valkey),
    ):
        if reference is None:
            continue
        if reference.kind == "env":
            value = (env_vars or {}).get(reference.value, "")
            if not (isinstance(value, str) and value.strip()) and (
                reference.value not in secret_env_names
            ):
                raise ValueError(
                    f"Kubernetes {label} reference {reference.value} "
                    "must be supplied by a Secret"
                )
            continue
        if not _profile_mounts_secret_file(profile_values or {}, reference.value):
            raise ValueError(
                f"Kubernetes {label} file {reference.value} must be "
                "mounted read-only from a Secret"
            )


def _profile_mounts_secret_file(profile: dict, file_path: str) -> bool:
    volumes = profile.get("extraVolumes", [])
    mounts = profile.get("extraVolumeMounts", [])
    if not isinstance(volumes, list) or not isinstance(mounts, list):
        return False
    secret_volumes = {
        item.get("name")
        for item in volumes
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("secret"), dict)
    }
    for mount in mounts:
        if not isinstance(mount, dict) or mount.get("name") not in secret_volumes:
            continue
        if mount.get("readOnly") is not True:
            continue
        mount_path = mount.get("mountPath")
        if not isinstance(mount_path, str) or not mount_path.startswith("/"):
            continue
        if mount.get("subPath"):
            if mount_path == file_path:
                return True
            continue
        prefix = mount_path.rstrip("/") + "/"
        if file_path.startswith(prefix):
            return True
    return False


def _validate_profile_configmap_credentials(profile_values: object) -> None:
    """Reject profile credentials the chart copies into the Router ConfigMap."""

    if profile_values is None:
        return
    if not isinstance(profile_values, dict):
        raise ValueError("Helm profile values must be a mapping")
    bound_secret_names = _profile_secret_reference_names(profile_values)
    for path in _PROFILE_CONFIGMAP_CREDENTIAL_PATHS:
        value = _nested_value(profile_values, path)
        if value in (None, ""):
            continue
        match = _PURE_ENV_REFERENCE.fullmatch(value) if isinstance(value, str) else None
        environment_name = value[2:-1] if match is not None else None
        if environment_name is None or not runtime_env_name_is_allowed(
            environment_name
        ):
            raise ValueError(
                f"Helm profile credential {'.'.join(path)} must use an "
                "uppercase, non-reserved environment reference such as "
                "${CREDENTIAL_ENV}"
            )
        if environment_name not in bound_secret_names:
            raise ValueError(
                f"Helm profile credential {'.'.join(path)} must bind "
                f"{environment_name} through env or extraEnv Secret secretKeyRef"
            )


def _profile_secret_reference_names(profile_values: dict[object, object]) -> set[str]:
    """Collect Router env names bound to external Kubernetes Secrets."""

    names: set[str] = set()
    for field in ("env", "extraEnv"):
        configured = profile_values.get(field, [])
        if not isinstance(configured, list):
            continue
        for entry in configured:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                continue
            value_from = entry.get("valueFrom")
            secret_ref = (
                value_from.get("secretKeyRef") if isinstance(value_from, dict) else None
            )
            if (
                "value" not in entry
                and isinstance(secret_ref, dict)
                and isinstance(secret_ref.get("name"), str)
                and secret_ref["name"]
                and isinstance(secret_ref.get("key"), str)
                and secret_ref["key"]
            ):
                names.add(entry["name"])
    return names


def _nested_value(document: dict[object, object], path: tuple[str, ...]) -> object:
    value: object = document
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _apply_cli_deployment_overrides(
    values: dict,
    *,
    namespace: str | None,
    minimal: bool,
    readonly: bool,
    image_overridden: bool,
) -> None:
    """Make explicit CLI deployment flags authoritative over profile defaults."""

    if namespace is not None or image_overridden:
        configured_global = values.get("global", {})
        if not isinstance(configured_global, dict):
            raise ValueError("Helm global values must be a mapping")
        global_values = dict(configured_global)
        if namespace is not None:
            global_values["namespace"] = namespace
        if image_overridden:
            global_values["imageRegistry"] = ""
        values["global"] = global_values

    if minimal or readonly:
        configured_dashboard = values.get("dashboard", {})
        if not isinstance(configured_dashboard, dict):
            raise ValueError("Helm dashboard values must be a mapping")
        dashboard_values = dict(configured_dashboard)
        if minimal:
            dashboard_values["enabled"] = False
        elif readonly:
            dashboard_values["readonly"] = True
        values["dashboard"] = dashboard_values


def _split_image_reference(image: str) -> tuple[str, str]:
    """Split a tag-based image reference without confusing registry ports."""

    if not image or image != image.strip() or any(char.isspace() for char in image):
        raise ValueError(
            "Kubernetes image references must be non-empty and whitespace-free"
        )
    if "@" in image:
        raise ValueError(
            "Kubernetes digest image references are not supported by this chart; "
            "select a tagged image through --image or a Helm profile"
        )

    last_slash = image.rfind("/")
    last_colon = image.rfind(":")
    if last_colon > last_slash:
        repository = image[:last_colon]
        tag = image[last_colon + 1 :]
        if not repository or not tag:
            raise ValueError("Kubernetes image references must use repository[:tag]")
        return repository, tag
    return image, "latest"


def _translate_env_vars(
    env_vars: dict[str, str] | None,
    values: dict,
    secret_name: str | None = None,
    config_file: str | None = None,
) -> None:
    """Map non-sensitive env vars into ``env:`` and wire a secret via ``envFromSecrets:``."""
    from cli.commands.runtime_support import sensitive_env_names  # noqa: PLC0415

    sensitive_names = sensitive_env_names(config_file) | _environment_reference_names(
        values
    )
    env_list = _validated_env_entries(values.get("env", []), "env", sensitive_names)
    extra_env = _validated_env_entries(
        values.get("extraEnv", []), "extraEnv", sensitive_names
    )
    dashboard = values.get("dashboard", {})
    if not isinstance(dashboard, dict):
        raise ValueError("Helm dashboard values must be a mapping")
    _validated_env_entries(
        dashboard.get("extraEnv", []), "dashboard.extraEnv", sensitive_names
    )

    if env_vars:
        existing_names = {e["name"] for e in (*env_list, *extra_env)}
        for name, value in sorted(env_vars.items()):
            if name in sensitive_names or name in existing_names:
                continue
            env_list.append({"name": name, "value": value})
    if env_list:
        values["env"] = env_list

    if secret_name:
        configured_secrets = values.get("envFromSecrets", [])
        if not isinstance(configured_secrets, list):
            raise ValueError("Helm envFromSecrets must be a list")
        secret_names = list(configured_secrets)
        if not all(isinstance(name, str) and name for name in secret_names):
            raise ValueError("Helm envFromSecrets entries must be non-empty strings")
        if secret_name not in secret_names:
            secret_names.append(secret_name)
        values["envFromSecrets"] = secret_names


def _validated_env_entries(
    configured: object, field: str, sensitive_names: set[str]
) -> list[dict[str, object]]:
    """Copy EnvVar entries and reject plaintext sensitive overrides."""

    if not isinstance(configured, list):
        raise ValueError(f"Helm {field} must be a list")
    entries: list[dict[str, object]] = []
    for configured_entry in configured:
        if not isinstance(configured_entry, dict) or not isinstance(
            configured_entry.get("name"), str
        ):
            raise ValueError(f"Helm {field} entries must be mappings with a name")
        entry = dict(configured_entry)
        name = entry["name"]
        if name in sensitive_names or _SENSITIVE_ENV_NAME.search(name):
            value_from = entry.get("valueFrom")
            secret_ref = (
                value_from.get("secretKeyRef") if isinstance(value_from, dict) else None
            )
            if "value" in entry or not (
                isinstance(secret_ref, dict)
                and isinstance(secret_ref.get("name"), str)
                and secret_ref["name"]
                and isinstance(secret_ref.get("key"), str)
                and secret_ref["key"]
            ):
                raise ValueError(
                    f"Helm {field} environment variable {name} must use a Secret "
                    "secretKeyRef, not a plaintext value"
                )
        entries.append(entry)
    return entries


def _environment_reference_names(value: object) -> set[str]:
    """Collect named secret and exact braced environment references.

    File-authored backend credentials use ``secret_env``. Keeping this discovery in
    the deployment translator lets Helm render a Secret reference without ever
    copying the secret value into generated values or Envoy configuration.
    """

    names: set[str] = set()
    pending = [value]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            secret_env = node.get("secret_env")
            if isinstance(secret_env, str) and secret_env.strip():
                names.add(secret_env.strip())
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
        elif isinstance(node, str):
            names.update(_ENV_REFERENCE_NAME.findall(node))
    return names


def _translate_observability(enable: bool, values: dict) -> None:
    """Toggle the observability dependency flags."""
    deps = values.setdefault("dependencies", {}).setdefault("observability", {})
    deps.setdefault("jaeger", {})["enabled"] = enable
    deps.setdefault("prometheus", {})["enabled"] = enable
    deps.setdefault("grafana", {})["enabled"] = enable


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into a copy of *base*."""
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
