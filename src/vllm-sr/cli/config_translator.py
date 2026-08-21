"""Translate a vLLM-SR config.yaml into Helm values overrides.

The user-facing config format is the router's ``config.yaml``.  For the
Kubernetes deployment path, this module converts the relevant sections into a
Helm ``values.yaml`` compatible dictionary that can be written to a temporary
file and passed to ``helm upgrade --install -f <file>``.
"""

from __future__ import annotations

import os
import tempfile

import yaml

from cli.consts import PUBLISH_MANAGEMENT_API_ENV
from cli.utils import get_logger, load_config

log = get_logger(__name__)


def translate_config_to_helm_values(
    config_file: str,
    *,
    source_config_file: str | None = None,
    image: str | None = None,
    pull_policy: str | None = None,
    enable_observability: bool = True,
    profile_values: dict | None = None,
    env_vars: dict[str, str] | None = None,
    env_secret_name: str | None = None,
) -> dict:
    """Build a Helm values dict from the user's ``config.yaml``.

    The returned dict can be serialized to YAML and passed via ``-f`` to
    ``helm upgrade --install``.
    """
    user_config = load_config(config_file)
    values: dict = {}

    if image:
        repo, _, tag = image.rpartition(":")
        if repo:
            values.setdefault("image", {})["repository"] = repo
        if tag:
            values.setdefault("image", {})["tag"] = tag

    if pull_policy:
        policy_map = {
            "always": "Always",
            "ifnotpresent": "IfNotPresent",
            "never": "Never",
        }
        mapped = policy_map.get(pull_policy.lower(), pull_policy)
        values.setdefault("image", {})["pullPolicy"] = mapped

    _translate_config_section(user_config, values)
    _translate_observability(enable_observability, values)
    _translate_env_vars(
        env_vars,
        values,
        secret_name=env_secret_name,
        config_file=source_config_file or config_file,
    )
    _translate_management_api(env_vars, values)

    if profile_values:
        values = _deep_merge(profile_values, values)

    return values


def write_helm_values_file(values: dict, dest_dir: str | None = None) -> str:
    """Write *values* to a temporary YAML file and return its path."""
    if dest_dir is None:
        dest_dir = tempfile.mkdtemp(prefix="vllm-sr-helm-")

    os.makedirs(dest_dir, exist_ok=True)
    values_path = os.path.join(dest_dir, "values-override.yaml")
    with open(values_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(values, fh, default_flow_style=False)
    log.debug(f"Wrote Helm values override to {values_path}")
    return values_path


def load_profile_values(profile: str | None, chart_dir: str) -> dict | None:
    """Load a named profile values file (``values-dev.yaml``, etc.)."""
    if not profile:
        return None
    profile_file = os.path.join(chart_dir, f"values-{profile}.yaml")
    if not os.path.exists(profile_file):
        log.warning(f"Profile values file not found: {profile_file}")
        return None
    return load_config(profile_file)


def _translate_config_section(user_config: dict, values: dict) -> None:
    """Map router config.yaml keys into the Helm values ``config:`` block."""
    helm_config = values.setdefault("config", {})

    passthrough_keys = [
        "bert_model",
        "semantic_cache",
        "response_api",
        "tools",
        "prompt_guard",
        "classifier",
        "reasoning_families",
        "default_reasoning_effort",
        "api",
    ]
    for key in passthrough_keys:
        if key in user_config:
            helm_config[key] = user_config[key]

    if "listeners" in user_config:
        helm_config["listeners"] = user_config["listeners"]
    if "decisions" in user_config:
        helm_config["decisions"] = user_config["decisions"]
    if "mom_registry" in user_config:
        helm_config["mom_registry"] = user_config["mom_registry"]


def _translate_env_vars(
    env_vars: dict[str, str] | None,
    values: dict,
    secret_name: str | None = None,
    config_file: str | None = None,
) -> None:
    """Map non-sensitive env vars into ``env:`` and wire a secret via ``envFromSecrets:``."""
    if env_vars:
        from cli.commands.runtime_support import sensitive_env_names  # noqa: PLC0415

        sensitive_names = sensitive_env_names(config_file)
        env_list: list[dict[str, str]] = values.get("env", [])
        existing_names = {e["name"] for e in env_list}
        for name, value in sorted(env_vars.items()):
            if name in sensitive_names or name in existing_names:
                continue
            env_list.append({"name": name, "value": value})
        if env_list:
            values["env"] = env_list

    if secret_name:
        secrets = values.get("envFromSecrets", [])
        if secret_name not in secrets:
            secrets.append(secret_name)
        values["envFromSecrets"] = secrets


def _translate_management_api(env_vars: dict[str, str] | None, values: dict) -> None:
    """Map ``--publish-management-api`` / ``VLLM_SR_PUBLISH_MANAGEMENT_API`` onto
    the chart's ``managementApi.expose`` value.

    Without this, the flag/env var only ever reached the container runtime
    (docker) path: for ``--target k8s`` it was accepted, logged, and then
    silently discarded, because :8080 exposure on the primary Service is
    controlled solely by the Helm value, not by a pod env var (#2463 Phase 4
    follow-up). Passing the flag for a Kubernetes deploy now also exposes
    classify-api on the primary Service, matching the docker-target behavior
    of publishing the management API for host/CLI access.
    """
    if not env_vars:
        return
    raw = str(env_vars.get(PUBLISH_MANAGEMENT_API_ENV, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        values.setdefault("managementApi", {})["expose"] = True


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
