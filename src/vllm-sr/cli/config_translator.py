"""Translate a vLLM-SR config.yaml into Helm values overrides.

The user-facing config format is the router's ``config.yaml``.  For the
Kubernetes deployment path, this module converts the relevant sections into a
Helm ``values.yaml`` compatible dictionary that can be written to a temporary
file and passed to ``helm upgrade --install -f <file>``.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile

import yaml

from cli.utils import get_logger, load_config

log = get_logger(__name__)


def translate_config_to_helm_values(
    config_file: str,
    *,
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

    if profile_values:
        values = _deep_merge(profile_values, values)

    # Profiles may already declare operator-managed env/envFrom entries. Add
    # the CLI-owned runtime environment after the merge so those references are
    # preserved and the release-scoped credential Secret is appended exactly
    # once instead of replacing the profile list.
    _translate_env_vars(env_vars, values, secret_name=env_secret_name)

    return values


def write_helm_values_file(values: dict, dest_dir: str | None = None) -> str:
    """Write *values* to a private YAML file and return its path.

    A caller-provided directory remains caller-owned. When this helper creates
    the directory, it is private, but the caller that receives the path is
    still responsible for removing it after the consuming process exits.
    """
    owned_directory: str | None = None
    try:
        if dest_dir is None:
            dest_dir = tempfile.mkdtemp(prefix="vllm-sr-helm-")
            owned_directory = dest_dir
            os.chmod(dest_dir, 0o700)
        else:
            os.makedirs(dest_dir, mode=0o700, exist_ok=True)

        values_path = os.path.join(dest_dir, "values-override.yaml")
        with contextlib.suppress(FileNotFoundError):
            os.unlink(values_path)

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        file_descriptor = os.open(values_path, flags, 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(file_descriptor, 0o600)
            else:
                os.chmod(values_path, 0o600)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as values_file:
                file_descriptor = -1
                yaml.safe_dump(values, values_file, default_flow_style=False)
        except BaseException:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(values_path)
            raise
    except BaseException:
        if owned_directory is not None:
            shutil.rmtree(owned_directory, ignore_errors=True)
        raise

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
) -> None:
    """Map non-sensitive env vars into ``env:`` and wire a secret via ``envFromSecrets:``."""
    if env_vars:
        from cli.commands.runtime_support import PASSTHROUGH_ENV_RULES  # noqa: PLC0415

        sensitive_names = {name for name, masked in PASSTHROUGH_ENV_RULES if masked}
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
