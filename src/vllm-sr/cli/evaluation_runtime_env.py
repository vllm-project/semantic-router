"""Dashboard-only environment wiring for production Evaluation services."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from cli.commands.runtime_management_credentials import (
    management_credential_env_names,
)
from cli.evaluation_deployment_snapshot import (
    materialize_evaluation_deployment_snapshot,
)
from cli.runtime_env_names import runtime_env_name_is_allowed

ROUTER_EVALUATION_API_KEY_REF_ENV = "EVALUATION_ROUTER_API_KEY_ENV"
ENVOY_EVALUATION_API_KEY_REF_ENV = "EVALUATION_ENVOY_API_KEY_ENV"
EVALUATION_ENABLED_ENV = "EVALUATION_ENABLED"
ROUTER_MANAGEMENT_API_KEY_ENV = "VLLM_SR_DASHBOARD_RECIPE_TOKEN"
EVALUATION_DEPLOYMENTS_DIR_ENV = "EVALUATION_DEPLOYMENTS_DIR"
EVALUATION_DEPLOYMENTS_CONTAINER_DIR = "/app/evaluation-deployments"

EVALUATION_LEDGER_PREFIXES = (
    "EVALUATION_AGENT_TASK_LEDGER",
    "EVALUATION_FAULT_RECOVERY_LEDGER",
    "EVALUATION_HARD_POLICY_LEDGER",
    "EVALUATION_PRODUCTION_EXPERIMENT_LEDGER",
)

EVALUATION_ENDPOINT_CONFIG_ENV_NAMES = tuple(
    name
    for prefix in EVALUATION_LEDGER_PREFIXES
    for name in (f"{prefix}_URL", f"{prefix}_API_KEY_ENV", f"{prefix}_TIMEOUT")
)

EVALUATION_CREDENTIAL_REF_ENV_NAMES = (
    ROUTER_EVALUATION_API_KEY_REF_ENV,
    ENVOY_EVALUATION_API_KEY_REF_ENV,
    *(f"{prefix}_API_KEY_ENV" for prefix in EVALUATION_LEDGER_PREFIXES),
)

EVALUATION_DASHBOARD_CONFIG_ENV_NAMES = (
    EVALUATION_ENABLED_ENV,
    ROUTER_EVALUATION_API_KEY_REF_ENV,
    ENVOY_EVALUATION_API_KEY_REF_ENV,
    *EVALUATION_ENDPOINT_CONFIG_ENV_NAMES,
)


def configure_dashboard_evaluation_env(
    dashboard_env: MutableMapping[str, str],
    *,
    source_config_path: str | None,
    host_env: Mapping[str, str] | None = None,
) -> set[str]:
    """Forward Evaluation config plus referenced secrets to Dashboard only.

    Secret-bearing entries are marker keys with empty values. The container
    command renders them as inheriting ``-e NAME`` arguments, so values never
    enter Docker argv, manifests, or logs.
    """

    environment = os.environ if host_env is None else host_env
    _forward_evaluation_config(dashboard_env, environment)
    refs, owners = _resolve_credential_refs(dashboard_env, environment)
    _validate_router_credential(refs, source_config_path)
    for ref in owners:
        dashboard_env[ref] = ""
    return set(owners)


def _forward_evaluation_config(
    dashboard_env: MutableMapping[str, str],
    environment: Mapping[str, str],
) -> None:
    enabled = dashboard_env.get(EVALUATION_ENABLED_ENV)
    if enabled is None:
        enabled = environment.get(EVALUATION_ENABLED_ENV)
    if enabled is not None:
        if enabled not in {"true", "false"}:
            raise ValueError(
                f"{EVALUATION_ENABLED_ENV} must be exactly 'true' or 'false'"
            )
        dashboard_env[EVALUATION_ENABLED_ENV] = enabled
    if enabled == "false":
        for name in EVALUATION_DASHBOARD_CONFIG_ENV_NAMES:
            if name != EVALUATION_ENABLED_ENV:
                dashboard_env.pop(name, None)
        return

    for name in EVALUATION_DASHBOARD_CONFIG_ENV_NAMES:
        if name == EVALUATION_ENABLED_ENV:
            continue
        value = dashboard_env.get(name)
        if value is None:
            value = environment.get(name)
        if value is not None:
            dashboard_env[name] = value


def _resolve_credential_refs(
    dashboard_env: Mapping[str, str],
    environment: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    refs: dict[str, str] = {}
    owners: dict[str, str] = {}
    for control_name in EVALUATION_CREDENTIAL_REF_ENV_NAMES:
        ref = dashboard_env.get(control_name, "")
        if not ref:
            continue
        if ref != ref.strip() or not runtime_env_name_is_allowed(ref):
            raise ValueError(
                f"{control_name} must name one uppercase, non-reserved environment variable"
            )
        if ref == ROUTER_MANAGEMENT_API_KEY_ENV:
            raise ValueError(
                f"{control_name} cannot reuse the Dashboard management credential"
            )
        owner = owners.get(ref)
        if owner is not None:
            raise ValueError(
                f"Evaluation credential references must be distinct: {owner} and {control_name}"
            )
        secret = environment.get(ref)
        if secret is None or not secret.strip():
            raise ValueError(
                f"Evaluation credential environment variable has no non-empty host value: {ref}"
            )
        owners[ref] = control_name
        refs[control_name] = ref
    return refs, owners


def _validate_router_credential(
    refs: Mapping[str, str],
    source_config_path: str | None,
) -> None:
    router_ref = refs.get(ROUTER_EVALUATION_API_KEY_REF_ENV)
    if router_ref is not None:
        configured_router_tokens = management_credential_env_names(source_config_path)
        if router_ref not in configured_router_tokens:
            raise ValueError(
                "Dedicated Router evaluation credential must be declared in "
                "global.services.management_api.auth.tokens"
            )


def evaluation_dashboard_secret_env_names(
    dashboard_env: Mapping[str, str],
) -> set[str]:
    """Return secret env names referenced by the Dashboard Evaluation config."""

    return {
        ref
        for control_name in EVALUATION_CREDENTIAL_REF_ENV_NAMES
        if (ref := dashboard_env.get(control_name, ""))
    }


def configure_dashboard_evaluation_deployments(
    dashboard_env: MutableMapping[str, str],
    mount_specs: list[str],
    *,
    staging_root: str,
    readable_gid: int,
    host_env: Mapping[str, str] | None = None,
) -> None:
    """Mount an optional deployment registry read-only into Dashboard only."""

    environment = os.environ if host_env is None else host_env
    raw = environment.get(EVALUATION_DEPLOYMENTS_DIR_ENV, "")
    if not raw:
        dashboard_env.pop(EVALUATION_DEPLOYMENTS_DIR_ENV, None)
        return
    if raw != raw.strip():
        raise ValueError(
            f"{EVALUATION_DEPLOYMENTS_DIR_ENV} must not contain surrounding whitespace"
        )
    host_root = Path(raw).absolute()
    snapshot = materialize_evaluation_deployment_snapshot(
        host_root,
        Path(staging_root).absolute(),
        readable_gid=readable_gid,
    )
    mount_specs.append(f"{snapshot}:{EVALUATION_DEPLOYMENTS_CONTAINER_DIR}:ro,z")
    dashboard_env[EVALUATION_DEPLOYMENTS_DIR_ENV] = EVALUATION_DEPLOYMENTS_CONTAINER_DIR
