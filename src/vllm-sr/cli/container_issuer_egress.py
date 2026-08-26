"""Materialize the local Dashboard issuer's system-owned egress boundary."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from pathlib import Path

import yaml

from cli.commands.runtime_paths import (
    CONTAINER_READABLE_STATE_FILE_MODE,
    private_runtime_state_subdirectory,
    write_private_state_bytes,
)
from cli.container_services import container_network_subnets
from cli.runtime_stack import RuntimeStackLayout

MANAGEMENT_ISSUER_EGRESS_POLICY_ENV = (
    "VLLM_SR_INTERNAL_MANAGEMENT_ISSUER_EGRESS_POLICY_FILE"
)
MANAGEMENT_ISSUER_EGRESS_POLICY_CONTAINER_PATH = (
    "/app/.vllm-sr/management-issuer-egress-policy.yaml"
)
_SYSTEM_POLICY_DIRECTORY = "system-policies"
_PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "fc00::/7",
    )
)


@dataclass(frozen=True)
class ManagementIssuerEgressPolicy:
    """Host and container locations for one regenerated system policy."""

    host_path: Path
    container_path: str = MANAGEMENT_ISSUER_EGRESS_POLICY_CONTAINER_PATH

    @property
    def mount_spec(self) -> str:
        return f"{self.host_path}:{self.container_path}:ro,z"


def materialize_management_issuer_egress_policy(
    *,
    runtime: str,
    network_name: str,
    state_root_dir: str,
    stack_layout: RuntimeStackLayout,
) -> ManagementIssuerEgressPolicy:
    """Regenerate the issuer exception from the network's current IPAM state.

    The file is replaced on every start.  Recreating the application network
    therefore cannot leave a stale private-range exception behind, and the
    exception remains scoped to the Dashboard issuer rather than inference
    backends.
    """

    return_code, raw_subnets, stderr = container_network_subnets(
        network_name, runtime=runtime
    )
    if return_code != 0:
        detail = stderr.strip() or "container runtime returned no details"
        raise RuntimeError(
            f"cannot inspect application network {network_name}: {detail}"
        )
    subnets = _canonical_private_subnets(raw_subnets)
    if not subnets:
        raise RuntimeError(
            f"application network {network_name} has no safe private IPAM subnet"
        )
    directory = private_runtime_state_subdirectory(
        state_root_dir, _SYSTEM_POLICY_DIRECTORY
    )
    path = directory / f"management-issuer-egress.{stack_layout.stack_name}.yaml"
    document = {
        "version": "v1",
        "schemes": ["https"],
        "hosts": [
            {
                "host": stack_layout.dashboard_container_name,
                "ports": [8743],
                "allow_cidrs": subnets,
            }
        ],
    }
    payload = yaml.safe_dump(document, sort_keys=False).encode("utf-8")
    write_private_state_bytes(
        path,
        payload,
        mode=CONTAINER_READABLE_STATE_FILE_MODE,
    )
    return ManagementIssuerEgressPolicy(host_path=path)


def _canonical_private_subnets(values: list[str]) -> list[str]:
    subnets = []
    for value in values:
        try:
            subnet = ipaddress.ip_network(value, strict=True)
        except ValueError as exc:
            raise RuntimeError(
                f"application network returned an invalid IPAM subnet: {value}"
            ) from exc
        if any(
            subnet.version == private.version and subnet.subnet_of(private)
            for private in _PRIVATE_NETWORKS
        ):
            subnets.append(str(subnet))
    return sorted(set(subnets), key=lambda value: (":" in value, value))
