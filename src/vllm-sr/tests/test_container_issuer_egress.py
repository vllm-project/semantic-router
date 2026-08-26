from pathlib import Path

import pytest
import yaml
from cli import container_issuer_egress
from cli.runtime_stack import resolve_runtime_stack


def test_management_issuer_policy_is_refreshed_from_current_network(
    monkeypatch, tmp_path
):
    layout = resolve_runtime_stack(stack_name="team_a")
    observed = []
    subnets = iter(
        [
            ["172.24.0.0/16", "203.0.113.0/24"],
            ["172.31.0.0/16"],
        ]
    )

    def inspect(network_name, *, runtime):
        observed.append((runtime, network_name))
        return (0, next(subnets), "")

    monkeypatch.setattr(container_issuer_egress, "container_network_subnets", inspect)

    first = container_issuer_egress.materialize_management_issuer_egress_policy(
        runtime="docker",
        network_name=layout.network_name,
        state_root_dir=str(tmp_path),
        stack_layout=layout,
    )
    assert yaml.safe_load(first.host_path.read_text())["hosts"][0] == {
        "host": layout.dashboard_container_name,
        "ports": [8743],
        "allow_cidrs": ["172.24.0.0/16"],
    }

    second = container_issuer_egress.materialize_management_issuer_egress_policy(
        runtime="docker",
        network_name=layout.network_name,
        state_root_dir=str(tmp_path),
        stack_layout=layout,
    )
    assert first.host_path == second.host_path
    assert yaml.safe_load(second.host_path.read_text())["hosts"][0]["allow_cidrs"] == [
        "172.31.0.0/16"
    ]
    assert observed == [
        ("docker", layout.network_name),
        ("docker", layout.network_name),
    ]


def test_management_issuer_policy_rejects_unsafe_networks(monkeypatch, tmp_path):
    monkeypatch.setattr(
        container_issuer_egress,
        "container_network_subnets",
        lambda *_args, **_kwargs: (0, ["203.0.113.0/24"], ""),
    )

    with pytest.raises(RuntimeError, match="no safe private IPAM subnet"):
        container_issuer_egress.materialize_management_issuer_egress_policy(
            runtime="docker",
            network_name="public-network",
            state_root_dir=str(tmp_path),
            stack_layout=resolve_runtime_stack(),
        )


def test_management_issuer_policy_mount_is_read_only(tmp_path):
    policy = container_issuer_egress.ManagementIssuerEgressPolicy(
        host_path=Path(tmp_path / "policy.yaml")
    )
    assert policy.mount_spec.endswith(
        ":/app/.vllm-sr/management-issuer-egress-policy.yaml:ro,z"
    )


def test_management_issuer_policy_preserves_exact_stack_service_name(
    monkeypatch, tmp_path
):
    layout = resolve_runtime_stack(stack_name="team_.blue")
    monkeypatch.setattr(
        container_issuer_egress,
        "container_network_subnets",
        lambda *_args, **_kwargs: (0, ["172.24.0.0/16"], ""),
    )

    policy = container_issuer_egress.materialize_management_issuer_egress_policy(
        runtime="docker",
        network_name=layout.network_name,
        state_root_dir=str(tmp_path),
        stack_layout=layout,
    )

    assert yaml.safe_load(policy.host_path.read_text())["hosts"][0]["host"] == (
        "team_.blue-vllm-sr-dashboard-container"
    )
