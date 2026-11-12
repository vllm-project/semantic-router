"""Shared fixtures for the CLI unit tests."""

import pytest
from cli import container_run_command, execution_platform, router_validation


@pytest.fixture(autouse=True)
def _no_router_image_validation(monkeypatch, request):
    """Keep `config validate`, `config migrate` and `serve` from running a
    local Router, and from inspecting this host's stack.

    Tests that cover the Router's validation mark themselves `router_image`,
    or name a fake Router binary in VLLM_SR_ROUTER_BINARY.
    """

    monkeypatch.delenv(router_validation.ROUTER_BINARY_ENV, raising=False)
    monkeypatch.setattr(router_validation, "_running_router", lambda: None)
    if request.node.get_closest_marker("router_image"):
        return

    def unavailable(*_args, **_kwargs):
        raise router_validation.RouterValidationUnavailableError(
            "Router image validation is off in unit tests"
        )

    monkeypatch.setattr(router_validation, "validation_image", unavailable)
    monkeypatch.setattr(router_validation, "served_router_image", unavailable)


@pytest.fixture(autouse=True)
def _default_host_gateway(monkeypatch):
    """Keep run-command tests independent of the host's Docker daemon."""

    monkeypatch.delenv(container_run_command.HOST_GATEWAY_IP_ENV, raising=False)
    monkeypatch.setattr(
        container_run_command, "_docker_default_bridge_available", lambda: True
    )


@pytest.fixture(autouse=True)
def _deterministic_execution_target(monkeypatch):
    """CLI unit tests never infer platform from the developer's hardware/cluster."""
    monkeypatch.setattr(
        execution_platform, "_host_devices", lambda: {"cuda": (), "rocm": ()}
    )
    monkeypatch.setattr(
        execution_platform,
        "_cluster_devices",
        lambda context=None: {"cuda": (), "rocm": ()},
    )
    for name in (
        "VLLM_SR_PLATFORM",
        "DASHBOARD_PLATFORM",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "VLLM_SR_AMD_ROUTER_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(name, raising=False)
