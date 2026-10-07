"""`vllm-sr config apply` against the local stack (#4698, #4699)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from cli import local_stack_config
from cli.commands import config_management
from cli.local_stack_config import LocalStack, save_pending_restart
from cli.main import main
from cli.pending_activation import ORIGIN_CLI, read_pending_activation
from cli.router_management_client import RouterManagementError, RouterResponse
from cli.runtime_stack import resolve_runtime_stack
from click.testing import CliRunner

MOVED_PORT = 8890
WARMUP_SECONDS = 120

SOURCE = f"""\
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: {MOVED_PORT}
providers:
  defaults:
    model: m
  models:
    - name: m
      backend_refs:
        - endpoint: host.docker.internal:8000
routing:
  modelCards:
    - name: m
"""


@pytest.fixture
def stack(tmp_path: Path) -> LocalStack:
    (tmp_path / "config.yaml").write_text(SOURCE, encoding="utf-8")
    state_dir = tmp_path / ".vllm-sr"
    state_dir.mkdir()
    return LocalStack(
        layout=resolve_runtime_stack(),
        state_dir=state_dir,
        platform=None,
        algorithm=None,
    )


class FakeClient:
    """Records what the CLI sends; answers with what the test scripted."""

    def __init__(self, *, plan=None, apply=None):
        self.sent: list[str] = []
        self._plan = plan or (lambda: {"changed": True, "current_etag": '"v1"'})
        self._apply = apply or (lambda: {"version": 2})

    def plan_config(self, yaml_text, mode):
        self.sent.append(yaml_text)
        return RouterResponse(payload=self._plan())

    def apply_config(self, yaml_text, mode, etag):
        self.sent.append(yaml_text)
        return RouterResponse(payload=self._apply())


def _apply(monkeypatch, stack, client, config_path: Path, *extra: str):
    monkeypatch.setattr(config_management, "local_stack", lambda: stack)
    monkeypatch.setattr(config_management, "_client", lambda *_args: client)
    return CliRunner().invoke(
        main, ["config", "apply", "--config", str(config_path), *extra]
    )


# #4698: what apply sends is what serve materializes, so the Router keeps the
# stack's management listener and a later serve finds its own document.
def test_apply_sends_the_stack_wiring_serve_adds(monkeypatch, stack, tmp_path):
    client = FakeClient()
    result = _apply(monkeypatch, stack, client, tmp_path / "config.yaml")

    assert result.exit_code == 0, result.output
    sent = yaml.safe_load(client.sent[-1])
    assert sent["global"]["services"]["management_api"] == {
        "bind_address": "0.0.0.0",
        "port": 8080,
    }
    assert client.sent[0] == client.sent[1]


def test_an_explicit_endpoint_of_the_local_stack_still_gets_its_wiring(
    monkeypatch, stack, tmp_path
):
    client = FakeClient()
    result = _apply(
        monkeypatch,
        stack,
        client,
        tmp_path / "config.yaml",
        "--endpoint",
        "http://127.0.0.1:8080",
    )

    assert result.exit_code == 0, result.output
    assert "management_api" in yaml.safe_load(client.sent[-1])["global"]["services"]


def test_apply_to_another_router_sends_the_file_as_written(
    monkeypatch, stack, tmp_path
):
    client = FakeClient()
    result = _apply(
        monkeypatch,
        stack,
        client,
        tmp_path / "config.yaml",
        "--endpoint",
        "http://router.example:8080",
    )

    assert result.exit_code == 0, result.output
    assert client.sent[-1] == SOURCE


def test_apply_of_a_file_outside_the_stack_says_how_to_serve_it(
    monkeypatch, stack, tmp_path
):
    elsewhere = tmp_path / "other" / "config.yaml"
    elsewhere.parent.mkdir()
    elsewhere.write_text(SOURCE, encoding="utf-8")

    result = _apply(monkeypatch, stack, FakeClient(), elsewhere)

    assert result.exit_code == 1
    assert "--replace-active-config" in result.output


# #4699: a change the Router can't take is saved like the Dashboard saves one.
def test_restart_required_is_saved_for_the_next_serve(monkeypatch, stack, tmp_path):
    def refuse():
        raise RouterManagementError(
            "Router management API returned HTTP 409: RESTART_REQUIRED: listeners changed",
            status=409,
            code="RESTART_REQUIRED",
            detail="listeners changed; standalone mode binds its listeners at startup",
        )

    saved = []
    monkeypatch.setattr(
        config_management,
        "save_pending_restart",
        lambda *args: saved.append(args) or tmp_path / "runtime-config.yaml",
    )
    result = _apply(
        monkeypatch, stack, FakeClient(plan=refuse), tmp_path / "config.yaml"
    )

    assert result.exit_code == 0, result.output
    assert "Restart required: run `vllm-sr serve` to apply." in result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "restart_required" and report["applied"] is False
    assert report["reason"].startswith("listeners changed")
    assert saved and saved[0][1] == tmp_path / "config.yaml"


def test_restart_required_without_a_local_stack_names_the_restart(
    monkeypatch, tmp_path
):
    (tmp_path / "config.yaml").write_text(SOURCE, encoding="utf-8")

    def refuse():
        raise RouterManagementError(
            "HTTP 409", status=409, code="RESTART_REQUIRED", detail="listeners changed"
        )

    result = _apply(
        monkeypatch, None, FakeClient(plan=refuse), tmp_path / "config.yaml"
    )

    assert result.exit_code == 1
    assert "Restart required: listeners changed" in result.output
    assert "Envoy" not in result.output


def test_a_timed_out_apply_says_the_change_may_still_activate(
    monkeypatch, stack, tmp_path
):
    def slow():
        raise RouterManagementError(
            "Router management API request timed out after 120s", timed_out=True
        )

    result = _apply(
        monkeypatch, stack, FakeClient(apply=slow), tmp_path / "config.yaml"
    )

    assert result.exit_code == 1
    assert "may still activate" in result.output
    assert "vllm-sr config versions" in result.output


def test_apply_waits_for_model_warmup_by_default():
    timeout = next(
        param
        for param in config_management.config_apply.params
        if param.name == "timeout"
    )
    assert (
        timeout.default == config_management.MUTATION_TIMEOUT_SECONDS == WARMUP_SECONDS
    )


def test_save_pending_restart_writes_what_serve_applies(stack, tmp_path):
    runtime_config = save_pending_restart(
        stack, tmp_path / "config.yaml", "listeners changed"
    )

    assert runtime_config == stack.runtime_config()
    active = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    assert active["listeners"][0]["port"] == MOVED_PORT
    assert active["global"]["services"]["management_api"]["bind_address"] == "0.0.0.0"
    pending = read_pending_activation(runtime_config)
    assert pending is not None and pending.reason == "restart"
    assert pending.origin == ORIGIN_CLI
    assert pending.saved_by() == "with `vllm-sr config apply`"
    assert pending.detail == "listeners changed"
    # serve of the same file finds the document it would write, so it applies it.
    receipt = runtime_config.with_suffix(".provenance.json")
    assert receipt.is_file()


def test_local_stack_reads_the_router_container(monkeypatch, tmp_path):
    state_dir = tmp_path / ".vllm-sr"
    monkeypatch.setattr(
        local_stack_config,
        "inspect_container_mounts",
        lambda _name: [{"Destination": "/app/.vllm-sr", "Source": str(state_dir)}],
    )
    monkeypatch.setattr(
        local_stack_config,
        "_container_env",
        lambda _name: {"VLLM_SR_PLATFORM": "amd", "VLLM_SR_ALGORITHM_OVERRIDE": "knn"},
    )

    found = local_stack_config.local_stack()

    assert found is not None
    assert found.state_dir == state_dir
    assert (found.platform, found.algorithm) == ("amd", "knn")
    assert found.owns(tmp_path / "config.yaml")
    assert not found.owns(tmp_path / "elsewhere" / "config.yaml")
