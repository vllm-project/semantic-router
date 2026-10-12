"""The CLI's config lifecycle commands and their compare-and-swap flow (#4477).

The client layer's header and guard contracts live in
``test_router_management_client.py``; this file covers what the commands
orchestrate around it: which ETag reaches the submit, what a plan without
one does, and how a rollback obtains its ETag before it sends.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from cli.commands import config_management
from cli.main import main
from cli.router_management_client import RouterManagementError, RouterResponse
from click.testing import CliRunner

SOURCE = "version: v0.3\n"


class FakeClient:
    """Records what the commands send; answers with what the test scripted."""

    def __init__(
        self,
        *,
        plan: Any = None,
        apply: Any = None,
        rollback: Any = None,
        versions: Any = None,
    ) -> None:
        self.plan_calls: list[tuple[str, str]] = []
        self.apply_calls: list[tuple[str, str, str]] = []
        self.rollback_calls: list[tuple[str, str]] = []
        self._plan = plan or (lambda: {"changed": True, "current_etag": '"v1"'})
        self._apply = apply or (lambda: {"version": 2})
        self._rollback = rollback or (lambda: {"version": 3})
        self._versions = versions or (lambda: {"versions": []})

    def get_config(self) -> RouterResponse:
        return RouterResponse(payload={}, etag='"v3"')

    def plan_config(self, yaml_text: str, mode: str) -> RouterResponse:
        self.plan_calls.append((yaml_text, mode))
        return RouterResponse(payload=self._plan())

    def apply_config(self, yaml_text: str, mode: str, etag: str) -> RouterResponse:
        self.apply_calls.append((yaml_text, mode, etag))
        return RouterResponse(payload=self._apply())

    def rollback_config(self, version: str, etag: str) -> RouterResponse:
        self.rollback_calls.append((version, etag))
        return RouterResponse(payload=self._rollback())

    def config_versions(self) -> RouterResponse:
        return RouterResponse(payload=self._versions())


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(SOURCE, encoding="utf-8")
    return path


def _run(monkeypatch: pytest.MonkeyPatch, client: FakeClient, *args: str) -> Any:
    monkeypatch.setattr(config_management, "local_stack", lambda: None)
    monkeypatch.setattr(config_management, "_client", lambda *_args: client)
    return CliRunner().invoke(main, ["config", *args])


def test_apply_submits_the_etag_the_plan_returned(
    monkeypatch: pytest.MonkeyPatch, config_file: Path
) -> None:
    client = FakeClient(plan=lambda: {"changed": True, "current_etag": '"v9"'})

    result = _run(monkeypatch, client, "apply", "--config", str(config_file))

    assert result.exit_code == 0, result.output
    assert client.apply_calls == [(SOURCE, "replace", '"v9"')]
    report = json.loads(result.stdout)
    assert report["applied"] is True


def test_apply_short_circuits_an_unchanged_plan(
    monkeypatch: pytest.MonkeyPatch, config_file: Path
) -> None:
    client = FakeClient(plan=lambda: {"changed": False})

    result = _run(monkeypatch, client, "apply", "--config", str(config_file))

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "applied": False,
        "reason": "unchanged",
        "plan": {"changed": False},
    }
    assert client.apply_calls == []


def test_apply_rejects_a_plan_without_the_current_etag(
    monkeypatch: pytest.MonkeyPatch, config_file: Path
) -> None:
    client = FakeClient(plan=lambda: {"changed": True})

    result = _run(monkeypatch, client, "apply", "--config", str(config_file))

    assert result.exit_code == 1
    assert "did not return current_etag" in result.stderr
    assert client.apply_calls == []


def test_apply_rejects_a_plan_that_is_not_a_document(
    monkeypatch: pytest.MonkeyPatch, config_file: Path
) -> None:
    client = FakeClient(plan=lambda: ["not", "a", "document"])

    result = _run(monkeypatch, client, "apply", "--config", str(config_file))

    assert result.exit_code == 1
    assert "invalid config plan" in result.stderr
    assert client.apply_calls == []


def test_plan_prints_the_plan_without_changing_the_router(
    monkeypatch: pytest.MonkeyPatch, config_file: Path
) -> None:
    plan = {"changed": True, "current_etag": '"v1"', "diff": []}
    client = FakeClient(plan=lambda: plan)

    result = _run(monkeypatch, client, "plan", "--config", str(config_file))

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == plan
    assert client.plan_calls == [(SOURCE, "replace")]
    assert client.apply_calls == []


def test_plan_names_apply_when_a_restart_is_required(
    monkeypatch: pytest.MonkeyPatch, config_file: Path
) -> None:
    def refuse() -> None:
        raise RouterManagementError(
            "Router management API returned HTTP 409: RESTART_REQUIRED",
            status=409,
            code="RESTART_REQUIRED",
            detail="listeners changed",
        )

    result = _run(
        monkeypatch, FakeClient(plan=refuse), "plan", "--config", str(config_file)
    )

    assert result.exit_code == 1
    assert "Restart required" in result.stderr
    assert "vllm-sr config apply" in result.stderr


def test_rollback_submits_the_version_with_the_etag_it_just_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()

    result = _run(monkeypatch, client, "rollback", "2")

    assert result.exit_code == 0, result.output
    assert client.rollback_calls == [("2", '"v3"')]
    assert json.loads(result.stdout) == {"version": 3}


def test_rollback_of_a_timed_out_mutation_says_it_may_still_activate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def slow() -> None:
        raise RouterManagementError(
            "Router management API request timed out after 120s", timed_out=True
        )

    result = _run(monkeypatch, FakeClient(rollback=slow), "rollback", "2")

    assert result.exit_code == 1
    assert "may still activate" in result.stderr
    assert "vllm-sr config versions" in result.stderr


def test_versions_prints_the_history(monkeypatch: pytest.MonkeyPatch) -> None:
    history = {"versions": [{"version": 3}, {"version": 2}]}

    result = _run(monkeypatch, FakeClient(versions=lambda: history), "versions")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == history


def test_the_config_group_exposes_the_lifecycle_commands() -> None:
    config = main.commands["config"]

    assert {"plan", "apply", "versions", "rollback"} <= set(config.commands)
