"""The stack's Router management credential: the CLI owns it, nothing else stores it."""

import json
import stat

import pytest
import yaml
from cli import core
from cli.commands.runtime_management_credentials import (
    management_credential_env_names,
)
from cli.commands.runtime_support import normalize_recipe_env_names
from cli.management_credential import (
    ManagementCredentialError,
    management_credential_path,
    stack_management_credential,
)
from cli.recipe_topology_contract import MANAGEMENT_CREDENTIAL_ENV
from cli.runtime_stack import resolve_runtime_stack

TOKEN = "0123456789abcdef" * 4


def _runtime_config(tmp_path, mode="bearer"):
    tokens = (
        "        tokens:\n"
        f"          - env: {MANAGEMENT_CREDENTIAL_ENV}\n"
        "            role: dashboard_control_plane\n"
        if mode == "bearer"
        else ""
    )
    path = tmp_path / "runtime-config.yaml"
    path.write_text(
        "global:\n  services:\n    management_api:\n      auth:\n"
        f"        mode: {mode}\n{tokens}",
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def _no_operator_credential(monkeypatch):
    monkeypatch.delenv(MANAGEMENT_CREDENTIAL_ENV, raising=False)


def test_a_config_that_binds_the_dashboard_credential_names_it(tmp_path):
    assert management_credential_env_names(_runtime_config(tmp_path)) == {
        MANAGEMENT_CREDENTIAL_ENV
    }
    assert (
        management_credential_env_names(_runtime_config(tmp_path, mode="disabled"))
        == set()
    )


def test_the_stack_generates_one_credential_and_keeps_it_owner_only(tmp_path):
    stack = resolve_runtime_stack()

    token = stack_management_credential(tmp_path, stack_layout=stack)

    path = management_credential_path(tmp_path, stack_layout=stack)
    assert len(token) == 64 and int(token, 16) >= 0
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema": "vllm-sr/management-credential/v1",
        "stack": stack.stack_name,
        "token": token,
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) & 0o077 == 0
    assert path.parent.parent == tmp_path / ".vllm-sr"
    # A restart keeps the value, so the Router and the Dashboard keep agreeing.
    assert stack_management_credential(tmp_path, stack_layout=stack) == token


def test_each_stack_keeps_its_own_credential(tmp_path):
    first = stack_management_credential(
        tmp_path, stack_layout=resolve_runtime_stack(stack_name="lane-a")
    )
    second = stack_management_credential(
        tmp_path, stack_layout=resolve_runtime_stack(stack_name="lane-b")
    )

    assert first != second
    assert management_credential_path(
        tmp_path, stack_layout=resolve_runtime_stack(stack_name="lane-a")
    ).name == ("dashboard.lane-a.json")


def test_the_operator_credential_wins_and_is_never_written(tmp_path, monkeypatch):
    monkeypatch.setenv(MANAGEMENT_CREDENTIAL_ENV, TOKEN)
    stack = resolve_runtime_stack()

    assert stack_management_credential(tmp_path, stack_layout=stack) == TOKEN
    assert not (tmp_path / ".vllm-sr" / "management-credential").exists()


def test_a_malformed_operator_credential_stops_the_start(tmp_path, monkeypatch):
    monkeypatch.setenv(MANAGEMENT_CREDENTIAL_ENV, "not-a-token")

    with pytest.raises(ManagementCredentialError, match="64 lowercase hexadecimal"):
        stack_management_credential(tmp_path, stack_layout=resolve_runtime_stack())


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("not json", "is invalid"),
        (
            json.dumps({"schema": "other", "stack": "vllm-sr", "token": TOKEN}),
            "is invalid",
        ),
        (
            json.dumps(
                {
                    "schema": "vllm-sr/management-credential/v1",
                    "stack": "vllm-sr",
                    "token": "x",
                }
            ),
            "stored management credential is invalid",
        ),
        (
            json.dumps(
                {
                    "schema": "vllm-sr/management-credential/v1",
                    "stack": "other",
                    "token": TOKEN,
                }
            ),
            "belongs to stack other",
        ),
    ],
)
def test_an_unusable_state_fails_closed_with_a_recovery_hint(
    tmp_path, document, message
):
    stack = resolve_runtime_stack()
    path = management_credential_path(tmp_path, stack_layout=stack)
    path.write_text(document, encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(ManagementCredentialError, match=message) as raised:
        stack_management_credential(tmp_path, stack_layout=stack)
    assert "rerun `vllm-sr serve`" in str(raised.value)
    assert TOKEN not in str(raised.value)


def test_serve_checks_readiness_with_the_credential_it_gives_the_router(tmp_path):
    runtime_config = _runtime_config(tmp_path)
    user_config = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    user_config["global"]["services"]["management_api"]["auth"]["roles"] = {
        "dashboard_control_plane": ["ready.read"]
    }
    stack = resolve_runtime_stack()

    assert (
        core._readiness_token_env(user_config, {}, runtime_config, tmp_path, stack)
        == MANAGEMENT_CREDENTIAL_ENV
    )
    # The probe reads the value inside the Router container; the CLI only
    # names it.
    assert management_credential_path(tmp_path, stack_layout=stack).is_file()


def test_a_recipe_cannot_bind_the_management_credential():
    with pytest.raises(ValueError, match="cannot be bound into a Recipe"):
        normalize_recipe_env_names([MANAGEMENT_CREDENTIAL_ENV])
    assert normalize_recipe_env_names(["MODEL_API_KEY"]) == ("MODEL_API_KEY",)
