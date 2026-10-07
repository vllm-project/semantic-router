import pytest
from cli.commands.runtime_management_credentials import (
    RECIPE_MANAGEMENT_CREDENTIAL_PATH,
    recipe_management_credential_env,
)
from cli.recipe_topology_contract import MANAGEMENT_CREDENTIAL_ENV

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


def _recipe_store(tmp_path, token):
    store = tmp_path / "recipe-store"
    credential = store / RECIPE_MANAGEMENT_CREDENTIAL_PATH
    credential.parent.mkdir(parents=True)
    if token is not None:
        credential.write_text(token + "\n", encoding="utf-8")
    return store


@pytest.fixture(autouse=True)
def _no_operator_credential(monkeypatch):
    monkeypatch.delenv(MANAGEMENT_CREDENTIAL_ENV, raising=False)


def test_a_router_bound_to_the_dashboard_credential_gets_it(tmp_path):
    assert recipe_management_credential_env(
        _runtime_config(tmp_path), _recipe_store(tmp_path, TOKEN)
    ) == {MANAGEMENT_CREDENTIAL_ENV: TOKEN}


def test_a_config_that_binds_no_credential_gets_none(tmp_path):
    assert (
        recipe_management_credential_env(
            _runtime_config(tmp_path, mode="disabled"), _recipe_store(tmp_path, TOKEN)
        )
        == {}
    )


def test_an_operator_credential_is_left_to_the_passthrough(tmp_path, monkeypatch):
    monkeypatch.setenv(MANAGEMENT_CREDENTIAL_ENV, "f" * 64)
    assert (
        recipe_management_credential_env(
            _runtime_config(tmp_path), _recipe_store(tmp_path, TOKEN)
        )
        == {}
    )


@pytest.mark.parametrize(
    ("token", "message"),
    [(None, "the Recipe store has none"), ("not-a-token", "is invalid")],
)
def test_a_missing_or_malformed_credential_stops_the_start(tmp_path, token, message):
    with pytest.raises(ValueError, match=message):
        recipe_management_credential_env(
            _runtime_config(tmp_path), _recipe_store(tmp_path, token)
        )
