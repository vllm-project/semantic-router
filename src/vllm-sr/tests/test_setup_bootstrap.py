import json
import stat

import yaml

from cli.bootstrap import (
    LOCAL_BOOTSTRAP_TOKEN_NAME,
    ensure_bootstrap_workspace,
    local_bootstrap_token_directory,
    local_dashboard_environment,
)
from cli.config_contract import DEFAULT_BACKEND_DISPATCH
from cli.consts import DEFAULT_LISTENER_PORT
from cli.runtime_stack import resolve_runtime_stack


def test_first_serve_bootstrap_creates_managed_config_and_private_secrets(tmp_path):
    config_path = tmp_path / "config.yaml"

    result = ensure_bootstrap_workspace(config_path)

    assert result.created_config is True
    assert result.created_output_dir is True
    assert result.created_secrets is True
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["version"] == "v0.4"
    assert config["global"]["control_plane"]["mode"] == "managed"
    assert config["global"]["services"]["backend_dispatch"] == DEFAULT_BACKEND_DISPATCH
    layout = resolve_runtime_stack()
    assert config["global"]["services"]["agent"] == {
        "public_inference_endpoint": (
            layout.envoy_listener_service_url(DEFAULT_LISTENER_PORT)
            + "/v1/chat/completions"
        )
    }
    assert "setup" not in config
    assert config["global"]["stores"]["access"]["postgres"]["dsn_env"]
    assert config["global"]["stores"]["access_runtime"]["redis"]["url_env"]
    serialized = config_path.read_text(encoding="utf-8")
    assert "postgresql://" not in serialized
    assert "redis://" not in serialized

    secret_files = tuple(
        path for path in result.secret_dir.rglob("*") if path.is_file()
    )
    assert secret_files
    assert stat.S_IMODE(result.secret_dir.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in secret_files)
    bootstrap_dir = local_bootstrap_token_directory(tmp_path, resolve_runtime_stack())
    assert stat.S_IMODE(bootstrap_dir.stat().st_mode) == 0o700
    assert (bootstrap_dir / LOCAL_BOOTSTRAP_TOKEN_NAME).is_file()
    keyring = json.loads((result.secret_dir / "api-key-hmac.json").read_text())
    assert keyring["activeVersion"] == "v1"
    assert len(keyring["keys"]) == 1


def test_bootstrap_is_idempotent_and_never_rotates_existing_material(tmp_path):
    config_path = tmp_path / "config.yaml"
    first = ensure_bootstrap_workspace(config_path)
    before = {
        str(path.relative_to(first.secret_dir)): path.read_bytes()
        for path in first.secret_dir.rglob("*")
        if path.is_file()
    }

    second = ensure_bootstrap_workspace(config_path)

    assert second.created_config is False
    assert second.created_secrets is False
    assert {
        str(path.relative_to(second.secret_dir)): path.read_bytes()
        for path in second.secret_dir.rglob("*")
        if path.is_file()
    } == before


def test_dashboard_bootstrap_authority_is_isolated_and_remains_finalized(tmp_path):
    config_path = tmp_path / "config.yaml"
    ensure_bootstrap_workspace(config_path)
    layout = resolve_runtime_stack()
    token = (
        local_bootstrap_token_directory(tmp_path, layout) / LOCAL_BOOTSTRAP_TOKEN_NAME
    )

    token.unlink()
    environment = local_dashboard_environment(tmp_path, layout)

    assert environment["DASHBOARD_ROUTER_BOOTSTRAP_TOKEN_FILE"] == str(token)
    assert (
        environment["DASHBOARD_ISSUER_TLS_KEY_FILE"]
        != environment["DASHBOARD_SIGNING_KEY_FILE"]
    )
    assert environment["DASHBOARD_ISSUER_TLS_KEY_FILE"] != str(
        tmp_path / ".vllm-sr" / "secrets" / layout.stack_name / "management-tls-key.pem"
    )


def test_bootstrap_preserves_an_explicit_existing_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    original = "version: v0.4\nlisteners: []\n"
    config_path.write_text(original, encoding="utf-8")

    result = ensure_bootstrap_workspace(config_path)

    assert result.created_config is False
    assert result.created_secrets is False
    assert config_path.read_text(encoding="utf-8") == original
    assert not result.secret_dir.exists()


def test_bootstrap_does_not_create_storage_credentials(tmp_path):
    ensure_bootstrap_workspace(tmp_path / "config.yaml")

    assert not (tmp_path / ".vllm-sr" / "storage-secrets").exists()
