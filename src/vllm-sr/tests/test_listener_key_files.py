"""A listener's client keys by reference: api_key_files, and ${NAME} api_keys."""

import sys
from pathlib import Path

import click
import pytest

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.commands.connect import _require_key  # noqa: E402
from cli.config_generator import generate_envoy_config_from_user_config  # noqa: E402
from cli.container_start import _build_common_runtime_env  # noqa: E402
from cli.parser import ConfigParseError, parse_user_config  # noqa: E402
from cli.runtime_stack import resolve_runtime_stack  # noqa: E402


def _document(keys):
    return f"""version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    {keys}
providers:
  defaults:
    model: local-model
  models:
    - name: local-model
      backend_refs:
        - provider: vllm
          endpoint: 127.0.0.1:8000
recipes:
  - name: default
"""


def _parse(tmp_path, document):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(document)
    return parse_user_config(str(config_path), log_summary=False)


def _render(tmp_path, monkeypatch, config):
    monkeypatch.setenv("ENVOY_EXTPROC_ADDRESS", "localhost")
    monkeypatch.setenv("ENVOY_ROUTER_API_ADDRESS", "localhost")
    output = tmp_path / "envoy.yaml"
    generate_envoy_config_from_user_config(config, str(output), log_summary=False)
    return output.read_text()


def test_a_listener_names_its_key_files(tmp_path):
    config = _parse(tmp_path, _document("api_key_files: [/run/keys/gateway]"))
    assert config.listeners[0].api_key_files == ["/run/keys/gateway"]
    with pytest.raises(ConfigParseError, match="absolute path"):
        _parse(tmp_path, _document("api_key_files: [keys/gateway]"))


def test_envoy_refuses_listener_key_files(tmp_path, monkeypatch):
    config = _parse(tmp_path, _document("api_key_files: [/run/keys/gateway]"))
    with pytest.raises(ValueError, match="api_key_files is read in standalone mode"):
        _render(tmp_path, monkeypatch, config)


def test_envoy_reads_listener_keys_as_the_router_does(tmp_path, monkeypatch):
    # The Router expands only uppercase names, with defaults, and reads $$ as $;
    # any other text, ${workshop_key} included, is a literal key.
    config = _parse(
        tmp_path,
        _document(
            'api_keys: ["${WORKSHOP_KEY}", "${SPARE_KEY:-sk-test-spare}", '
            '"${workshop_key}", "pay$$key"]'
        ),
    )
    monkeypatch.setenv("WORKSHOP_KEY", "sk-test-workshop")
    monkeypatch.delenv("SPARE_KEY", raising=False)
    rendered = _render(tmp_path, monkeypatch, config)
    # Envoy expands nothing: the key itself must be in its configuration.
    for key in ("sk-test-workshop", "sk-test-spare", "${workshop_key}", "pay$key"):
        assert f'["{key}"] = true' in rendered
    assert "${WORKSHOP_KEY}" not in rendered

    monkeypatch.delenv("WORKSHOP_KEY")
    with pytest.raises(
        ValueError,
        match=r"api_keys\[0\] reads WORKSHOP_KEY and expands to an empty key",
    ):
        _render(tmp_path, monkeypatch, config)


def test_connect_asks_for_the_key_of_a_listener_with_key_files():
    doc = {"listeners": [{"port": 8899, "api_key_files": ["/run/keys/gateway"]}]}
    with pytest.raises(click.ClickException, match="takes an API key"):
        _require_key(doc, None, False, "VLLM_SR_API_KEY", None)
    _require_key(doc, None, True, "VLLM_SR_API_KEY", None)


def test_serve_shares_saved_key_files_with_the_dashboard_group():
    # The Router container saves keys as root; the Dashboard (uid 65532) reads
    # them through the state directory's group, which its entrypoint sets.
    env = _build_common_runtime_env(
        {},
        resolve_runtime_stack(),
        runtime_container_config="/app/.vllm-sr/runtime-config.yaml",
    )
    assert env["VLLM_SR_KEY_FILE_GROUP"] == "65532"
