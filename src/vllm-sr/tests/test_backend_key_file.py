"""A backend's auth.api_key_file: the Dashboard writes it, serve and Envoy read the config."""

import sys
from pathlib import Path

import pytest

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.config_generator import generate_envoy_config_from_user_config  # noqa: E402
from cli.parser import ConfigParseError, parse_user_config  # noqa: E402


def _document(*auth_lines):
    backends = "".join(
        f"""        - name: openai-{index}
          provider: openai
          endpoint: https://api.openai.com/v1
          auth: {{{auth}}}
"""
        for index, auth in enumerate(auth_lines)
    )
    return f"""version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: gpt
  models:
    - name: gpt
      provider_model_id: gpt-4o-mini
      api_format: chat_completions
      backend_refs:
{backends}recipes:
  - name: default
"""


def _parse(tmp_path, document):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(document)
    return parse_user_config(str(config_path), log_summary=False)


def test_a_backend_names_its_key_file_and_envoy_renders(tmp_path, monkeypatch):
    key_file = "/app/.vllm-sr/provider-keys/key-1"
    config = _parse(tmp_path, _document(f"api_key_file: {key_file}"))
    assert config.providers.models[0].backend_refs[0].auth.api_key_file == key_file

    # The Dashboard regenerates a split stack's Envoy config from this file.
    monkeypatch.setenv("ENVOY_EXTPROC_ADDRESS", "localhost")
    monkeypatch.setenv("ENVOY_ROUTER_API_ADDRESS", "localhost")
    output = tmp_path / "envoy.yaml"
    generate_envoy_config_from_user_config(config, str(output), log_summary=False)
    assert "model_gpt_cluster" in output.read_text()


def test_a_relative_key_file_is_refused(tmp_path):
    with pytest.raises(ConfigParseError, match="absolute path"):
        _parse(tmp_path, _document("api_key_file: provider-keys/key-1"))


def test_replicas_with_different_key_files_cannot_share_a_cluster(
    tmp_path, monkeypatch
):
    config = _parse(
        tmp_path,
        _document("api_key_file: /keys/one", "api_key_file: /keys/two"),
    )
    monkeypatch.setenv("ENVOY_EXTPROC_ADDRESS", "localhost")
    monkeypatch.setenv("ENVOY_ROUTER_API_ADDRESS", "localhost")
    with pytest.raises(ValueError, match="credential"):
        generate_envoy_config_from_user_config(
            config, str(tmp_path / "envoy.yaml"), log_summary=False
        )
