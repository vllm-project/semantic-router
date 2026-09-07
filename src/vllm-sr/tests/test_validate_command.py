from types import SimpleNamespace

import pytest
from cli.commands.validate import (
    _aggregate_projection_summary_lines,
    _aggregate_signal_summary_lines,
    _projection_summary_lines,
    _provider_projection_errors,
    _signal_summary_lines,
)
from cli.main import main
from cli.parser import parse_user_config
from click.testing import CliRunner


def test_signal_summary_lines_cover_v03_signal_surface():
    signals = SimpleNamespace(
        keywords=[object()],
        embeddings=[object()],
        domains=[object()],
        fact_check=[object()],
        user_feedbacks=[object()],
        reasks=[object()],
        preferences=[object()],
        language=[object()],
        context=[object()],
        structure=[object()],
        complexity=[object()],
        modality=[object()],
        role_bindings=[object()],
        jailbreak=[object()],
        pii=[object()],
        kb=[object()],
        conversation=[object()],
        events=[object()],
    )

    lines = _signal_summary_lines(signals)

    assert "  Keyword signals: 1" in lines
    assert "  Embedding signals: 1" in lines
    assert "  Domains: 1" in lines
    assert "  Fact check signals: 1" in lines
    assert "  User feedback signals: 1" in lines
    assert "  Reask signals: 1" in lines
    assert "  Preference signals: 1" in lines
    assert "  Language signals: 1" in lines
    assert "  Context signals: 1" in lines
    assert "  Structure signals: 1" in lines
    assert "  Complexity signals: 1" in lines
    assert "  Modality signals: 1" in lines
    assert "  Authz signals: 1" in lines
    assert "  Jailbreak signals: 1" in lines
    assert "  PII signals: 1" in lines
    assert "  Knowledge-base signals: 1" in lines
    assert "  Conversation signals: 1" in lines
    assert "  Event signals: 1" in lines


def test_projection_summary_lines_cover_v03_projection_surface():
    projections = SimpleNamespace(
        partitions=[object()],
        scores=[object()],
        mappings=[object()],
    )

    assert _projection_summary_lines(projections) == [
        "  Projection partitions: 1",
        "  Projection scores: 1",
        "  Projection mappings: 1",
    ]


def test_aggregate_summary_lines_include_recipe_owned_routing():
    default = SimpleNamespace(
        signals=SimpleNamespace(keywords=[]),
        projections=SimpleNamespace(scores=[], mappings=[]),
    )
    recipe = SimpleNamespace(
        signals=SimpleNamespace(keywords=[object(), object()]),
        projections=SimpleNamespace(scores=[object()], mappings=[object()]),
    )
    profiles = [("default", default), ("accuracy-first", recipe)]

    assert _aggregate_signal_summary_lines(profiles) == ["  Keyword signals: 2"]
    assert _aggregate_projection_summary_lines(profiles) == [
        "  Projection scores: 1",
        "  Projection mappings: 1",
    ]


def _write_config(tmp_path, text: str):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(text)
    return config_path


def test_provider_projection_validation_is_structural_and_non_mutating(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("VALIDATION_ONLY_OPENAI_API_KEY", raising=False)
    config_path = _write_config(
        tmp_path,
        """
version: v0.3
providers:
  models:
    - name: frontier
      catalog: openai/gpt-5.4
      backend_refs:
        - provider: openai
          api_key_env: VALIDATION_ONLY_OPENAI_API_KEY
routing: {}
""",
    )
    user_config = parse_user_config(str(config_path), log_summary=False)
    authored_config = user_config.model_dump()

    assert _provider_projection_errors(user_config) == []
    assert user_config.model_dump() == authored_config


def test_validate_command_preserves_backendless_external_gateway_metadata(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
version: v0.3
listeners: []
providers:
  models:
    - name: claude-metadata-only
      api_format: anthropic
routing: {}
""",
    )

    result = CliRunner().invoke(main, ["validate", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "Configuration is valid" in result.output


def test_validate_command_accepts_routing_only_model_card_default(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
version: v0.3
providers:
  defaults:
    model: private-model
routing:
  modelCards:
    - name: private-model
      description: Metadata-only routing model
""",
    )

    result = CliRunner().invoke(main, ["validate", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "Configuration is valid" in result.output


def test_validate_command_rejects_empty_external_gateway_provider_model(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
version: v0.3
listeners: []
providers:
  models:
    - name: bare
routing: {}
""",
    )

    result = CliRunner().invoke(main, ["validate", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "must define backend_refs or model metadata" in result.output


def test_validate_command_accepts_backendless_explicit_zero_pricing(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
version: v0.3
listeners: []
providers:
  models:
    - name: free-model
      pricing:
        prompt_per_1m: 0
        completion_per_1m: 0
routing: {}
""",
    )

    result = CliRunner().invoke(main, ["validate", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "Configuration is valid" in result.output


def test_validate_command_rejects_blank_model_identity(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
version: v0.3
listeners: []
providers:
  models:
    - name: " "
      api_format: openai
routing: {}
""",
    )

    result = CliRunner().invoke(main, ["validate", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "Provider model name cannot be empty" in result.output


def test_validate_command_rejects_blank_routing_card_identity(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
version: v0.3
listeners: []
providers: {}
routing:
  modelCards:
    - name: " "
""",
    )

    result = CliRunner().invoke(main, ["validate", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "Model card name cannot be empty" in result.output


@pytest.mark.parametrize(
    ("base_url", "expected_error"),
    (
        (
            "ftp://example.com/v1#fragment",
            "base_url scheme 'ftp' is unsupported",
        ),
        (
            "https://example.com/v1#fragment",
            "base_url userinfo and fragments are not allowed",
        ),
        (
            "https://example.com/v1?region=west",
            "base_url query parameters are not supported by Envoy routing",
        ),
    ),
)
def test_validate_command_rejects_invalid_backend_url(
    tmp_path, base_url, expected_error
):
    config_path = _write_config(
        tmp_path,
        f"""
version: v0.3
listeners: []
providers:
  models:
    - name: local-model
      api_format: openai
      backend_refs:
        - provider: vllm
          base_url: {base_url}
routing: {{}}
""",
    )

    result = CliRunner().invoke(main, ["validate", "--config", str(config_path)])

    assert result.exit_code == 1
    assert expected_error in result.output


@pytest.mark.parametrize(
    ("backend_refs", "expected_error"),
    (
        (
            """
        - provider: vllm
          base_url: http://a.example/v1
          api_key_env: KEY_A
        - provider: vllm
          base_url: http://b.example/custom
          api_key_env: KEY_A
""",
            "base path differ",
        ),
        (
            """
        - provider: vllm
          base_url: http://a.example/v1
          api_key_env: KEY_A
        - provider: vllm
          base_url: http://b.example/v1
          api_key_env: KEY_B
""",
            "credential source differ",
        ),
        (
            """
        - provider: vllm
          base_url: http://a.example/v1
        - provider: sglang
          base_url: http://b.example/v1
""",
            "provider differ",
        ),
        (
            """
        - provider: vllm
          base_url: http://a.example/v1
        - provider: vllm
          base_url: https://a.example/v1
""",
            "URL scheme",
        ),
        (
            """
        - provider: vllm
          base_url: http://a.example/v1
          extra_headers:
            x-region: east
        - provider: vllm
          base_url: http://b.example/v1
          extra_headers:
            x-region: west
""",
            "extra headers differ",
        ),
        (
            """
        - provider: vllm
          base_url: https://a.example/v1
        - provider: vllm
          base_url: https://b.example/v1
""",
            "TLS server name differ",
        ),
        (
            """
        - provider: vllm
          base_url: https://127.0.0.1/v1
""",
            "HTTPS endpoint must use a DNS hostname",
        ),
        (
            """
        - provider: vllm
          base_url: http://a.example:not-a-port/v1
""",
            "has an invalid endpoint port",
        ),
        (
            """
        - provider: vllm
          base_url: http://a.example:0/v1
""",
            "expected 1..65535",
        ),
        (
            """
        - provider: vllm
          endpoint: a.example:65536/v1
""",
            "expected 1..65535",
        ),
        (
            """
        - provider: vllm
          base_url: http://a.example:65536/v1
""",
            "has an invalid endpoint port",
        ),
        (
            """
        - provider: vllm
          endpoint: a.example:-1/v1
""",
            "expected 1..65535",
        ),
        (
            """
        - name: replica
          provider: vllm
          base_url: http://a.example/v1
        - name: replica
          provider: vllm
          base_url: http://b.example/v1
""",
            "name 'replica' is duplicated",
        ),
        (
            """
        - provider: vllm
          base_url: http://base.example/v1
          endpoint: endpoint.example:8000/v1
""",
            "cannot set both base_url and endpoint",
        ),
    ),
)
def test_validate_command_reuses_envoy_backend_group_validation(
    tmp_path, backend_refs, expected_error
):
    config_path = _write_config(
        tmp_path,
        f"""
version: v0.3
listeners: []
providers:
  models:
    - name: local-model
      api_format: openai
      backend_refs:
{backend_refs}
routing: {{}}
""",
    )

    result = CliRunner().invoke(main, ["validate", "--config", str(config_path)])

    assert result.exit_code == 1
    assert expected_error in result.output


def test_validate_command_rejects_backendless_router_owned_model(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
version: v0.3
providers:
  models:
    - name: local-model
      api_format: openai
routing: {}
""",
    )

    result = CliRunner().invoke(main, ["validate", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "must define backend_refs with an explicit Provider ID" in result.output


@pytest.mark.parametrize(
    ("model_config", "expected_error"),
    (
        (
            """
    - name: frontier
      catalog: openai/gpt-5.4
      backend_refs:
        - provider: anthropic
""",
            "provider 'anthropic' has no catalog mapping for model 'openai/gpt-5.4'",
        ),
        (
            """
    - name: frontier
      catalog: openai/gpt-5.4
      api_format: bogus
      backend_refs:
        - provider: openai
""",
            "api_format 'bogus' is unsupported",
        ),
        (
            """
    - name: private-deployment
      provider_model_id: deployment-name
      api_format: openai
      backend_refs:
        - provider: azure-openai
""",
            "requires endpoint or base_url because provider 'azure-openai' has no default",
        ),
    ),
)
def test_validate_command_rejects_provider_projection_errors(
    tmp_path, model_config, expected_error
):
    config_path = _write_config(
        tmp_path,
        f"""
version: v0.3
listeners: []
providers:
  models:
{model_config}
routing: {{}}
""",
    )

    result = CliRunner().invoke(main, ["validate", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "Configuration validation failed" in result.output
    assert expected_error in result.output


@pytest.mark.parametrize(
    "model_config",
    (
        """
    - name: frontier
      catalog: openai/gpt-5.4
      backend_refs:
        - provider: openai
""",
        """
    - name: local-model
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: http://127.0.0.1:8000/v1
""",
        """
    - name: local-model
      api_format: openai
      backend_refs:
        - provider: vllm
          base_url: http://[::1]:8000/v1
""",
        """
    - name: local-model
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: "[::1]:9123/v1"
          protocol: HTTP
""",
        """
    - name: local-model
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: api.example.test:9443/v1
          protocol: HTTPS
""",
    ),
)
def test_validate_command_accepts_sparse_builtin_and_custom_models(
    tmp_path, model_config
):
    config_path = _write_config(
        tmp_path,
        f"""
version: v0.3
providers:
  defaults:
    model: {"frontier" if "catalog:" in model_config else "local-model"}
  models:
{model_config}
routing: {{}}
""",
    )

    result = CliRunner().invoke(main, ["validate", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "Configuration is valid" in result.output
