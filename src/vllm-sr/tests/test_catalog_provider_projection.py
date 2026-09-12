"""Contracts for projecting sparse provider bindings into Envoy config."""

import sys
from pathlib import Path

import pytest
import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.catalog_provider_projection import (  # noqa: E402
    project_provider_models_for_envoy,
)
from cli.config_generator import generate_envoy_config_from_user_config  # noqa: E402
from cli.parser import ConfigParseError, parse_user_config  # noqa: E402
from cli.validator import validate_user_config  # noqa: E402

REPO_ROOT = CLI_ROOT.parents[1]


def _render_envoy_config(tmp_path, monkeypatch, config_text):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "envoy.yaml"
    config_path.write_text(config_text)
    monkeypatch.setenv("ENVOY_EXTPROC_ADDRESS", "localhost")
    monkeypatch.setenv("ENVOY_ROUTER_API_ADDRESS", "localhost")
    user_config = parse_user_config(str(config_path))
    generate_envoy_config_from_user_config(user_config, str(output_path))
    return yaml.safe_load(output_path.read_text())


def _cluster_by_name(rendered_config, cluster_name):
    for cluster in rendered_config["static_resources"]["clusters"]:
        if cluster["name"] == cluster_name:
            return cluster
    raise AssertionError(f"cluster {cluster_name!r} not found")


def _model_route(rendered_config, model_name):
    listener = rendered_config["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    routes = hcm["route_config"]["virtual_hosts"][0]["routes"]
    for route in routes:
        headers = route.get("match", {}).get("headers", [])
        if any(
            header.get("name") == "x-selected-model"
            and header.get("string_match", {}).get("exact") == model_name
            for header in headers
        ):
            return route
    raise AssertionError(f"route for model {model_name!r} not found")


def _default_route(rendered_config):
    listener = rendered_config["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    routes = hcm["route_config"]["virtual_hosts"][0]["routes"]
    for route in reversed(routes):
        if not route.get("match", {}).get("headers"):
            return route
    raise AssertionError("default route not found")


def test_catalog_provider_defaults_materialize_complete_cloud_route(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "catalog-openai-key")
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: frontier
  models:
    - name: frontier
      catalog: openai/gpt-5.4
      backend_refs:
        - name: primary
          provider: openai
          api_key_env: OPENAI_API_KEY
routing:
  decisions:
    - name: default-route
      priority: 100
      modelRefs:
        - model: frontier
""",
    )

    cluster = _cluster_by_name(rendered, "frontier_cluster")
    assert cluster["type"] == "LOGICAL_DNS"
    assert cluster["transport_socket"]["name"] == "envoy.transport_sockets.tls"
    endpoint = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]
    assert endpoint["endpoint"]["address"]["socket_address"] == {
        "address": "api.openai.com",
        "port_value": 443,
    }
    assert endpoint["load_balancing_weight"] == 1

    route = _model_route(rendered, "frontier")
    assert route["route"]["cluster"] == "frontier_cluster"
    assert route["route"]["host_rewrite_literal"] == "api.openai.com"
    assert "regex_rewrite" not in route["route"]
    assert (
        _default_route(rendered)["route"]["regex_rewrite"]["substitution"] == "/v1\\1"
    )
    headers = {
        header["header"]["key"]: header["header"]["value"]
        for header in route.get("request_headers_to_add", [])
    }
    # Credentials are owned by Router/extproc so per-request credentials can
    # override the static fallback without a second Envoy injection point.
    assert "Authorization" not in headers
    assert "catalog-openai-key" not in yaml.safe_dump(rendered)


def test_catalog_anthropic_binding_keeps_credentials_and_default_headers(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "catalog-anthropic-key")
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
providers:
  defaults:
    model: claude
  models:
    - name: claude
      catalog: anthropic/claude-sonnet-5
      backend_refs:
        - provider: anthropic
          api_key_env: ANTHROPIC_API_KEY
routing: {}
""",
    )

    cluster = _cluster_by_name(rendered, "claude_cluster")
    assert cluster["type"] == "LOGICAL_DNS"
    assert cluster["transport_socket"]["name"] == "envoy.transport_sockets.tls"
    endpoint = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]
    assert endpoint["endpoint"]["address"]["socket_address"] == {
        "address": "api.anthropic.com",
        "port_value": 443,
    }
    route = _model_route(rendered, "claude")
    assert route["route"]["cluster"] == "claude_cluster"
    headers = {
        header["header"]["key"]: header["header"]["value"]
        for header in route.get("request_headers_to_add", [])
    }
    assert headers == {"anthropic-version": "2023-06-01"}
    assert "catalog-anthropic-key" not in yaml.safe_dump(rendered)
    with pytest.raises(AssertionError):
        _cluster_by_name(rendered, "anthropic_api_cluster")


@pytest.mark.parametrize(
    ("api_format", "model_name"),
    (("anthropic", "claude-test"), ("openai", "generic-chat-model")),
)
def test_router_owned_physical_model_requires_explicit_provider(
    tmp_path, api_format, model_name
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
version: v0.3
listeners:
  - name: public
    address: 0.0.0.0
    port: 8899
providers:
  models:
    - name: {model_name}
      api_format: {api_format}
routing: {{}}
"""
    )
    config = parse_user_config(str(config_path))

    errors = validate_user_config(config, log_summary=False)
    assert any(
        error.field == f"providers.models.{model_name}.backend_refs"
        and "must define backend_refs with an explicit Provider ID" in error.message
        for error in errors
    )

    with pytest.raises(
        ValueError,
        match="must define backend_refs with an explicit Provider ID",
    ):
        generate_envoy_config_from_user_config(
            config,
            str(tmp_path / "envoy.yaml"),
        )


def test_router_owned_virtual_model_may_resolve_its_recipe_pool(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: v0.3
listeners:
  - name: public
    address: 0.0.0.0
    port: 8899
providers:
  models:
    - name: auto
      catalog: vllm-sr/mom-v1-lite
routing: {}
"""
    )

    errors = validate_user_config(
        parse_user_config(str(config_path)),
        log_summary=False,
    )

    assert not any(error.field.endswith(".backend_refs") for error in errors)


def test_backendless_api_model_is_valid_metadata_but_cannot_generate_envoy(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: v0.3
listeners: []
providers:
  models:
    - name: claude-test
      api_format: anthropic
routing: {}
"""
    )

    config = parse_user_config(str(config_path))

    assert validate_user_config(config, log_summary=False) == []
    with pytest.raises(
        ValueError,
        match="must define backend_refs with an explicit Provider ID",
    ):
        generate_envoy_config_from_user_config(
            config,
            str(tmp_path / "envoy.yaml"),
        )


def test_catalog_openai_responses_binding_keeps_header_and_base_path(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "responses-key")
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
providers:
  defaults:
    model: frontier-responses
  models:
    - name: frontier-responses
      catalog: openai/gpt-5.4
      api_format: responses
      backend_refs:
        - provider: openai
          api_key_env: OPENAI_API_KEY
routing: {}
""",
    )

    route = _model_route(rendered, "frontier-responses")
    assert route["route"]["cluster"] == "frontier_responses_cluster"
    assert "regex_rewrite" not in route["route"]
    assert (
        _default_route(rendered)["route"]["regex_rewrite"]["substitution"] == "/v1\\1"
    )
    headers = {
        header["header"]["key"]: header["header"]["value"]
        for header in route.get("request_headers_to_add", [])
    }
    assert "Authorization" not in headers
    assert "responses-key" not in yaml.safe_dump(rendered)


def test_explicit_empty_auth_prefix_overrides_catalog_default(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: v0.3
providers:
  models:
    - name: private
      provider_model_id: private
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: https://private.example/v1
          auth_header: x-api-key
          auth_prefix: ""
          api_key_env: PRIVATE_API_KEY
routing: {}
"""
    )

    projected = project_provider_models_for_envoy(parse_user_config(str(config_path)))

    assert projected[0].backend_refs[0].auth_header == "x-api-key"
    assert projected[0].backend_refs[0].auth_prefix == ""


def test_catalog_provider_keeps_operator_custom_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "custom-gateway-key")
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
providers:
  defaults:
    model: frontier
  models:
    - name: frontier
      catalog: openai/gpt-5.4
      backend_refs:
        - provider: openai
          base_url: https://gateway.example.test/openai/v1
          api_key_env: OPENAI_API_KEY
routing:
  decisions:
    - name: default-route
      priority: 100
      modelRefs:
        - model: frontier
""",
    )

    cluster = _cluster_by_name(rendered, "frontier_cluster")
    endpoint = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]
    assert (
        endpoint["endpoint"]["address"]["socket_address"]["address"]
        == "gateway.example.test"
    )
    route = _model_route(rendered, "frontier")
    assert route["route"]["host_rewrite_literal"] == "gateway.example.test"
    assert "regex_rewrite" not in route["route"]
    assert (
        _default_route(rendered)["route"]["regex_rewrite"]["substitution"]
        == "/openai/v1\\1"
    )


def test_catalog_provider_projection_rejects_missing_model_mapping(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: v0.3
providers:
  models:
    - name: frontier
      catalog: openai/gpt-5.4
      backend_refs:
        - provider: anthropic
          api_key_env: ANTHROPIC_API_KEY
routing: {}
"""
    )
    config = parse_user_config(str(config_path))

    with pytest.raises(
        ValueError,
        match=r"provider 'anthropic' has no catalog mapping for model 'openai/gpt-5.4'",
    ):
        generate_envoy_config_from_user_config(
            config,
            str(tmp_path / "envoy.yaml"),
        )


def test_catalog_provider_projection_rejects_provider_without_endpoint(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: v0.3
providers:
  models:
    - name: private-deployment
      provider_model_id: deployment-name
      api_format: openai
      backend_refs:
        - provider: azure-openai
          api_key_env: AZURE_OPENAI_API_KEY
routing: {}
"""
    )
    config = parse_user_config(str(config_path))

    with pytest.raises(
        ValueError,
        match=r"requires endpoint or base_url because provider 'azure-openai' has no default",
    ):
        generate_envoy_config_from_user_config(
            config,
            str(tmp_path / "envoy.yaml"),
        )


@pytest.mark.parametrize(
    ("base_url", "expected_error"),
    (
        (
            "ftp://example.com/v1",
            "base_url scheme 'ftp' is unsupported",
        ),
        (
            "https://user@example.com/v1",
            "base_url userinfo and fragments are not allowed",
        ),
        (
            "https://example.com/v1#fragment",
            "base_url userinfo and fragments are not allowed",
        ),
        (
            "example.com/v1",
            "base_url must be an absolute http(s) URL",
        ),
        (
            "https://example.com/v1?region=west",
            "base_url query parameters are not supported by Envoy routing",
        ),
    ),
)
def test_catalog_provider_projection_rejects_invalid_backend_url(
    tmp_path, base_url, expected_error
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
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
"""
    )

    with pytest.raises(ValueError) as error:
        project_provider_models_for_envoy(parse_user_config(str(config_path)))
    assert expected_error in str(error.value)


def test_catalog_provider_projection_rejects_base_url_with_endpoint(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: v0.3
listeners: []
providers:
  models:
    - name: local-model
      api_format: openai
      backend_refs:
        - provider: vllm
          base_url: http://base.example/v1
          endpoint: endpoint.example:8000/v1
routing: {}
"""
    )

    with pytest.raises(ValueError, match="cannot set both base_url and endpoint"):
        project_provider_models_for_envoy(parse_user_config(str(config_path)))


def test_catalog_provider_projection_accepts_backend_url_templates(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: v0.3
listeners: []
providers:
  models:
    - name: private-deployment
      provider_model_id: deployment-name
      api_format: openai
      backend_refs:
        - provider: azure-openai
          base_url: https://${AZURE_RESOURCE}.openai.azure.com/openai/deployments/${AZURE_DEPLOYMENT}
          api_version: "2025-04-01"
routing: {}
"""
    )

    projected = project_provider_models_for_envoy(parse_user_config(str(config_path)))

    assert projected[0].backend_refs[0].base_url.startswith("https://${AZURE_RESOURCE}")
    assert projected[0].backend_refs[0].api_version == "2025-04-01"


@pytest.mark.parametrize(
    (
        "backend_config",
        "expected_address",
        "expected_port",
        "expected_authority",
        "expected_https",
    ),
    (
        (
            "base_url: http://[::1]:8000/v1",
            "::1",
            8000,
            "[::1]:8000",
            False,
        ),
        (
            'endpoint: "[::1]:9123/v1"\n          protocol: HTTP',
            "::1",
            9123,
            "[::1]:9123",
            False,
        ),
        (
            "endpoint: api.example.test:9443/v1\n          protocol: HTTPS",
            "api.example.test",
            9443,
            "api.example.test:9443",
            True,
        ),
    ),
)
def test_shared_backend_projection_normalizes_protocol_and_ipv6_authority(
    tmp_path,
    monkeypatch,
    backend_config,
    expected_address,
    expected_port,
    expected_authority,
    expected_https,
):
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        f"""
version: v0.3
providers:
  models:
    - name: local-model
      api_format: openai
      backend_refs:
        - provider: vllm
          {backend_config}
routing: {{}}
""",
    )

    cluster = _cluster_by_name(rendered, "local_model_cluster")
    endpoint = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]
    assert endpoint["endpoint"]["address"]["socket_address"] == {
        "address": expected_address,
        "port_value": expected_port,
    }
    route = _model_route(rendered, "local-model")
    assert route["route"]["host_rewrite_literal"] == expected_authority
    assert "regex_rewrite" not in route["route"]
    assert (
        _default_route(rendered)["route"]["regex_rewrite"]["substitution"]
        == "/v1" + r"\1"
    )
    assert ("transport_socket" in cluster) is expected_https


def test_catalog_deployment_name_requires_explicit_provider_model_id(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: v0.3
providers:
  models:
    - name: mai
      catalog: microsoft/mai-thinking-1
      backend_refs:
        - provider: microsoft-foundry
          base_url: https://foundry.example.test
routing: {}
"""
    )
    config = parse_user_config(str(config_path))

    with pytest.raises(
        ValueError,
        match=r"provider_model_id.*operator-defined deployment name",
    ):
        project_provider_models_for_envoy(config)


def test_catalog_deployment_name_preserves_operator_provider_model_id(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: v0.3
providers:
  models:
    - name: mai
      catalog: microsoft/mai-thinking-1
      provider_model_id: operator-mai-production
      backend_refs:
        - provider: microsoft-foundry
          base_url: https://foundry.example.test
routing: {}
"""
    )
    config = parse_user_config(str(config_path))

    projected = project_provider_models_for_envoy(config)

    assert projected[0].api_format == "openai"
    assert projected[0].external_model_ids == {
        "microsoft-foundry": "operator-mai-production"
    }


def test_backend_ref_rejects_negative_weight_during_cli_parse(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: v0.3
providers:
  models:
    - name: local-model
      backend_refs:
        - provider: vllm
          endpoint: localhost:8000
          weight: -1
routing: {}
"""
    )

    with pytest.raises(ConfigParseError, match="greater than or equal to 0"):
        parse_user_config(str(config_path))


@pytest.mark.parametrize(
    ("backend_refs", "difference"),
    [
        (
            [
                {"provider": "vllm", "endpoint": "10.0.0.1:8000"},
                {"provider": "sglang", "endpoint": "10.0.0.2:8000"},
            ],
            "provider",
        ),
        (
            [
                {"provider": "vllm", "endpoint": "10.0.0.1:8000/v1"},
                {
                    "provider": "vllm",
                    "endpoint": "10.0.0.2:8000/compatible/v1",
                },
            ],
            "base path",
        ),
        (
            [
                {
                    "provider": "vllm",
                    "endpoint": "10.0.0.1:8000",
                    "api_key_env": "PRIMARY_API_KEY",
                },
                {
                    "provider": "vllm",
                    "endpoint": "10.0.0.2:8000",
                    "api_key_env": "SECONDARY_API_KEY",
                },
            ],
            "credential source",
        ),
        (
            [
                {"provider": "vllm", "endpoint": "https://api-1.example.test"},
                {"provider": "vllm", "endpoint": "https://api-2.example.test"},
            ],
            "TLS server name",
        ),
    ],
)
def test_generator_rejects_backend_groups_envoy_cannot_represent(
    tmp_path,
    backend_refs,
    difference,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "providers": {
                    "models": [
                        {
                            "name": "mixed-upstreams",
                            "backend_refs": backend_refs,
                        }
                    ]
                },
                "routing": {},
            }
        )
    )
    config = parse_user_config(str(config_path))

    with pytest.raises(
        ValueError,
        match=rf"cannot share one Envoy cluster.*{difference} differ",
    ):
        generate_envoy_config_from_user_config(
            config,
            str(tmp_path / "envoy.yaml"),
        )


def test_catalog_virtual_model_without_backends_keeps_router_fallback(
    tmp_path,
    monkeypatch,
):
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
providers:
  defaults:
    model: virtual-entrypoint
  models:
    - name: virtual-entrypoint
      catalog: vllm-sr/mom-v1-blend
routing: {}
""",
    )

    with pytest.raises(AssertionError):
        _cluster_by_name(rendered, "virtual_entrypoint_cluster")
    assert _default_route(rendered)["route"]["cluster"] == "vllm_static_cluster"


def test_reference_config_projects_a_homogeneous_weighted_pool(tmp_path, monkeypatch):
    output_path = tmp_path / "envoy.yaml"
    monkeypatch.setenv("ENVOY_EXTPROC_ADDRESS", "localhost")
    monkeypatch.setenv("ENVOY_ROUTER_API_ADDRESS", "localhost")
    config = parse_user_config(str(REPO_ROOT / "config/config.yaml"))

    generate_envoy_config_from_user_config(config, str(output_path))

    rendered = yaml.safe_load(output_path.read_text())
    cluster = _cluster_by_name(rendered, "qwen3_8b_cluster")
    endpoints = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"]
    assert [endpoint["load_balancing_weight"] for endpoint in endpoints] == [80, 20]
    assert {
        endpoint["endpoint"]["address"]["socket_address"]["address"]
        for endpoint in endpoints
    } == {"127.0.0.1"}
