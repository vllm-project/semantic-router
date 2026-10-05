"""Deterministic config proposal artifact for issue #3217."""

import json
from pathlib import Path

import pytest
import yaml
from cli.config_proposal import ProposalUnsupportedError, propose, supported_intents
from cli.main import main
from click.testing import CliRunner

SECRET = "sk-live-secret-value"
LISTENER_PORT = 8899
ACCESS_KEY = "AKIASECRETVALUE"
CLIENT_SECRET = "CLIENTSECRETVALUE"
URL_SECRET = "URLSECRET"
BEARER_IN_DESCRIPTION = "Bearer superSecretTokenValue12"
REPO_ROOT = Path(__file__).resolve().parents[3]

BASE_CONFIG = f"""
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: {LISTENER_PORT}
    timeout: 300s
providers:
  defaults:
    model: local-model
  models:
    - name: local-model
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: 127.0.0.1:8000
          protocol: http
          weight: 100
          api_key: {SECRET}
          api_key_env: LOCAL_MODEL_API_KEY
    - name: cheap-model
      backend_refs:
        - name: cheap
          provider: vllm
          endpoint: 127.0.0.1:8001
          protocol: http
          weight: 100
routing:
  modelCards:
    - name: local-model
    - name: cheap-model
  decisions:
    - name: default-route
      description: Catch-all route
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: local-model
          use_reasoning: false
        - model: cheap-model
          use_reasoning: false
    - name: keep-me
      description: Unrelated decision that must survive {BEARER_IN_DESCRIPTION} http://user:URLSECRET@example.com/v1 /home/pranav/private/router.yaml
      priority: 10
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: cheap-model
          use_reasoning: false
"""


def _write_config(tmp_path: Path, text: str = BASE_CONFIG) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_supported_intent_preserves_unrelated_config_and_redacts_secrets(tmp_path):
    config_path = _write_config(tmp_path)

    document = propose(
        config_path,
        "selection.latency-aware",
        "default-route",
        repo_root=REPO_ROOT,
    )

    assert document["schema"] == "vllm-sr/config-proposal/v1"
    assert document["applied"] is False
    assert document["validation"]["status"] == "valid"
    assert [check["name"] for check in document["validation"]["checks"]] == [
        "schema",
        "canonical_parse",
        "policy",
    ]
    assert all(check["ok"] for check in document["validation"]["checks"])
    assert document["provenance"]["input"]["name"] == "config.yaml"
    assert document["provenance"]["input"]["version"] == "v0.3"
    assert document["provenance"]["sources"] == [
        {
            "kind": "fragment",
            "path": "config/fragments/algorithm/selection/latency-aware.yaml",
            "sha256": document["provenance"]["sources"][0]["sha256"],
        }
    ]
    assert str(tmp_path) not in json.dumps(document)
    assert SECRET not in json.dumps(document)

    candidate = yaml.safe_load(document["candidate_yaml"])
    default_route = candidate["routing"]["decisions"][0]
    keep_me = candidate["routing"]["decisions"][1]
    assert default_route["algorithm"]["type"] == "latency_aware"
    assert default_route["algorithm"]["latency_aware"] == {
        "tpot_percentile": 90,
        "ttft_percentile": 95,
    }
    assert default_route["description"] == "Catch-all route"
    assert keep_me["name"] == "keep-me"
    assert keep_me["description"].startswith("Unrelated decision that must survive")
    assert URL_SECRET not in keep_me["description"]
    assert BEARER_IN_DESCRIPTION not in json.dumps(document)
    assert BEARER_IN_DESCRIPTION not in document["diff"]
    assert "superSecretTokenValue12" not in keep_me["description"]
    assert "***" in keep_me["description"]
    assert "/home/" not in json.dumps(document)
    assert "http://***@example.com/v1" in keep_me["description"]
    assert "algorithm" not in keep_me
    assert candidate["listeners"][0]["port"] == LISTENER_PORT
    assert candidate["providers"]["models"][0]["backend_refs"][0]["api_key"] == "***"
    assert (
        candidate["providers"]["models"][0]["backend_refs"][0]["api_key_env"]
        == "LOCAL_MODEL_API_KEY"
    )
    assert "latency_aware" in document["diff"]
    removed = [
        line[1:]
        for line in document["diff"].splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    assert not any("keep-me" in line for line in removed)
    assert config_path.read_text(encoding="utf-8") == BASE_CONFIG


def test_same_inputs_produce_the_same_proposal(tmp_path):
    config_path = _write_config(tmp_path)

    first = propose(
        config_path,
        "selection.latency-aware",
        "default-route",
        repo_root=REPO_ROOT,
    )
    second = propose(
        config_path,
        "selection.latency-aware",
        "default-route",
        repo_root=REPO_ROOT,
    )

    assert first == second


def test_unknown_intent_fails_explicitly(tmp_path):
    config_path = _write_config(tmp_path)

    with pytest.raises(ProposalUnsupportedError, match="unsupported: intent"):
        propose(
            config_path,
            "deploy-private-gateway",
            "default-route",
            repo_root=REPO_ROOT,
        )

    assert config_path.read_text(encoding="utf-8") == BASE_CONFIG


def test_missing_decision_is_unsupported(tmp_path):
    config_path = _write_config(tmp_path)

    with pytest.raises(ProposalUnsupportedError, match="do not invent decisions"):
        propose(
            config_path,
            "selection.latency-aware",
            "does-not-exist",
            repo_root=REPO_ROOT,
        )


def test_non_canonical_version_is_unsupported(tmp_path):
    config_path = _write_config(tmp_path, "version: v0.2\nrouting: {}\n")

    with pytest.raises(ProposalUnsupportedError, match=r"not canonical v0\.3"):
        propose(
            config_path,
            "selection.latency-aware",
            "default-route",
            repo_root=REPO_ROOT,
        )


def test_cli_prints_proposal_without_writing_config(tmp_path):
    config_path = _write_config(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "config",
            "propose",
            "--config",
            str(config_path),
            "--intent",
            "selection.latency-aware",
            "--decision",
            "default-route",
        ],
    )

    assert result.exit_code == 0, result.output
    document = json.loads(result.output)
    assert document["applied"] is False
    assert document["validation"]["status"] == "valid"
    assert SECRET not in result.output
    assert config_path.read_text(encoding="utf-8") == BASE_CONFIG
    assert "selection.latency-aware" in supported_intents()


def test_cli_rejects_unsupported_intent(tmp_path):
    config_path = _write_config(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "config",
            "propose",
            "--config",
            str(config_path),
            "--intent",
            "install-a-new-model-provider",
            "--decision",
            "default-route",
        ],
    )

    assert result.exit_code != 0
    assert "unsupported: intent" in result.output
    assert "candidate_yaml" not in result.output


def _config_with_extra_credentials(text: str = BASE_CONFIG) -> str:
    return text.replace(
        f"api_key: {SECRET}",
        (
            f"api_key: {SECRET}\n"
            f"          access_key: {ACCESS_KEY}\n"
            f"          client_secret: {CLIENT_SECRET}"
        ),
    )


def test_free_form_field_redacts_embedded_bearer_literal(tmp_path):
    config_path = _write_config(
        tmp_path,
        BASE_CONFIG.replace(
            "description: Catch-all route",
            f"description: Notes may include {BEARER_IN_DESCRIPTION} inline",
        ),
    )

    document = propose(
        config_path,
        "selection.latency-aware",
        "default-route",
        repo_root=REPO_ROOT,
    )

    rendered = json.dumps(document)
    assert BEARER_IN_DESCRIPTION not in rendered
    assert "superSecretTokenValue12" not in rendered
    description = yaml.safe_load(document["candidate_yaml"])["routing"]["decisions"][0][
        "description"
    ]
    assert "superSecretTokenValue12" not in description
    assert "***" in description


def test_access_key_and_client_secret_are_redacted(tmp_path):
    config_path = _write_config(tmp_path, _config_with_extra_credentials())

    document = propose(
        config_path,
        "selection.latency-aware",
        "default-route",
        repo_root=REPO_ROOT,
    )

    rendered = json.dumps(document)
    assert ACCESS_KEY not in rendered
    assert CLIENT_SECRET not in rendered
    assert SECRET not in rendered
    assert str(tmp_path) not in rendered
    backend = yaml.safe_load(document["candidate_yaml"])["providers"]["models"][0][
        "backend_refs"
    ][0]
    assert backend["access_key"] == "***"
    assert backend["client_secret"] == "***"
    assert backend["api_key"] == "***"
    for check in document["validation"]["checks"]:
        assert ACCESS_KEY not in " ".join(check["diagnostics"])
        assert CLIENT_SECRET not in " ".join(check["diagnostics"])


RECIPE_CONFIG = f"""
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: {LISTENER_PORT}
    timeout: 300s
providers:
  defaults:
    model: local-model
  models:
    - name: local-model
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: http://user:{URL_SECRET}@127.0.0.1:8000/v1
          protocol: http
          weight: 100
          api_key: {SECRET}
          api_key_env: LOCAL_MODEL_API_KEY
    - name: cheap-model
      backend_refs:
        - name: cheap
          provider: vllm
          endpoint: 127.0.0.1:8001
          protocol: http
          weight: 100
routing:
  modelCards:
    - name: local-model
    - name: cheap-model
  decisions:
    - name: keep-default
      description: Top-level decision that must survive
      priority: 50
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: local-model
          use_reasoning: false
recipes:
  - name: privacy-lane
    description: Target recipe
    routing:
      decisions:
        - name: old-privacy
          priority: 20
          rules:
            operator: AND
            conditions: []
          modelRefs:
            - model: local-model
              use_reasoning: false
  - name: keep-recipe
    description: Leave this recipe alone
    routing:
      decisions:
        - name: stay
          priority: 10
          rules:
            operator: AND
            conditions: []
          modelRefs:
            - model: cheap-model
              use_reasoning: false
"""


def test_recipe_privacy_replaces_only_one_routing_block(tmp_path):
    config_path = _write_config(tmp_path, RECIPE_CONFIG)

    document = propose(
        config_path,
        "recipe.privacy",
        recipe="privacy-lane",
        repo_root=REPO_ROOT,
    )

    rendered = json.dumps(document)
    assert document["applied"] is False
    assert document["intent"]["recipe"] == "privacy-lane"
    assert document["provenance"]["diff_source"] == "textual-unified-until-3477"
    assert document["provenance"]["sources"][0]["kind"] == "recipe"
    assert (
        document["provenance"]["sources"][0]["path"]
        == "config/recipes/privacy/config.yaml"
    )
    assert SECRET not in rendered
    assert URL_SECRET not in rendered
    assert str(tmp_path) not in rendered
    assert [check["name"] for check in document["validation"]["checks"]] == [
        "schema",
        "canonical_parse",
        "policy",
    ]

    candidate = yaml.safe_load(document["candidate_yaml"])
    privacy = next(
        item for item in candidate["recipes"] if item["name"] == "privacy-lane"
    )
    kept = next(item for item in candidate["recipes"] if item["name"] == "keep-recipe")
    assert privacy["description"] == "Target recipe"
    assert any(
        card["name"] == "local/private-qwen"
        for card in privacy["routing"]["modelCards"]
    )
    assert any(
        decision["name"] == "local_privacy_policy"
        for decision in privacy["routing"]["decisions"]
    )
    assert "modelCards" in document["diff"]
    assert kept["description"] == "Leave this recipe alone"
    assert kept["routing"]["decisions"][0]["name"] == "stay"
    assert candidate["routing"]["decisions"][0]["name"] == "keep-default"
    assert candidate["listeners"][0]["port"] == LISTENER_PORT
    assert candidate["providers"]["models"][1]["name"] == "cheap-model"
    assert config_path.read_text(encoding="utf-8") == RECIPE_CONFIG


def test_missing_recipe_is_unsupported(tmp_path):
    config_path = _write_config(tmp_path, RECIPE_CONFIG)

    with pytest.raises(ProposalUnsupportedError, match="do not invent recipes"):
        propose(
            config_path,
            "recipe.privacy",
            recipe="does-not-exist",
            repo_root=REPO_ROOT,
        )

    assert config_path.read_text(encoding="utf-8") == RECIPE_CONFIG


def test_recipe_intent_does_not_accept_a_decision_target(tmp_path):
    config_path = _write_config(tmp_path, RECIPE_CONFIG)

    with pytest.raises(ProposalUnsupportedError, match="unsupported: intent"):
        propose(
            config_path,
            "recipe.privacy",
            "privacy-lane",
            repo_root=REPO_ROOT,
        )


def test_cli_prints_invalid_recipe_proposal_without_writing(tmp_path):
    config_path = _write_config(tmp_path, RECIPE_CONFIG)

    result = CliRunner().invoke(
        main,
        [
            "config",
            "propose",
            "--config",
            str(config_path),
            "--intent",
            "recipe.privacy",
            "--recipe",
            "privacy-lane",
        ],
    )

    document = json.loads(result.output)
    assert document["applied"] is False
    assert "local_privacy_policy" in document["candidate_yaml"]
    assert SECRET not in result.output
    assert URL_SECRET not in result.output
    assert config_path.read_text(encoding="utf-8") == RECIPE_CONFIG
    if document["validation"]["status"] != "valid":
        assert result.exit_code != 0
    else:
        assert result.exit_code == 0
