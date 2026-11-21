"""Tests for router_replay's shadow block, a decision's retention, and the
cache, shadow and retention forms the CLI refuses like the Router."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from cli.config_reuse_forms import removed_reuse_forms
from cli.main import main
from cli.models import (
    DecisionRetention,
    FallbackPolicy,
    PluginConfig,
    PluginType,
    RouterReplayPluginConfig,
    RouterReplayShadowConfig,
    UserConfig,
)
from cli.parser import ConfigParseError, parse_user_config
from cli.validation_error import ValidationError
from cli.validator import validate_user_config
from click.testing import CliRunner
from pydantic import ValidationError as PydanticValidationError


class TestRouterReplayShadowConfig:
    def test_defaults_and_round_trip(self):
        cfg = RouterReplayPluginConfig(
            enabled=False, shadow={"models": [" candidate ", "other"], "sample_rate": 0}
        )
        shadow = cfg.shadow
        assert shadow.models == ["candidate", "other"]
        assert shadow.sample_rate == 0
        assert shadow.execution.max_concurrency is None
        assert shadow.execution.max_queue_size is None
        assert shadow.execution.timeout == "30s"
        assert shadow.budget.max_response_bytes is None
        assert shadow.capture.enabled is False
        assert shadow.capture.max_bytes is None
        assert cfg.model_dump(exclude_unset=True) == {
            "enabled": False,
            "shadow": {"models": ["candidate", "other"], "sample_rate": 0},
        }
        assert len(RouterReplayShadowConfig(models=list("abcdefgh")).models) == 8
        assert RouterReplayShadowConfig(models=["m"]).sample_rate is None

    @pytest.mark.parametrize(
        "models", [[], [" "], ["a", " a "], list("abcdefghi"), [1], "a", None]
    )
    @pytest.mark.parametrize("enabled", [False, True])
    def test_models_required_even_when_replay_disabled(self, models, enabled):
        with pytest.raises(PydanticValidationError):
            RouterReplayPluginConfig(enabled=enabled, shadow={"models": models})
        with pytest.raises(PydanticValidationError, match="models"):
            RouterReplayPluginConfig(enabled=enabled, shadow={})

    @pytest.mark.parametrize(
        "value", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"]
    )
    def test_sample_rate_is_strict(self, value):
        with pytest.raises(PydanticValidationError):
            RouterReplayShadowConfig(models=["m"], sample_rate=value)

    @pytest.mark.parametrize("value", [None, 0, 0.5, 1])
    def test_sample_rate_boundaries(self, value):
        assert (
            RouterReplayShadowConfig(models=["m"], sample_rate=value).sample_rate
            == value
        )

    @pytest.mark.parametrize("shadow", [[], "candidate", False])
    def test_declared_shadow_requires_object(self, shadow):
        assert RouterReplayPluginConfig(enabled=False).shadow is None
        with pytest.raises(PydanticValidationError):
            RouterReplayPluginConfig(enabled=False, shadow=shadow)

    def test_null_shadow_clears_inherited_config(self):
        assert RouterReplayPluginConfig(shadow=None).model_dump(exclude_unset=True) == {
            "shadow": None,
        }

    @pytest.mark.parametrize(
        "section,field",
        [
            ("execution", "max_concurrency"),
            ("execution", "max_queue_size"),
            ("budget", "max_response_bytes"),
            ("capture", "max_bytes"),
        ],
    )
    def test_positive_integer_limits(self, section, field):
        for value in [0, -1, True, 2.5, "2"]:
            with pytest.raises(PydanticValidationError):
                RouterReplayShadowConfig(models=["m"], **{section: {field: value}})
        for value in [None, 2.0, 17]:
            cfg = RouterReplayShadowConfig(models=["m"], **{section: {field: value}})
            assert getattr(getattr(cfg, section), field) == value

    def test_budget_reserves_probe_byte_per_model(self):
        for budget in [1, 2, 3]:
            with pytest.raises(
                PydanticValidationError,
                match=r"budget\.max_response_bytes must allocate at least two bytes to each model",
            ):
                RouterReplayShadowConfig(
                    models=["a", "b"], budget={"max_response_bytes": budget}
                )
        RouterReplayShadowConfig(models=["a", "b"], budget={"max_response_bytes": 4})

    @pytest.mark.parametrize(
        "timeout",
        [
            "1ms",
            "0.5s",
            "1m30s",
            "2m500ms",
            "5m",
            "300s",
            "300.0000000009s",
            "+1s",
            ".5s",
            "1.s",
            "0.000000001s",
        ],
    )
    def test_go_durations(self, timeout):
        assert (
            RouterReplayShadowConfig(
                models=["m"], execution={"timeout": timeout}
            ).execution.timeout
            == timeout
        )

    @pytest.mark.parametrize(
        "timeout",
        [
            "0s",
            "-1s",
            "5m1ms",
            "301s",
            "1h",
            "30",
            "",
            "nan",
            "1s junk",
            " 1s",
            30,
            None,
            "0.0000000001s",
            "1" * 129 + "s",
        ],
    )
    def test_bad_durations(self, timeout):
        with pytest.raises(PydanticValidationError):
            RouterReplayShadowConfig(models=["m"], execution={"timeout": timeout})

    @pytest.mark.parametrize("section", ["execution", "budget", "capture"])
    def test_nested_objects_are_strict(self, section):
        for value in [None, [], "bad", 1, False, {"bogus": None}]:
            with pytest.raises(PydanticValidationError):
                RouterReplayShadowConfig(models=["m"], **{section: value})

    @pytest.mark.parametrize("value", [None, "true", 1])
    def test_capture_enabled_is_strict(self, value):
        with pytest.raises(PydanticValidationError):
            RouterReplayShadowConfig(models=["m"], capture={"enabled": value})

    @pytest.mark.parametrize(
        "field",
        [
            "enabled",
            "model",
            "arms",
            "max_concurrency",
            "max_queue_depth",
            "timeout_seconds",
            "max_response_bytes",
            "max_retries",
            "capture_response_body",
            "max_capture_bytes",
            "tls_skip_verify",
        ],
    )
    def test_no_flat_or_legacy_fields(self, field):
        with pytest.raises(PydanticValidationError):
            RouterReplayShadowConfig(models=["m"], **{field: None})

    def test_forward_headers_accepts_allowlist_and_rejects_credentials(self):
        config = RouterReplayShadowConfig(models=["m"], forward_headers=["X-Tenant"])
        assert config.forward_headers == ["X-Tenant"]
        with pytest.raises(PydanticValidationError, match="forward_headers"):
            RouterReplayShadowConfig(models=["m"], forward_headers=["Authorization"])

    def test_shadow_dispatch_is_no_plugin_type(self):
        assert "shadow_dispatch" not in {member.value for member in PluginType}
        with pytest.raises(PydanticValidationError):
            PluginConfig(type="shadow_dispatch", configuration={"enabled": True})


class TestDecisionRetention:
    def test_fields(self):
        retention = DecisionRetention(ttl_turns=3, keep_current_model=True)
        assert retention.ttl_turns == 3 and retention.drop is None

    def test_rules(self):
        with pytest.raises(PydanticValidationError, match="conflicts with ttl_turns"):
            DecisionRetention(drop=True, ttl_turns=2)
        with pytest.raises(PydanticValidationError):
            DecisionRetention(ttl_turns=-1)
        with pytest.raises(PydanticValidationError):
            DecisionRetention(mystery=True)


@pytest.mark.parametrize(
    "header",
    [
        "Authorization",
        "Proxy-Authorization",
        "Cookie",
        "X-Api-Key",
        "Api-Key",
        "X-Goog-Api-Key",
        "X-User-OpenAI-Key",
        "X-User-Anthropic-Key",
        "X-User-Azure-OpenAI-Key",
        "X-User-Bedrock-Key",
        "X-User-Gemini-Key",
        "X-User-Vertex-AI-Key",
        "X-User-Minimax-Key",
        "Ocp-Apim-Subscription-Key",
        "Api-Subscription-Key",
        "X-User-Cloudflare-Workers-AI-Key",
    ],
)
def test_forward_request_headers_still_reject_credentials(header):
    with pytest.raises(PydanticValidationError, match="credential header"):
        FallbackPolicy(forward_request_headers=[header])


def _decisions(*plugins, **decision):
    return {
        "recipes": [
            {
                "name": "default",
                "decisions": [{"name": "d", "plugins": list(plugins), **decision}],
            }
        ]
    }


class TestRemovedReuseForms:
    def test_lines_match_the_router(self):
        lines = removed_reuse_forms(
            {
                "global": {"stores": {"semantic_cache": {"enabled": True}}},
                **_decisions(
                    {"type": "semantic-cache", "configuration": {"enabled": True}},
                    {
                        "type": "response_cache",
                        "configuration": {
                            "enabled": True,
                            "similarity_threshold": 0.9,
                            "control_header": "x",
                        },
                    },
                    {"type": "shadow_dispatch", "configuration": {"enabled": True}},
                    emits=[{"kind": "retention", "retention": {"drop": True}}],
                ),
            }
        )
        assert lines == [
            "global.stores.semantic_cache: removed_field: global.stores.semantic_cache "
            "is removed; run `vllm-sr config migrate` (global.stores.response_cache)",
            "recipes[default].decisions[d].plugins[0]: removed_plugin_type: plugin "
            "semantic-cache is removed; run `vllm-sr config migrate` "
            "(type: response_cache)",
            "recipes[default].decisions[d].plugins[1]: removed_field: response_cache fields "
            "similarity_threshold, control_header are removed; run `vllm-sr config "
            "migrate` (the similarity threshold is semantic.match.recall.min_similarity; "
            "allow_request_controls and control_header are request_controls.enabled "
            "and request_controls.header)",
            "recipes[default].decisions[d].plugins[2]: removed_plugin_type: plugin "
            "shadow_dispatch is removed; run `vllm-sr config "
            "migrate` (a shadow block in the decision's router_replay plugin)",
            "recipes[default].decisions[d].emits: removed_field: emits is removed; run "
            "`vllm-sr config migrate` (the decision's retention attribute)",
        ]

    def test_canonical_forms_pass(self):
        assert (
            removed_reuse_forms(
                _decisions(
                    {
                        "type": "response_cache",
                        "configuration": {
                            "enabled": True,
                            "semantic": {"match": {"recall": {"top_k": 3}}},
                        },
                    },
                    {
                        "type": "router_replay",
                        "configuration": {"shadow": {"models": ["c"]}},
                    },
                    retention={"drop": True},
                )
            )
            == []
        )

    def test_a_recipes_data_policy_is_named_with_the_routers_line(self):
        golden = (
            Path(__file__).resolve().parents[3]
            / "src/semantic-router/pkg/config/testdata/migrate/removed-field-data-policy"
        )
        before = yaml.safe_load((golden / "before.yaml").read_text(encoding="utf-8"))
        hint = (golden / "hint.txt").read_text(encoding="utf-8").strip()

        assert hint in removed_reuse_forms(before)
        after = yaml.safe_load((golden / "after.yaml").read_text(encoding="utf-8"))
        assert removed_reuse_forms(after) == []


@pytest.fixture
def shadow_document():
    plugin = {
        "type": "router_replay",
        "configuration": {
            "enabled": False,
            "shadow": {"models": ["candidate", "other"], "sample_rate": 0},
        },
    }
    return {
        "version": "v0.5",
        "listeners": [],
        "providers": {
            "defaults": {"model": "primary"},
            "models": [
                {
                    "name": name,
                    "provider_model_id": "upstream-" + name,
                    "backend_refs": [
                        {"endpoint": "127.0.0.1:8000", "provider": "vllm"}
                    ],
                }
                for name in ("primary", "candidate", "other")
            ],
        },
        "recipes": [
            {
                "name": "private",
                "decisions": [
                    {
                        "name": "shadow-route",
                        "priority": 1,
                        "rules": {"operator": "AND", "conditions": []},
                        "modelRefs": [{"model": "primary"}],
                        "plugins": [plugin],
                    }
                ],
            }
        ],
    }, plugin


@pytest.mark.parametrize("scope", ["decision", "recipe", "global"])
@pytest.mark.parametrize(
    "target", [" candidate ", "candidate-lora", "unknown", "upstream-candidate"]
)
def test_shadow_backend_references_in_all_scopes(shadow_document, scope, target):
    document, plugin = shadow_document
    candidate = document["providers"]["models"][1]
    candidate["loras"] = [{"name": "candidate-lora"}]
    plugin["configuration"]["shadow"]["models"][0] = target
    recipe = document["recipes"][0]
    if scope != "decision":
        recipe["decisions"] = []
        if scope == "recipe":
            recipe["plugins"] = [plugin]
        else:
            document["global"] = {"plugins": [plugin]}
    errors = validate_user_config(
        UserConfig.model_validate(document), log_summary=False
    )
    if target in ("unknown", "upstream-candidate"):
        assert any(
            f'router_replay shadow model "{target}" has no configured backend'
            in error.message
            for error in errors
        )
        assert any(
            ("global" if scope == "global" else "recipes.private") in error.field
            for error in errors
        )
    else:
        assert errors == []


@pytest.mark.parametrize("delegated", [False, True])
@pytest.mark.parametrize("target", ["candidate", "candidate-lora"])
def test_shadow_requires_static_backend_including_lora(
    shadow_document, delegated, target
):
    document, plugin = shadow_document
    candidate = document["providers"]["models"][1]
    candidate["loras"] = [{"name": "candidate-lora"}]
    plugin["configuration"]["shadow"]["models"][0] = target
    if delegated:
        candidate["backend_refs"][0]["auth"] = {"credential": "delegated"}
    else:
        candidate["backend_refs"] = []
    errors = validate_user_config(
        UserConfig.model_validate(document), log_summary=False
    )
    assert any(
        ("auth.credential: delegated" if delegated else "has no configured backend")
        in error.message
        for error in errors
    )


@pytest.mark.parametrize("scope", ["decision", "recipe", "global"])
def test_declared_shadow_is_validated_in_all_scopes(shadow_document, scope):
    document, plugin = shadow_document
    plugin["configuration"]["shadow"]["budget"] = {"max_response_bytes": 3}
    recipe = document["recipes"][0]
    if scope != "decision":
        recipe["decisions"] = []
        if scope == "recipe":
            recipe["plugins"] = [plugin]
        else:
            document["global"] = {"plugins": [plugin]}
    errors = validate_user_config(
        UserConfig.model_validate(document), log_summary=False
    )
    assert any(
        "budget.max_response_bytes must allocate at least two bytes to each model"
        in error.message
        for error in errors
    )


def test_parser_uses_nested_shadow_schema(tmp_path, shadow_document):
    document, plugin = shadow_document
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    parsed = parse_user_config(str(path), log_summary=False)
    assert validate_user_config(parsed, log_summary=False) == []
    plugin["configuration"]["shadow"]["budget"] = {"max_tokens": 1}
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigParseError, match="max_tokens"):
        parse_user_config(str(path), log_summary=False)


def test_each_shadow_candidate_needs_a_backend(shadow_document):
    document, _ = shadow_document
    document["providers"]["models"][2]["backend_refs"] = []
    errors = validate_user_config(
        UserConfig.model_validate(document), log_summary=False
    )
    assert any(
        'router_replay shadow model "other" has no configured backend' in error.message
        for error in errors
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"execution": None},
        {"budget": []},
        {"capture": {"enabled": None}},
        {"execution": {"max_concurrency": True}},
        {"budget": {"max_response_bytes": 4.5}},
        {"models": ["candidate", "other"], "enabled": False},
    ],
)
def test_parser_rejects_malformed_shadow(tmp_path, shadow_document, overrides):
    document, plugin = shadow_document
    plugin["configuration"]["shadow"].update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigParseError):
        parse_user_config(str(path), log_summary=False)


@pytest.mark.parametrize(
    "decision_override",
    [
        {"shadow": {"capture": {"enabled": True}}},
        {"enabled": False, "shadow": {"execution": {"max_queue_size": 3}}},
        {"shadow": None},
    ],
)
def test_partial_shadow_overrides_inherit_models(
    tmp_path, shadow_document, decision_override
):
    document, global_plugin = shadow_document
    document["global"] = {"plugins": [deepcopy(global_plugin)]}
    recipe = document["recipes"][0]
    recipe["plugins"] = [
        {
            "type": "router_replay",
            "configuration": {
                "shadow": {"sample_rate": 0.5, "execution": {"timeout": "1s"}},
            },
        }
    ]
    recipe["decisions"][0]["plugins"] = [
        {
            "type": "router_replay",
            "configuration": decision_override,
        }
    ]
    config = UserConfig.model_validate(document)
    before = config.model_dump()
    assert validate_user_config(config, log_summary=False) == []
    assert config.model_dump() == before
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    parsed = parse_user_config(str(path), log_summary=False)
    assert validate_user_config(parsed, log_summary=False) == []


@pytest.mark.parametrize(
    "override",
    [
        {"budget": {"max_response_bytes": 3}},
        {"execution": None},
        {"capture": {"enabled": None}},
    ],
)
def test_disabled_partial_shadow_overrides_remain_strict(shadow_document, override):
    document, global_plugin = shadow_document
    document["global"] = {"plugins": [deepcopy(global_plugin)]}
    document["recipes"][0]["decisions"][0]["plugins"] = [
        {
            "type": "router_replay",
            "configuration": {"enabled": False, "shadow": override},
        }
    ]
    config = UserConfig.model_validate(document)
    assert validate_user_config(config, log_summary=False)


@pytest.mark.parametrize("scope", ["global", "recipe", "decision"])
@pytest.mark.parametrize(
    "first_configuration",
    [
        {"max_records": -1},
        {"shadow": {"models": ["missing"]}},
        {},
    ],
    ids=["invalid-limit", "missing-model", "empty"],
)
def test_duplicate_replay_in_one_layer_is_rejected(
    shadow_document, scope, first_configuration
):
    document, _ = shadow_document
    recipe = document["recipes"][0]
    decision = recipe["decisions"][0]
    decision["plugins"] = []
    layer, owner = {
        "global": (document.setdefault("global", {}), "global"),
        "recipe": (recipe, "recipes.private"),
        "decision": (decision, "recipes.private.decisions.shadow-route"),
    }[scope]
    layer["plugins"] = [
        {"type": "router_replay", "configuration": first_configuration},
        {
            "type": "router_replay",
            "configuration": {"max_records": 1, "shadow": None},
        },
    ]
    config = UserConfig.model_validate(document)
    before = config.model_dump()
    errors = validate_user_config(config, log_summary=False)
    assert any(
        isinstance(error, ValidationError)
        and "duplicate" in error.message.lower()
        and "router_replay" in error.message
        and error.field.startswith(f"{owner}.plugins")
        for error in errors
    )
    assert config.model_dump() == before


def test_offline_cli_rejects_duplicate_replay(tmp_path, shadow_document):
    document, _ = shadow_document
    decision = document["recipes"][0]["decisions"][0]
    decision["plugins"] = [
        {
            "type": "router_replay",
            "configuration": {"max_records": -1, "shadow": {"models": ["missing"]}},
        },
        {
            "type": "router_replay",
            "configuration": {"max_records": 1, "shadow": None},
        },
    ]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    result = CliRunner().invoke(
        main, ["config", "validate", "--offline", "--config", str(path)]
    )
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)
    assert "Configuration validation failed" in result.output
    assert "duplicate" in result.output.lower()
    assert "router_replay" in result.output
