"""Tests for router_replay's shadow block, a decision's retention, and the
cache, shadow and retention forms the CLI refuses like the Router."""

import pytest
from cli.config_reuse_forms import removed_reuse_forms
from cli.models import (
    DecisionRetention,
    PluginConfig,
    PluginType,
    RouterReplayPluginConfig,
    RouterReplayShadowConfig,
)
from pydantic import ValidationError as PydanticValidationError


class TestRouterReplayShadowConfig:
    def test_defaults_are_bounded(self):
        cfg = RouterReplayShadowConfig(model="candidate")
        assert cfg.tls_skip_verify is False
        assert cfg.sample_rate is None
        assert cfg.max_concurrency == 2
        assert cfg.max_queue_depth == 8
        assert cfg.timeout_seconds == 30
        assert cfg.max_response_bytes == 1048576
        assert cfg.max_retries == 0
        assert cfg.capture_response_body is False
        assert cfg.max_capture_bytes == 4096

    def test_model_is_required(self):
        with pytest.raises(PydanticValidationError, match="model"):
            RouterReplayShadowConfig()
        with pytest.raises(PydanticValidationError, match="shadow model is required"):
            RouterReplayShadowConfig(model=" ")

    def test_bounds_and_headers(self):
        with pytest.raises(PydanticValidationError):
            RouterReplayShadowConfig(model="m", sample_rate=1.5)
        with pytest.raises(PydanticValidationError):
            RouterReplayShadowConfig(model="m", max_retries=4)
        with pytest.raises(PydanticValidationError, match="credential header"):
            RouterReplayShadowConfig(model="m", forward_headers=["Authorization"])
        with pytest.raises(PydanticValidationError, match="pseudo-header"):
            RouterReplayShadowConfig(model="m", forward_headers=[":authority"])
        with pytest.raises(PydanticValidationError, match="bogus"):
            RouterReplayShadowConfig(model="m", bogus=1)

    def test_router_replay_carries_the_shadow(self):
        cfg = RouterReplayPluginConfig(
            enabled=True, shadow={"model": "candidate", "sample_rate": 0.05}
        )
        assert cfg.shadow is not None and cfg.shadow.sample_rate == 0.05

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


def _decisions(*plugins, **decision):
    return {
        "routing": {"decisions": [{"name": "d", "plugins": list(plugins), **decision}]}
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
            "routing.decisions[d].plugins[0]: removed_plugin_type: plugin "
            "semantic-cache is removed; run `vllm-sr config migrate` "
            "(type: response_cache)",
            "routing.decisions[d].plugins[1]: removed_field: response_cache fields "
            "similarity_threshold, control_header are removed; run `vllm-sr config "
            "migrate` (the similarity threshold is semantic.match.recall.min_similarity; "
            "allow_request_controls and control_header are request_controls.enabled "
            "and request_controls.header)",
            "routing.decisions[d].plugins[2]: removed_plugin_type: plugin "
            "shadow_dispatch is removed; run `vllm-sr config "
            "migrate` (a shadow block in the decision's router_replay plugin)",
            "routing.decisions[d].emits: removed_field: emits is removed; run "
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
                        "configuration": {"shadow": {"model": "c"}},
                    },
                    retention={"drop": True},
                )
            )
            == []
        )
