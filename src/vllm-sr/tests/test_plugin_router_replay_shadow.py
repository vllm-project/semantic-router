"""Tests for router_replay's shadow block, a decision's retention, and the
cache, shadow and retention forms the CLI refuses like the Router."""

from pathlib import Path

import pytest
import yaml
from cli.config_reuse_forms import removed_reuse_forms
from cli.models import (
    DecisionRetention,
    PluginConfig,
    PluginType,
    RouterReplayPluginConfig,
    RouterReplayShadowConfig,
    ShadowDispatchBudgetConfig,
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

    def test_model_or_arms_required(self):
        with pytest.raises(PydanticValidationError, match="shadow model or arms"):
            RouterReplayShadowConfig()
        with pytest.raises(PydanticValidationError, match="shadow model or arms"):
            RouterReplayShadowConfig(model=" ")
        with pytest.raises(PydanticValidationError, match="empty model names"):
            RouterReplayShadowConfig(model="candidate", arms=["arm-a", " "])
        # Arms-only is a valid multi-arm shadow (issue #3376).
        cfg = RouterReplayShadowConfig(arms=["arm-a", "arm-b"])
        assert cfg.model is None
        assert cfg.arms == ["arm-a", "arm-b"]

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
            enabled=True,
            shadow={
                "model": "candidate",
                "arms": ["arm-a"],
                "budget": {"max_calls_per_request": 2},
                "sample_rate": 0.05,
            },
        )
        assert cfg.shadow is not None and cfg.shadow.sample_rate == 0.05
        assert cfg.shadow.arms == ["arm-a"]
        assert cfg.shadow.budget is not None
        assert cfg.shadow.budget.max_calls_per_request == 2

    def test_budget_field_parses(self):
        cfg = RouterReplayShadowConfig(
            model="candidate",
            arms=["arm-a"],
            budget=ShadowDispatchBudgetConfig(
                max_calls_per_request=2,
                max_tokens_per_request=512,
                reserve_tokens_per_arm=256,
                max_concurrency_per_request=2,
            ),
        )
        assert cfg.budget is not None
        assert cfg.budget.max_calls_per_request == 2
        assert cfg.budget.max_tokens_per_request == 512
        assert cfg.budget.reserve_tokens_per_arm == 256
        assert cfg.budget.max_concurrency_per_request == 2

    def test_budget_caps_reject_silent_combinations(self):
        # Mirrors the Router: caps that cannot bind before dispatch are refused
        # here instead of failing silently at run time.
        with pytest.raises(
            PydanticValidationError, match="requires reserve_tokens_per_arm"
        ):
            RouterReplayShadowConfig(
                model="m",
                budget=ShadowDispatchBudgetConfig(max_tokens_per_request=100),
            )
        with pytest.raises(
            PydanticValidationError,
            match="requires reserve_tokens_per_arm and price_per_million_tokens",
        ):
            RouterReplayShadowConfig(
                model="m",
                budget=ShadowDispatchBudgetConfig(max_cost_per_request=1.0),
            )
        with pytest.raises(
            PydanticValidationError, match="requires reserve_tokens_per_arm"
        ):
            RouterReplayShadowConfig(
                model="m",
                budget=ShadowDispatchBudgetConfig(
                    max_cost_per_request=1.0, price_per_million_tokens=2.0
                ),
            )

    def test_budget_rejects_negative_and_unknown_fields(self):
        with pytest.raises(PydanticValidationError):
            ShadowDispatchBudgetConfig(max_tokens_per_request=-1)
        with pytest.raises(PydanticValidationError, match="max_calls"):
            ShadowDispatchBudgetConfig(max_calls=1)

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
                        "configuration": {"shadow": {"model": "c"}},
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
