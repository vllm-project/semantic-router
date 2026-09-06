"""Tests for the canonical context compression CLI contract."""

import pytest
from cli.models import ContextCompressionPluginConfig
from pydantic import ValidationError as PydanticValidationError


class TestContextCompressionPluginConfig:
    def test_canonical_nested_contract(self):
        config = ContextCompressionPluginConfig(
            enabled=True,
            mode="auto",
            budget={
                "trigger_tokens": "auto",
                "target_tokens": "auto",
                "reserve_output_tokens": 4096,
            },
            targets={
                "tool_outputs": {
                    "mode": "extractive",
                    "min_tokens": 2000,
                    "target_tokens": 1000,
                },
                "rag": {"mode": "preserve"},
            },
            scoring={"method": "bm25"},
            request_controls={
                "enabled": True,
                "allowed": ["bypass", "target"],
                "max_target_tokens": 16000,
            },
        )
        assert config.targets is not None
        assert config.targets.tool_outputs.target_tokens == 1000
        assert config.budget is not None
        assert config.budget.reserve_output_tokens == 4096

    def test_rejects_flat_fields(self):
        with pytest.raises(PydanticValidationError, match="Extra inputs"):
            ContextCompressionPluginConfig(enabled=True, min_tokens=1000)

    @pytest.mark.parametrize(
        ("min_tokens", "target_tokens"),
        [(1000, 1000), (1000, 1200)],
    )
    def test_rejects_invalid_budget(self, min_tokens, target_tokens):
        with pytest.raises(
            PydanticValidationError,
            match="target_tokens must be less than min_tokens",
        ):
            ContextCompressionPluginConfig(
                enabled=True,
                targets={
                    "tool_outputs": {
                        "mode": "extractive",
                        "min_tokens": min_tokens,
                        "target_tokens": target_tokens,
                    }
                },
            )
