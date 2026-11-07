"""Tests for the context plugin's compress block and budget in the CLI contract."""

import pytest
from cli.models import ContextPluginConfig
from pydantic import ValidationError as PydanticValidationError


def _compress(**compress):
    return ContextPluginConfig(enabled=True, compress=compress)


class TestContextCompressConfig:
    def test_canonical_nested_contract(self):
        config = ContextPluginConfig(
            enabled=True,
            compress={
                "mode": "auto",
                "targets": {
                    "tool_outputs": {
                        "mode": "extractive",
                        "min_tokens": 2000,
                        "target_tokens": 1000,
                    },
                    "rag": {"mode": "preserve"},
                },
                "match": {"recall": {"method": "bm25"}},
                "request_controls": {
                    "enabled": True,
                    "allowed": ["bypass", "target"],
                    "max_target_tokens": 16000,
                },
            },
            budget={"max_tokens": "auto", "reserve_output_tokens": 4096},
        )
        assert config.compress.targets.tool_outputs.target_tokens == 1000
        assert config.compress.match == {"recall": {"method": "bm25"}}
        assert config.budget.reserve_output_tokens == 4096

    def test_rejects_removed_fields(self):
        with pytest.raises(PydanticValidationError, match="Extra inputs"):
            _compress(scoring={"method": "bm25"})
        with pytest.raises(PydanticValidationError, match="Extra inputs"):
            ContextPluginConfig(
                enabled=True, compress={}, budget={"trigger_tokens": 4000}
            )

    def test_budget_needs_compress(self):
        with pytest.raises(PydanticValidationError, match="add compress"):
            ContextPluginConfig(
                enabled=True,
                sources=[{"type": "memory"}],
                budget={"max_tokens": 4000},
            )

    @pytest.mark.parametrize("max_tokens", [0, -1])
    def test_rejects_an_empty_budget(self, max_tokens):
        with pytest.raises(PydanticValidationError, match="at least 1"):
            ContextPluginConfig(
                enabled=True, compress={}, budget={"max_tokens": max_tokens}
            )

    @pytest.mark.parametrize(
        ("min_tokens", "target_tokens"),
        [(1000, 1000), (1000, 1200)],
    )
    def test_rejects_invalid_target(self, min_tokens, target_tokens):
        with pytest.raises(
            PydanticValidationError,
            match="target_tokens must be less than min_tokens",
        ):
            _compress(
                targets={
                    "tool_outputs": {
                        "mode": "extractive",
                        "min_tokens": min_tokens,
                        "target_tokens": target_tokens,
                    }
                }
            )

    def test_current_user_is_explicit_opt_in(self):
        default = _compress(targets={})
        assert default.compress.targets.current_user.mode == "preserve"
        configured = _compress(targets={"current_user": {"mode": "truncate"}})
        assert configured.compress.targets.current_user.mode == "truncate"
        restored = ContextPluginConfig.model_validate_json(configured.model_dump_json())
        assert restored.compress.targets.current_user.mode == "truncate"

    @pytest.mark.parametrize("mode", ["extractive", "recoverable", "drop"])
    def test_current_user_rejects_unsupported_modes(self, mode):
        with pytest.raises(PydanticValidationError):
            _compress(targets={"current_user": {"mode": mode}})
