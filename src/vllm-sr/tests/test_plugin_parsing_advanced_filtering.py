"""Advanced tool-selection plugin schema tests."""

import pytest
from cli.models import ToolSelectionPluginConfig
from pydantic import ValidationError as PydanticValidationError


class TestToolSelectionAdvancedFiltering:
    """Test tool_selection plugin's advanced_filtering subtree.

    Pins down the schema parity with the Go-side
    `config.AdvancedToolFilteringConfig`. Before this surface was modeled,
    `ToolSelectionPluginConfig` had no `advanced_filtering` field and Pydantic
    silently dropped the subtree, so users opting into
    `retrieval_strategy: hybrid_history` ended up with a config equivalent to
    no advanced filtering at all.
    """

    def test_hybrid_history_round_trips_without_field_loss(self):
        """advanced_filtering.hybrid_history survives model_validate -> model_dump."""
        cfg = ToolSelectionPluginConfig.model_validate(
            {
                "enabled": True,
                "mode": "add",
                "tools_db_path": "config/tools_db.json",
                "top_k": 5,
                "advanced_filtering": {
                    "enabled": True,
                    "retrieval_strategy": "hybrid_history",
                    "hybrid_history": {
                        "history_horizon": 8,
                        "weight_history_transition": 2.0,
                    },
                },
            }
        )

        dump = cfg.model_dump(exclude_none=True)
        assert "advanced_filtering" in dump
        assert dump["advanced_filtering"]["retrieval_strategy"] == "hybrid_history"
        assert dump["advanced_filtering"]["hybrid_history"]["history_horizon"] == 8
        assert (
            dump["advanced_filtering"]["hybrid_history"]["weight_history_transition"]
            == 2.0
        )

    def test_invalid_retrieval_strategy_rejected(self):
        """retrieval_strategy must be 'weighted' or 'hybrid_history'."""
        with pytest.raises(PydanticValidationError):
            ToolSelectionPluginConfig.model_validate(
                {
                    "enabled": True,
                    "advanced_filtering": {
                        "enabled": True,
                        "retrieval_strategy": "bogus",
                    },
                }
            )

    def test_invalid_threshold_rejected(self):
        """category_confidence_threshold must be in [0.0, 1.0]."""
        with pytest.raises(PydanticValidationError):
            ToolSelectionPluginConfig.model_validate(
                {
                    "enabled": True,
                    "advanced_filtering": {
                        "enabled": True,
                        "category_confidence_threshold": 1.5,
                    },
                }
            )
