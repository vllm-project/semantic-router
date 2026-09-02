"""Tests for the shadow_dispatch plugin configuration model."""

import pytest
from cli.models import PluginConfig, PluginType, ShadowDispatchPluginConfig
from pydantic import ValidationError as PydanticValidationError


class TestShadowDispatchPluginConfig:
    def test_defaults_are_bounded(self):
        cfg = ShadowDispatchPluginConfig(enabled=True, model="candidate")
        assert cfg.enabled is True
        assert cfg.tls_skip_verify is False
        assert cfg.sample_rate is None
        assert cfg.max_concurrency == 2
        assert cfg.max_queue_depth == 8
        assert cfg.timeout_seconds == 30
        assert cfg.max_response_bytes == 1048576
        assert cfg.max_retries == 0
        assert cfg.capture_response_body is False
        assert cfg.max_capture_bytes == 4096

    def test_model_required_when_enabled(self):
        with pytest.raises(PydanticValidationError, match="model is required"):
            ShadowDispatchPluginConfig(enabled=True)
        ShadowDispatchPluginConfig(enabled=False)

    def test_enabled_is_required(self):
        with pytest.raises(PydanticValidationError, match="enabled"):
            ShadowDispatchPluginConfig(model="candidate")

    @pytest.mark.parametrize(
        "field,value",
        [
            ("sample_rate", 1.5),
            ("sample_rate", -0.1),
            ("max_retries", 4),
            ("max_concurrency", -1),
        ],
    )
    def test_out_of_range_bounds_rejected(self, field, value):
        with pytest.raises(PydanticValidationError):
            ShadowDispatchPluginConfig(enabled=True, model="candidate", **{field: value})

    def test_unknown_fields_rejected(self):
        with pytest.raises(PydanticValidationError):
            ShadowDispatchPluginConfig(enabled=True, model="candidate", bogus=1)

    def test_plugin_type_round_trips(self):
        plugin = PluginConfig(
            type="shadow_dispatch",
            configuration={"enabled": True, "model": "candidate", "sample_rate": 0.1},
        )
        assert plugin.type is PluginType.SHADOW_DISPATCH
        assert plugin.model_dump()["type"] == "shadow_dispatch"
