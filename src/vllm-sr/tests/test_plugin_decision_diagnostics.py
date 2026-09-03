"""CLI contract for bounded decision diagnostics."""

import pytest
from cli.models import UserConfig
from cli.validator import validate_plugin_configurations


@pytest.mark.parametrize("named_recipe", [False, True])
@pytest.mark.parametrize(
    "configuration,valid",
    [
        ({}, True),
        ({"enabled": True, "max_signals": 0}, True),
        ({"enabled": True, "max_payload_bytes": 65536}, True),
        ({"max_signals": 129}, False),
        ({"max_projections": 65}, False),
        ({"max_text_runes": 513}, False),
        ({"max_payload_bytes": 65537}, False),
        ({"enabled": False, "max_signals": -1}, False),
        ({"enabled": "true"}, False),
        ({"max_signals": "32"}, False),
    ],
)
def test_decision_diagnostics_plugin_contract(named_recipe, configuration, valid):
    routing = {
        "decisions": [
            {
                "name": "diagnostics",
                "priority": 1,
                "modelRefs": [{"model": "model-a"}],
                "plugins": [
                    {"type": "decision_diagnostics", "configuration": configuration}
                ],
            }
        ]
    }
    data = {"version": "v0.3"}
    if named_recipe:
        data["recipes"] = [{"name": "diagnostics", "routing": routing}]
    else:
        data["routing"] = routing
    config = UserConfig.model_validate(data)
    assert bool(validate_plugin_configurations(config)) is not valid
