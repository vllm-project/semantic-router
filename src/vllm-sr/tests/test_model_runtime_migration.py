"""The CLI's own migration leaves the removed guard module to the Router's."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli.config_application_forms import removed_application_forms
from cli.config_migration import migrate_config_data


def legacy_document():
    return {
        "version": "v0.3",
        "listeners": [],
        "routing": {},
        "global": {
            "model_catalog": {
                "external": [
                    {
                        "model_role": "guardrail",
                        "llm_model_name": "served-guard",
                        "llm_endpoint": {"address": "guard.example", "port": 8080},
                    }
                ],
                "modules": {
                    "prompt_guard": {
                        "enabled": True,
                        "protocol": "http_classify",
                        "threshold": 0.7,
                    }
                },
            }
        },
    }


def test_the_retired_guard_protocol_binds_nothing_and_is_left_to_the_router():
    source = legacy_document()
    original = copy.deepcopy(source)
    result = migrate_config_data(source)
    assert source == original
    catalog = result["global"]["model_catalog"]
    assert "jailbreak" not in (catalog.get("bindings") or {})
    assert removed_application_forms(result) == [
        "global.model_catalog.modules.prompt_guard: removed_section: "
        "global.model_catalog.modules.prompt_guard is removed; run `vllm-sr config "
        "migrate` (drop it; its threshold is a jailbreak rule's predicate)"
    ]
