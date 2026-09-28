from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_model_catalog.py"
SPEC = importlib.util.spec_from_file_location(
    "generate_model_catalog_reasoning_validation", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
catalog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = catalog
SPEC.loader.exec_module(catalog)


class ModelCatalogReasoningValidationTests(unittest.TestCase):
    def test_provider_reasoning_transport_must_project_the_family_type(self) -> None:
        binding = {
            "catalog": "example/model",
            "relationship": "first_party",
            "id": "native-model",
            "protocols": ["openai/chat-completions@1"],
        }
        providers = {
            "example": {
                "protocols": ["openai/chat-completions@1"],
                "supported_operations": ["openai/chat-completions@1#create"],
                "reasoning_transport": "thinking_object",
                "models": [binding],
            }
        }
        models = {
            "example/model": {
                "kind": "physical",
                "lifecycle": "active",
                "reasoning_family": "effort",
            }
        }
        families = {
            "effort": {
                "type": "reasoning_effort",
                "levels": ["low", "high"],
                "default": "high",
                "modes": ["enabled"],
                "default_mode": "enabled",
            }
        }

        with self.assertRaisesRegex(
            catalog.CatalogBuildError,
            r"thinking_object.*cannot project.*reasoning_effort",
        ):
            catalog._validate_provider_bindings(
                providers,
                models,
                {"openai/chat-completions@1"},
                families,
            )

        binding["reasoning_transport"] = "thinking_object_effort"
        catalog._validate_provider_bindings(
            providers,
            models,
            {"openai/chat-completions@1"},
            families,
        )

    def test_effort_flags_ignore_the_disabled_sentinel_for_omission(self) -> None:
        catalog._validate_reasoning(
            [
                {
                    "id": "hybrid",
                    "type": "reasoning_effort",
                    "parameter": "reasoning_effort",
                    "activation_parameter": "enable_thinking",
                    "effort_flags": {
                        "low": "low_effort",
                        "medium": "medium_effort",
                    },
                    "levels": ["none", "low", "medium", "high"],
                    "default": "high",
                    "modes": ["enabled", "disabled"],
                    "default_mode": "enabled",
                    "disabled": "none",
                }
            ]
        )

    def test_plain_effort_transport_cannot_drop_an_independent_switch(self) -> None:
        providers = {
            "example": {
                "protocols": ["openai/chat-completions@1"],
                "supported_operations": ["openai/chat-completions@1#create"],
                "reasoning_transport": "top_level_effort",
                "models": [
                    {
                        "catalog": "example/model",
                        "relationship": "first_party",
                        "id": "native-model",
                        "protocols": ["openai/chat-completions@1"],
                    }
                ],
            }
        }
        models = {
            "example/model": {
                "kind": "physical",
                "lifecycle": "active",
                "reasoning_family": "hybrid",
            }
        }
        families = {
            "hybrid": {
                "type": "reasoning_effort",
                "parameter": "reasoning_effort",
                "activation_parameter": "enable_thinking",
                "levels": ["low", "high"],
                "default": "high",
                "modes": ["enabled", "disabled"],
                "default_mode": "enabled",
            }
        }

        with self.assertRaisesRegex(
            catalog.CatalogBuildError,
            r"top_level_effort.*cannot project.*disabled switch",
        ):
            catalog._validate_provider_bindings(
                providers,
                models,
                {"openai/chat-completions@1"},
                families,
            )

        providers["example"]["models"][0]["reasoning_modes"] = ["enabled"]
        catalog._validate_provider_bindings(
            providers,
            models,
            {"openai/chat-completions@1"},
            families,
        )


if __name__ == "__main__":
    unittest.main()
