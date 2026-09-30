from __future__ import annotations

import importlib
import unittest

from support import PLUGIN_ROOT, has

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None


@unittest.skipIf(tomllib is None, "needs tomllib")
class EntryPointTest(unittest.TestCase):
    def test_entry_points_resolve(self) -> None:
        project = tomllib.loads((PLUGIN_ROOT / "pyproject.toml").read_text())
        groups = project["project"]["entry-points"]
        self.assertEqual(
            groups["vllm.general_plugins"],
            {"vllm_sr_decision2": "vllm_sr_plugins:register"},
        )
        self.assertEqual(
            groups["vllm.endpoint_plugins"],
            {
                "vllm_sr_system_one": "vllm_sr_plugins.decision2.endpoint:SystemOneEndpoint"
            },
        )
        for targets in groups.values():
            for target in targets.values():
                module, attribute = target.split(":")
                self.assertTrue(
                    callable(getattr(importlib.import_module(module), attribute))
                )


@unittest.skipUnless(has("vllm") and has("fastapi"), "needs vLLM")
class RegistrationTest(unittest.TestCase):
    def test_register_is_idempotent_and_installs_the_config_hook(self) -> None:
        from vllm import ModelRegistry
        from vllm.model_executor.models.config import (
            MODELS_CONFIG_MAP,
            Qwen3_5ForCausalLMConfig,
        )

        import vllm_sr_plugins

        vllm_sr_plugins.register()
        vllm_sr_plugins.register()
        self.assertIn("Decision2Qwen3_5ForScoring", ModelRegistry.get_supported_archs())
        self.assertIs(
            MODELS_CONFIG_MAP["Decision2Qwen3_5ForScoring"], Qwen3_5ForCausalLMConfig
        )

    def test_endpoint_attaches_system_one_route(self) -> None:
        from fastapi import FastAPI

        from vllm_sr_plugins.decision2.endpoint import SystemOneEndpoint

        plugin = SystemOneEndpoint()
        app = FastAPI()
        plugin.attach_router(app)
        self.assertEqual(plugin.required_tasks, ("plugin",))
        self.assertIn("/v1/system_one", {route.path for route in app.routes})


if __name__ == "__main__":
    unittest.main()
