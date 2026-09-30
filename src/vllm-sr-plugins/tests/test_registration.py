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
                "vllm_sr_decisions": "vllm_sr_plugins.decision2.endpoint:DecisionsEndpoint"
            },
        )
        for targets in groups.values():
            for target in targets.values():
                module, attribute = target.split(":")
                self.assertTrue(
                    callable(getattr(importlib.import_module(module), attribute))
                )


@unittest.skipUnless(
    has("fastapi") and has("httpx") and has("torch") and has("transformers"),
    "needs FastAPI, httpx, torch and transformers",
)
class DecisionsRouteTest(unittest.TestCase):
    """Requests through FastAPI, the service and a fake engine; no vLLM needed."""

    def client(self, service=None):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_sr_plugins.decision2.endpoint import STATE_KEY, DecisionsEndpoint

        plugin = DecisionsEndpoint()
        app = FastAPI()
        plugin.attach_router(app)
        self.assertEqual(plugin.required_tasks, ("token_classify",))
        if service is not None:
            setattr(app.state, STATE_KEY, service)
        return TestClient(app)

    def service(self):
        from test_service import RecordingEngine, fake_package

        from vllm_sr_plugins.decision2.service import SystemOneService

        return SystemOneService(fake_package(), RecordingEngine())

    def test_decisions_and_alias_answer_the_same(self) -> None:
        from test_service import QUESTIONS

        client = self.client(self.service())
        body = {"state": "ticket: printer on fire", "questions": QUESTIONS}
        first = client.post("/v1/decisions", json=body)
        alias = client.post("/v1/system_one", json={**body, "model": "DEV2.0-test"})
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(alias.status_code, 200, alias.text)
        self.assertEqual(first.json(), alias.json())
        self.assertEqual(list(first.json()["answers"]), list(QUESTIONS))
        self.assertEqual(first.json()["model"], "DEV2.0-test")

    def test_rejections(self) -> None:
        client = self.client(self.service())
        noul = {"q": {"type": "noul", "instructions": "Is it urgent?"}}
        self.assertEqual(client.post("/v1/decisions", content=b"{").status_code, 400)
        self.assertEqual(
            client.post("/v1/decisions", json={"questions": noul}).status_code, 400
        )
        self.assertEqual(
            client.post(
                "/v1/decisions", json={"state": "s", "questions": noul, "extra": 1}
            ).status_code,
            400,
        )
        self.assertEqual(
            client.post(
                "/v1/decisions", json={"state": "s", "questions": {}}
            ).status_code,
            400,
        )
        self.assertEqual(
            client.post(
                "/v1/decisions",
                json={"state": "s", "questions": noul, "model": "other"},
            ).status_code,
            404,
        )

    def test_uninitialized(self) -> None:
        response = self.client().post(
            "/v1/decisions", json={"state": "s", "questions": {}}
        )
        self.assertEqual(response.status_code, 503)


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


if __name__ == "__main__":
    unittest.main()
