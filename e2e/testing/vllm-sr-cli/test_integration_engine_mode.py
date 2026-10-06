#!/usr/bin/env python3
"""Engine mode: `vllm-sr serve MODEL ...` runs the model runtime and serves it.

These tests start the real CLI in the current Python environment, which must
have the model runtime installed (`make model-runtime-install`). They serve
tiny random-weight packages written by `vllm-sr-runtime fixture`, so no model
is downloaded, and send the requests the Quickstart page tells readers to send,
read from the page itself.
"""

import math
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from runtime_http import HTTP_OK, ServeProcess, call, page_requests

QUICKSTART = (
    Path(__file__).resolve().parents[3] / "website/docs/model-runtime/quickstart.md"
)
HTTP_BAD_REQUEST = 400
HTTP_UNSUPPORTED = 422


class TestEngineMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="vllm-sr-engine-"))
        cls.addClassCleanup(shutil.rmtree, cls.root, ignore_errors=True)
        cls.decision = cls._fixture("decision", "decision2", "qwen3")
        cls.classifier = cls._fixture("classifier", "task_heads", "sequence")
        cls.requests = page_requests(QUICKSTART)

    @classmethod
    def _fixture(cls, name: str, family: str, variant: str) -> str:
        output = cls.root / name
        subprocess.run(
            [
                "vllm-sr-runtime",
                "fixture",
                str(output),
                "--family",
                family,
                "--variant",
                variant,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return str(output)

    def _serve(self, *models: str) -> ServeProcess:
        engine = ServeProcess(
            ["vllm-sr", "serve", *models, "--device", "cpu"],
            self.root / f"serve-{len(models)}-{time.monotonic_ns()}.log",
        )
        self.addCleanup(engine.stop)
        engine.wait_ready()
        return engine

    def test_quickstart_decision_model(self):
        engine = self._serve(self.decision)

        status, models = call(engine.base, "/v1/models")
        self.assertEqual(status, HTTP_OK)
        (card,) = models["data"]
        self.assertEqual(card["family"], "decision2")
        self.assertIn("decisions", card["surfaces"])
        self.assertTrue(card["ready"])

        body = self.requests["/v1/decisions"]
        status, response = call(engine.base, "/v1/decisions", body)
        self.assertEqual(status, HTTP_OK, response)
        kind, reasoning = response["answers"]["kind"], response["answers"]["reasoning"]
        self.assertIn(kind["choice"], body["questions"]["kind"]["criteria"])
        self.assertTrue(
            math.isclose(sum(kind["probabilities"].values()), 1.0, abs_tol=1e-6)
        )
        self.assertGreaterEqual(reasoning["noul"], 0.0)
        self.assertLessEqual(reasoning["noul"], 1.0)

        status, alias = call(engine.base, "/v1/systemone", body)
        self.assertEqual(status, HTTP_OK)
        self.assertEqual(alias["answers"], response["answers"])

        status, refused = call(engine.base, "/v1/classify", {"input": ["x"]})
        self.assertEqual(status, HTTP_UNSUPPORTED, refused)
        self.assertEqual(refused["error"]["code"], "unsupported_surface")

        status, metrics = call(engine.base, "/metrics")
        self.assertEqual(status, HTTP_OK)
        self.assertIn(
            'vllm_sr_runtime_requests_total{endpoint="/v1/decisions"', metrics
        )
        self.assertEqual(engine.stop(), 0)

    def test_quickstart_classifier(self):
        engine = self._serve(self.classifier)

        status, response = call(
            engine.base, "/v1/classify", self.requests["/v1/classify"]
        )

        self.assertEqual(status, HTTP_OK, response)
        self.assertEqual(response["kind"], "sequence")
        (result,) = response["results"]
        self.assertIn(result["label"], response["labels"])
        self.assertTrue(math.isclose(sum(result["probabilities"]), 1.0, abs_tol=1e-6))
        self.assertGreater(result["input"]["tokens"], 0)

    def test_one_process_serves_several_models_and_bundles(self):
        engine = self._serve(self.decision, self.classifier)

        status, models = call(engine.base, "/v1/models")
        self.assertEqual(status, HTTP_OK)
        ids = {card["family"]: card["id"] for card in models["data"]}
        self.assertEqual(set(ids), {"decision2", "task_heads"})
        status, health = call(engine.base, "/health")
        self.assertEqual(status, HTTP_OK)
        self.assertEqual(set(health["models"]), set(ids.values()))

        decisions = {**self.requests["/v1/decisions"], "model": ids["decision2"]}
        classify = {**self.requests["/v1/classify"], "model": ids["task_heads"]}
        status, bundle = call(
            engine.base,
            "/v1/bundle",
            {
                "tasks": [
                    {"id": "kind", "decisions": decisions},
                    {"id": "domain", "classify": classify},
                ]
            },
        )
        self.assertEqual(status, HTTP_OK, bundle)
        self.assertEqual(
            [result["id"] for result in bundle["results"]], ["kind", "domain"]
        )
        self.assertEqual(
            [result["status"] for result in bundle["results"]], [HTTP_OK, HTTP_OK]
        )
        alone = {
            "kind": call(engine.base, "/v1/decisions", decisions)[1],
            "domain": call(engine.base, "/v1/classify", classify)[1],
        }
        self.assertEqual(
            bundle["results"][0]["decisions"]["answers"], alone["kind"]["answers"]
        )
        self.assertEqual(
            bundle["results"][1]["classify"]["results"], alone["domain"]["results"]
        )

        status, missing = call(
            engine.base, "/v1/classify", self.requests["/v1/classify"]
        )
        self.assertEqual(status, HTTP_BAD_REQUEST, missing)
        self.assertEqual(missing["error"]["code"], "invalid_request")


if __name__ == "__main__":
    unittest.main()
