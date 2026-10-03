#!/usr/bin/env python3
"""Engine mode: `vllm-sr serve MODEL ...` runs the model runtime and serves it.

These tests start the real CLI in the current Python environment, which must
have the model runtime installed (`make model-runtime-install`). They serve
tiny random-weight packages written by `vllm-sr-runtime fixture`, so no model
is downloaded, and send the requests the Quickstart page tells readers to send,
read from the page itself.
"""

import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

QUICKSTART = (
    Path(__file__).resolve().parents[3] / "website/docs/model-runtime/quickstart.md"
)
# A runtime request on the page: the path a curl command calls and its JSON body.
CURL_REQUEST = re.compile(
    r"curl[^\n]*?(/v1/(?:decisions|classify|embeddings|rerank|bundle))"
    r"(?:(?!\ncurl).)*?-d '(\{.*?\})'",
    re.S,
)
READY_TIMEOUT_SECONDS = 180
STOP_TIMEOUT_SECONDS = 30
HTTP_TIMEOUT_SECONDS = 60
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNSUPPORTED = 422


def quickstart_requests() -> dict[str, dict]:
    """The Quickstart's runtime requests, by path."""
    page = QUICKSTART.read_text(encoding="utf-8")
    return {path: json.loads(body) for path, body in CURL_REQUEST.findall(page)}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _call(base: str, path: str, body: dict | None = None) -> tuple[int, object]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib_request.Request(
        base + path,
        data=data,
        method="GET" if body is None else "POST",
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib_request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            payload = response.read()
            status = response.status
    except urllib_error.HTTPError as failure:
        payload, status = failure.read(), failure.code
    text = payload.decode()
    try:
        return status, json.loads(text)
    except json.JSONDecodeError:
        return status, text


class EngineProcess:
    """One `vllm-sr serve MODEL ...` process on a free local port."""

    def __init__(self, models: list[str], log_path: Path):
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log = log_path.open("w")
        self.process = subprocess.Popen(
            ["vllm-sr", "serve", *models, "--device", "cpu", "--port", str(self.port)],
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
        )

    def wait_ready(self) -> dict:
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        interval = 0.25
        last = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(
                    f"vllm-sr serve exited with {self.process.returncode}"
                )
            try:
                status, last = _call(self.base, "/health")
                if status == HTTP_OK:
                    return last
            except (urllib_error.URLError, ConnectionError):
                pass
            time.sleep(interval)
            interval = min(interval * 2, 2.0)
        raise AssertionError(f"not ready within {READY_TIMEOUT_SECONDS}s: {last}")

    def stop(self) -> int:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as error:
                self.process.kill()
                self.process.wait()
                raise AssertionError("vllm-sr serve did not stop on SIGINT") from error
        self.log.close()
        return self.process.returncode


class TestEngineMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="vllm-sr-engine-"))
        cls.addClassCleanup(shutil.rmtree, cls.root, ignore_errors=True)
        cls.decision = cls._fixture("decision", "decision2", "qwen3")
        cls.classifier = cls._fixture("classifier", "task_heads", "sequence")
        cls.requests = quickstart_requests()

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

    def _serve(self, *models: str) -> EngineProcess:
        engine = EngineProcess(
            list(models), self.root / f"serve-{len(models)}-{time.monotonic_ns()}.log"
        )
        self.addCleanup(engine.stop)
        engine.wait_ready()
        return engine

    def test_quickstart_decision_model(self):
        engine = self._serve(self.decision)

        status, models = _call(engine.base, "/v1/models")
        self.assertEqual(status, HTTP_OK)
        (card,) = models["data"]
        self.assertEqual(card["family"], "decision2")
        self.assertIn("decisions", card["surfaces"])
        self.assertTrue(card["ready"])

        body = self.requests["/v1/decisions"]
        status, response = _call(engine.base, "/v1/decisions", body)
        self.assertEqual(status, HTTP_OK, response)
        kind, reasoning = response["answers"]["kind"], response["answers"]["reasoning"]
        self.assertIn(kind["choice"], body["questions"]["kind"]["criteria"])
        self.assertTrue(
            math.isclose(sum(kind["probabilities"].values()), 1.0, abs_tol=1e-6)
        )
        self.assertGreaterEqual(reasoning["noul"], 0.0)
        self.assertLessEqual(reasoning["noul"], 1.0)

        status, alias = _call(engine.base, "/v1/systemone", body)
        self.assertEqual(status, HTTP_OK)
        self.assertEqual(alias["answers"], response["answers"])

        status, refused = _call(engine.base, "/v1/classify", {"input": ["x"]})
        self.assertEqual(status, HTTP_UNSUPPORTED, refused)
        self.assertEqual(refused["error"]["code"], "unsupported_surface")

        status, metrics = _call(engine.base, "/metrics")
        self.assertEqual(status, HTTP_OK)
        self.assertIn(
            'vllm_sr_runtime_requests_total{endpoint="/v1/decisions"', metrics
        )
        self.assertEqual(engine.stop(), 0)

    def test_quickstart_classifier(self):
        engine = self._serve(self.classifier)

        status, response = _call(
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

        status, models = _call(engine.base, "/v1/models")
        self.assertEqual(status, HTTP_OK)
        ids = {card["family"]: card["id"] for card in models["data"]}
        self.assertEqual(set(ids), {"decision2", "task_heads"})
        status, health = _call(engine.base, "/health")
        self.assertEqual(status, HTTP_OK)
        self.assertEqual(set(health["models"]), set(ids.values()))

        decisions = {**self.requests["/v1/decisions"], "model": ids["decision2"]}
        classify = {**self.requests["/v1/classify"], "model": ids["task_heads"]}
        status, bundle = _call(
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
            "kind": _call(engine.base, "/v1/decisions", decisions)[1],
            "domain": _call(engine.base, "/v1/classify", classify)[1],
        }
        self.assertEqual(
            bundle["results"][0]["decisions"]["answers"], alone["kind"]["answers"]
        )
        self.assertEqual(
            bundle["results"][1]["classify"]["results"], alone["domain"]["results"]
        )

        status, missing = _call(
            engine.base, "/v1/classify", self.requests["/v1/classify"]
        )
        self.assertEqual(status, HTTP_BAD_REQUEST, missing)
        self.assertEqual(missing["error"]["code"], "invalid_request")


if __name__ == "__main__":
    unittest.main()
