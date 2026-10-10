"""HTTP layer of decision25_server.py with a stub model (needs fastapi + httpx; skipped otherwise)."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "package"
FORMAT = Path(__file__).resolve().parents[2] / "common" / "decision_format.py"
READY = all(importlib.util.find_spec(m) for m in ("fastapi", "httpx", "starlette"))


class StubModel:
    model_name = "stub"
    max_length = 40

    def __init__(self, rt):
        self.rt = rt

    def prepare(self, state, questions):
        prepared = self.rt.Prepared(keys=list(questions))
        for key, question in questions.items():
            try:
                normalized = self.rt.normalize_question(question)
            except ValueError as exc:
                prepared.errors[key] = {
                    "type": None,
                    "error": "invalid_question",
                    "message": str(exc),
                }
                continue
            prepared.questions[key] = normalized
            if len(str(state)) > self.max_length:
                prepared.errors[key] = {
                    "type": normalized.kind,
                    "error": "max_length_exceeded",
                    "message": "over the maximum context length of 40 tokens",
                }
                continue
            prepared.sequences[key] = [0] * len(str(state))
        return prepared

    def warmup(self):
        self.warmed = True
        return 0.0

    def run(self, prepared):
        return {
            k: [1 / len(prepared.questions[k].keys)] * len(prepared.questions[k].keys)
            for k in prepared.runnable
        }, 7

    def respond(self, prepared, probabilities, tokens):
        return self.rt.Decision25.respond(self, prepared, probabilities, tokens)


@unittest.skipUnless(READY, "needs fastapi and httpx")
class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.created = not (PACKAGE / "decision25_format.py").exists()
        if cls.created:
            (PACKAGE / "decision25_format.py").write_bytes(FORMAT.read_bytes())
        sys.path.insert(0, str(PACKAGE))
        import decision25_runtime
        import decision25_server

        cls.rt, cls.server = decision25_runtime, decision25_server
        decision25_server.Decision25.from_pretrained = staticmethod(
            lambda *a, **k: StubModel(decision25_runtime)
        )

    @classmethod
    def tearDownClass(cls):
        if cls.created:
            (PACKAGE / "decision25_format.py").unlink()

    def client(self):
        from fastapi.testclient import TestClient

        args = argparse.Namespace(
            model="stub",
            revision=None,
            device="cpu",
            batch_size=8,
            verify="none",
            name=None,
            no_warmup=False,
        )
        return TestClient(self.server.build_app(args))

    def test_requests(self):
        questions = {
            "route": {"type": "choice", "criteria": {"a": None, "b": "second"}},
            "yes": {"type": "noul", "instructions": "Yes?"},
            "level": {"type": "score", "criteria": ["low", "high"]},
        }
        with self.client() as client:
            self.assertEqual(client.get("/health").json()["status"], "ready")
            response = client.post(
                "/v1/systemone",
                json={"model": "x", "state": "short", "questions": questions},
            )
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(
                {k: a["type"] for k, a in body["answers"].items()},
                {"route": "choice", "yes": "noul", "level": "score"},
            )
            self.assertEqual(body["usage"], {"input_tokens": 7, "output_tokens": 0})
            long = client.post(
                "/v1/systemone",
                json={"model": "x", "state": "y" * 50, "questions": questions},
            )
            self.assertEqual(long.status_code, 422)
            self.assertIn("maximum context length", long.text)
            bad = client.post(
                "/v1/systemone",
                json={
                    "model": "x",
                    "state": "s",
                    "questions": {"q": {"type": "choice"}},
                },
            )
            self.assertEqual(bad.status_code, 422)
            extra = client.post(
                "/v1/systemone",
                json={"model": "x", "state": "s", "questions": questions, "samples": 2},
            )
            self.assertEqual(extra.status_code, 422)
            self.assertEqual(
                client.get("/v1/models").json()["models"][0]["name"], "stub"
            )

    def test_api_key(self):
        os.environ["DECISION_API_KEY"] = "test-key"
        try:
            with self.client() as client:
                body = {
                    "model": "x",
                    "state": "s",
                    "questions": {"q": {"type": "noul"}},
                }
                self.assertEqual(
                    client.post("/v1/systemone", json=body).status_code, 401
                )
                ok = client.post(
                    "/v1/systemone",
                    json=body,
                    headers={"Authorization": "Bearer test-key"},
                )
                self.assertEqual(ok.status_code, 200)
        finally:
            del os.environ["DECISION_API_KEY"]


if __name__ == "__main__":
    unittest.main()
