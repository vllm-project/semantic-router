"""Offline regressions for malformed Kai responses."""

from __future__ import annotations

import contextlib
import http.server
import importlib.util
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

PILOT_DIR = Path(__file__).parents[1]
RUNNER_PATH = PILOT_DIR / "kai" / "run_kai.py"
SPEC = importlib.util.spec_from_file_location("kai_runner", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import failure
    raise ImportError(f"cannot load Kai runner from {RUNNER_PATH}")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

LABELS = json.loads((PILOT_DIR / "protocol.json").read_text("utf-8"))["label_order"]


def answer(probabilities: dict) -> dict:
    return {
        "model": RUNNER.MODEL,
        "answers": {
            RUNNER.QUESTION_KEY: {
                "type": "choice",
                "choice": LABELS[0],
                "confidence": 0.5,
                "probabilities": probabilities,
            }
        },
    }


VALID = answer(dict.fromkeys(LABELS, 1 / len(LABELS)))
STRING_PROBABILITY = answer(
    {**VALID["answers"]["intent"]["probabilities"], "math": "0.1"}
)
MALFORMED = {
    "empty list body": [],
    "null answer": {"model": RUNNER.MODEL, "answers": {RUNNER.QUESTION_KEY: None}},
    "string probability": STRING_PROBABILITY,
    "integer beyond float range": answer(
        {**VALID["answers"]["intent"]["probabilities"], "math": 10**400}
    ),
}


def fake_call(responses: list) -> mock.Mock:
    def call(base_url: str, body_bytes: bytes) -> tuple:
        body = responses.pop(0)
        return 200, json.dumps(body), body, None, 1.0

    return mock.Mock(side_effect=call)


class ScriptedRuntime(http.server.BaseHTTPRequestHandler):
    """Answers each request with the next (body, bytes to send) pair. Sending
    fewer bytes than the declared length truncates the body, and None closes
    the connection without a response."""

    replies: ClassVar[list] = []

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        self.close_connection = True
        body, sent = self.replies.pop(0)
        if sent is None:
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body[:sent])

    def log_message(self, *args) -> None:
        pass


class TestMalformedResponses(unittest.TestCase):
    def test_malformed_body_is_a_recorded_contract_failure(self) -> None:
        for name, body in MALFORMED.items():
            with self.subTest(name), mock.patch.object(
                RUNNER, "call", fake_call([body])
            ):
                record = RUNNER.run_one("http://kai", "state", {}, LABELS)

                self.assertFalse(record["contract_valid"])
                self.assertTrue(record["contract_errors"])
                self.assertEqual(record["raw_response"], json.dumps(body))
                self.assertNotIn("top1_probability", record)

    def test_valid_body_keeps_its_top_label(self) -> None:
        with mock.patch.object(RUNNER, "call", fake_call([VALID])):
            record = RUNNER.run_one("http://kai", "state", {}, LABELS)

        self.assertTrue(record["contract_valid"])
        self.assertEqual(record["top1_from_probabilities"], sorted(LABELS))

    def test_run_records_every_case_after_a_malformed_one(self) -> None:
        cases = len((PILOT_DIR / "inputs.jsonl").read_text("utf-8").splitlines())
        bodies = [VALID, *MALFORMED.values()]
        bodies += [VALID] * (cases + 1 - len(bodies))
        with tempfile.TemporaryDirectory() as out_dir, mock.patch.object(
            RUNNER, "call", fake_call(bodies)
        ), mock.patch.object(
            sys,
            "argv",
            ["run_kai.py", "--pilot-dir", str(PILOT_DIR), "--out-dir", out_dir],
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            RUNNER.main()
            lines = (Path(out_dir) / "kai-results.jsonl").read_text("utf-8")

        records = [json.loads(line) for line in lines.splitlines()]
        valid = [record["contract_valid"] for record in records]
        self.assertEqual(
            valid, [False] * len(MALFORMED) + [True] * (cases - len(MALFORMED))
        )

    def test_run_records_a_truncated_and_a_dropped_response(self) -> None:
        cases = len((PILOT_DIR / "inputs.jsonl").read_text("utf-8").splitlines())
        valid = json.dumps(VALID).encode()
        ScriptedRuntime.replies = [(valid, len(valid)), (valid, 20), (valid, None)]
        ScriptedRuntime.replies += [(valid, len(valid))] * (cases - 2)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ScriptedRuntime)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        argv = ["run_kai.py", "--base-url", f"http://127.0.0.1:{server.server_port}"]
        with tempfile.TemporaryDirectory() as out_dir, mock.patch.object(
            sys, "argv", [*argv, "--pilot-dir", str(PILOT_DIR), "--out-dir", out_dir]
        ), contextlib.redirect_stdout(io.StringIO()):
            RUNNER.main()
            lines = (Path(out_dir) / "kai-results.jsonl").read_text("utf-8")

        truncated, dropped, *healthy = [json.loads(line) for line in lines.splitlines()]
        self.assertIn("IncompleteRead", truncated["error"])
        self.assertEqual(truncated["raw_response"], valid[:20].decode())
        self.assertIn("RemoteDisconnected", dropped["error"])
        self.assertFalse(truncated["contract_valid"] or dropped["contract_valid"])
        self.assertEqual([r["contract_valid"] for r in healthy], [True] * (cases - 2))


if __name__ == "__main__":
    unittest.main()
