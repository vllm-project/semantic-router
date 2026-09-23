"""Small loopback checks for paired request bytes and public-safe receipts."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from .cases import DEFAULT_CASES, MODELS, load_cases
from decision_runtime.confidence import choice_confidence, score_confidence

from .__main__ import main
from .report import build_matrix, percentile, summarize
from .transport import Endpoint, Sample, measure


def _answer(question):
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": 0.5}
    if kind == "choice":
        names = list(question["criteria"])
        probabilities = dict.fromkeys(names, 1 / len(names))
        return {
            "type": "choice",
            "choice": names[0],
            "confidence": choice_confidence(tuple(probabilities.values())),
            "probabilities": probabilities,
        }
    levels = question["criteria"]
    probabilities = {str(i): 1 / len(levels) for i in range(len(levels))}
    return {
        "type": "score",
        "score": (len(levels) - 1) / 2,
        "confidence": score_confidence(tuple(probabilities.values())),
        "legend": {str(i): value for i, value in enumerate(levels)},
        "probabilities": probabilities,
    }


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.bodies.append(body)
        request = json.loads(body)
        response = {
            "model": request["model"],
            "answers": {
                name: _answer(question)
                for name, question in request["questions"].items()
            },
            "usage": {"input_tokens": 10, "output_tokens": 1},
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


class _InvalidHandler(_Handler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        payload = b'{"detail":"private service error"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class HarnessTests(TestCase):
    def test_nearest_rank(self):
        self.assertEqual(percentile([9, 1, 5, 3], 0.5), 3)
        self.assertEqual(percentile([9, 1, 5, 3], 0.95), 9)
        self.assertIsNone(percentile([], 0.99))

    def test_identical_bytes_and_private_values_absent_from_receipt(self):
        servers = [ThreadingHTTPServer(("127.0.0.1", 0), _Handler) for _ in range(2)]
        for server in servers:
            server.bodies = []
        threads = [
            threading.Thread(target=server.serve_forever, daemon=True)
            for server in servers
        ]
        for thread in threads:
            thread.start()
        try:
            with TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                args = [
                    "run",
                    "--model",
                    MODELS[0],
                    "--old-url",
                    f"http://127.0.0.1:{servers[0].server_port}/v1/systemone",
                    "--new-url",
                    f"http://127.0.0.1:{servers[1].server_port}/v1/systemone",
                    "--old-token-env",
                    "DECISION_TEST_OLD_TOKEN",
                    "--new-token-env",
                    "DECISION_TEST_NEW_TOKEN",
                    "--old-source-ref",
                    "a" * 40,
                    "--new-source-ref",
                    "b" * 40,
                    "--old-model-revision",
                    "c" * 40,
                    "--new-model-revision",
                    "c" * 40,
                    "--old-hardware",
                    "test GPU x1",
                    "--new-hardware",
                    "test GPU x1",
                    "--old-network-scope",
                    "loopback",
                    "--new-network-scope",
                    "loopback",
                    "--warmup",
                    "1",
                    "--latency-pairs",
                    "4",
                    "--throughput-requests",
                    "4",
                    "--rounds",
                    "2",
                    "--concurrency",
                    "2",
                    "--output-dir",
                    str(output),
                ]
                with patch.dict(
                    os.environ,
                    {
                        "DECISION_TEST_OLD_TOKEN": "old-private-token",
                        "DECISION_TEST_NEW_TOKEN": "new-private-token",
                    },
                ):
                    self.assertEqual(main(args), 0)
                self.assertEqual(sorted(servers[0].bodies), sorted(servers[1].bodies))
                samples = [
                    json.loads(line)
                    for line in (output / "samples.jsonl").read_text().splitlines()
                ]
                self.assertEqual(len(samples), 26)
                self.assertTrue(all(sample["success"] for sample in samples))
                receipt_text = (output / "receipt.json").read_text()
                self.assertNotIn("127.0.0.1", receipt_text)
                self.assertNotIn("old-private-token", receipt_text)
                self.assertNotIn("new-private-token", receipt_text)
                receipt = json.loads(receipt_text)
                self.assertTrue(receipt["summary"]["comparison"]["comparable"])
                self.assertEqual(
                    receipt["summary"]["arms"]["old"]["latency"]["attempts"], 4
                )
                self.assertEqual(
                    receipt["summary"]["arms"]["new"]["throughput"]["attempts"], 8
                )
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    def test_matrix_requires_six_distinct_models(self):
        with self.assertRaisesRegex(ValueError, "exactly six"):
            build_matrix([])

    def test_invalid_response_is_counted_without_body(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _InvalidHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            case = load_cases(DEFAULT_CASES, MODELS[0])[0]
            sample = measure(
                Endpoint(
                    "old", f"http://127.0.0.1:{server.server_port}/v1/systemone", None
                ),
                case,
                phase="latency",
                round_number=0,
                sequence=0,
                timeout_seconds=5,
            )
            self.assertEqual(sample.error_code, "response_contract")
            self.assertEqual(sample.status_code, 200)
            self.assertNotIn(
                "private service error", json.dumps(sample.public_record(0))
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_scope_mismatch_suppresses_ratios(self):
        samples = [
            Sample(
                arm=arm,
                phase=phase,
                round=0,
                sequence=0,
                case_id="case",
                request_sha256="a" * 64,
                started_ns=1,
                ended_ns=1_000_001,
                status_code=200,
                error_code=None,
                response_sha256="b" * 64,
            )
            for arm in ("old", "new")
            for phase in ("latency", "throughput")
        ]
        old = {
            "model_revision": "a",
            "hardware": "MI300X x1",
            "network_scope": "public",
        }
        new = {
            "model_revision": "a",
            "hardware": "MI300X x1",
            "network_scope": "loopback",
        }
        comparison = summarize(samples, old, new)["comparison"]
        self.assertFalse(comparison["comparable"])
        self.assertIn("different_network_scope", comparison["reasons"])
        self.assertIsNone(comparison["old_p50_over_new_p50"])
