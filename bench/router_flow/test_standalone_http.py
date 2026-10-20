"""Exercise benchmark response validation against a real local HTTP server."""

import importlib.util
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "standalone_http", Path(__file__).with_name("standalone_http.py")
)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class BenchmarkTests(unittest.TestCase):
    def server(self, wrong_model=False, malformed=False, delay_general=False):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                code = "code" in body["messages"][0]["content"]
                model = "code-model" if code else "general-model"
                if delay_general and not code:
                    time.sleep(0.01)
                result = json.dumps(
                    {
                        "model": "wrong" if wrong_model else model,
                        "choices": (
                            None if malformed else [{"message": {"content": "ok"}}]
                        ),
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(result)))
                self.end_headers()
                self.wfile.write(result)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def test_valid_routes_produce_timings_and_counts(self):
        result = BENCH.phase(self.server(), 2, 0.05)
        self.assertTrue(result["passed"])
        self.assertGreater(result["requests"], 0)
        self.assertEqual(len(result["latencies_ms"]), result["requests"])
        self.assertGreater(result["p50_ms"], 0)
        self.assertGreaterEqual(result["p95_ms"], result["p50_ms"])

    def test_fast_wrong_response_is_not_a_successful_measurement(self):
        result = BENCH.phase(self.server(wrong_model=True), 1, 0.05)
        self.assertFalse(result["passed"])
        self.assertEqual(result["requests"], 0)
        self.assertEqual(result["errors"], ["ValueError"])

    def test_invalid_duration_and_concurrency_are_rejected(self):
        for concurrency, seconds in ((0, 1), (1, 0), (1, float("nan"))):
            with self.assertRaises(ValueError):
                BENCH.phase("http://127.0.0.1:1", concurrency, seconds)

    def test_malformed_choices_are_recorded_as_a_failed_measurement(self):
        result = BENCH.phase(self.server(malformed=True), 1, 0.05)
        self.assertFalse(result["passed"])
        self.assertEqual(result["errors"], ["ValueError"])

    def test_each_worker_alternates_routes_and_reports_counts(self):
        calls = {}
        original = BENCH.request

        def record(endpoint, index):
            calls.setdefault(threading.get_ident(), []).append(index % 2)
            return original(endpoint, index)

        with patch.object(BENCH, "request", side_effect=record):
            result = BENCH.phase(self.server(delay_general=True), 4, 0.25)
        self.assertTrue(any(len(routes) > 1 for routes in calls.values()))
        for routes in calls.values():
            self.assertTrue(all(left != right for left, right in pairwise(routes)))
        counts = result["route_requests"]
        self.assertEqual(sum(counts.values()), result["requests"])
        self.assertLessEqual(abs(counts["code-model"] - counts["general-model"]), 4)

    def test_failed_warmup_still_writes_a_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            status = BENCH.main(
                [
                    "--endpoint",
                    self.server(wrong_model=True),
                    "--seconds",
                    "0.05",
                    "--repeats",
                    "1",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 1)
            report = json.loads(output.read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["phases"], [])
            self.assertEqual(report["warmup"]["errors"], ["ValueError"])


if __name__ == "__main__":
    unittest.main()
