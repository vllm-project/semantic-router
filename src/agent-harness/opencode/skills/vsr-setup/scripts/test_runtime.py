import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import runtime


def serve(routes: dict) -> tuple[ThreadingHTTPServer, str]:
    """A runtime at 127.0.0.1 that answers each path in `routes` with (status, body)."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            status, body = routes.get(self.path, (404, {"error": "not found"}))
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1/decisions"


class Health(unittest.TestCase):
    def check(self, routes: dict) -> dict:
        server, url = serve(routes)
        try:
            return runtime.health(url)
        finally:
            server.shutdown()

    def test_a_bare_runtime_lists_its_model_at_v1_models_with_what_it_reads(self):
        out = self.check(
            {
                "/health": (200, {"status": "ready"}),
                "/v1/models": (
                    200,
                    {
                        "data": [
                            {
                                "id": "vela3-lite",
                                "presets": ["pii", "halu"],
                                "modalities": ["text", "image"],
                                "question_types": ["choice", "noul", "score", "span"],
                            }
                        ]
                    },
                ),
            }
        )
        self.assertTrue(out["ready"])
        self.assertEqual(
            out["models"],
            [
                {
                    "id": "vela3-lite",
                    "presets": ["pii", "halu"],
                    "modalities": ["text", "image"],
                    "questionTypes": ["choice", "noul", "score", "span"],
                }
            ],
        )

    def test_a_hosted_api_has_no_health_and_lists_several_models(self):
        out = self.check(
            {
                "/v1/systemone/models": (
                    200,
                    {
                        "data": [
                            {"id": "vllm-sr/d3-ultra"},
                            {"id": "vllm-sr/vela3-flash"},
                        ]
                    },
                )
            }
        )
        self.assertTrue(out["ready"])
        self.assertIn(
            "models vllm-sr/d3-ultra, vllm-sr/vela3-flash: name one in /connect",
            runtime.summary(out)[0],
        )

    def test_a_runtime_still_loading_is_not_ready(self):
        out = self.check({"/health": (200, {"status": "loading"})})
        self.assertFalse(out["ready"])
        self.assertTrue(out["loading"])

    def test_a_runtime_that_refuses_requests_without_a_token_says_so(self):
        out = self.check(
            {
                "/health": (401, {}),
                "/v1/systemone/models": (401, {}),
                "/v1/models": (401, {}),
            }
        )
        self.assertFalse(out["ready"])
        self.assertTrue(out["needsToken"])
        self.assertIn("refuses requests without a token", runtime.summary(out)[0])

    def test_nothing_listening_never_raises(self):
        out = runtime.health("http://127.0.0.1:9/v1/decisions", timeout=0.5)
        self.assertFalse(out["ready"])
        self.assertEqual(out["models"], [])


class Network(unittest.TestCase):
    def test_plain_http_to_another_machine_is_cleartext(self):
        self.assertTrue(runtime.cleartext_remote("http://10.0.0.5:8790/v1/decisions"))
        for url in (
            "https://app.vllm-sr.ai/v1/decisions",
            "http://127.0.0.1:8790/v1/decisions",
            "http://localhost:8790/v1/decisions",
            "http://[::1]:8790/v1/decisions",
        ):
            with self.subTest(url=url):
                self.assertFalse(runtime.cleartext_remote(url))

    def test_slow_only_past_a_second_a_question(self):
        self.assertIsNone(runtime.slow_note(300))
        self.assertIsNone(runtime.slow_note(None))
        self.assertIn("slow runtime, 1450 ms", runtime.slow_note(1450))


if __name__ == "__main__":
    unittest.main()
