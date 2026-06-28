#!/usr/bin/env python3
"""Mock LLM router server for end-to-end validation.

The server records the rendered query it receives and returns a deterministic
Router-R1-compatible response. Use this to confirm the runtime actually sends
the built prompt to the router server.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockLLMRouterHandler(BaseHTTPRequestHandler):
    selected_model = "mock-model-a"
    log_prefix = "[MockLLMRouter]"

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": self.selected_model})
            return
        if self.path == "/models":
            self._send_json(
                200,
                {
                    "models": [
                        {
                            "name": "mock-model-a",
                            "description": "Baseline mock router target",
                        },
                        {
                            "name": "mock-model-b",
                            "description": "Secondary mock router target",
                        },
                    ]
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/route":
            self._send_json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length).decode("utf-8")
        try:
            payload = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid json: {exc}"})
            return

        query = str(payload.get("query", ""))
        if not query:
            self._send_json(400, {"error": "Missing 'query' field"})
            return

        self._log("received route request")
        self._log(f"query={query}")
        self._log(f"payload={json.dumps(payload, ensure_ascii=False)}")

        selected_model = self._select_model(query)
        response = {
            "selected_model": selected_model,
            "thinking": f"mock router selected {selected_model}",
            "full_response": f"<think>mock router</think><route>{selected_model}</route>",
        }
        self._send_json(200, response)

    def _select_model(self, query: str) -> str:
        lowered = query.lower()
        if "32b" in lowered or "complex" in lowered:
            return "mock-model-b"
        return self.selected_model

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _log(self, message: str):
        line = f"{self.log_prefix} {message}"
        print(line, flush=True)

    def log_message(self, format, *args):
        print(f"{self.log_prefix} {format % args}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock LLM router server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--selected-model", default="mock-model-a")
    args = parser.parse_args()

    MockLLMRouterHandler.selected_model = args.selected_model
    server = HTTPServer((args.host, args.port), MockLLMRouterHandler)
    print(f"{MockLLMRouterHandler.log_prefix} listening on {args.host}:{args.port}", flush=True)
    print(f"{MockLLMRouterHandler.log_prefix} health=/health route=/route", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"{MockLLMRouterHandler.log_prefix} shutting down", flush=True)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
