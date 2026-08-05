"""OpenAI -> Ollama-native proxy that disables Qwen3 'thinking'.

Why: the semantic-router looper calls panel/judge models over the OpenAI
``/v1/chat/completions`` endpoint, but Ollama's OpenAI endpoint ignores the
``think`` flag, so Qwen3 burns the whole token budget (and minutes) on reasoning
before emitting any answer. Ollama's NATIVE ``/api/chat`` *does* honor
``"think": false``. This shim accepts OpenAI requests, forwards them to the native
endpoint with thinking off, and returns an OpenAI-shaped response.

Point provider ``backend_refs[].base_url`` at this proxy:
    http://localhost:11435/v1

Run:
    .venv-bench/bin/python -m bench.grounded_fusion.ollama_proxy --port 11435
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OLLAMA_NATIVE = "http://localhost:11434/api/chat"


def _to_native(body: dict) -> dict:
    options = {}
    if "temperature" in body:
        options["temperature"] = body["temperature"]
    if body.get("max_tokens"):
        options["num_predict"] = body["max_tokens"]
    return {
        "model": body["model"],
        "messages": body.get("messages", []),
        "think": False,  # the whole point
        "stream": False,
        "options": options,
    }


def _to_openai(native: dict, model: str) -> dict:
    msg = native.get("message", {}) or {}
    prompt_tok = native.get("prompt_eval_count", 0) or 0
    completion_tok = native.get("eval_count", 0) or 0
    return {
        "id": f"chatcmpl-proxy-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": msg.get("content", "")},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tok,
            "completion_tokens": completion_tok,
            "total_tokens": prompt_tok + completion_tok,
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):  # quiet
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid json"})
            return
        payload = json.dumps(_to_native(body)).encode()
        # Retry transient Ollama failures (e.g. a model load racing under memory
        # pressure) so a hiccup doesn't silently thin the fusion panel.
        last_err = None
        for attempt in range(3):
            req = urllib.request.Request(
                OLLAMA_NATIVE,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=1800) as resp:
                    native = json.loads(resp.read())
                self._send(200, _to_openai(native, body.get("model", "")))
                return
            except Exception as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
        # client may have already given up; nothing to send if so
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self._send(502, {"error": str(last_err)})

    def _send(self, code: int, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11435)
    args = ap.parse_args()
    print(f"ollama no-think proxy on :{args.port} -> {OLLAMA_NATIVE}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
