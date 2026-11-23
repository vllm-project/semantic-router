"""Minimal mock OpenAI backend for the recipe-lab replay smoke test.

Serves deterministic chat completions so the router can dispatch end to end
without a real model. Answer text is fixed; correctness for the smoke is
judged by the case list, not by this server.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWER_TABLE = [
    ("capital of france", "The capital of France is Paris."),
    ("hello in french", "Bonjour!"),
    ("for loop", "A for loop repeats a block of code over a sequence."),
    (
        "fun fact about honey",
        "Honey never spoils; edible honey was found in ancient tombs.",
    ),
    (
        "quicksort",
        "Both are O(n log n) on average; quicksort often wins on small arrays due to cache locality.",
    ),
    (
        "caching strategies",
        "LRU adapts to skew; TTL is simpler but can serve stale entries.",
    ),
]

# The local lane answers the honey case badly on purpose, so the smoke has a
# real E3 (capability error): routing was right, the model still got it wrong.
WRONG_LOCAL_ANSWERS = {"fun fact about honey": "The capital of France is Paris."}


_CAPABILITY_TABLE: dict = {}


def _answer_for(model: str, prompt: str) -> str:
    # capability model from the cases file: hard questions fail on the local
    # lane and succeed on the frontier lane, so routing decisions move accuracy
    if prompt in _CAPABILITY_TABLE:
        case = _CAPABILITY_TABLE[prompt]
        hard = case.get("truth") == "cloud_frontier_reasoning"
        if not hard or not model.startswith("local"):
            return str(case.get("expected_contains", "mock answer"))
        return "That requires deeper analysis than I can provide right now."
    lowered = prompt.lower()
    for needle, good_answer in ANSWER_TABLE:
        if needle in lowered:
            if model.startswith("local") and needle in WRONG_LOCAL_ANSWERS:
                return WRONG_LOCAL_ANSWERS[needle]
            return good_answer
    return "mock answer"


# A request that caps output below this many tokens finishes with "length":
# the M3 integration truncation scenario drives sr-bench's output-limit
# accounting (engine marks the case `quality_failure: "output_limit"`).
TRUNCATION_MAX_TOKENS = 16

# Per-completion artificial latency (seconds): keeps sr-bench arms in
# flight long enough for fault-injection polls to observe them, standing
# in for duplicating the case set many times over.
_DELAY_SECONDS = 0.0


def _completion(model: str, content: str, finish_reason: str = "stop") -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        }
    ).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def _send(
        self, status: int, body: bytes, content_type: str = "application/json"
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/v1/models"):
            body = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {"id": "local/private-qwen", "object": "model"},
                        {"id": "local/omni", "object": "model"},
                        {"id": "cloud/frontier-reasoning", "object": "model"},
                    ],
                }
            ).encode("utf-8")
            self._send(200, body)
        else:
            self._send(404, b'{"error": "not found"}')

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        if self.path.startswith("/v1/chat/completions"):
            try:
                payload = json.loads(raw.decode("utf-8"))
                model = str(payload.get("model", "unknown"))
                prompt = str(payload.get("messages", [{}])[-1].get("content", ""))
                stream = bool(payload.get("stream"))
                max_tokens = payload.get("max_tokens")
            except (json.JSONDecodeError, AttributeError):
                model, prompt, stream, max_tokens = "unknown", "", False, None
            if _DELAY_SECONDS > 0:
                time.sleep(_DELAY_SECONDS)
            finish_reason = (
                "length"
                if isinstance(max_tokens, int) and max_tokens < TRUNCATION_MAX_TOKENS
                else "stop"
            )
            content = _answer_for(model, prompt)
            if stream:
                self._send_sse(model, content, finish_reason)
            else:
                self._send(200, _completion(model, content, finish_reason))
        else:
            self._send(404, b'{"error": "not found"}')

    def _send_sse(self, model: str, content: str, finish_reason: str = "stop") -> None:
        chunk = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": content}}
            ],
        }
        final = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": 16,
                "completion_tokens": 2,
                "total_tokens": 18,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
        body = (
            f"data: {json.dumps(chunk)}\n\n"
            f"data: {json.dumps(final)}\n\n"
            "data: [DONE]\n\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # silence default stderr noise
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="mock OpenAI-compatible backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--cases",
        default=None,
        help="cases JSONL (prompt/expected_contains/truth): enables the capability model - "
        "the local lane fails hard questions, the frontier lane answers correctly",
    )
    parser.add_argument(
        "--delay-ms",
        type=float,
        default=0.0,
        help="artificial per-completion latency; keeps sr-bench arms in flight "
        "for fault-injection polls (env: RECIPE_LAB_MOCK_DELAY_MS)",
    )
    args = parser.parse_args()
    global _DELAY_SECONDS
    _DELAY_SECONDS = args.delay_ms / 1000.0
    if args.cases:
        _CAPABILITY_TABLE.clear()
        with open(args.cases, encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                case = json.loads(line)
                _CAPABILITY_TABLE[case["prompt"]] = case
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock backend listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
