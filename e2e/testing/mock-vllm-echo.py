#!/usr/bin/env python3
"""Echo-mode mock vLLM server for memory retrieval testing.

Echoes back ALL messages (system + user + injected memory context) so that
tests can verify memory injection by checking keywords in the response.

Usage:
    python3 e2e/testing/mock-vllm-echo.py --port 8003
    python3 e2e/testing/mock-vllm-echo.py --port 8003 --host 0.0.0.0
"""

import json
import time
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler


class EchoVLLMHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._json_response(200, {"data": [{"id": "qwen3", "object": "model"}]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()
        req = json.loads(body) if body else {}

        messages = req.get("messages", [])

        all_text = " ".join(
            m.get("content", "")
            for m in messages
            if isinstance(m.get("content", ""), str)
        )

        if "extract" in all_text.lower() and (
            "fact" in all_text.lower() or "memory" in all_text.lower()
        ):
            echo_text = self._handle_extraction(messages)
        elif "rewrite" in all_text.lower() and "query" in all_text.lower():
            echo_text = self._handle_rewrite(messages)
        else:
            parts = []
            for m in messages:
                role = m.get("role", "unknown")
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
                parts.append(f"[{role}] {content}")
            echo_text = "\n".join(parts)

        response = {
            "id": f"chatcmpl-echo-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", "qwen3"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": echo_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(echo_text.split()),
                "completion_tokens": len(echo_text.split()),
                "total_tokens": len(echo_text.split()) * 2,
            },
        }
        self._json_response(200, response)

    def _handle_rewrite(self, messages):
        """Extract the query from the rewrite template and return it as-is.

        The user message has format:
            History:\n...\n\nQuery: <actual query>\n\nRewritten query:
        """
        import re

        user_msg = next(
            (
                m.get("content", "")
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        match = re.search(
            r"Query:\s*(.+?)(?:\n\nRewritten query:|$)", user_msg, re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return user_msg.strip()

    def _handle_extraction(self, messages):
        """Return JSON facts in the format the Go extractor expects:
        a bare JSON array of {"type": "semantic", "content": "..."} objects.

        Only scans user-role messages (which contain the embedded conversation),
        skipping the system prompt that contains example sentences.
        """
        import re

        user_text = " ".join(
            m.get("content", "")
            for m in messages
            if m.get("role") == "user" and isinstance(m.get("content", ""), str)
        )
        facts = []
        sentences = re.split(r"[.!?\n]+", user_text)
        for s in sentences:
            s = s.strip()
            if s.startswith("[user]:"):
                s = s[len("[user]:") :].strip()
            elif s.startswith("[assistant]:"):
                continue
            if len(s) > 15 and any(
                kw in s.lower()
                for kw in [
                    "i ",
                    "my ",
                    "i'm",
                    "i've",
                    "i am",
                    "i have",
                    "name is",
                    "favorite",
                    "work",
                    "live",
                    "play",
                    "learning",
                    "visit",
                    "run ",
                    "cook",
                    "drive",
                    "best time",
                    "every ",
                    "walked",
                    "signature",
                ]
            ):
                facts.append({"type": "semantic", "content": s[:200]})
        if not facts:
            cleaned = re.sub(
                r"Extract important information.*?Return JSON array:",
                "",
                user_text,
                flags=re.DOTALL,
            ).strip()
            if cleaned:
                facts = [{"type": "semantic", "content": cleaned[:200]}]
        return json.dumps(facts[:10])

    def _json_response(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        print(f"[EchoVLLM] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="Echo-mode mock vLLM for memory tests")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), EchoVLLMHandler)
    print(f"Echo vLLM mock running on {args.host}:{args.port}")
    print("All messages (system + user + injected) will be echoed in responses")
    server.serve_forever()


if __name__ == "__main__":
    main()
