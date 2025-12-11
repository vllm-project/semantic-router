#!/usr/bin/env python3
"""
Mock vLLM server for testing hallucination detection.

This server returns controllable responses to test the hallucination detection pipeline:
1. Factual questions that need fact-checking
2. Responses that contain hallucinations (claims not in context)
3. Responses that are grounded in context

Usage:
    python mock-vllm-hallucination.py --port 8002
"""

import argparse
import json
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

# Predefined responses for testing hallucination detection
# Each response includes tool_calls context to simulate RAG
RESPONSES = {
    # Response with hallucination - claims facts not in context
    "hallucination": {
        "content": "The Eiffel Tower was built in 1887 by architect Gustave Eiffel. It stands 324 meters tall and was originally painted red. The tower receives over 7 million visitors annually and has a secret apartment at the top.",
        "context": "The Eiffel Tower is located in Paris, France. It was completed in 1889 for the World's Fair. The tower is 330 meters tall.",
    },
    # Response grounded in context - no hallucination
    "grounded": {
        "content": "The Eiffel Tower is located in Paris, France. It was completed in 1889 for the World's Fair.",
        "context": "The Eiffel Tower is located in Paris, France. It was completed in 1889 for the World's Fair. The tower is 330 meters tall.",
    },
    # Default response
    "default": {
        "content": "I can help you with that question. Based on the information available, I would say the answer depends on the specific context.",
        "context": "",
    },
}

# Track the last question to determine response type
last_question = ""


class MockVLLMHandler(BaseHTTPRequestHandler):
    """Handler for mock vLLM requests"""

    def log_message(self, format, *args):
        """Custom logging"""
        print(f"[Mock vLLM] {args[0]}")

    def send_json_response(self, data: Dict[str, Any], status: int = 200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/health":
            self.send_json_response({"status": "healthy"})
        elif self.path == "/v1/models":
            self.send_json_response(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "qwen3",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "mock-vllm",
                        }
                    ],
                }
            )
        else:
            self.send_json_response({"error": "Not found"}, 404)

    def do_POST(self):
        """Handle POST requests"""
        global last_question

        if self.path == "/v1/chat/completions":
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                request = json.loads(body)
                messages = request.get("messages", [])

                # Find the user message
                user_message = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        user_message = msg.get("content", "").lower()
                        last_question = user_message

                # Determine response type based on keywords
                if "hallucination" in user_message or "eiffel" in user_message:
                    response_type = "hallucination"
                elif "grounded" in user_message or "fact" in user_message:
                    response_type = "grounded"
                else:
                    response_type = "default"

                response_data = RESPONSES[response_type]

                # Create OpenAI-compatible response
                response = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.get("model", "qwen3"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_data["content"],
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 100,
                        "total_tokens": 150,
                    },
                }

                print(f"[Mock vLLM] Responding with {response_type} response")
                self.send_json_response(response)

            except json.JSONDecodeError as e:
                self.send_json_response({"error": f"Invalid JSON: {e}"}, 400)
        else:
            self.send_json_response({"error": "Not found"}, 404)


def main():
    parser = argparse.ArgumentParser(
        description="Mock vLLM server for hallucination testing"
    )
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), MockVLLMHandler)
    print(f"[Mock vLLM] Starting server on {args.host}:{args.port}")
    print(f"[Mock vLLM] Endpoints:")
    print(f"  - GET  /health")
    print(f"  - GET  /v1/models")
    print(f"  - POST /v1/chat/completions")
    print(f"[Mock vLLM] Use keywords in prompts to control response:")
    print(f"  - 'hallucination' or 'eiffel' -> returns hallucinated content")
    print(f"  - 'grounded' or 'fact' -> returns grounded content")
    print(f"  - (other) -> returns default response")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Mock vLLM] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
