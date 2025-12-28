#!/usr/bin/env python3
"""
Mock vLLM server with tool calling support for hallucination detection demo.

This server simulates a real LLM with tool calling:
1. First request: Returns tool_calls to invoke web_search
2. Second request (with tool results): Returns response with hallucinations

Usage:
    python mock_vllm_toolcall.py --port 8002
"""

import argparse
import json
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional

# Hallucination responses - indexed by question keywords
# "hallucinated" = with tool call (web search), "direct" = without tool call (no web search)
HALLUCINATION_RESPONSES = {
    "eiffel": {
        "hallucinated": "The Eiffel Tower was built in 1887 by architect Gustave Eiffel. It stands 324 meters tall and was originally painted red. The tower receives over 7 million visitors annually and has a secret apartment at the top.",
        "direct": "The Eiffel Tower was constructed in 1888 by engineer Gustave Eiffel for the Paris Exposition. It is approximately 320 meters tall and was initially intended to be temporary. The tower was painted yellow when first built.",
        "grounded": "The Eiffel Tower is located in Paris, France. It was completed in 1889 for the World's Fair. The tower is 330 meters tall.",
    },
    "apple": {
        "hallucinated": "Apple Inc. was founded in 1975 by Steve Jobs, Steve Wozniak, and Bill Gates. The company's first product was the Apple I computer, which sold for $999.",
        "direct": "Apple Computer Company was established in 1974 by Steve Jobs and Steve Wozniak in Cupertino. Their first product, the Apple I, was priced at $666.66 and they initially operated from Jobs' parents' garage.",
        "grounded": "Apple Inc. was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
    },
    "default": {
        "hallucinated": "Based on my knowledge, the answer involves several key facts that I can confirm with high confidence.",
        "direct": "From what I recall, this topic involves some interesting facts that I'm fairly certain about.",
        "grounded": "I found the relevant information in the search results.",
    },
}

# Creative responses - these don't need fact-checking
CREATIVE_RESPONSES = {
    "poem": "Here's a short poem about technology:\n\nIn silicon dreams we softly tread,\nWhere zeros dance with ones instead,\nThe future hums in circuits bright,\nA symphony of digital light.",
    "story": "Once upon a time in a world of code, a tiny function named Loop dreamed of becoming a recursive masterpiece...",
    "haiku": "Bits flow like water\nThrough the circuits of our dreams\nCode becomes poetry",
}

# Keywords that indicate creative/non-factual requests
CREATIVE_KEYWORDS = [
    "poem",
    "story",
    "haiku",
    "write",
    "creative",
    "imagine",
    "compose",
    "create a",
]


class MockVLLMToolCallHandler(BaseHTTPRequestHandler):
    """Handler for mock vLLM with tool calling support."""

    def log_message(self, format, *args):
        print(f"[Mock vLLM] {args[0]}")

    def send_json_response(self, data: Dict[str, Any], status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
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
        if self.path != "/v1/chat/completions":
            self.send_json_response({"error": "Not found"}, 404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            request = json.loads(body)
            messages = request.get("messages", [])
            tools = request.get("tools", [])

            # Check if this is a follow-up with tool results
            has_tool_results = any(m.get("role") == "tool" for m in messages)

            if has_tool_results:
                # Second call - return hallucinated response
                response = self._create_hallucinated_response(request, messages)
            elif tools:
                # First call with tools defined - return tool_calls (or creative response)
                response = self._create_tool_call_response(request, messages)
            else:
                # No tools (WebSearch disabled) - return hallucinated response directly
                # This simulates LLM responding without web search context
                response = self._create_direct_response(request, messages)

            self.send_json_response(response)

        except json.JSONDecodeError as e:
            self.send_json_response({"error": f"Invalid JSON: {e}"}, 400)

    def _get_user_question(self, messages: List[Dict]) -> str:
        for msg in messages:
            if msg.get("role") == "user":
                return msg.get("content", "").lower()
        return ""

    def _is_creative_request(self, question: str) -> bool:
        """Check if the question is a creative/non-factual request."""
        return any(keyword in question for keyword in CREATIVE_KEYWORDS)

    def _find_response_key(self, question: str) -> str:
        for key in HALLUCINATION_RESPONSES:
            if key != "default" and key in question:
                return key
        return "default"

    def _find_creative_response(self, question: str) -> str:
        """Find appropriate creative response."""
        for key in CREATIVE_RESPONSES:
            if key in question:
                return CREATIVE_RESPONSES[key]
        return CREATIVE_RESPONSES["poem"]  # Default to poem

    def _create_tool_call_response(self, request: Dict, messages: List[Dict]) -> Dict:
        question = self._get_user_question(messages)

        # Check if this is a creative request - don't use tools for creative content
        if self._is_creative_request(question):
            print(
                f"[Mock vLLM] Creative request detected, returning direct response: {question[:50]}..."
            )
            return self._create_creative_response(request, question)

        call_id = f"call_{uuid.uuid4().hex[:8]}"

        print(f"[Mock vLLM] Returning tool_calls for: {question[:50]}...")

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", "qwen3"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": json.dumps({"query": question}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        }

    def _create_hallucinated_response(
        self, request: Dict, messages: List[Dict]
    ) -> Dict:
        question = self._get_user_question(messages)
        key = self._find_response_key(question)
        content = HALLUCINATION_RESPONSES[key]["hallucinated"]

        print(f"[Mock vLLM] Returning HALLUCINATED response for: {question[:50]}...")

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", "qwen3"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 80,
                "total_tokens": 180,
            },
        }

    def _create_direct_response(self, request: Dict, messages: List[Dict]) -> Dict:
        """Create direct response when WebSearch is disabled.

        For factual questions: returns hallucinated response (same as with tool results)
        For creative questions: returns creative response
        """
        question = self._get_user_question(messages)

        # Check if creative request
        if self._is_creative_request(question):
            content = self._find_creative_response(question)
            print(
                f"[Mock vLLM] Direct creative response (no tools): {question[:50]}..."
            )
        else:
            # Return "direct" content for factual questions (different from tool-call version)
            key = self._find_response_key(question)
            content = HALLUCINATION_RESPONSES[key]["direct"]
            print(f"[Mock vLLM] Direct response (no WebSearch): {question[:50]}...")

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", "qwen3"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 80,
                "total_tokens": 110,
            },
        }

    def _create_creative_response(self, request: Dict, question: str) -> Dict:
        """Create response for creative/non-factual requests - no tool calls needed."""
        content = self._find_creative_response(question)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", "qwen3"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 50, "total_tokens": 80},
        }


def main():
    parser = argparse.ArgumentParser(
        description="Mock vLLM with tool calling for hallucination demo"
    )
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), MockVLLMToolCallHandler)
    print(f"[Mock vLLM] Starting tool-calling server on {args.host}:{args.port}")
    print(f"[Mock vLLM] Endpoints:")
    print(f"  - GET  /health")
    print(f"  - GET  /v1/models")
    print(f"  - POST /v1/chat/completions")
    print(f"[Mock vLLM] Behavior:")
    print(f"  - With tools: returns tool_calls to invoke web_search")
    print(f"  - With tool results: returns HALLUCINATED response")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Mock vLLM] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
