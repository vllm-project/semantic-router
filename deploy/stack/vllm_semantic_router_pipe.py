"""
title: vLLM Semantic Router Pipe
author: open-webui
date: 2025-10-01
version: 1.1
license: Apache-2.0
description: A pipe for proxying requests to vLLM Semantic Router and displaying decision headers (category, reasoning, model, injection) and security alerts (PII violations, jailbreak detection).
requirements: requests, pydantic
"""

import json
from typing import Generator, Iterator, List, Union

import requests
from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        # vLLM Semantic Router endpoint URL
        # For stack deployment, use localhost since all services run in one container
        vsr_base_url: str = "http://localhost:8801"

        # API key for authentication (if required)
        api_key: str = ""

        # Enable/disable displaying VSR headers in the UI
        show_vsr_info: bool = True

        # Enable/disable logging VSR headers to console
        log_vsr_info: bool = True

        # Enable/disable debug logging
        debug: bool = True

        # Request timeout in seconds
        timeout: int = 300

    def __init__(self):
        # Important: type should be "manifold" instead of "pipe"
        # manifold type Pipeline will be displayed in the model list
        self.type = "manifold"
        self.id = "auto"
        self.name = "vsr/"

        # Initialize valves
        self.valves = self.Valves(
            **{
                "vsr_base_url": "http://localhost:8801",
                "api_key": "",
                "show_vsr_info": True,
                "log_vsr_info": True,
                "debug": True,
                "timeout": 300,
            }
        )

        # Store VSR headers from the last request
        self.last_vsr_headers = {}

        print("=" * 80)
        print("🚀 vLLM Semantic Router Pipe - Initialization")
        print("=" * 80)
        print(f"  Type: {self.type}")
        print(f"  ID: {self.id}")
        print(f"  Name: {self.name}")
        print(f"  VSR Base URL: {self.valves.vsr_base_url}")
        print(f"  Debug Mode: {self.valves.debug}")
        print("=" * 80)

    async def on_startup(self):
        print("\n" + "=" * 80)
        print("🔥 on_startup: vLLM Semantic Router Pipe initialized")
        print("=" * 80)
        print(f"  VSR Base URL: {self.valves.vsr_base_url}")
        print(f"  API Key: {'***' if self.valves.api_key else '(not set)'}")
        print(f"  Show VSR Info: {self.valves.show_vsr_info}")
        print(f"  Log VSR Info: {self.valves.log_vsr_info}")
        print(f"  Debug: {self.valves.debug}")
        print(f"  Timeout: {self.valves.timeout}s")

        # Test if pipelines() is being called
        pipes_list = self.pipelines()
        print(f"\n📋 Available Pipes/Models:")
        for pipe in pipes_list:
            print(f"    - ID: {pipe['id']}")
            print(f"      Name: {pipe['name']}")
        print("=" * 80 + "\n")

    async def on_shutdown(self):
        print("\n" + "=" * 80)
        print("🛑 on_shutdown: vLLM Semantic Router Pipe")
        print("=" * 80 + "\n")

    async def on_valves_updated(self):
        print("\n" + "=" * 80)
        print("⚙️  on_valves_updated: vLLM Semantic Router Pipe valves updated")
        print("=" * 80)
        print(f"  VSR Base URL: {self.valves.vsr_base_url}")
        print(f"  API Key: {'***' if self.valves.api_key else '(not set)'}")
        print(f"  Show VSR Info: {self.valves.show_vsr_info}")
        print(f"  Log VSR Info: {self.valves.log_vsr_info}")
        print(f"  Debug: {self.valves.debug}")
        print(f"  Timeout: {self.valves.timeout}s")
        print("=" * 80 + "\n")

    def pipes(self) -> List[dict]:
        """
        Deprecated: manifold type uses pipelines() method instead of pipes()
        The returned model list will be displayed in Open WebUI's model selector
        """
        return self.pipelines()

    def pipelines(self) -> List[dict]:
        """
        Important: manifold type uses pipelines() method instead of pipes()
        The returned model list will be displayed in Open WebUI's model selector
        """
        pipelines_list = [
            {
                "id": "auto",
                "name": "auto",
            }
        ]

        if self.valves.debug:
            print("\n" + "=" * 80)
            print("📞 pipelines() method called - Returning available models")
            print("=" * 80)
            for pipeline in pipelines_list:
                print(f"  - ID: {pipeline['id']}")
                print(f"    Name: {pipeline['name']}")
            print("=" * 80 + "\n")

        return pipelines_list

    def _extract_vsr_headers(self, headers: dict) -> dict:
        """
        Extract VSR-specific headers from response headers.
        """
        vsr_headers = {}

        # List of VSR headers to extract
        vsr_header_keys = [
            # Decision headers
            "x-vsr-selected-category",
            "x-vsr-selected-reasoning",
            "x-vsr-selected-model",
            "x-vsr-injected-system-prompt",
            "x-vsr-cache-hit",
            # Security headers
            "x-vsr-pii-violation",
            "x-vsr-jailbreak-blocked",
            "x-vsr-jailbreak-type",
            "x-vsr-jailbreak-confidence",
        ]

        # Extract headers (case-insensitive)
        for key in vsr_header_keys:
            # Try lowercase
            value = headers.get(key)
            if not value:
                # Try uppercase
                value = headers.get(key.upper())
            if not value:
                # Try title case
                value = headers.get(key.title())

            if value:
                vsr_headers[key] = value

        return vsr_headers

    def _format_vsr_info(self, vsr_headers: dict, position: str = "prefix") -> str:
        """
        Format VSR headers into a readable message for display.
        Shows the semantic router's decision chain in 3 stages (multi-line format):
        Stage 1: Security Validation
        Stage 2: Cache Check
        Stage 3: Intelligent Routing

        Args:
            vsr_headers: VSR decision headers
            position: "prefix" (before response) or "suffix" (after response)
        """
        if not vsr_headers:
            return ""

        # Build decision chain in stages (multi-line format)
        lines = ["**🔀 vLLM Semantic Router - Chain-Of-Thought 🔀**"]

        # ============================================================
        # Stage 1: Security Validation (🛡️)
        # ============================================================
        security_parts = []

        has_jailbreak = vsr_headers.get("x-vsr-jailbreak-blocked") == "true"
        has_pii = vsr_headers.get("x-vsr-pii-violation") == "true"
        is_blocked = has_jailbreak or has_pii

        # Jailbreak check
        if has_jailbreak:
            jailbreak_type = vsr_headers.get("x-vsr-jailbreak-type", "unknown")
            jailbreak_confidence = vsr_headers.get("x-vsr-jailbreak-confidence", "N/A")
            security_parts.append(
                f"🚨 *Jailbreak Detected, Confidence: {jailbreak_confidence}*"
            )
        else:
            security_parts.append("✅ *No Jailbreak*")

        # PII check
        if has_pii:
            security_parts.append("🚨 *PII Detected*")
        else:
            security_parts.append("✅ *No PII*")

        # Result
        if is_blocked:
            security_parts.append("❌ ***BLOCKED***")
        else:
            security_parts.append("💯 ***Continue***")

        lines.append(
            "  → 🛡️ ***Stage 1 - Prompt Guard***: " + " → ".join(security_parts)
        )

        # If blocked, stop here
        if is_blocked:
            result = "\n".join(lines)
            if position == "prefix":
                return result + "\n\n---\n\n"
            else:
                return "\n\n---\n\n" + result

        # ============================================================
        # Stage 2: Cache Check (🔥)
        # ============================================================
        cache_parts = []
        has_cache_hit = vsr_headers.get("x-vsr-cache-hit") == "true"

        if has_cache_hit:
            cache_parts.append("🔥 *HIT*")
            cache_parts.append("⚡️ *Retrieve Memory*")
            cache_parts.append("💯 ***Fast Response***")
        else:
            cache_parts.append("🌊 *MISS*")
            cache_parts.append("🧠 *Update Memory*")
            cache_parts.append("💯 ***Continue***")

        lines.append("  → 🔥 ***Stage 2 - Router Memory***: " + " → ".join(cache_parts))

        # If cache hit, stop here
        if has_cache_hit:
            result = "\n".join(lines)
            if position == "prefix":
                return result + "\n\n---\n\n"
            else:
                return "\n\n---\n\n" + result

        # ============================================================
        # Stage 3: Intelligent Routing (🧠)
        # ============================================================
        routing_parts = []

        # Domain
        category = vsr_headers.get("x-vsr-selected-category", "").strip()
        if not category:
            category = "other"
        routing_parts.append(f"📂 *{category}*")

        # Reasoning mode
        if vsr_headers.get("x-vsr-selected-reasoning"):
            reasoning = vsr_headers["x-vsr-selected-reasoning"]
            if reasoning == "on":
                routing_parts.append("🧠 *Reasoning On*")
            else:
                routing_parts.append("⚡ *Reasoning Off*")

        # Model
        if vsr_headers.get("x-vsr-selected-model"):
            model = vsr_headers["x-vsr-selected-model"]
            routing_parts.append(f"🥷 *{model}*")

        # Prompt optimization
        if vsr_headers.get("x-vsr-injected-system-prompt") == "true":
            routing_parts.append("🎯 *Prompt Optimized*")

        routing_parts.append(f"💯 ***Continue***")

        if routing_parts:
            lines.append(
                "  → 🧠 ***Stage 3 - Smart Routing***: " + " → ".join(routing_parts)
            )

        # Combine all lines
        result = "\n".join(lines)

        if position == "prefix":
            return result + "\n\n---\n\n"
        else:
            return "\n\n---\n\n" + result

    def _log_vsr_info(self, vsr_headers: dict):
        """
        Log VSR information to console.
        """
        if not vsr_headers or not self.valves.log_vsr_info:
            return

        # Check if there are security violations
        has_security_violation = (
            vsr_headers.get("x-vsr-pii-violation") == "true"
            or vsr_headers.get("x-vsr-jailbreak-blocked") == "true"
        )

        print("=" * 60)
        if has_security_violation:
            print("🛡️  SECURITY ALERT & Routing Decision:")
        else:
            print("vLLM Semantic Router Decision:")
        print("=" * 60)

        # Log security violations first
        if vsr_headers.get("x-vsr-pii-violation") == "true":
            print("  🚨 PII VIOLATION: Request blocked")

        if vsr_headers.get("x-vsr-jailbreak-blocked") == "true":
            print("  🚨 JAILBREAK BLOCKED: Potential attack detected")
            if vsr_headers.get("x-vsr-jailbreak-type"):
                print(f"     Type: {vsr_headers['x-vsr-jailbreak-type']}")
            if vsr_headers.get("x-vsr-jailbreak-confidence"):
                print(f"     Confidence: {vsr_headers['x-vsr-jailbreak-confidence']}")

        # Log routing decision information
        if vsr_headers.get("x-vsr-selected-category"):
            print(f"  Category: {vsr_headers['x-vsr-selected-category']}")

        if vsr_headers.get("x-vsr-selected-reasoning"):
            print(f"  Reasoning Mode: {vsr_headers['x-vsr-selected-reasoning']}")

        if vsr_headers.get("x-vsr-selected-model"):
            print(f"  Selected Model: {vsr_headers['x-vsr-selected-model']}")

        if vsr_headers.get("x-vsr-injected-system-prompt"):
            print(
                f"  System Prompt Injected: {vsr_headers['x-vsr-injected-system-prompt']}"
            )

        if vsr_headers.get("x-vsr-cache-hit"):
            cache_hit = vsr_headers["x-vsr-cache-hit"].lower()
            print(f"  Cache Hit: {cache_hit}")

        print("=" * 60)

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Main pipe function that handles the request/response flow.

        Manifold type pipe() method signature:
        - user_message: User's last message
        - model_id: Selected model ID
        - messages: Complete message history
        - body: Complete request body
        """

        if self.valves.debug:
            print("\n" + "=" * 80)
            print("🔄 pipe() method called - Processing request")
            print("=" * 80)
            print(
                f"  User message: {user_message[:100]}..."
                if len(user_message) > 100
                else f"  User message: {user_message}"
            )
            print(f"  Model ID: {model_id}")
            print(f"  Model requested: {body.get('model', 'N/A')}")
            print(f"  Stream mode: {body.get('stream', False)}")
            print(f"  Messages count: {len(messages)}")
            print("=" * 80)

        # Prepare the request to vLLM Semantic Router
        url = f"{self.valves.vsr_base_url}/v1/chat/completions"

        if self.valves.debug:
            print(f"\n📡 Sending request to: {url}")

        headers = {
            "Content-Type": "application/json",
        }

        if self.valves.api_key:
            headers["Authorization"] = f"Bearer {self.valves.api_key}"
            if self.valves.debug:
                print(f"  Authorization: Bearer ***")

        # Important: Change model in body to "auto"
        # VSR backend only accepts model="auto", then automatically selects model based on request content
        request_body = body.copy()
        original_model = request_body.get("model", "N/A")
        request_body["model"] = "auto"

        if self.valves.debug:
            print(f"\n🔄 Model mapping:")
            print(f"  Original model: {original_model}")
            print(f"  Sending to VSR: auto")

        # Check if streaming is requested
        is_streaming = request_body.get("stream", False)

        if self.valves.debug:
            print(f"  Streaming: {is_streaming}")
            print(f"  Timeout: {self.valves.timeout}s")

        try:
            if self.valves.debug:
                print(f"\n🔌 Connecting to vLLM Semantic Router...")

            response = requests.post(
                url,
                json=request_body,  # Use modified request_body
                headers=headers,
                timeout=self.valves.timeout,
                stream=request_body.get("stream", False),
            )

            if self.valves.debug:
                print(f"✅ Response received - Status: {response.status_code}")
                print(f"  Response headers count: {len(response.headers)}")

            # Check for HTTP errors
            if response.status_code != 200:
                error_msg = f"Error: vLLM Semantic Router returned status {response.status_code}"
                if self.valves.debug:
                    print(f"\n❌ {error_msg}")
                    print(f"  Response text: {response.text[:500]}")
                    print("=" * 80 + "\n")
                return f"{error_msg}: {response.text}"

            # Extract VSR headers from response
            vsr_headers = self._extract_vsr_headers(dict(response.headers))
            self.last_vsr_headers = vsr_headers

            if self.valves.debug:
                print(f"  VSR headers found: {len(vsr_headers)}")
                for key, value in vsr_headers.items():
                    print(f"    {key}: {value}")

                # Print all response headers for debugging
                print(f"\n  All response headers:")
                for key, value in response.headers.items():
                    if key.lower().startswith("x-vsr"):
                        print(f"    {key}: {value}")

            # Log VSR information
            self._log_vsr_info(vsr_headers)

            if is_streaming:
                if self.valves.debug:
                    print(f"\n📺 Handling streaming response...")
                # Handle streaming response
                return self._handle_streaming_response(response, vsr_headers)
            else:
                if self.valves.debug:
                    print(f"\n📄 Handling non-streaming response...")
                    print(f"  Response status: {response.status_code}")
                    print(f"  Response content length: {len(response.content)}")
                    print(
                        f"  Response content type: {response.headers.get('content-type', 'unknown')}"
                    )

                # Check if response is empty
                if not response.content:
                    error_msg = "Error: Empty response from vLLM Semantic Router"
                    if self.valves.debug:
                        print(f"\n❌ {error_msg}")
                        print("=" * 80 + "\n")
                    return error_msg

                # Try to parse JSON response
                try:
                    response_data = response.json()
                except json.JSONDecodeError as e:
                    error_msg = (
                        f"Error: Invalid JSON response from vLLM Semantic Router"
                    )
                    if self.valves.debug:
                        print(f"\n❌ {error_msg}")
                        print(f"  JSON error: {str(e)}")
                        print(
                            f"  Response text (first 500 chars): {response.text[:500]}"
                        )
                        print("=" * 80 + "\n")
                    return f"{error_msg}: {str(e)}"

                if self.valves.debug:
                    print(f"  Response data keys: {list(response_data.keys())}")
                    if "choices" in response_data:
                        print(f"  Choices count: {len(response_data['choices'])}")

                # Add VSR info to the response if enabled
                if self.valves.show_vsr_info and vsr_headers:
                    vsr_info = self._format_vsr_info(vsr_headers, position="prefix")

                    if self.valves.debug:
                        print(
                            f"  Adding VSR info to response (length: {len(vsr_info)})"
                        )

                    # Prepend to the assistant's message
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        for choice in response_data["choices"]:
                            if "message" in choice and "content" in choice["message"]:
                                choice["message"]["content"] = (
                                    vsr_info + choice["message"]["content"]
                                )
                                if self.valves.debug:
                                    print(f"  ✅ VSR info prepended to response")

                if self.valves.debug:
                    print(f"\n✅ Request completed successfully")
                    print("=" * 80 + "\n")

                return response_data

        except requests.exceptions.Timeout:
            error_msg = f"Error: Request to vLLM Semantic Router timed out after {self.valves.timeout} seconds"
            if self.valves.debug:
                print(f"\n❌ {error_msg}")
                print("=" * 80 + "\n")
            return error_msg
        except Exception as e:
            error_msg = (
                f"Error: Failed to communicate with vLLM Semantic Router: {str(e)}"
            )
            if self.valves.debug:
                print(f"\n❌ {error_msg}")
                print(f"  Exception type: {type(e).__name__}")
                print(f"  Exception details: {str(e)}")
                print("=" * 80 + "\n")
            return error_msg

    def _handle_streaming_response(
        self, response: requests.Response, vsr_headers: dict
    ) -> Generator:
        """
        Handle streaming SSE response from vLLM Semantic Router.
        Manually parse SSE stream, no need for sseclient-py dependency.

        Strategy:
        1. Add VSR info before the first content chunk (if enabled)
        2. Detect VSR header updates during streaming (via SSE events)
        3. Ensure it's only added once
        """
        vsr_info_added = False
        first_content_chunk = True  # Mark whether it's the first content chunk
        # Use initial vsr_headers, but may be updated during streaming
        current_vsr_headers = vsr_headers.copy()

        if self.valves.debug:
            print(f"\n📝 Initial VSR headers:")
            for key, value in current_vsr_headers.items():
                print(f"    {key}: {value}")

        # Read streaming response line by line
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            # SSE format: data: {...}
            if line.startswith("data: "):
                data_str = line[6:].strip()  # Remove "data: " prefix

                if data_str == "[DONE]":
                    yield f"data: [DONE]\n\n"

                    if self.valves.debug:
                        print(
                            f"✅ Streaming completed, VSR info added: {vsr_info_added}"
                        )
                else:
                    try:
                        chunk_data = json.loads(data_str)

                        # Check if chunk contains updated VSR header information
                        # Some SSE implementations may include updated headers in chunk metadata
                        if "vsr_headers" in chunk_data:
                            if self.valves.debug:
                                print(f"🔄 VSR headers updated in stream:")
                            for key, value in chunk_data["vsr_headers"].items():
                                full_key = (
                                    f"x-vsr-{key}"
                                    if not key.startswith("x-vsr-")
                                    else key
                                )
                                if current_vsr_headers.get(full_key) != value:
                                    if self.valves.debug:
                                        print(
                                            f"    {full_key}: {current_vsr_headers.get(full_key)} → {value}"
                                        )
                                    current_vsr_headers[full_key] = value

                        # Add VSR info before the first content chunk
                        if (
                            first_content_chunk
                            and self.valves.show_vsr_info
                            and not vsr_info_added
                        ):
                            if (
                                "choices" in chunk_data
                                and len(chunk_data["choices"]) > 0
                            ):
                                choice = chunk_data["choices"][0]
                                delta = choice.get("delta", {})

                                # Check if there is content (role or content)
                                if "role" in delta or "content" in delta:
                                    if self.valves.debug:
                                        print(
                                            f"✅ Adding VSR info at first content chunk"
                                        )
                                        print(f"    VSR headers:")
                                        for key, value in current_vsr_headers.items():
                                            print(f"      {key}: {value}")

                                    # Format VSR info (using prefix mode)
                                    vsr_info = self._format_vsr_info(
                                        current_vsr_headers, position="prefix"
                                    )

                                    # Add VSR info before the first content
                                    current_content = delta.get("content", "")
                                    delta["content"] = vsr_info + current_content
                                    chunk_data["choices"][0]["delta"] = delta
                                    vsr_info_added = True
                                    first_content_chunk = False

                        # If not the first chunk, mark as False
                        if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                            choice = chunk_data["choices"][0]
                            delta = choice.get("delta", {})
                            if "role" in delta or "content" in delta:
                                first_content_chunk = False

                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    except json.JSONDecodeError:
                        # If not valid JSON, pass through as-is
                        yield f"data: {data_str}\n\n"
