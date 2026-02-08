"""Envoy Model wrapper that routes LLM calls through the Semantic Router."""

import base64
import json
import logging
import os
import re
from io import BytesIO
from typing import Any, Generator

import httpx
from PIL import Image
from smolagents import Model
from smolagents.models import ChatMessageStreamDelta

logger = logging.getLogger(__name__)


class EnvoyModel(Model):
    """
    Model wrapper that routes all LLM calls through Envoy proxy.

    This enables the Semantic Router to:
    - Classify requests and route to appropriate vision models
    - Apply decision-based routing (computer_use, vision, etc.)
    - Execute plugin chains (PII detection, caching, prompts)
    - Provide unified observability and metrics
    """

    def __init__(
        self,
        envoy_url: str = None,
        model_id: str = "MoM",
        timeout: float = 300.0,
        use_ollama_direct: bool = True,  # Bypass Envoy for vision models
        ollama_model: str = "qwen3-vl:8b",  # Upgraded from qwen2.5vl:7b
    ):
        """
        Initialize the Envoy model wrapper.

        Args:
            envoy_url: URL of the Envoy proxy (default: http://localhost:8801)
            model_id: Model ID to send (default: "MoM" which triggers decision engine)
            timeout: Request timeout in seconds
            use_ollama_direct: If True, call Ollama directly for vision (bypasses Envoy)
            ollama_model: Ollama vision model to use when bypassing Envoy
        """
        super().__init__()
        self.envoy_url = envoy_url or os.getenv("ENVOY_URL", "http://localhost:8801")
        self.model_id = model_id
        self.timeout = timeout
        self.ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_model = ollama_model

        # Check if HuggingFace token is available for faster cloud inference
        self.hf_token = os.getenv("HF_TOKEN")
        self.use_huggingface = self.hf_token is not None
        self.hf_model = "Qwen/Qwen3-VL-8B-Instruct"

        # Only use Ollama if HF not available
        self.use_ollama_direct = use_ollama_direct and not self.use_huggingface

        if self.use_huggingface:
            logger.info(f"Using HuggingFace Inference API with model: {self.hf_model}")
        elif self.use_ollama_direct:
            logger.info(f"Using Ollama directly with model: {self.ollama_model}")
        else:
            logger.info(f"Using Envoy proxy at: {self.envoy_url}")

        self.client = httpx.Client(timeout=self.timeout)

    def __call__(
        self,
        messages: list[dict[str, Any]],
        stop_sequences: list[str] | None = None,
        grammar: str | None = None,
        tools_to_call_from: list | None = None,
        **kwargs,
    ) -> str:
        """
        Call the model through Envoy proxy.

        Args:
            messages: List of message dicts with role and content
            stop_sequences: Optional stop sequences
            grammar: Optional grammar constraint
            tools_to_call_from: Optional list of tools (not used for vision)
            **kwargs: Additional arguments (may include images)

        Returns:
            Model response text
        """
        # Convert smolagents message format to OpenAI format
        openai_messages = self._convert_messages(messages, kwargs.get("images"))

        # Build the request payload
        payload = {
            "model": self.model_id,
            "messages": openai_messages,
            "stream": False,
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        # Add any additional parameters
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        # Make the request through Envoy
        try:
            response = self.client.post(
                f"{self.envoy_url}/v1/chat/completions",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Agent-Mode": "computer_use",
                },
            )
            response.raise_for_status()
            result = response.json()

            # Extract the response text
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
                elif "text" in choice:
                    return choice["text"]

            raise ValueError(f"Unexpected response format: {result}")

        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Envoy request failed with status {e.response.status_code}: {e.response.text}"
            )
        except httpx.RequestError as e:
            raise RuntimeError(f"Envoy request failed: {e}")

    def generate_stream(
        self,
        messages: list[Any],
        stop_sequences: list[str] | None = None,
        grammar: str | None = None,
        tools_to_call_from: list | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta, None, None]:
        """
        Stream the model response through Envoy proxy or Ollama directly.

        Args:
            messages: List of message dicts with role and content
            stop_sequences: Optional stop sequences
            grammar: Optional grammar constraint
            tools_to_call_from: Optional list of tools (not used for vision)
            **kwargs: Additional arguments (may include images)

        Yields:
            ChatMessageStreamDelta objects with content chunks
        """
        # Convert smolagents message format to OpenAI format
        openai_messages = self._convert_messages(messages, kwargs.get("images"))

        # Choose endpoint based on configuration
        headers = {"Content-Type": "application/json"}

        if self.use_huggingface:
            # Use HuggingFace Inference API (faster, cloud-based)
            url = f"https://router.huggingface.co/nebius/v1/chat/completions"
            model = self.hf_model
            headers["Authorization"] = f"Bearer {self.hf_token}"
        elif self.use_ollama_direct:
            url = f"{self.ollama_url}/v1/chat/completions"
            model = self.ollama_model
        else:
            url = f"{self.envoy_url}/v1/chat/completions"
            model = self.model_id

        # Build the request payload - use non-streaming for vision models
        use_streaming = False  # Disable streaming for reliability
        payload = {
            "model": model,
            "messages": openai_messages,
            "stream": use_streaming,
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        # Add any additional parameters
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        # Log request details for debugging
        logger.info(f"Making request to {url} with model {model} (HF: {self.use_huggingface})")
        logger.info(f"Number of messages: {len(openai_messages)}")
        for i, msg in enumerate(openai_messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content_desc = f"[{len(content)} parts]"
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type", "unknown")
                        if part_type == "image_url":
                            img_url = part.get("image_url", {}).get("url", "")
                            content_desc += f" image:{len(img_url)} chars"
                        else:
                            text_preview = part.get("text", "")[:50] if part.get("text") else ""
                            content_desc += f" {part_type}:'{text_preview}...'"
                    else:
                        content_desc += f" unknown_part:{type(part)}"
            else:
                content_desc = f"{len(str(content))} chars: '{str(content)[:100]}...'"
            logger.info(f"  Message {i}: role={role}, content={content_desc}")

        # Also log full payload structure (without image data)
        debug_payload = {
            "model": payload["model"],
            "stream": payload["stream"],
            "messages": []
        }
        for msg in payload["messages"]:
            debug_msg = {"role": msg.get("role")}
            content = msg.get("content")
            if isinstance(content, list):
                debug_msg["content"] = [
                    {"type": p.get("type"), "text": p.get("text", "")[:50] + "..." if p.get("text") else None}
                    if p.get("type") != "image_url" else {"type": "image_url", "size": len(str(p))}
                    for p in content if isinstance(p, dict)
                ]
            else:
                debug_msg["content"] = str(content)[:100] + "..." if content else None
            debug_payload["messages"].append(debug_msg)
        logger.info(f"Payload structure: {json.dumps(debug_payload, indent=2)}")

        # Make the request
        try:
            if use_streaming:
                with httpx.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if not line:
                            continue

                        # Handle SSE format: "data: {...}"
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix

                            if data == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield ChatMessageStreamDelta(content=content)
                            except json.JSONDecodeError:
                                continue
            else:
                # Non-streaming request
                response = self.client.post(
                    url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()

                # Log raw response for debugging
                logger.info(f"Raw response: {json.dumps(result, indent=2)[:1000]}")

                # Extract and yield the full response
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "")

                    # Handle case where content might be a list (multimodal response)
                    if isinstance(content, list):
                        logger.info(f"Content is a list with {len(content)} parts, extracting text")
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                            elif isinstance(part, str):
                                text_parts.append(part)
                        content = "\n".join(text_parts)

                    # Clean up malformed response where model echoes input format
                    # The model sometimes outputs: [{'type': 'text', 'text': '...'}]
                    if isinstance(content, str) and (
                        content.startswith("[{'type':") or
                        content.startswith("[{\"type\":") or
                        "'type': 'text'" in content[:100]
                    ):
                        logger.warning("Detected malformed response with input format echo, cleaning...")
                        try:
                            # Try multiple patterns to extract text
                            # Pattern 1: 'text': 'content' (single quotes)
                            # Pattern 2: "text": "content" (double quotes)
                            # Pattern 3: 'text': "content" (mixed quotes)

                            # First, try to find content after 'text': or "text":
                            # This handles multi-line content better
                            text_match = re.search(r"['\"]text['\"]\s*:\s*['\"](.+)", content, re.DOTALL)
                            if text_match:
                                extracted = text_match.group(1)
                                # Remove trailing quote and bracket
                                extracted = re.sub(r"['\"]?\s*\}?\s*\]?\s*$", "", extracted)
                                # Also remove escaped newlines
                                extracted = extracted.replace("\\n", "\n")
                                logger.info(f"Extracted text from malformed response: {extracted[:100]}...")
                                content = extracted
                            else:
                                # Fallback: try simpler pattern
                                matches = re.findall(r"['\"]text['\"]\s*:\s*['\"]([^'\"]+)['\"]", content)
                                if matches:
                                    extracted = "\n".join(matches)
                                    logger.info(f"Extracted text (fallback): {extracted[:100]}...")
                                    content = extracted
                        except Exception as e:
                            logger.warning(f"Failed to clean malformed response: {e}")

                    logger.info(f"Extracted content ({len(content)} chars): {content[:200]}...")

                    if content:
                        yield ChatMessageStreamDelta(content=content)

        except httpx.HTTPStatusError as e:
            # Try to read the error response for debugging
            try:
                if hasattr(e.response, 'read'):
                    e.response.read()
                error_text = e.response.text[:500] if e.response.text else "No error body"
            except Exception:
                error_text = "Could not read error response"
            logger.error(f"Request failed: {error_text}")
            raise RuntimeError(
                f"Request failed with status {e.response.status_code}: {error_text}"
            )
        except httpx.RequestError as e:
            raise RuntimeError(f"Request failed: {e}")

    def _convert_messages(
        self,
        messages: list[Any],
        images: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Convert smolagents message format to OpenAI vision format.

        Args:
            messages: List of smolagents ChatMessage objects or dicts
            images: Optional list of base64-encoded images

        Returns:
            List of OpenAI-formatted message dicts
        """
        result = []

        for idx, msg in enumerate(messages):
            # Handle both dict and ChatMessage objects
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls")
                tool_call_id = msg.get("tool_call_id")
            else:
                # ChatMessage object - access attributes directly
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", None)
                tool_calls = getattr(msg, "tool_calls", None)
                tool_call_id = getattr(msg, "tool_call_id", None)

            # Convert MessageRole enum to string value if needed
            # smolagents uses MessageRole enum (e.g., MessageRole.USER) not strings
            if hasattr(role, "value"):
                role = role.value  # Extract string value from enum

            # Log original message info for debugging
            content_preview = str(content)[:100] if content else "None"
            logger.debug(f"Converting message {idx}: role={role}, content_type={type(content).__name__}, preview={content_preview}")

            # Normalize role to valid roles for Ollama
            # Ollama's OpenAI-compatible API supports: system, user, assistant
            # smolagents uses: user, assistant, system, tool-call, tool-response
            if role in ("tool-response", "tool_response", "observation", "tool"):
                # Convert tool responses to user messages with observation prefix
                content_str = str(content) if content else ""
                result.append({
                    "role": "user",
                    "content": [{"type": "text", "text": f"[Observation]: {content_str}"}]
                })
                logger.info(f"Converted {role} message to user message")
                continue
            elif role in ("tool-call",):
                # Tool call messages - convert to assistant action
                content_str = str(content) if content else "[Action]"
                result.append({
                    "role": "assistant",
                    "content": content_str
                })
                logger.info(f"Converted {role} message to assistant message")
                continue
            elif role not in ("system", "user", "assistant"):
                # Map unknown roles to user
                logger.warning(f"Unknown role '{role}', mapping to 'user'")
                role = "user"

            # Handle assistant messages - strip tool_calls as Ollama doesn't support them
            # The content is what matters for the conversation history
            if role == "assistant":
                content_str = str(content) if content else ""
                if not content_str and tool_calls:
                    # If no content but has tool_calls, generate a summary
                    content_str = "[Action executed]"
                result.append({
                    "role": "assistant",
                    "content": content_str if content_str else " "
                })
                continue

            content_parts = []

            # Handle None content
            if content is None:
                content = ""

            # Handle text content
            if isinstance(content, str):
                if content:  # Only add non-empty text
                    content_parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                # Already in parts format
                for part in content:
                    if isinstance(part, str):
                        if part:  # Only add non-empty text
                            content_parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        # Check for embedded images in dict
                        converted_part = self._convert_content_part(part)
                        # Only add if converted part has meaningful content
                        if converted_part.get("type") == "image_url" or converted_part.get("text"):
                            content_parts.append(converted_part)
                    elif isinstance(part, Image.Image):
                        # Direct PIL Image in content
                        image_url = self._image_to_data_url(part)
                        if image_url:
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            })
                    else:
                        # Try to convert if it has image attribute
                        if hasattr(part, "image"):
                            image_url = self._image_to_data_url(getattr(part, "image"))
                            if image_url:
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": image_url},
                                })
                        else:
                            part_str = str(part)
                            if part_str:  # Only add non-empty text
                                content_parts.append({"type": "text", "text": part_str})

            # Ensure content is valid - Ollama requires non-empty content
            # Use simple string format when possible to avoid model confusion
            if content_parts:
                # Check if we have only text parts (no images)
                has_images = any(p.get("type") == "image_url" for p in content_parts if isinstance(p, dict))
                if has_images:
                    # Keep multipart format for messages with images
                    result.append({"role": role, "content": content_parts})
                else:
                    # Use simple string format for text-only messages
                    # This reduces the chance of model echoing the format
                    text_content = "\n".join(
                        p.get("text", "") for p in content_parts
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                    result.append({"role": role, "content": text_content if text_content else " "})
            elif isinstance(content, str) and content:
                # Fallback to string content if no parts
                result.append({"role": role, "content": content})
            else:
                # Add placeholder for empty messages (required by some models)
                result.append({"role": role, "content": " "})

        # Attach images to the last user message
        if images and result:
            # Find the last user message
            for i in range(len(result) - 1, -1, -1):
                if result[i]["role"] == "user":
                    if isinstance(result[i]["content"], str):
                        result[i]["content"] = [{"type": "text", "text": result[i]["content"]}]
                    elif not isinstance(result[i]["content"], list):
                        result[i]["content"] = []

                    # Add images
                    for img in images:
                        image_url = self._image_to_data_url(img)
                        if image_url:
                            result[i]["content"].append({
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            })
                    break

        # Post-process: merge consecutive messages with the same role
        # This can happen when tool messages are converted to user messages
        merged_result = []
        for msg in result:
            if merged_result and merged_result[-1]["role"] == msg["role"]:
                # Merge with previous message of same role
                prev_content = merged_result[-1]["content"]
                curr_content = msg["content"]

                # Normalize to list format
                if isinstance(prev_content, str):
                    prev_content = [{"type": "text", "text": prev_content}]
                if isinstance(curr_content, str):
                    curr_content = [{"type": "text", "text": curr_content}]

                # Merge content lists
                merged_result[-1]["content"] = prev_content + curr_content
                logger.info(f"Merged consecutive {msg['role']} messages")
            else:
                merged_result.append(msg)

        # Log final message structure (at INFO level for visibility)
        logger.info(f"Converted {len(messages)} messages → {len(merged_result)} final messages")
        for i, msg in enumerate(merged_result):
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, list):
                parts_desc = ", ".join(p.get("type", "?") for p in content[:5])
                if len(content) > 5:
                    parts_desc += f", ... ({len(content)} total)"
                logger.info(f"  Message {i}: role={role}, parts=[{parts_desc}]")
            else:
                content_len = len(str(content)) if content else 0
                logger.info(f"  Message {i}: role={role}, content_len={content_len}")

        return merged_result

    def _convert_content_part(self, part: dict) -> dict:
        """
        Convert a content part dict, handling any embedded images.

        Args:
            part: A content part dict that may contain images

        Returns:
            Converted dict with images as data URLs
        """
        result = {}
        for key, value in part.items():
            if isinstance(value, Image.Image):
                # Convert PIL Image to data URL
                result[key] = self._image_to_data_url(value)
            elif isinstance(value, dict):
                # Recursively handle nested dicts
                result[key] = self._convert_content_part(value)
            elif isinstance(value, list):
                # Handle lists
                result[key] = [
                    self._convert_content_part(item) if isinstance(item, dict)
                    else self._image_to_data_url(item) if isinstance(item, Image.Image)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def _image_to_data_url(self, img: Any) -> str | None:
        """
        Convert various image formats to data URL.

        Args:
            img: PIL Image, base64 string, or data URL

        Returns:
            Data URL string or None if conversion fails
        """
        # Handle PIL Image objects
        if isinstance(img, Image.Image):
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{img_base64}"

        # Handle string formats
        if isinstance(img, str):
            # Already a data URL
            if img.startswith("data:"):
                return img
            # Assume base64 encoded
            return f"data:image/png;base64,{img}"

        # Handle bytes
        if isinstance(img, bytes):
            img_base64 = base64.b64encode(img).decode("utf-8")
            return f"data:image/png;base64,{img_base64}"

        return None

    def __del__(self):
        """Clean up the HTTP client."""
        if hasattr(self, "client"):
            self.client.close()
