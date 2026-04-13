"""HTTP client helpers for vllm-sr chat (OpenAI-compatible chat completions)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from cli.deployment_backend import DEFAULT_TARGET, resolve_target
from cli.runtime_stack import resolve_runtime_stack
from cli.utils import load_config

DEFAULT_CHAT_MODEL = "MoM"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


def normalize_base_url(url: str) -> str:
    """Return base URL without trailing slash."""
    return url.rstrip("/")


def resolve_listener_host_port(config_path: str) -> tuple[str, int]:
    """Resolve host port for the first HTTP listener (Docker / local Envoy mapping)."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Use --config or --base-url to target the router."
        )
    user_config = load_config(str(path)) or {}
    listeners = user_config.get("listeners") or []
    if not listeners:
        raise ValueError(
            f"No listeners configured in {config_path}; cannot derive local API URL."
        )
    first = listeners[0]
    if not isinstance(first, dict):
        raise ValueError("First listener entry must be a mapping.")
    raw_port = first.get("port")
    if not isinstance(raw_port, int):
        raise ValueError(
            "First listener must declare an integer 'port' to derive the public URL."
        )
    stack_layout = resolve_runtime_stack()
    host_port = raw_port + stack_layout.port_offset
    return "localhost", host_port


def resolve_chat_base_url(
    *,
    config_path: str,
    target: str | None,
) -> str:
    """Resolve the HTTP base URL for chat completions (no trailing slash)."""
    resolved_target = resolve_target(target)
    if resolved_target != DEFAULT_TARGET:
        raise ValueError(
            "Non-Docker targets are not yet supported by `vllm-sr chat`. "
            "Use `curl` or another HTTP client to reach the routed endpoint."
        )
    host, port = resolve_listener_host_port(config_path)
    return normalize_base_url(f"http://{host}:{port}")


def build_chat_payload(
    *,
    model: str,
    user_text: str,
    system_text: str | None,
    temperature: float | None,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


def chat_completions_url(base: str) -> str:
    return urljoin(base.rstrip("/") + "/", CHAT_COMPLETIONS_PATH.lstrip("/"))


def extract_assistant_text(data: dict[str, Any]) -> str:
    """Best-effort extraction of assistant text from an OpenAI-style JSON body."""
    err = data.get("error")
    if isinstance(err, dict):
        msg = err.get("message", json.dumps(err))
        raise ValueError(f"API error: {msg}")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(
            "Response contained no choices; raw response was not a chat result."
        )
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        raise ValueError("Malformed chat completion: missing message object.")
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    parts.append(str(part.get("text", "")))
                elif "text" in part:
                    parts.append(str(part["text"]))
        return "".join(parts)
    return str(content)


def post_chat_completions(
    *,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> requests.Response:
    headers = {"Content-Type": "application/json"}
    return requests.post(url, headers=headers, json=payload, timeout=timeout)
