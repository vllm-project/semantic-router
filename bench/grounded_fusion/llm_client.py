"""Minimal OpenAI-compatible chat client for the grounded-fusion benchmark.

Used to talk to:
- Ollama directly (``http://localhost:11434/v1``) for the rubric grader and for
  generating the cached panel when running offline.
- The semantic router's OpenAI endpoint for full E2E fusion runs (the harness in
  ``evaluate.py`` reads the extra ``fusion`` trace block from those responses).

Kept dependency-light (``requests`` only) to match ``bench/hallucination``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import requests

# Qwen3 (and other hybrid-reasoning models) emit <think>...</think> blocks inside
# the content. We split those out so downstream JSON parsing sees clean output.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class ChatResult:
    content: str
    reasoning: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None


class ChatClient:
    """Thin OpenAI-compatible /chat/completions client with retries."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "ollama",
        timeout: int = 600,
        temperature: float = 0.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
    ) -> ChatResult:
        payload: dict[str, object] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_body:
            payload.update(extra_body)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err = None
        for attempt in range(self.max_retries + 1):
            start = time.time()
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                latency_ms = (time.time() - start) * 1000
                if resp.status_code != requests.codes.ok:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    continue
                data = resp.json()
                content, reasoning = self._extract_content(data)
                return ChatResult(
                    content=content,
                    reasoning=reasoning,
                    usage=data.get("usage", {}) or {},
                    raw=data,
                    latency_ms=latency_ms,
                )
            except requests.exceptions.RequestException as e:
                last_err = str(e)
            time.sleep(1.5 * (attempt + 1))
        return ChatResult(content="", error=last_err or "unknown error")

    @staticmethod
    def _extract_content(data: dict) -> tuple[str, str]:
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError):
            return "", ""
        content = msg.get("content") or ""
        # Some servers surface reasoning separately; otherwise it's inline <think>.
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        think_blocks = _THINK_RE.findall(content)
        if think_blocks:
            reasoning = (reasoning + "\n" + "\n".join(think_blocks)).strip()
            content = _THINK_RE.sub("", content).strip()
        return content, reasoning
