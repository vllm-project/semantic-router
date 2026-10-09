"""Async client for the OpenAI-compatible vLLM servers, with a resumable request cache.

Every request is keyed by the sha256 of its payload; answers are appended to a JSONL cache, so a
restarted shard replays finished calls instead of re-sending them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import httpx


def payload_key(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class RequestCache:
    """Append-only JSONL map from request key to response record."""

    def __init__(self, path: Path | None):
        self.path = path
        self.data: dict[str, dict] = {}
        if path and path.exists():
            with open(path, encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.data[record["key"]] = record["value"]
        self.stream = open(path, "a", encoding="utf-8") if path else None

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, value: dict) -> None:
        self.data[key] = value
        if self.stream:
            self.stream.write(
                json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n"
            )
            self.stream.flush()

    def close(self) -> None:
        if self.stream:
            self.stream.close()


class Stats:
    def __init__(self):
        self.calls = 0
        self.cached = 0
        self.errors = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.started = time.time()

    def snapshot(self) -> dict:
        elapsed = max(time.time() - self.started, 1e-6)
        return {
            "calls": self.calls,
            "cached": self.cached,
            "errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_s": round(elapsed, 1),
            "completion_tok_per_s": round(self.completion_tokens / elapsed, 1),
            "prompt_tok_per_s": round(self.prompt_tokens / elapsed, 1),
        }


class LLM:
    def __init__(
        self, base_url: str, model: str, concurrency: int = 256, timeout: float = 3600.0
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=60.0),
            limits=httpx.Limits(
                max_connections=concurrency + 8,
                max_keepalive_connections=concurrency + 8,
            ),
        )
        self.stats = Stats()
        self.healthy = asyncio.Event()
        self.healthy.set()
        self.health_lock = asyncio.Lock()

    async def close(self) -> None:
        await self.client.aclose()

    async def wait_healthy(self, max_wait: float = 3 * 3600) -> None:
        """Block every caller until ``/health`` answers 200 again (one poller for all coroutines)."""
        if self.health_lock.locked():
            await self.healthy.wait()
            return
        async with self.health_lock:
            self.healthy.clear()
            deadline = time.time() + max_wait
            while time.time() < deadline:
                try:
                    response = await self.client.get(
                        f"{self.base_url}/health", timeout=10.0
                    )
                    if response.status_code == 200:
                        break
                except (httpx.TransportError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(20)
            self.healthy.set()

    async def _post(self, path: str, payload: dict, retries: int = 8) -> dict:
        delay = 2.0
        for attempt in range(retries):
            await self.healthy.wait()
            try:
                async with self.semaphore:
                    response = await self.client.post(
                        f"{self.base_url}{path}", json=payload
                    )
                if response.status_code == 200:
                    return response.json()
                if response.status_code in (400, 404, 422):
                    return {
                        "error": {
                            "status": response.status_code,
                            "body": response.text[:2000],
                        }
                    }
            except (httpx.TransportError, httpx.TimeoutException, json.JSONDecodeError):
                self.stats.errors += 1
                await self.wait_healthy()
                continue
            self.stats.errors += 1
            await asyncio.sleep(delay + random.random())
            delay = min(delay * 2, 60.0)
        return {"error": {"status": "retries_exhausted"}}

    async def chat(
        self,
        messages: list[dict],
        *,
        cache: RequestCache | None = None,
        schema: dict | None = None,
        thinking: bool = False,
        budget: int | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        seed: int | None = None,
        presence_penalty: float = 0.0,
        extra: dict | None = None,
    ) -> dict:
        """Chat completion. Returns {content, reasoning, finish, usage} or {error}."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if seed is not None:
            payload["seed"] = seed
        if thinking and budget:
            payload["thinking_token_budget"] = budget
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema, "strict": True},
            }
        if extra:
            payload.update(extra)
        key = payload_key(payload)
        if cache is not None and (hit := cache.get(key)) is not None:
            self.stats.cached += 1
            return hit
        body = await self._post("/v1/chat/completions", payload)
        self.stats.calls += 1
        if "error" in body:
            return {"error": body["error"]}
        choice = body["choices"][0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}
        self.stats.prompt_tokens += usage.get("prompt_tokens", 0)
        self.stats.completion_tokens += usage.get("completion_tokens", 0)
        result = {
            "content": message.get("content") or "",
            "reasoning": message.get("reasoning_content")
            or message.get("reasoning")
            or "",
            "finish": choice.get("finish_reason"),
            "usage": {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
            },
            "model": body.get("model"),
        }
        if cache is not None and result["finish"] in ("stop", "length"):
            cache.put(key, result)
        return result

    async def code_probs(
        self, prompt: str, token_ids: list[int], *, cache: RequestCache | None = None
    ) -> dict:
        """Next-token probabilities of ``token_ids`` after a fully rendered ``prompt`` (completions API).

        Returns {probs: [p per id, renormalised], mass: total raw probability on the ids} or {error}.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 1,
            "logprob_token_ids": token_ids,
            "return_tokens_as_token_ids": True,
            "add_special_tokens": False,
        }
        key = payload_key(payload)
        if cache is not None and (hit := cache.get(key)) is not None:
            self.stats.cached += 1
            return hit
        body = await self._post("/v1/completions", payload)
        self.stats.calls += 1
        if "error" in body:
            return {"error": body["error"]}
        usage = body.get("usage") or {}
        self.stats.prompt_tokens += usage.get("prompt_tokens", 0)
        self.stats.completion_tokens += usage.get("completion_tokens", 0)
        logprobs = body["choices"][0].get("logprobs") or {}
        tops = (logprobs.get("top_logprobs") or [{}])[0] or {}
        by_id: dict[int, float] = {}
        for token, value in tops.items():
            if isinstance(token, str) and token.startswith("token_id:"):
                by_id[int(token.split(":", 1)[1])] = value
        if len(by_id) < len(token_ids):
            return {
                "error": {
                    "status": "missing_logprobs",
                    "got": len(by_id),
                    "want": len(token_ids),
                }
            }
        raw = [math.exp(by_id[t]) for t in token_ids]
        mass = sum(raw)
        result = {"probs": [v / mass for v in raw], "mass": mass}
        if cache is not None:
            cache.put(key, result)
        return result


def parse_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def endpoint(name: str, default: str) -> str:
    return os.environ.get(name, default)
