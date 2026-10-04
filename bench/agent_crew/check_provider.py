#!/usr/bin/env python3
"""Check whether a provider's replies will get through vLLM Semantic Router.

The router reads provider replies strictly: a field it does not know turns the
whole reply into a 502. This script calls the provider directly, once plainly
and once with a tool, and lists every field the router would reject.

    python check_provider.py --base-url https://api.x.ai/v1 --api-key-env XAI_API_KEY --model grok-4.20-0309-non-reasoning

Field lists match the canonical chat response contract in
src/semantic-router/pkg/protocolcodec, whose per-object field names are listed in
that package's testdata/contracts/chat-nested-fields.json. A provider vendor
policy relaxes this boundary, so a field named here can still get through when
the router is configured for that vendor. PASS is a strong signal, not a
guarantee.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

MAX_FINGERPRINT_CHARS = 256

# Go decodes a numeric stop reason into int64; a string only has to be non-empty.
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

ALLOWED = {
    "response": {
        "id",
        "object",
        "created",
        "model",
        "provider",
        "choices",
        "usage",
        "metadata",
        "moderation",
        "error",
        "service_tier",
        "system_fingerprint",
        "prompt_logprobs",
        "prompt_token_ids",
        "prompt_text",
        "prompt_routed_experts",
        "kv_transfer_params",
        "ec_transfer_params",
        "metrics",
        "do_remote_decode",
        "do_remote_prefill",
        "remote_block_ids",
        "remote_engine_id",
        "remote_host",
        "remote_port",
        "usage_breakdown",
        "x_groq",
    },
    "choice": {
        "index",
        "message",
        "finish_reason",
        "native_finish_reason",
        "logprobs",
        "stop_reason",
        "token_ids",
        "routed_experts",
    },
    "message": {
        "id",
        "role",
        "content",
        "refusal",
        "reasoning_content",
        "reasoning",
        "audio",
        "function_call",
        "tool_calls",
        "tool_call_id",
        "annotations",
        "name",
    },
    "tool_call": {"id", "type", "function", "custom", "index"},
    "function": {"name", "arguments", "TokenizedArguments"},
    "custom_call": {"name", "input"},
    "audio": {"id", "data", "transcript", "expires_at"},
    "legacy_function_call": {"name", "arguments"},
    "annotation": {"type", "url_citation"},
    "url_citation": {"url", "title", "start_index", "end_index"},
    "logprobs": {"content", "refusal"},
    "token_logprob": {"token", "logprob", "bytes", "top_logprobs"},
    "top_token_logprob": {"token", "logprob", "bytes"},
    "error": {"message", "type", "param", "code"},
    # The only supported kv-transfer marker is an empty object.
    "kv_transfer_params": set(),
    "usage": {
        "service_tier",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "compute_units",
        "prompt_tokens_details",
        "completion_tokens_details",
        "cost_in_usd_ticks",
        "num_sources_used",
        "queue_time",
        "prompt_time",
        "completion_time",
        "total_time",
        "cost",
        "is_byok",
        "cost_details",
        "server_tool_use",
    },
    "cost_details": {
        "upstream_inference_cost",
        "upstream_inference_prompt_cost",
        "upstream_inference_completions_cost",
        "server_tool_cost",
    },
    "server_tool_use": {"web_search_requests"},
    "prompt_tokens_details": {
        "cached_tokens",
        "cache_write_tokens",
        "created_cache_tokens",
        "cache_creation_tokens",
        "multimodal_tokens",
        "audio_tokens",
        "text_tokens",
        "image_tokens",
        "video_tokens",
    },
    "completion_tokens_details": {
        "accepted_prediction_tokens",
        "audio_tokens",
        "reasoning_tokens",
        "text_tokens",
        "image_tokens",
        "rejected_prediction_tokens",
    },
}

# Nested objects the codec also closes, as field name to (shape, schema).
CHILDREN = {
    "response": {
        "choices": ("array", "choice"),
        "usage": ("object", "usage"),
        "error": ("object", "error"),
        "kv_transfer_params": ("object", "kv_transfer_params"),
    },
    "choice": {"message": ("object", "message"), "logprobs": ("object", "logprobs")},
    "message": {
        "tool_calls": ("array", "tool_call"),
        "audio": ("object", "audio"),
        "function_call": ("object", "legacy_function_call"),
        "annotations": ("array", "annotation"),
    },
    "tool_call": {
        "function": ("object", "function"),
        "custom": ("object", "custom_call"),
    },
    "annotation": {"url_citation": ("object", "url_citation")},
    "logprobs": {
        "content": ("array", "token_logprob"),
        "refusal": ("array", "token_logprob"),
    },
    "token_logprob": {"top_logprobs": ("array", "top_token_logprob")},
    "usage": {
        "prompt_tokens_details": ("object", "prompt_tokens_details"),
        "completion_tokens_details": ("object", "completion_tokens_details"),
        "cost_details": ("object", "cost_details"),
        "server_tool_use": ("object", "server_tool_use"),
    },
}
NULL_ONLY = {
    "prompt_logprobs",
    "prompt_text",
    "prompt_routed_experts",
    "ec_transfer_params",
    "metrics",
}
SERVICE_TIERS = {
    "auto",
    "default",
    "flex",
    "priority",
    "scale",
    "on_demand",
    "performance",
}
# Mistral reports the served tier inside usage, with its own value set.
USAGE_SERVICE_TIERS = {"standard", "priority"}
TOOL = {
    "type": "function",
    "function": {
        "name": "list_changed_files",
        "description": "List the files changed in the pull request.",
        "parameters": {"type": "object", "properties": {}},
    },
}
CASES = (
    (
        "plain reply",
        {"messages": [{"role": "user", "content": "Reply with the single word: ok"}]},
    ),
    (
        "tool call",
        {
            "messages": [
                {"role": "user", "content": "Which files changed? Use the tool."}
            ],
            "tools": [TOOL],
        },
    ),
)


def problems(body: dict[str, Any]) -> list[str]:
    found: list[str] = []

    def walk(obj: Any, schema: str, path: str) -> None:
        if not isinstance(obj, dict):
            return
        found.extend(
            f"{path}.{key} is a field the router does not accept"
            for key in sorted(set(obj) - ALLOWED[schema])
        )
        for key, (shape, child) in CHILDREN.get(schema, {}).items():
            value = obj.get(key)
            if shape == "object":
                walk(value, child, f"{path}.{key}")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    walk(item, child, f"{path}.{key}[{i}]")

    walk(body, "response", "response")
    found.extend(
        f"response.{key} must be null"
        for key in sorted(NULL_ONLY)
        if body.get(key) is not None
    )
    if (
        body.get("service_tier") is not None
        and body["service_tier"] not in SERVICE_TIERS
    ):
        found.append(
            f"response.service_tier {body['service_tier']!r} is not one the router accepts"
        )
    fingerprint = body.get("system_fingerprint")
    if (
        fingerprint is not None
        and not 1 <= len(str(fingerprint)) <= MAX_FINGERPRINT_CHARS
    ):
        found.append(
            "response.system_fingerprint must be 1 to 256 characters when present"
        )
    for i, choice in enumerate(body.get("choices") or []):
        if not isinstance(choice, dict):
            continue
        if choice.get("routed_experts") is not None:
            found.append(f"choices[{i}].routed_experts must be null")
        found.extend(stop_reason_problems(choice.get("stop_reason"), i))
    usage = body.get("usage")
    if isinstance(usage, dict):
        tier = usage.get("service_tier")
        if tier is not None and tier not in USAGE_SERVICE_TIERS:
            found.append(f"usage.service_tier {tier!r} is not one the router accepts")
    return found


def stop_reason_problems(reason: Any, index: int) -> list[str]:
    """The codec accepts a non-empty string and decodes numbers as int64."""
    if reason is None:
        return []
    if isinstance(reason, int) and not isinstance(reason, bool):
        if INT64_MIN <= reason <= INT64_MAX:
            return []
    elif isinstance(reason, str) and reason:
        return []
    return [
        f"choices[{index}].stop_reason must be a signed 64-bit integer or a non-empty string"
    ]


def call(
    base_url: str, api_key: str, payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        # Cloudflare-fronted APIs (Groq) block urllib's default user agent: error 1010.
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
            "user-agent": "vllm-sr-check-provider/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--api-key-env",
        required=True,
        help="name of the environment variable holding the key",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(
            f"{args.api_key_env} is not set. Export it in this terminal first.",
            file=sys.stderr,
        )
        return 2
    failed = refused = False
    for name, extra in CASES:
        payload = {"model": args.model, "max_tokens": args.max_tokens, **extra}
        try:
            body = call(args.base_url, api_key, payload, args.timeout)
        except urllib.error.HTTPError as exc:
            print(
                f"\n{name}: the provider itself returned HTTP {exc.code}: {exc.read()[:300].decode(errors='replace')}"
            )
            refused = True
            continue
        found = problems(body)
        usage = body.get("usage") or {}
        print(f"\n{name}: {'PASS' if not found else 'FAIL'}")
        for item in found:
            print(f"  - {item}")
        print(
            f"  tokens: prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}"
        )
        print(
            f"  service_tier: {body.get('service_tier')!r}  system_fingerprint: {body.get('system_fingerprint')!r}"
        )
        if name == "tool call" and not (
            (body.get("choices") or [{}])[0].get("message") or {}
        ).get("tool_calls"):
            print(
                "  note: the model answered without calling the tool; the crew needs tool calls"
            )
        failed = failed or bool(found)
    if failed:
        print("\nVerdict: FAIL, the router would reply 502 for this provider")
        print(
            "  A provider configured with a matching vendor drops its own extra\n"
            "  fields instead of rejecting them, so check whether this provider\n"
            "  has one before ruling it out."
        )
        return 1
    if refused:
        print(
            "\nVerdict: NOT CHECKED, the provider refused the request (key, model name or quota)"
        )
        return 2
    print("\nVerdict: PASS, safe to add to the router config")
    return 0


if __name__ == "__main__":
    sys.exit(main())
