#!/usr/bin/env python3
"""Check whether a provider's replies will get through vLLM Semantic Router.

The router reads provider replies strictly: a field it does not know turns the
whole reply into a 502. This script calls the provider directly, once plainly
and once with a tool, and lists every field the router would reject.

    python check_provider.py --base-url https://api.x.ai/v1 --api-key-env XAI_API_KEY --model grok-4.20-0309-non-reasoning

Field lists match src/semantic-router/pkg/protocolcodec with the xAI/Groq field
fix (branch fix/provider-usage-fields). Router images built before that fix
still reject x_groq and the xAI/Groq usage fields. PASS is a strong signal,
not a guarantee.
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

ALLOWED = {
    "response": {
        "id",
        "object",
        "created",
        "model",
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
    "tool_call": {"id", "type", "function", "custom"},
    "function": {"name", "arguments", "TokenizedArguments"},
    "usage": {
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
    },
    "prompt_tokens_details": {
        "cached_tokens",
        "cache_write_tokens",
        "audio_tokens",
        "text_tokens",
        "image_tokens",
    },
    "completion_tokens_details": {
        "accepted_prediction_tokens",
        "audio_tokens",
        "reasoning_tokens",
        "text_tokens",
        "rejected_prediction_tokens",
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

    def unknown(obj: Any, level: str, path: str) -> None:
        if isinstance(obj, dict):
            found.extend(
                f"{path}.{key} is a field the router does not accept"
                for key in sorted(set(obj) - ALLOWED[level])
            )

    unknown(body, "response", "response")
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
    if fingerprint is not None and not 1 <= len(str(fingerprint)) <= MAX_FINGERPRINT_CHARS:
        found.append(
            "response.system_fingerprint must be 1 to 256 characters when present"
        )
    for i, choice in enumerate(body.get("choices") or []):
        unknown(choice, "choice", f"choices[{i}]")
        message = choice.get("message") if isinstance(choice, dict) else None
        unknown(message, "message", f"choices[{i}].message")
        for j, call in enumerate((message or {}).get("tool_calls") or []):
            unknown(call, "tool_call", f"choices[{i}].message.tool_calls[{j}]")
            unknown(
                call.get("function") if isinstance(call, dict) else None,
                "function",
                f"choices[{i}].message.tool_calls[{j}].function",
            )
    usage = body.get("usage")
    if isinstance(usage, dict):
        unknown(usage, "usage", "usage")
        unknown(
            usage.get("prompt_tokens_details"),
            "prompt_tokens_details",
            "usage.prompt_tokens_details",
        )
        unknown(
            usage.get("completion_tokens_details"),
            "completion_tokens_details",
            "usage.completion_tokens_details",
        )
    return found


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
