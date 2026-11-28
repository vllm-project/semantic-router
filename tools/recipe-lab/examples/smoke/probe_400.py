"""Probe the router's 400: replay sr-bench-style requests with/without
stream_options to isolate what the gateway rejects. Run inside the GPU job
next to a live router."""

from __future__ import annotations

import argparse

import requests

_PROBE_TEXT_CAP = 400  # stop reading the probe response after this many chars


def probe(
    url: str, model: str, payload_extra: dict, label: str, headers: dict | None = None
) -> None:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True,
        **payload_extra,
    }
    response = requests.post(
        url.rstrip("/") + "/chat/completions",
        json=body,
        stream=True,
        timeout=120,
        headers=headers or {},
    )
    text = ""
    for chunk in response.iter_content(chunk_size=4096, decode_unicode=True):
        text += chunk
        if len(text) > _PROBE_TEXT_CAP:
            break
    print(f"[{label}] HTTP {response.status_code}: {text[:220]!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", default="http://127.0.0.1:8899/v1")
    parser.add_argument("--model", default="vllm-sr/auto")
    parser.add_argument("--config-hash", default="")
    parser.add_argument(
        "--only-mom-headers",
        action="store_true",
        help="first request carries the mom headers",
    )
    args = parser.parse_args()
    if args.only_mom_headers:
        hash_value = args.config_hash
        singles = {
            "hdr-vsr-debug": {"X-VSR-Debug": "true"},
            "hdr-config-hash": {"X-SR-Bench-Expected-Config-Hash": hash_value},
            "hdr-max-inference": {"X-SR-Bench-Max-Inference-Calls": "12"},
        }
        for label, headers in singles.items():
            probe(
                args.router,
                args.model,
                {"stream_options": {"include_usage": True}},
                label,
                headers,
            )
        return 0
    probe(args.router, args.model, {}, "plain-stream")
    probe(
        args.router,
        args.model,
        {"stream_options": {"include_usage": True}},
        "with-stream-options",
    )
    if args.config_hash:
        probe(
            args.router,
            args.model,
            {"stream_options": {"include_usage": True}},
            "mom-headers",
            {
                "X-VSR-Debug": "true",
                "X-SR-Bench-Expected-Config-Hash": args.config_hash,
                "X-SR-Bench-Max-Inference-Calls": "12",
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
