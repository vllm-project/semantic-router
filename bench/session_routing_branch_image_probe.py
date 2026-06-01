#!/usr/bin/env python3
"""Probe a running branch-image router stack for GA diagnostic readiness."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HTTP_OK = 200
HTTP_REDIRECT_START = 300
REQUIRED_DIAGNOSTIC_HEADERS = (
    "x-vsr-selected-model",
    "x-vsr-selected-decision",
    "x-vsr-replay-id",
    "x-vsr-session-phase",
    "x-vsr-selected-confidence",
    "x-vsr-context-token-count",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8899/v1")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--session-header", default="x-session-id")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--ref", default="")
    parser.add_argument("--image-tag", default="")
    parser.add_argument("--label", default="branch-image")
    parser.add_argument("--extra-header", action="append", default=[])
    parser.add_argument(
        "--required-header",
        action="append",
        default=list(REQUIRED_DIAGNOSTIC_HEADERS),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path(".agent-harness/experiments/branch-image-diagnostic") / stamp


def parse_extra_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"extra header must be NAME:VALUE, got {value!r}")
        name, raw = value.split(":", 1)
        headers[name.strip()] = raw.strip()
    return headers


def request_headers(args: argparse.Namespace) -> dict[str, str]:
    session_id = args.session_id or f"branch-image-probe-{int(time.time())}"
    headers = {
        "Content-Type": "application/json",
        args.session_header: session_id,
    }
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    headers.update(parse_extra_headers(args.extra_header))
    return headers


def chat_body(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": "Return one short sentence for a routing diagnostics probe.",
            }
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }


def send_chat(args: argparse.Namespace) -> dict[str, Any]:
    url = args.base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(chat_body(args)).encode("utf-8"),
        headers=request_headers(args),
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
            return response_record(
                int(response.status), dict(response.headers), payload, started
            )
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        return response_record(int(exc.code), dict(exc.headers), payload, started)
    except Exception as exc:  # pragma: no cover - platform network errors vary
        return {
            "status": 0,
            "headers": {},
            "json": {},
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": str(exc),
        }


def response_record(
    status: int, headers: dict[str, str], payload: str, started: float
) -> dict[str, Any]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {}
    return {
        "status": status,
        "headers": normalize_headers(headers),
        "json": parsed if isinstance(parsed, dict) else {},
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "error": (
            error_message(parsed, payload) if status >= HTTP_REDIRECT_START else ""
        ),
    }


def normalize_headers(headers: dict[str, str]) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items()}


def error_message(parsed: Any, payload: str) -> str:
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        return str(error.get("message", ""))
    if isinstance(error, str):
        return error
    return payload[:300]


def validate_header_value(header: str, value: str) -> bool:
    if header == "x-vsr-selected-confidence":
        try:
            parsed = float(value)
        except ValueError:
            return False
        return 0.0 <= parsed <= 1.0
    if header == "x-vsr-context-token-count":
        try:
            parsed = int(value)
        except ValueError:
            return False
        return parsed >= 0
    if header == "x-vsr-session-phase":
        return value.strip() in {
            "user_turn",
            "tool_loop",
            "provider_state",
        }
    return bool(value.strip())


def summarize(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    required = list(dict.fromkeys(header.lower() for header in args.required_header))
    headers = result.get("headers") or {}
    missing = [
        header for header in required if not str(headers.get(header, "")).strip()
    ]
    invalid = [
        header
        for header in required
        if header not in missing
        and not validate_header_value(header, str(headers[header]))
    ]
    status = int(result.get("status") or 0)
    success = HTTP_OK <= status < HTTP_REDIRECT_START
    validation_failures: list[str] = []
    if not success:
        validation_failures.append(f"chat status {status} is not successful")
    validation_failures.extend(
        f"missing diagnostic header: {header}" for header in missing
    )
    validation_failures.extend(
        f"invalid diagnostic header: {header}" for header in invalid
    )
    response_json = result.get("json") or {}
    return {
        "validation_kind": "branch-image-diagnostic-probe",
        "label": args.label,
        "ref": args.ref,
        "image_tag": args.image_tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": {
            "chat_completion_ok": success,
            "diagnostic_headers_ok": not missing and not invalid,
        },
        "chat": {
            "status": status,
            "latency_ms": result.get("latency_ms"),
            "response_model": response_json.get("model", ""),
            "usage": response_json.get("usage", {}),
            "error": result.get("error", ""),
        },
        "diagnostic_headers": {header: headers.get(header, "") for header in required},
        "missing_diagnostic_headers": missing,
        "invalid_diagnostic_headers": invalid,
        "validation_failures": validation_failures,
    }


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "summary.md").write_text(render_markdown(summary))


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Branch Image Diagnostic Probe",
        "",
        f"- label: {summary['label']}",
        f"- ref: {summary['ref'] or 'unspecified'}",
        f"- image_tag: {summary['image_tag'] or 'unspecified'}",
        f"- chat_completion_ok: {summary['checks']['chat_completion_ok']}",
        f"- diagnostic_headers_ok: {summary['checks']['diagnostic_headers_ok']}",
        f"- validation_failures: {summary['validation_failures']}",
        "",
        "| Header | Value |",
        "| --- | --- |",
    ]
    for header, value in summary["diagnostic_headers"].items():
        lines.append(f"| `{header}` | `{value}` |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or default_output_dir()
    result = send_chat(args)
    summary = summarize(args, result)
    write_outputs(summary, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "validation_failures": summary["validation_failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if summary["validation_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
