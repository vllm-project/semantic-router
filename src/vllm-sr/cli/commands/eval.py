"""Prompt evaluation command for vLLM Semantic Router.

This command calls the router evaluation endpoint (POST /api/v1/eval) and prints
signal evaluation results.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import click
import requests

from cli.commands.common import exit_with_logged_error
from cli.consts import DEFAULT_API_PORT
from cli.utils import get_logger

log = get_logger(__name__)

_MAX_SIGNAL_DISPLAY = 10
_MAX_CONFIDENCE_DISPLAY = 5
_HTTP_FORBIDDEN = 403


@dataclass(frozen=True)
class EvalRequest:
    """Request payload for /api/v1/eval."""

    messages: list[dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        # The API endpoint forces evaluate_all_signals=true server-side, but it
        # does not hurt to be explicit.
        return {
            "messages": self.messages,
            "evaluate_all_signals": True,
        }


def _default_endpoint() -> str:
    # Support port offset via environment variable (set during vllm-sr serve)
    port_offset = int(os.getenv("VLLM_SR_PORT_OFFSET", "0"))
    api_port = DEFAULT_API_PORT + port_offset
    return f"http://localhost:{api_port}/api/v1/eval"


def _normalize_endpoint(endpoint: str) -> str:
    """Normalize endpoint so we always end up calling /api/v1/eval."""

    endpoint = endpoint.strip()
    if not endpoint:
        return _default_endpoint()

    # If user passes a base URL (e.g. http://localhost:8080), append path.
    if endpoint.endswith("/"):
        endpoint = endpoint[:-1]

    if endpoint.endswith("/api/v1/eval"):
        return endpoint

    # Handle passing /api/v1 or /api
    if endpoint.endswith("/api/v1"):
        return endpoint + "/eval"

    return urljoin(endpoint + "/", "api/v1/eval")


def _parse_messages_json(messages_json: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(messages_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --messages JSON: {exc}") from exc

    if not isinstance(value, list):
        raise ValueError("--messages must be a JSON array of message objects")

    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(
                f"--messages[{idx}] must be an object, got {type(item).__name__}"
            )

    return value


def _prompt_to_messages(prompt: str) -> list[dict[str, Any]]:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("--prompt must be non-empty")
    return [{"role": "user", "content": prompt}]


def _format_error_response(resp: Any) -> str:
    """Extract a clean error message from a non-200 router response.

    The router returns structured JSON errors:
        {"error": {"code": "INVALID_INPUT", "message": "...", "timestamp": "..."}}
    Fall back to raw text if the body is not that shape.
    """
    try:
        body = resp.json()
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            err = body["error"]
            code = err.get("code", "")
            message = err.get("message", "")
            if code and message:
                return f"Router returned {resp.status_code} {code}: {message}"
            if message:
                return f"Router returned {resp.status_code}: {message}"
    except ValueError:
        pass
    content_type = getattr(resp, "headers", {}).get("Content-Type", "")
    if resp.status_code == _HTTP_FORBIDDEN and "text/html" in content_type:
        url = getattr(resp, "url", "the endpoint")
        return (
            f"Router returned 403 from {url}. "
            "This looks like a proxy or gateway — check that --endpoint points directly "
            "to the router API port (default: 8080), not to Envoy or another proxy."
        )
    return f"Router eval request failed: HTTP {resp.status_code} - {resp.text}"


def _count_grouped_signals(signals: Any) -> int:
    """Count total signals in a grouped structure (by signal type)."""
    if not isinstance(signals, dict):
        return 0
    return sum(
        len(sig_list)
        for sig_list in signals.values()
        if isinstance(sig_list, (list, dict))
    )


def _append_grouped_signals(
    lines: list[str], signals: Any, limit: int = _MAX_CONFIDENCE_DISPLAY
) -> None:
    """Append grouped signals to output, limited to N items total."""
    if not isinstance(signals, dict):
        return
    count = 0
    for sig_type, sig_list in signals.items():
        if not isinstance(sig_list, (list, dict)):
            continue
        items = sig_list if isinstance(sig_list, list) else list(sig_list.keys())
        for sig_name in items[:limit]:
            lines.append(f"  - {sig_type}:{sig_name}")
            count += 1
            if count >= limit:
                return


def _summarize_used_signals(lines: list[str], used_signals: Any) -> None:
    """Append used-signals block to lines."""
    if isinstance(used_signals, dict):
        total = sum(
            len(v) if isinstance(v, (list, dict)) else 1 for v in used_signals.values()
        )
        lines.append(f"used signals: {total}")
        for sig_type, sig_list in used_signals.items():
            if isinstance(sig_list, (list, dict)):
                for sig_name in (
                    sig_list if isinstance(sig_list, list) else sig_list.keys()
                ):
                    lines.append(f"  - {sig_type}:{sig_name}")
    elif isinstance(used_signals, list):
        lines.append(f"used signals: {len(used_signals)}")
        for sig_name in used_signals[:_MAX_SIGNAL_DISPLAY]:
            lines.append(f"  - {sig_name}")
        if len(used_signals) > _MAX_SIGNAL_DISPLAY:
            lines.append(f"  ... and {len(used_signals) - _MAX_SIGNAL_DISPLAY} more")


def _summarize_signal_confidences(
    lines: list[str], signal_confidences: dict[str, float]
) -> None:
    """Append top signal confidences to lines."""
    lines.append("signal confidences:")
    top = sorted(signal_confidences.items(), key=lambda x: -x[1])[
        :_MAX_CONFIDENCE_DISPLAY
    ]
    for sig_key, confidence in top:
        lines.append(f"  - {sig_key}: {confidence:.2f}")
    if len(signal_confidences) > _MAX_CONFIDENCE_DISPLAY:
        lines.append(
            f"  ... and {len(signal_confidences) - _MAX_CONFIDENCE_DISPLAY} more"
        )


def _summarize_decision_result(
    payload: dict[str, Any], decision_result: dict[str, Any]
) -> list[str]:
    """Build summary lines for the decision_result (current EvalResponse format)."""
    lines: list[str] = []
    lines.append(f"decision: {decision_result.get('decision_name') or '(none)'}")

    used_signals = decision_result.get("used_signals", {})
    if used_signals:
        _summarize_used_signals(lines, used_signals)

    matched = decision_result.get("matched_signals", {})
    unmatched = decision_result.get("unmatched_signals", {})
    matched_count = _count_grouped_signals(matched)
    unmatched_count = _count_grouped_signals(unmatched)

    if matched_count > 0:
        lines.append(f"matched signals: {matched_count}")
        _append_grouped_signals(lines, matched)
    if unmatched_count > 0:
        lines.append(f"unmatched signals: {unmatched_count}")

    signal_confidences = payload.get("signal_confidences") or {}
    if signal_confidences:
        _summarize_signal_confidences(lines, signal_confidences)

    routing = payload.get("routing_decision")
    if routing:
        lines.append(f"routing: {routing}")

    return lines


def _summarize_legacy_signal_item(name: str, item: Any) -> str:
    """Format a single legacy signal entry."""
    if not isinstance(item, dict):
        return f"- {name}: {item}"
    extras = []
    if item.get("score") is not None:
        extras.append(f"score={item['score']}")
    if item.get("fired") is not None:
        extras.append(f"fired={item['fired']}")
    suffix = (" " + ", ".join(extras)) if extras else ""
    return f"- {name}{suffix}"


def _summarize_legacy_signals(signals: Any) -> list[str] | None:
    """Build summary lines for the legacy signals format, or None if not applicable."""
    if isinstance(signals, list):
        lines = [f"signals: {len(signals)}"]
        for item in signals:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("signal") or "<unknown>"
            lines.append(_summarize_legacy_signal_item(name, item))
        return lines
    if isinstance(signals, dict):
        lines = [f"signals: {len(signals)}"]
        for name, item in signals.items():
            lines.append(_summarize_legacy_signal_item(name, item))
        return lines
    return None


def _summarize_response(payload: dict[str, Any]) -> str:
    """Best-effort human-readable summary.

    The exact router response schema can evolve; we should be robust.
    """
    if not isinstance(payload, dict):
        return json.dumps(payload, indent=2, ensure_ascii=False)

    decision_result = payload.get("decision_result")
    if isinstance(decision_result, dict):
        lines = _summarize_decision_result(payload, decision_result)
        if lines:
            return "\n".join(lines)

    # chat.completion format (legacy / pass-through)
    if payload.get("object") == "chat.completion":
        lines = ["Evaluation successful"]
        if payload.get("model"):
            lines.append(f"model: {payload['model']}")
        tokens = (payload.get("usage") or {}).get("total_tokens", 0)
        if tokens > 0:
            lines.append(f"tokens: {tokens}")
        return "\n".join(lines)

    # Legacy signals field
    legacy = _summarize_legacy_signals(payload.get("signals"))
    if legacy is not None:
        return "\n".join(legacy)

    return json.dumps(payload, indent=2, ensure_ascii=False)


@click.command()
@click.option(
    "--prompt",
    default=None,
    help="Plain text prompt to evaluate.",
)
@click.option(
    "--messages",
    "messages_json",
    default=None,
    help="OpenAI-style messages JSON array string.",
)
@click.option(
    "--endpoint",
    default=None,
    help=(
        "Router base URL or full eval endpoint. " f"Defaults to {_default_endpoint()}."
    ),
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Print the full JSON response payload.",
)
@click.option(
    "--timeout",
    default=15,
    show_default=True,
    help="HTTP request timeout in seconds.",
)
@exit_with_logged_error(log)
def eval(
    prompt: str | None,
    messages_json: str | None,
    endpoint: str | None,
    output_json: bool,
    timeout: int,
) -> None:
    """Evaluate a prompt/messages against the router /api/v1/eval endpoint."""

    if (prompt is None and messages_json is None) or (
        prompt is not None and messages_json is not None
    ):
        raise ValueError("Provide exactly one of --prompt or --messages")

    if messages_json is not None:
        messages = _parse_messages_json(messages_json)
    else:
        messages = _prompt_to_messages(prompt or "")

    url = _normalize_endpoint(endpoint or "")

    req = EvalRequest(messages=messages)

    try:
        resp = requests.post(url, json=req.to_json(), timeout=timeout)
    except requests.ConnectionError as exc:
        raise ValueError(
            f"Router is not running at {url}. Start the router with 'vllm-sr serve' and retry."
        ) from exc
    except requests.Timeout as exc:
        raise ValueError(
            f"Request to {url} timed out after {timeout}s. Is the router healthy?"
        ) from exc
    except requests.RequestException as exc:
        raise ValueError(f"Failed to call router eval endpoint {url}: {exc}") from exc

    if resp.status_code != requests.codes.ok:
        raise ValueError(_format_error_response(resp))

    try:
        payload = resp.json()
    except ValueError as exc:
        raise ValueError(f"Router returned non-JSON response: {resp.text}") from exc

    if output_json:
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    click.echo(_summarize_response(payload))
