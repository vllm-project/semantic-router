"""Inspect and exercise routing without depending on Dashboard."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import click
import requests

from cli.chat_client import chat_completions_url, resolve_chat_base_url
from cli.commands.common import exit_with_logged_error
from cli.commands.eval_rendering import render_route_preview_summary
from cli.router_management_client import RouterManagementClient
from cli.terminal import echo
from cli.url_display import redact_url
from cli.utils import get_logger

log = get_logger(__name__)

_MAX_SIGNAL_DISPLAY = 10
_MAX_CONFIDENCE_DISPLAY = 5


@dataclass(frozen=True)
class RoutePreviewRequest:
    """Request payload for /api/v1/routing/preview."""

    messages: list[dict[str, Any]]
    model: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload = {"messages": self.messages}
        if self.model:
            payload["model"] = self.model
        return payload


def _parse_messages_json(messages_json: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(messages_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --messages JSON: {exc}") from exc

    if not isinstance(value, list):
        raise ValueError("--messages must be a JSON array of message objects")

    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(
                f"--messages[{idx}] must be an object, got {type(item).__name__}"
            )

    return value


def _prompt_to_messages(prompt: str) -> list[dict[str, Any]]:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("--prompt must be non-empty")
    return [{"role": "user", "content": prompt}]


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
    if requested_model := payload.get("requested_model"):
        lines.append(f"model: {requested_model}")
    if recipe := payload.get("recipe"):
        lines.append(f"recipe: {recipe}")
    lines.append(f"decision: {decision_result.get('decision_name') or '(none)'}")
    if algorithm := decision_result.get("algorithm"):
        lines.append(f"algorithm: {algorithm}")
    if plugins := decision_result.get("plugins"):
        lines.append(f"plugins: {', '.join(str(plugin) for plugin in plugins)}")

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


def _summarize_response(payload: dict[str, Any]) -> str:
    """Render the current routing-preview response contract."""
    if not isinstance(payload, dict):
        return json.dumps(payload, indent=2, ensure_ascii=False)

    decision_result = payload.get("decision_result")
    if isinstance(decision_result, dict):
        lines = _summarize_decision_result(payload, decision_result)
        if lines:
            return "\n".join(lines)

    return json.dumps(payload, indent=2, ensure_ascii=False)


@click.group()
def route() -> None:
    """Preview routing decisions or probe the routed inference path."""


@route.command("preview")
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
    "--model",
    default=None,
    help="Routing model or entrypoint whose recipe should be evaluated.",
)
@click.option(
    "--endpoint",
    default=None,
    help="Router management origin or /api/v1 root; defaults to the local management port.",
)
@click.option(
    "--token-env",
    default="VSR_MGMT_TOKEN",
    show_default=True,
    help="Environment variable containing the Router management bearer token.",
)
@click.option(
    "--trace/--no-trace",
    default=False,
    help="Include per-decision routing trace trees.",
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
    type=float,
    default=15.0,
    show_default=True,
    help="HTTP request timeout in seconds.",
)
@exit_with_logged_error(log)
def preview(
    prompt: str | None,
    messages_json: str | None,
    model: str | None,
    endpoint: str | None,
    token_env: str,
    trace: bool,
    output_json: bool,
    timeout: float,
) -> None:
    """Preview signals and the selected route without calling a model backend."""

    if (prompt is None and messages_json is None) or (
        prompt is not None and messages_json is not None
    ):
        raise ValueError("Provide exactly one of --prompt or --messages")

    if messages_json is not None:
        messages = _parse_messages_json(messages_json)
    else:
        messages = _prompt_to_messages(prompt or "")

    req = RoutePreviewRequest(messages=messages, model=(model or "").strip() or None)
    payload = (
        RouterManagementClient(
            endpoint,
            timeout=timeout,
            token_env=token_env,
        )
        .preview_route(req.to_json(), trace=trace)
        .payload
    )

    if output_json:
        echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    render_route_preview_summary(_summarize_response(payload))


_ROUTING_RECEIPT_HEADERS = (
    "x-request-id",
    "x-vsr-replay-id",
    "x-vsr-selected-recipe",
    "x-vsr-selected-decision",
    "x-vsr-selected-confidence",
    "x-vsr-selected-algorithm",
    "x-vsr-selected-model",
    "x-vsr-response-path",
)


def _response_body(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _routing_receipt_headers(response: requests.Response) -> dict[str, str]:
    return {
        name: value
        for name in _ROUTING_RECEIPT_HEADERS
        if (value := response.headers.get(name))
    }


def _probe_assertions(
    *,
    response: requests.Response,
    response_body: Any,
    expected_status: int,
    expected_recipe: str | None,
    expected_decision: str | None,
    expected_algorithm: str | None,
    expected_selected_model: str | None,
    expected_response_model: str | None,
) -> list[dict[str, Any]]:
    assertions: list[dict[str, Any]] = [
        {
            "field": "status",
            "expected": expected_status,
            "actual": response.status_code,
            "passed": response.status_code == expected_status,
        }
    ]
    expectations = {
        "x-vsr-selected-recipe": expected_recipe,
        "x-vsr-selected-decision": expected_decision,
        "x-vsr-selected-algorithm": expected_algorithm,
        "x-vsr-selected-model": expected_selected_model,
    }
    for header, expected in expectations.items():
        if expected is None:
            continue
        actual = response.headers.get(header, "")
        assertions.append(
            {
                "field": header,
                "expected": expected,
                "actual": actual,
                "passed": actual == expected,
            }
        )
    if expected_response_model is not None:
        actual = (
            response_body.get("model", "") if isinstance(response_body, dict) else ""
        )
        assertions.append(
            {
                "field": "response.body.model",
                "expected": expected_response_model,
                "actual": actual,
                "passed": actual == expected_response_model,
            }
        )
    return assertions


@route.command("probe")
@click.option("--prompt", default=None, help="Plain-text user prompt.")
@click.option(
    "--messages",
    "messages_json",
    default=None,
    help="OpenAI-style messages JSON array string.",
)
@click.option("--model", default="vllm-sr/auto", show_default=True)
@click.option("--config", default="config.yaml", show_default=True)
@click.option(
    "--base-url",
    default=None,
    help=(
        "Explicit Envoy listener origin or OpenAI /v1 base URL; otherwise "
        "derive it from --config."
    ),
)
@click.option(
    "--api-key-env",
    default="OPENAI_API_KEY",
    show_default=True,
    help="Environment variable containing the bearer token; omitted when unset.",
)
@click.option("--temperature", type=float, default=None)
@click.option("--timeout", type=float, default=120.0, show_default=True)
@click.option(
    "--target", default=None, help="Deployment target used for URL resolution."
)
@click.option("--debug/--no-debug", default=True, show_default=True)
@click.option("--expect-status", type=int, default=HTTPStatus.OK, show_default=True)
@click.option("--expect-recipe", default=None)
@click.option("--expect-decision", default=None)
@click.option("--expect-algorithm", default=None)
@click.option(
    "--expect-selected-model",
    default=None,
    help="Assert the Router's x-vsr-selected-model receipt header.",
)
@click.option(
    "--expect-response-model",
    default=None,
    help="Assert the upstream OpenAI response body's top-level model field.",
)
@exit_with_logged_error(log)
def probe(
    prompt: str | None,
    messages_json: str | None,
    model: str,
    config: str,
    base_url: str | None,
    api_key_env: str,
    temperature: float | None,
    timeout: float,
    target: str | None,
    debug: bool,
    expect_status: int,
    expect_recipe: str | None,
    expect_decision: str | None,
    expect_algorithm: str | None,
    expect_selected_model: str | None,
    expect_response_model: str | None,
) -> None:
    """Send one real routed request and emit a machine-readable evidence receipt."""

    if (prompt is None) == (messages_json is None):
        raise ValueError("Provide exactly one of --prompt or --messages")
    messages = (
        _parse_messages_json(messages_json)
        if messages_json is not None
        else _prompt_to_messages(prompt or "")
    )
    base = resolve_chat_base_url(
        config_path=config,
        target=target,
        base_url=base_url,
    )
    url = chat_completions_url(base)
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if debug:
        headers["x-vsr-debug"] = "true"
    if token := os.getenv(api_key_env, "").strip():
        headers["Authorization"] = f"Bearer {token}"

    started = time.monotonic()
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise ValueError(
            f"Failed to probe routed endpoint {redact_url(url)}: {exc}"
        ) from exc
    latency_ms = round((time.monotonic() - started) * 1000, 3)
    response_body = _response_body(response)
    assertions = _probe_assertions(
        response=response,
        response_body=response_body,
        expected_status=expect_status,
        expected_recipe=expect_recipe,
        expected_decision=expect_decision,
        expected_algorithm=expect_algorithm,
        expected_selected_model=expect_selected_model,
        expected_response_model=expect_response_model,
    )
    passed = all(assertion["passed"] for assertion in assertions)
    receipt = {
        "schema_version": "vllm-sr.route-probe.v1",
        "passed": passed,
        "request": {
            "url": redact_url(url),
            "model": model,
            "message_count": len(messages),
        },
        "response": {
            "status": response.status_code,
            "latency_ms": latency_ms,
            "routing": _routing_receipt_headers(response),
            "body": response_body,
        },
        "assertions": assertions,
    }
    click.echo(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True))
    if not passed:
        raise click.exceptions.Exit(2)
