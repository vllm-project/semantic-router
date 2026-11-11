"""What a harness needs from the Router, and one call that shows the gateway answers it.

The check reads a v0.5 configuration and reports; it never edits it, since one
gateway can serve other clients too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import requests

CLAUDE_HEADERS = {
    "session": "x-claude-code-session-id",
    "agent": "x-claude-code-agent-id",
}
CODEX_HEADERS = {"session": "session-id", "agent": "thread-id"}
MAYBE_ANTHROPIC = (
    " (if a model it can pick is served over Anthropic Messages;"
    " its api_format is not set here)"
)
MAYBE_OPENAI = (
    " (if a model is served over an OpenAI API; its api_format is not set here)"
)
# what Claude Code sends that an openai backend cannot carry as sent
OPENAI_LOSSY = ("tool_result.is_error", "messages[].role", "content.file")
PROBE_TOOL = {
    "name": "get_time",
    "description": "Return the current time.",
    "parameters": {"type": "object", "properties": {}},
}


@dataclass
class RouterFinding:
    """One thing the harness needs that the configuration lacks, and the YAML for it."""

    need: str
    snippet: str


@dataclass
class ProbeResult:
    ok: bool
    model: str = ""
    detail: str = ""
    warnings: list[str] = field(default_factory=list)


def _get(doc: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(doc, dict):
            return None
        doc = doc.get(key)
    return doc


def _protection_finding(router: dict, headers: dict[str, str]) -> RouterFinding | None:
    learning = router.get("learning") or {}
    protection = learning.get("protection") or {}
    configured = _get(protection, "identity", "headers") or {}
    if (
        learning.get("enabled") is True
        and protection.get("enabled") is True
        and protection.get("scope") == "session"
        and all(configured.get(k) == v for k, v in headers.items())
    ):
        return None
    lines = "\n".join(f"            {k}: {v}" for k, v in headers.items())
    return RouterFinding(
        "session protection on this harness's identity headers, so each session and subagent keeps its model",
        "global:\n  router:\n    learning:\n      enabled: true\n      protection:\n"
        "        enabled: true\n        scope: session\n        identity:\n          headers:\n"
        + lines,
    )


def _lossy_finding(router: dict, field_name: str, why: str) -> RouterFinding | None:
    if field_name in (router.get("allow_lossy_translation") or []):
        return None
    return RouterFinding(
        why, f"global:\n  router:\n    allow_lossy_translation: [{field_name}]"
    )


def _model_formats(doc: dict) -> dict[str, str]:
    """Each provider model's api_format, or "" when the configuration leaves it to its
    backend bindings, which only the Router resolves."""
    return {
        m["name"]: str(m.get("api_format") or "")
        for m in _get(doc, "providers", "models") or []
        if isinstance(m, dict) and m.get("name")
    }


def _decisions(doc: dict) -> list[dict]:
    return [
        decision
        for recipe in doc.get("recipes") or []
        if isinstance(recipe, dict)
        for decision in recipe.get("decisions") or []
        if isinstance(decision, dict)
    ]


def _sets_default_max_tokens(decision: dict) -> bool:
    return any(
        isinstance(plugin, dict)
        and plugin.get("type") == "request_params"
        and (plugin.get("configuration") or {}).get("default_max_tokens") is not None
        for plugin in decision.get("plugins") or []
    )


def _claude_findings(doc: dict, router: dict) -> list[RouterFinding | None]:
    findings = [
        _lossy_finding(
            router,
            "tools.defer_loading",
            "tools.defer_loading, or the Router refuses the first calls of every Claude Code session"
            " and Claude Code goes on without tool search; a Router too old to know this field"
            " refuses the configuration when it loads",
        ),
        _protection_finding(router, CLAUDE_HEADERS),
    ]
    formats = set(_model_formats(doc).values())
    allowed = router.get("allow_lossy_translation") or []
    missing = [f for f in OPENAI_LOSSY if f not in allowed]
    if missing and formats & {"openai", ""}:
        findings.append(
            RouterFinding(
                f"{', '.join(missing)}, or the Router refuses Claude Code's error tool results,"
                " system turns and files toward an openai backend"
                + ("" if "openai" in formats else MAYBE_OPENAI),
                f"global:\n  router:\n    allow_lossy_translation: [{', '.join(missing)}]",
            )
        )
    return findings


def _codex_findings(doc: dict, router: dict) -> list[RouterFinding | None]:
    formats = _model_formats(doc)
    findings: list[RouterFinding | None] = []
    # each decision that can pick an anthropic model, with a hint when none of its
    # models says so and one leaves api_format to its backend bindings
    to_anthropic = []
    for decision in _decisions(doc):
        kinds = {
            formats.get(ref.get("model"), "")
            for ref in decision.get("modelRefs") or []
            if isinstance(ref, dict)
        }
        if "anthropic" in kinds:
            to_anthropic.append((decision, ""))
        elif "" in kinds:
            to_anthropic.append((decision, MAYBE_ANTHROPIC))
    if to_anthropic:
        known = any(not maybe for _, maybe in to_anthropic)
        findings.append(
            _lossy_finding(
                router,
                "reasoning.signature",
                "reasoning.signature, or a stream from an anthropic backend ends in lossy_translation"
                " where its thinking block closes" + ("" if known else MAYBE_ANTHROPIC),
            )
        )
    for decision, maybe in to_anthropic:
        if not _sets_default_max_tokens(decision):
            findings.append(
                RouterFinding(
                    f"an output limit on decision {decision.get('name')}: Codex sends none, and an"
                    " anthropic backend then gets 4096" + maybe,
                    "plugins:\n  - type: request_params\n    configuration:\n      default_max_tokens: 32000",
                )
            )
    if _get(doc, "global", "services", "response_api", "enabled") is False:
        findings.append(
            RouterFinding(
                "the Responses service, which Codex's endpoint needs (it is on by default)",
                "global:\n  services:\n    response_api:\n      enabled: true",
            )
        )
    findings.append(_protection_finding(router, CODEX_HEADERS))
    return findings


def system_one_models(doc: dict | None) -> list[str]:
    """The System One models the first listener publishes, which vSR can ask there."""
    listeners = (doc or {}).get("listeners") or []
    first = listeners[0] if listeners else None
    return [str(m) for m in _get(first, "systemone", "models") or []]


def check_router_config(doc: dict, harness: str) -> list[RouterFinding]:
    """What this harness needs from a v0.5 Router configuration and does not find."""
    router = _get(doc, "global", "router") or {}
    found = (
        _claude_findings(doc, router)
        if harness == "claude-code"
        else _codex_findings(doc, router)
    )
    return [f for f in found if f]


# --- probes ------------------------------------------------------------------------


def _probe_request(harness: str, model: str) -> tuple[str, dict, dict]:
    if harness == "claude-code":
        return (
            "/v1/messages",
            {"anthropic-version": "2023-06-01"},
            {
                "model": model,
                "max_tokens": 256,
                "messages": [
                    {"role": "user", "content": "What time is it? Use the tool."}
                ],
                "tools": [
                    {
                        "name": PROBE_TOOL["name"],
                        "description": PROBE_TOOL["description"],
                        "input_schema": PROBE_TOOL["parameters"],
                    }
                ],
            },
        )
    return (
        "/v1/responses",
        {},
        {
            "model": model,
            "max_output_tokens": 256,
            "input": "What time is it? Use the tool.",
            "tools": [{"type": "function", **PROBE_TOOL}],
        },
    )


def _reply_of(harness: str, events: list[dict]) -> tuple[dict | None, str]:
    """The finished reply a stream carries, or None and why it is not one."""
    for event in events:
        if event.get("type") in ("error", "response.failed"):
            return None, f"the stream ended in an error: {json.dumps(event)[:300]}"
    if harness == "claude-code":
        if not any(event.get("type") == "message_stop" for event in events):
            return None, "the stream did not finish a Messages reply"
        blocks = [
            event.get("content_block") or {}
            for event in events
            if event.get("type") == "content_block_start"
        ]
        return {"content": blocks}, ""
    done = next(
        (
            event["response"]
            for event in events
            if event.get("type") in ("response.completed", "response.incomplete")
            and isinstance(event.get("response"), dict)
        ),
        None,
    )
    if done is None:
        return None, "the stream did not finish a Responses reply"
    return done, ""


def _events(response: requests.Response) -> list[dict]:
    """The JSON data of each server-sent event in the body."""
    events = []
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data:") and line[5:].strip() != "[DONE]":
            try:
                events.append(json.loads(line[5:]))
            except ValueError:
                continue
    return [event for event in events if isinstance(event, dict)]


def _called_tool(harness: str, answer: dict) -> bool:
    if harness == "claude-code":
        return any(
            block.get("type") == "tool_use" for block in answer.get("content") or []
        )
    return any(
        item.get("type") == "function_call" for item in answer.get("output") or []
    )


def probe_gateway(
    gateway: str, harness: str, model: str, token: str, timeout: float = 120
) -> ProbeResult:
    """One streamed tool-call request in the harness's own protocol, since both
    harnesses stream and some failures show only where a stream ends."""
    path, extra, body = _probe_request(harness, model)
    url = gateway.rstrip("/") + path
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        **extra,
    }
    try:
        with requests.post(
            url,
            json={**body, "stream": True},
            headers=headers,
            timeout=timeout,
            stream=True,
        ) as response:
            if response.status_code != requests.codes.ok:
                return ProbeResult(
                    False,
                    detail=f"{url}: HTTP {response.status_code} {response.text[:300]}",
                )
            events = _events(response)
    except requests.RequestException as exc:
        return ProbeResult(False, detail=f"{url}: {exc}")
    answer, problem = _reply_of(harness, events)
    if answer is None:
        return ProbeResult(False, detail=f"{url}: {problem}")
    warnings = [
        w.strip()
        for w in response.headers.get("x-vsr-warnings", "").split(",")
        if w.strip()
    ]
    called = (
        "called the tool"
        if _called_tool(harness, answer)
        else "answered without calling the tool"
    )
    return ProbeResult(
        True,
        model=response.headers.get("x-vsr-selected-model", ""),
        detail=called,
        warnings=warnings,
    )
