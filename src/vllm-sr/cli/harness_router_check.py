"""What a harness needs from the Router, and one call that shows the gateway answers it.

The check reads a v0.5 configuration and reports; it never edits it, since one
gateway can serve other clients too.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import requests

from cli.model_catalog import DEFAULT_CHANNEL, _load_catalog_document
from cli.model_catalog_types import ModelCatalogError

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
# the canonical name of each api_format alias, as the Router reads it
API_FORMAT_ALIASES = {"openai": "chat_completions", "anthropic": "messages"}
OPENAI_FORMATS = {"chat_completions", "responses"}
NOT_CHAT = {"images", "systemone"}
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
    name: str = ""
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
    """Each provider model's canonical api_format, or "" when the configuration leaves
    it to its backend bindings, which only the Router resolves."""
    formats = {
        m["name"]: str(m.get("api_format") or "")
        for m in _get(doc, "providers", "models") or []
        if isinstance(m, dict) and m.get("name")
    }
    return {name: API_FORMAT_ALIASES.get(f, f) for name, f in formats.items()}


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
    if missing and formats & (OPENAI_FORMATS | {""}):
        findings.append(
            RouterFinding(
                f"{', '.join(missing)}, or the Router refuses Claude Code's error tool results,"
                " system turns and files toward an openai backend"
                + ("" if formats & OPENAI_FORMATS else MAYBE_OPENAI),
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
        if "messages" in kinds:
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
    return [str(m) for m in _get(first, "apis", "systemone", "models") or []]


def entrypoint_models(doc: dict, model: str) -> list[str] | None:
    """The provider models a request for model can reach, as the Router maps it: a
    model's own name, or an entrypoint's recipe, whose decisions name models and
    whose requests no decision takes go to providers.defaults.model. None when the
    configuration maps model nowhere."""
    formats = _model_formats(doc)
    if model in formats:
        return [model]
    recipes = [r for r in doc.get("recipes") or [] if isinstance(r, dict)]
    chat = [
        e
        for e in doc.get("entrypoints") or []
        if isinstance(e, dict) and e.get("api") in (None, "", "chat")
    ]
    recipe = next(
        (e.get("recipe") for e in chat if model in (e.get("model_names") or [])), None
    )
    if recipe is None and model == "vllm-sr/auto":
        # the Router serves vllm-sr/auto itself when no chat entrypoint maps the default recipe
        marked = [e.get("recipe") for e in chat if e.get("default")]
        default = (
            marked[0]
            if marked
            else recipes[0].get("name") if len(recipes) == 1 else "default"
        )
        if all(e.get("recipe") != default for e in chat):
            recipe = default
    if recipe is None:
        return None
    # a decision's modelRefs, and the models its candidateIterations walk
    picked = [
        ref.get("model")
        for r in recipes
        if r.get("name") == recipe
        for decision in r.get("decisions") or []
        if isinstance(decision, dict)
        for ref in [
            *(decision.get("modelRefs") or []),
            *(
                one
                for it in decision.get("candidateIterations") or []
                if isinstance(it, dict)
                for one in it.get("models") or []
            ),
        ]
        if isinstance(ref, dict) and ref.get("model")
    ]
    fallback = _get(doc, "providers", "defaults", "model")
    # an image or System One model never answers a harness's turn
    return [
        name
        for name in dict.fromkeys([*picked, *([fallback] if fallback else [])])
        if formats.get(name, "") not in NOT_CHAT
    ]


def context_windows(doc: dict, names: list[str]) -> dict[str, int | None]:
    """Each model's context window, as the Router reads it: the model's own
    context_window_size, else its built-in catalog card's limit."""
    cards = _catalog_windows()
    models = {
        m.get("name"): m
        for m in _get(doc, "providers", "models") or []
        if isinstance(m, dict)
    }
    found = {}
    for name in names:
        model = models.get(name) or {}
        window = model.get("context_window_size") or cards.get(model.get("catalog"))
        found[name] = window if isinstance(window, int) and window > 0 else None
    return found


def _catalog_windows() -> dict[str, int]:
    try:
        _, document = _load_catalog_document(DEFAULT_CHANNEL)
    except ModelCatalogError:
        return {}
    return {
        card["id"]: (card.get("limits") or {}).get("context_window_size")
        for card in document.get("models") or []
        if isinstance(card, dict) and card.get("id")
    }


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


# Room for a model that thinks before it calls the tool or answers its result.
PROBE_TOKENS = 1024


def _probe_request(harness: str, model: str) -> tuple[str, dict, dict]:
    if harness == "claude-code":
        return (
            "/v1/messages",
            {"anthropic-version": "2023-06-01"},
            {
                "model": model,
                "max_tokens": PROBE_TOKENS,
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
            "max_output_tokens": PROBE_TOKENS,
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
        text = "".join(
            (event.get("delta") or {}).get("text") or ""
            for event in events
            if event.get("type") == "content_block_delta"
        )
        stop = next(
            (
                (event.get("delta") or {}).get("stop_reason")
                for event in events
                if event.get("type") == "message_delta"
            ),
            None,
        )
        return {"content": blocks, "text": text, "stop_reason": stop}, ""
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
    response.encoding = "utf-8"  # what server-sent events always are
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


def _plain_problem(harness: str, answer: Any) -> str:
    """Why a 200 answer is not a finished reply in the harness's protocol, or ""."""
    if not isinstance(answer, dict):
        return "the answer is not a JSON object"
    if harness == "claude-code":
        if answer.get("type") != "message" or not isinstance(
            answer.get("content"), list
        ):
            return "the answer is not a Messages reply"
        return ""
    if answer.get("object") != "response" or answer.get("status") not in (
        "completed",
        "incomplete",
    ):
        return f"the answer is not a finished Responses reply (status {answer.get('status')!r})"
    return ""


def _plain_request(harness: str, model: str) -> dict:
    question = "Reply with the word ok."
    if harness == "claude-code":
        return {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": question}],
        }
    return {"model": model, "max_output_tokens": 64, "input": question}


def _text_of(harness: str, reply: dict) -> str:
    """The text a finished reply gives, which the answer to a tool result needs."""
    if harness == "claude-code":
        return reply.get("text") or ""
    return "".join(
        part.get("text") or ""
        for item in reply.get("output") or []
        if item.get("type") == "message"
        for part in item.get("content") or []
    )


def _with_result(harness: str, call: dict, first: dict) -> dict:
    """The tool call's request again, carrying every call it answered with and a
    result for each, as the harness sends them next."""
    if harness == "claude-code":
        tools = [b for b in first["content"] if b.get("type") == "tool_use"]
        results = [
            {"type": "tool_result", "tool_use_id": tool["id"], "content": "12:00"}
            for tool in tools
        ]
        messages = [
            *call["messages"],
            {"role": "assistant", "content": tools},
            {"role": "user", "content": results},
        ]
        return {**call, "stream": True, "messages": messages}
    # the items go back as Codex sends them, without the status a reply gives them
    output = [
        {k: v for k, v in item.items() if k != "status"}
        for item in first.get("output") or []
    ]
    results = [
        {"type": "function_call_output", "call_id": item["call_id"], "output": "12:00"}
        for item in output
        if item.get("type") == "function_call"
    ]
    items = [{"role": "user", "content": call["input"]}, *output, *results]
    return {**call, "stream": True, "input": items}


def _post(url: str, headers: dict, body: dict, timeout: float):
    """The response and its reply: a stream's events when the body streams, else
    the JSON body; or None and why it failed."""
    try:
        with requests.post(
            url, json=body, headers=headers, timeout=timeout, stream=True
        ) as response:
            if response.status_code != requests.codes.ok:
                return (
                    response,
                    None,
                    f"HTTP {response.status_code} {response.text[:300]}",
                )
            if body.get("stream"):
                return response, _events(response), ""
            try:
                return response, response.json(), ""
            except ValueError:
                return response, None, "the answer is not JSON"
    except requests.RequestException as exc:
        return None, None, str(exc)


def _result(name: str, url: str, response, problem: str, detail: str) -> ProbeResult:
    headers = response.headers if response is not None else {}
    return ProbeResult(
        not problem,
        name=name,
        model=headers.get("x-vsr-selected-model", ""),
        detail=f"{url}: {problem}" if problem else detail,
        warnings=warning_notes(headers.get("x-vsr-warnings", "")),
    )


# A member of an RFC 8941 list or parameter list: anything up to the separator,
# with quoted strings, which may hold the separator, kept whole.
_LIST_MEMBER = re.compile(r'(?:[^,"]|"(?:[^"\\]|\\.)*")+')
_PARAM = re.compile(r'(?:[^;"]|"(?:[^"\\]|\\.)*")+')


def warning_notes(value: str) -> list[str]:
    """x-vsr-warnings in words: a protocol loss as its field, action and reason, a
    truncated list as how many were left out, and any other code as it is."""
    notes = []
    for member in _LIST_MEMBER.findall(value):
        code, *params = [part.strip() for part in _PARAM.findall(member)]
        if not code:
            continue
        values = {}
        for param in params:
            key, _, raw = param.partition("=")
            if raw.startswith('"'):
                raw = re.sub(r"\\(.)", r"\1", raw[1:-1])
            values[key] = raw
        if code == "protocol":
            notes.append(
                f"{values.get('field') or 'the request'} {values.get('action', 'changed')}:"
                f" {values.get('reason', 'no reason given')}"
            )
        elif code == "truncated":
            notes.append(f"{values.get('count', 'more')} more left out of the header")
        else:
            notes.append(code)
    return notes


def probe_gateway(
    gateway: str, harness: str, model: str, token: str, timeout: float = 120
) -> list[ProbeResult]:
    """Two checks in the harness's own protocol, each run only after the one before
    passed: a plain reply, then a streamed tool loop, a tool call and the answer to
    its result, since both harnesses stream and some failures show only where a
    stream ends or once a tool result goes back."""
    path, extra, call = _probe_request(harness, model)
    url = gateway.rstrip("/") + path
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        **extra,
    }
    response, answer, problem = _post(
        url, headers, _plain_request(harness, model), timeout
    )
    plain = _result(
        "Basic reply",
        url,
        response,
        problem or _plain_problem(harness, answer),
        "answered",
    )
    if not plain.ok:
        return [plain]
    response, events, problem = _post(url, headers, {**call, "stream": True}, timeout)
    first, problem = (None, problem) if problem else _reply_of(harness, events)
    detail = "called the tool and answered its result"
    if first is not None and not _called_tool(harness, first):
        detail = (
            "the model answered without calling the tool, so the loop was not tested"
        )
        if _hit_token_limit(first):
            detail = _token_limit("before calling the tool, so the loop was not tested")
    elif first is not None:
        response, events, problem = _post(
            url, headers, _with_result(harness, call, first), timeout
        )
        if not problem:
            final, problem = _reply_of(harness, events)
            if final is not None and not _text_of(harness, final):
                problem = "the answer to the tool result has no text"
                if _hit_token_limit(final):
                    problem += ": " + _token_limit("before writing any")
    return [plain, _result("Streamed tool loop", url, response, problem, detail)]


def _token_limit(where: str) -> str:
    return (
        f"the model reached the check's {PROBE_TOKENS}-token output limit {where},"
        " most likely while thinking; turn reasoning off for this route or run the"
        " check again"
    )


def _hit_token_limit(reply: dict) -> bool:
    """Whether a finished reply stopped at its output-token limit: a Messages
    stop reason, or a Responses reply left incomplete for that reason."""
    return (
        reply.get("stop_reason") == "max_tokens"
        or (reply.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
    )
