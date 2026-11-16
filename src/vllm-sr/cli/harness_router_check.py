"""What a harness needs from the Router, and one call that shows the gateway answers it.

The check reads a v0.5 configuration and reports; it never edits it, since one
gateway can serve other clients too.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import requests

from cli.durations import parse_duration
from cli.harness_connect import MAX_OUTPUT_TOKENS
from cli.model_catalog import DEFAULT_CHANNEL, _load_catalog_document
from cli.model_catalog_types import ModelCatalogError
from cli.models import _fallback_duration_seconds
from cli.runtime_stack import resolve_runtime_stack

CLAUDE_HEADERS = {
    "session": "x-claude-code-session-id",
    "agent": "x-claude-code-agent-id",
}
CODEX_HEADERS = {"session": "session-id", "agent": "thread-id"}
# what Pi sends with the session-affinity compat connect writes; it names no agent
PI_HEADERS = {"session": "x-session-id"}
# what the Router reads when protection's identity.headers names none: the session
# a request pins with a session_id_headers default, then these, as
# defaultIdentityHeaders in its config package lists them
SESSION_ID_HEADERS = ("x-session-id", "x-claude-code-session-id")
IDENTITY_DEFAULTS = {
    "session": ("session-id",),
    "agent": ("x-claude-code-agent-id", "thread-id"),
}
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


PROTECTION_NEED = "session protection, so each session and subagent keeps its model"
IDENTITY_NEED = (
    "session protection on this harness's identity headers, which the configuration"
    " names otherwise, so each session and subagent keeps its model"
)


def _reads_identity(router: dict, headers: dict[str, str]) -> bool:
    """Whether protection reads the harness's identity headers: as identity.headers
    names them, or, where it names none, by the Router's defaults."""
    named = _get(router, "learning", "protection", "identity", "headers") or {}
    pinned = _get(router, "headers", "session_id_headers") or SESSION_ID_HEADERS
    return all(
        (
            named[key] == name
            if named.get(key)
            else name in IDENTITY_DEFAULTS[key] or (key == "session" and name in pinned)
        )
        for key, name in headers.items()
    )


def _protection_finding(router: dict, headers: dict[str, str]) -> RouterFinding | None:
    learning = router.get("learning") or {}
    protection = learning.get("protection") or {}
    reads = _reads_identity(router, headers)
    if (
        learning.get("enabled") is True
        and protection.get("enabled") is True
        and protection.get("scope") == "session"
        and reads
    ):
        return None
    snippet = (
        "global:\n  router:\n    learning:\n      enabled: true\n      protection:\n"
        "        enabled: true\n        scope: session"
    )
    if reads:
        return RouterFinding(PROTECTION_NEED, snippet)
    lines = "\n".join(f"            {k}: {v}" for k, v in headers.items())
    return RouterFinding(
        IDENTITY_NEED,
        snippet + "\n        identity:\n          headers:\n" + lines,
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


def _claude_findings(doc: dict, router: dict, reach) -> list[RouterFinding | None]:
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
    formats = {f for name, f in _model_formats(doc).items() if reach(name)}
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


def _to_anthropic(doc: dict, reach) -> list[tuple[dict, str]]:
    """Each decision that can pick an anthropic model the entrypoint reaches, with
    a hint when none of its models says so and one leaves api_format to its
    backend bindings."""
    formats = _model_formats(doc)
    to_anthropic = []
    for decision in _decisions(doc):
        kinds = {
            formats.get(ref.get("model"), "")
            for ref in decision.get("modelRefs") or []
            if isinstance(ref, dict) and reach(ref.get("model"))
        }
        if "messages" in kinds:
            to_anthropic.append((decision, ""))
        elif "" in kinds:
            to_anthropic.append((decision, MAYBE_ANTHROPIC))
    return to_anthropic


def _signature_finding(router: dict, to_anthropic) -> RouterFinding | None:
    if not to_anthropic:
        return None
    known = any(not maybe for _, maybe in to_anthropic)
    return _lossy_finding(
        router,
        "reasoning.signature",
        "reasoning.signature, or a reply in which an anthropic backend thinks fails with"
        " lossy_translation" + ("" if known else MAYBE_ANTHROPIC),
    )


def _codex_findings(doc: dict, router: dict, reach) -> list[RouterFinding | None]:
    to_anthropic = _to_anthropic(doc, reach)
    findings = [_signature_finding(router, to_anthropic)]
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


def _pi_findings(doc: dict, router: dict, reach) -> list[RouterFinding | None]:
    # Pi sends its own output limit, so an anthropic backend needs no default. On
    # its own Anthropic login Pi speaks Messages, which keeps thinking signatures,
    # and sends no session header, so neither finding applies.
    if _caller_of(doc, reach) == "anthropic":
        return []
    return [
        _signature_finding(router, _to_anthropic(doc, reach)),
        _protection_finding(router, PI_HEADERS),
    ]


def system_one_models(doc: dict | None, gateway: str | None = None) -> list[str]:
    """The System One models a listener publishes, which vSR can ask there: the
    listener the gateway URL names, or the first. The local stack publishes each
    listener on its port plus VLLM_SR_PORT_OFFSET, so a gateway on this host
    names the listener that many ports lower."""
    listeners = [
        one for one in (doc or {}).get("listeners") or [] if isinstance(one, dict)
    ]
    url = urlsplit(gateway or "")
    if url.port is not None:
        port = url.port
        if url.hostname in ("localhost", "127.0.0.1", "::1"):
            port -= resolve_runtime_stack().port_offset
        listeners = [one for one in listeners if str(one.get("port")) == str(port)]
    first = listeners[0] if listeners else None
    return [str(m) for m in _get(first, "apis", "systemone", "models") or []]


def entrypoint_models(doc: dict, model: str) -> list[str] | None:
    """The provider models a request for model can reach, as the Router maps it: a
    model's own name, or an entrypoint's recipe, whose decisions name models and
    whose requests no decision takes go to providers.defaults.model. None when the
    configuration maps model nowhere."""
    groups = entrypoint_candidates(doc, model)
    if groups is None:
        return None
    return list(dict.fromkeys(name for names, _ in groups for name in names))


def _entrypoint_recipe(doc: dict, model: str) -> dict | None:
    """The recipe an entrypoint maps model to, empty when the document does not
    write it out, None when no entrypoint maps model."""
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
    return next((r for r in recipes if r.get("name") == recipe), {})


# The Router's selector algorithms answer with one eligible candidate, so a
# request has to fit only that one, or algorithm.minimum_candidates of them.
# Any other algorithm may call every candidate.
SELECTOR_ALGORITHMS = {"static", "random", "multi_factor", "systemone"}


def _candidates_needed(decision: dict, models: list[str]) -> int:
    algorithm = decision.get("algorithm") or {}
    if algorithm.get("type") not in (None, "", *SELECTOR_ALGORITHMS):
        return len(models)
    return max(1, int(algorithm.get("minimum_candidates") or 1))


def entrypoint_candidates(doc: dict, model: str) -> list[tuple[list[str], int]] | None:
    """entrypoint_models, as the candidate lists the Router picks from, one per
    decision and providers.defaults.model for requests no decision takes, each
    with how many of its candidates a request has to fit."""
    formats = _model_formats(doc)
    if model in formats:
        return [([model], 1)]
    recipe = _entrypoint_recipe(doc, model)
    if recipe is None:
        return None
    groups = []
    for decision in recipe.get("decisions") or []:
        if not isinstance(decision, dict):
            continue
        names = _decision_models(decision)
        groups.append((names, _candidates_needed(decision, names)))
    fallback = _get(doc, "providers", "defaults", "model")
    if fallback:
        groups.append(([fallback], 1))
    # an image or System One model never answers a harness's turn
    groups = [
        (
            list(dict.fromkeys(n for n in names if formats.get(n, "") not in NOT_CHAT)),
            need,
        )
        for names, need in groups
    ]
    return [(names, min(need, len(names))) for names, need in groups if names]


def _decision_models(decision: dict) -> list[str]:
    """A decision's modelRefs, and the models its candidateIterations walk."""
    refs = [
        *(decision.get("modelRefs") or []),
        *(
            one
            for it in decision.get("candidateIterations") or []
            if isinstance(it, dict)
            for one in it.get("models") or []
        ),
    ]
    return [r.get("model") for r in refs if isinstance(r, dict) and r.get("model")]


def recipe_known_limits(doc: dict, model: str) -> bool:
    """Whether the recipe model maps to drops the candidates a request no longer
    fits (candidate_requirements.context: known_limits), so a decision falls
    over to its larger models instead of refusing the request."""
    recipe = _entrypoint_recipe(doc, model) or {}
    return _get(recipe, "candidate_requirements", "context") == "known_limits"


def caller_provider(doc: dict | None, entrypoint: str | None = None) -> str | None:
    """The provider whose login the Router presents upstream for each client
    (auth.credential: caller), which then has to come from the client in place of
    a gateway key, among the models a request for entrypoint can reach (any model
    when it is None or the configuration names none it reaches). Every caller
    backend names one provider; other models beside them take a key of their
    own or none."""
    reachable = (
        entrypoint_models(doc, entrypoint) if doc and entrypoint else None
    ) or None
    return _caller_of(doc, lambda name: reachable is None or name in reachable)


def _caller_of(doc: dict | None, reach) -> str | None:
    """The provider the caller backends of the models reach accepts name."""
    return next(
        (
            str(ref.get("provider") or "").lower()
            for model in _get(doc, "providers", "models") or []
            if isinstance(model, dict) and reach(model.get("name"))
            for ref in model.get("backend_refs") or []
            if isinstance(ref, dict)
            and (ref.get("auth") or {}).get("credential") == "caller"
        ),
        None,
    )


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


# The fallback the check asks for: room for an agent call that runs for
# minutes (the longest on the agent harness page took about 17). 1200s is also
# the ext_proc message_timeout of the Envoy the CLI renders, the longest the
# Router can hold a response there.
AGENT_FALLBACK_TIMEOUT = "1200s"
# pkg/fallback DefaultPolicy's timeouts, in seconds, which a 0 inherits
FALLBACK_DEFAULT_SECONDS = {"total_timeout": 30, "per_attempt_timeout": 10}
FALLBACK_WHY = (
    " so a model call that fails moves to the decision's next model (which answers"
    " without the first one's prompt cache) instead of returning the error to the"
    f" agent, with {AGENT_FALLBACK_TIMEOUT} for it: the default 30s in all and 10s"
    " for each model end a slower answer"
)
TIMEOUT_WHY = (
    ": the Router cuts a call off at that time even while it streams, and long"
    " agent calls run past it; the idle timeout still ends a stream that stalls"
)


def _agent_fallback(enable: bool) -> str:
    """The fallback block with room for an agent call, and the switch when the
    layer it goes in has to turn fallback on."""
    lines = ["fallback:", *(["  enabled: true"] if enable else [])]
    lines += [f"  {key}: {AGENT_FALLBACK_TIMEOUT}" for key in FALLBACK_DEFAULT_SECONDS]
    return "\n".join(lines)


def _reachable_decisions(doc: dict, model: str | None) -> list[tuple[dict, dict]]:
    """The recipe and decision of each decision a request for model can take: the
    entrypoint's recipe, every recipe when model is None or maps nowhere, and none
    for a provider model asked by name."""
    if model in _model_formats(doc):
        return []
    recipe = _entrypoint_recipe(doc, model) if model else None
    recipes = (
        [recipe]
        if recipe is not None
        else [r for r in doc.get("recipes") or [] if isinstance(r, dict)]
    )
    return [
        (recipe, decision)
        for recipe in recipes
        for decision in recipe.get("decisions") or []
        if isinstance(decision, dict)
    ]


def _fallback_fix(layers: list[dict]) -> tuple[int, bool] | None:
    """Where to change fallback, given its layers from the most specific (the
    decision's, its recipe's, global.router's): the most specific layer that
    turns it off or gives it a timeout shorter than an agent call, and whether
    the block there has to turn it on. None when it is on with room."""
    last = len(layers) - 1
    enabled_at = next(
        (i for i, f in enumerate(layers) if f.get("enabled") is not None), None
    )
    off = enabled_at is None or layers[enabled_at]["enabled"] is not True
    found = [last if enabled_at is None else enabled_at] if off else []
    agent = float(parse_duration(AGENT_FALLBACK_TIMEOUT))
    for key, default in FALLBACK_DEFAULT_SECONDS.items():
        at, seconds = last, default
        for i, layer in enumerate(layers):
            try:
                value = _fallback_duration_seconds(layer[key]) if layer.get(key) else 0
            except ValueError:
                value = 0
            if value:
                at, seconds = i, value
                break
        if float(seconds) < agent:
            found.append(at)
    if not found:
        return None
    fix = min(found)
    return fix, off or enabled_at >= fix


def _fallback_findings(doc: dict, decisions) -> list[RouterFinding]:
    """Where to change fallback so that each decision moves a failed call to its
    next model with room for an agent call, and each decision that names one
    model, which leaves fallback nowhere to go."""
    default = _get(doc, "global", "router", "fallback")
    fixes: dict[tuple[str, bool], list[str]] = {}
    one_model = []
    for recipe, decision in decisions:
        name = str(decision.get("name"))
        places = [
            (f"decision {name}'s own fallback", decision.get("fallback")),
            (f"recipe {recipe.get('name')}'s own fallback", recipe.get("fallback")),
            ("", default),
        ]
        layers = [f if isinstance(f, dict) else {} for _, f in places]
        fix = _fallback_fix(layers)
        if fix is not None:
            fixes.setdefault((places[fix[0]][0], fix[1]), []).append(name)
            continue
        models = list(dict.fromkeys(_decision_models(decision)))
        if len(models) <= 1:
            one_model.append(
                RouterFinding(
                    f"a second model on decision {name}: fallback moves only to the"
                    f" decision's next model, and decision {name} names one",
                    "modelRefs:"
                    + "".join(f"\n  - model: {m}" for m in models)
                    + "\n  - model: <another model>",
                )
            )
    findings = []
    for (at, enable), names in fixes.items():
        block = _agent_fallback(enable)
        findings.append(
            RouterFinding(
                f"fallback on decision {', '.join(names)}"
                + (f" (set in {at})" if at else "")
                + f",{FALLBACK_WHY}",
                (
                    block
                    if at
                    else "global:\n  router:\n" + textwrap.indent(block, "    ")
                ),
            )
        )
    return findings + one_model


def _total_timeout_findings(doc: dict, reach, model, decisions) -> list[RouterFinding]:
    """What bounds a whole call to a model a request can reach: the decision's
    reliability.total_timeout for the calls it routes, else the model's own,
    else the listener timeout (300s unless set), which also bounds how long a
    stream may stay silent."""
    listeners = [one for one in doc.get("listeners") or [] if isinstance(one, dict)]
    listener = next(iter(listeners), {}).get("timeout")
    listener = "300s" if listener in (None, "") else str(listener)

    def bounds(value) -> bool:
        try:
            return parse_duration(str(value)) > 0
        except ValueError:
            return False

    # the models whose own timeout, or the listener's, bounds some call
    covered = {_get(doc, "providers", "defaults", "model"), model}
    by_decision = []
    for _, decision in decisions:
        own = _get(decision, "reliability", "total_timeout")
        if own is None:
            covered.update(_decision_models(decision))
        elif bounds(own):
            by_decision.append(f"{decision.get('name')} ({own})")
    formats = _model_formats(doc)
    if not decisions and model not in formats:
        covered = None  # no decision names the models a request takes: all of them
    by_model = []
    for entry in _get(doc, "providers", "models") or []:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not name or not reach(name) or formats[name] in NOT_CHAT:
            continue
        if covered is not None and name not in covered:
            continue
        own = _get(entry, "reliability", "total_timeout")
        value, where = (
            (listener, "the listener's timeout")
            if own is None
            else (own, "its total_timeout")
        )
        if bounds(value):
            by_model.append((name, f"{name} ({value}, {where})"))
    findings = []
    if by_model:
        findings.append(
            RouterFinding(
                f"total_timeout: 0s on {', '.join(said for _, said in by_model)}"
                + TIMEOUT_WHY,
                "providers:\n  models:"
                + "".join(
                    f"\n    - name: {name}\n      reliability:\n        total_timeout: 0s"
                    for name, _ in by_model
                ),
            )
        )
    if by_decision:
        findings.append(
            RouterFinding(
                f"reliability.total_timeout: 0s on decision {', '.join(by_decision)}"
                + TIMEOUT_WHY,
                "reliability:\n  total_timeout: 0s",
            )
        )
    return findings


def check_router_config(
    doc: dict, harness: str, model: str | None = None
) -> list[RouterFinding]:
    """What this harness needs from a v0.5 Router configuration and does not find,
    for the models a request for model can reach (all of them when model is None
    or maps nowhere)."""
    router = _get(doc, "global", "router") or {}
    reachable = entrypoint_models(doc, model) if model else None
    reach = (lambda _: True) if reachable is None else set(reachable).__contains__
    findings = {"claude-code": _claude_findings, "pi": _pi_findings}
    found = findings.get(harness, _codex_findings)(doc, router, reach)
    decisions = _reachable_decisions(doc, model)
    found += _fallback_findings(doc, decisions)
    found += _total_timeout_findings(doc, reach, model, decisions)
    # an entrypoint that reaches one model has no model to keep
    single = len(set(reachable or [])) == 1
    needs = (PROTECTION_NEED, IDENTITY_NEED)
    return [f for f in found if f and not (single and f.need in needs)]


def output_limit(doc: dict | None, entrypoint: str) -> int:
    """The output limit a harness asks for: MAX_OUTPUT_TOKENS, or the smallest
    max_output_tokens of the models a request for entrypoint can reach when that
    is lower, since a model refuses a request for more."""
    reachable = set(entrypoint_models(doc, entrypoint) or []) if doc else set()
    limits = [
        int(model["max_output_tokens"])
        for model in _get(doc, "providers", "models") or []
        if isinstance(model, dict)
        and model.get("name") in reachable
        and isinstance(model.get("max_output_tokens"), int)
        and model["max_output_tokens"] > 0
    ]
    return min([int(MAX_OUTPUT_TOKENS), *limits])


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
    if harness == "pi":
        return (
            "/v1/chat/completions",
            {},
            {
                "model": model,
                "max_completion_tokens": PROBE_TOKENS,
                "messages": [
                    {"role": "user", "content": "What time is it? Use the tool."}
                ],
                "tools": [{"type": "function", "function": PROBE_TOOL}],
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
        # a Chat Completions stream says so with an error object of its own
        failed = isinstance(event.get("error"), dict)
        if failed or event.get("type") in ("error", "response.failed"):
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
    if harness == "pi":
        return _chat_reply(events)
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


def _chat_reply(chunks: list[dict]) -> tuple[dict | None, str]:
    """A streamed Chat Completions reply put together: its text, each tool call
    with the arguments its deltas carry, and the finish reason."""
    choices = [
        choice
        for chunk in chunks
        for choice in chunk.get("choices") or []
        if isinstance(choice, dict)
    ]
    finish = next(
        (c["finish_reason"] for c in reversed(choices) if c.get("finish_reason")),
        None,
    )
    if finish is None:
        return None, "the stream did not finish a Chat Completions reply"
    deltas = [c.get("delta") or {} for c in choices]
    calls: dict[int, dict] = {}
    for part in (call for delta in deltas for call in delta.get("tool_calls") or []):
        call = calls.setdefault(part.get("index", 0), {"arguments": ""})
        function = part.get("function") or {}
        call["id"] = part.get("id") or call.get("id")
        call["name"] = function.get("name") or call.get("name")
        call["arguments"] += function.get("arguments") or ""
    text = "".join(delta.get("content") or "" for delta in deltas)
    return {"text": text, "tool_calls": list(calls.values()), "finish": finish}, ""


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
    if harness == "pi":
        return bool(answer.get("tool_calls"))
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
    if harness == "pi":
        # the Router names the object, which it also accepts from a backend unnamed
        choices = answer.get("choices")
        if answer.get("object") not in (None, "chat.completion") or not (
            isinstance(choices, list) and choices
        ):
            return "the answer is not a Chat Completions reply"
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
    if harness == "pi":
        return {
            "model": model,
            "max_completion_tokens": 64,
            "messages": [{"role": "user", "content": question}],
        }
    return {"model": model, "max_output_tokens": 64, "input": question}


def _text_of(harness: str, reply: dict) -> str:
    """The text a finished reply gives, which the answer to a tool result needs."""
    if harness in ("claude-code", "pi"):
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
    if harness == "pi":
        calls = first["tool_calls"]
        messages = [
            *call["messages"],
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": one["id"],
                        "type": "function",
                        "function": {
                            "name": one["name"],
                            "arguments": one["arguments"],
                        },
                    }
                    for one in calls
                ],
            },
            *(
                {"role": "tool", "tool_call_id": one["id"], "content": "12:00"}
                for one in calls
            ),
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
                problem = f"HTTP {response.status_code} {response.text[:300]}"
                if _refused_key(response.text):
                    problem = f"the gateway did not accept the key ({problem})"
                return response, None, problem
            if body.get("stream"):
                return response, _events(response), ""
            try:
                return response, response.json(), ""
            except ValueError:
                return response, None, "the answer is not JSON"
    except requests.Timeout:
        return None, None, f"no answer within {timeout:g} s"
    except requests.ConnectionError as exc:
        # requests nests the socket error a few reprs deep
        found = re.search(r"\[Errno -?\d+\] ([^'\")]+)", str(exc))
        why = found[1] if found else "the connection failed"
        return None, None, f"no connection ({why}). Is the gateway running there?"
    except requests.RequestException as exc:
        return None, None, str(exc)


# The listener's answer to a client key it does not hold, the same from the Envoy
# template and the Router's own listener. A 401 from past it, the Router's when it
# cannot present the backend's credential or a backend's own, says something else,
# even with the same code.
LISTENER_REFUSAL = {
    "code": "invalid_api_key",
    "message": "Unauthorized: missing or invalid API key",
}


def _refused_key(text: str) -> bool:
    try:
        error = json.loads(text).get("error")
    except (ValueError, AttributeError):
        return False
    return isinstance(error, dict) and LISTENER_REFUSAL.items() <= error.items()


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


def _probe_target(harness: str, model: str, gateway: str, token: str):
    path, extra, call = _probe_request(harness, model)
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        **extra,
    }
    return gateway.rstrip("/") + path, headers, call


def basic_reply(
    gateway: str, harness: str, model: str, token: str, timeout: float = 120
) -> ProbeResult:
    """A plain request in the harness's own protocol, and whether a finished
    reply came back."""
    url, headers, _ = _probe_target(harness, model, gateway, token)
    response, answer, problem = _post(
        url, headers, _plain_request(harness, model), timeout
    )
    return _result(
        "Basic reply",
        url,
        response,
        problem or _plain_problem(harness, answer),
        "answered",
    )


def probe_gateway(
    gateway: str, harness: str, model: str, token: str, timeout: float = 120
) -> list[ProbeResult]:
    """Two checks in the harness's own protocol, each run only after the one before
    passed: a plain reply, then a streamed tool loop, a tool call and the answer to
    its result, since both harnesses stream and some failures show only where a
    stream ends or once a tool result goes back."""
    url, headers, call = _probe_target(harness, model, gateway, token)
    plain = basic_reply(gateway, harness, model, token, timeout)
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
    stop reason, a Chat Completions finish reason, or a Responses reply left
    incomplete for that reason."""
    return (
        reply.get("stop_reason") == "max_tokens"
        or reply.get("finish") == "length"
        or (reply.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
    )
