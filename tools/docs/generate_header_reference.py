#!/usr/bin/env python3
"""Generate the website's Router header reference from the header registry.

tools/codegen/headers exports the registry in src/semantic-router/pkg/headers
to reference.generated.json; this renders the page from that export, so the
website build checks the page without a Go toolchain.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/semantic-router/pkg/headers/reference.generated.json"
OUTPUT = ROOT / "website/docs/api/headers.md"

SURFACES = {
    "public": (
        "[Response headers](#response-headers)",
        "Every client, on each response a header applies to.",
    ),
    "debug": (
        "[`x-vsr-trace`](#the-trace)",
        "Only a caller `global.router.headers.debug` allows, on a request that "
        "asks with `x-vsr-debug: true`.",
    ),
    "control": (
        "[Request headers](#request-headers)",
        "Sent by clients to steer or identify a request.",
    ),
    "internal": (
        "[Internal headers](#internal-headers)",
        "No client: they pass between the Router, the gateway and authenticator "
        "in front of it, and providers, and no response the Router writes "
        "carries them.",
    ),
}
SENSITIVITIES = ("none", "identity", "secret")
DEBUG_MODES = {
    "none": ("`none` (default)", "Nobody."),
    "any": (
        "`any`",
        "Every client that sends `x-vsr-debug: true`. Use it for development and "
        "tests, or behind a gateway that removes the header from untrusted clients.",
    ),
    "authenticated": (
        "`authenticated`",
        "Only the users and groups in `debug_principals`, as the identity headers "
        "your authenticator sets assert them (`global.services.authz.identity`).",
    ),
}
EXAMPLE_RESPONSE = (
    ("x-vsr-schema-version", None),
    ("x-vsr-config-version", "7"),
    ("x-vsr-response-path", "upstream"),
    ("x-vsr-selected-recipe", "default"),
    ("x-vsr-selected-decision", "complex-request"),
    ("x-vsr-selected-model", "reasoning-model"),
    ("x-vsr-replay-id", "replay_01J..."),
)
TRACE_NOTES = (
    (
        ("confidence",),
        "`confidence` is absent when the decision rests on policy leaves, "
        "aggregates several evidence leaves, or resolves through an error policy.",
    ),
    (
        ("fallback_attempts",),
        "`fallback_attempts` counts the same way in both gateway modes and is "
        "omitted when the primary answered.",
    ),
    (
        ("ranking",),
        "`ranking` is omitted when only one decision matched; parse "
        "`decision_ranking` in Router Replay for the full ranking.",
    ),
    (
        ("learning",),
        "`learning` lists the Router Learning methods with their `action`. Show "
        "the actions as user-facing phrases such as `tool/protocol pinned`, "
        "`model switched`, or `learning bypassed`; fresh conversation or "
        "session-start diagnostics are usually useful only in debug views, as "
        "neutral status text.",
    ),
    (
        ("effective_input_tokens", "effective_max_output_tokens"),
        "`effective_input_tokens` and `effective_max_output_tokens` appear "
        "together on a successful upstream response when automatic output was "
        "resolved, in the initial headers of streaming and buffered responses "
        "alike. Explicit output limits, cache hits, skipped processing, and "
        "Looper responses omit them. Their values come from the final dispatch "
        "after request changes and backend rendering; they are not estimates or "
        "the model's configured maximum context length.",
    ),
    (
        ("config_hash", "inference_calls", "model_usage"),
        "`config_hash`, `inference_calls` and `model_usage` are the receipts of "
        "a config-bound evaluation request, one sent with "
        "`x-vsr-bench-expected-config-hash`.",
    ),
    (
        ("retention",),
        "`retention` shows the fields a matched decision's retention directive "
        "set, so operators can verify policy wiring; clients should not use them "
        "as commands.",
    ),
    (
        ("prompt_cache",),
        "`prompt_cache` is content-free: it reports counts and outcomes, never "
        "the cached text itself, and it never claims a cache hit or "
        "provider-side savings.",
    ),
)


# A header name in a code span, such as `x-vsr-trace` or `x-vsr-debug: true`;
# a prefix such as `x-vsr-` ends in a hyphen and is not matched.
HEADER_IN_CODE = re.compile(r"`(x-[a-z0-9-]*[a-z0-9])[`:]")


class ReferenceError(ValueError):
    """The export holds something this page has no words for."""


def cell(value: str) -> str:
    if "|" in value or "\n" in value:
        raise ReferenceError(f"{value!r} does not fit a table cell")
    return value


def code_list(names: list[str]) -> str:
    quoted = [f"`{name}`" for name in names]
    if len(quoted) < 3:
        return " and ".join(quoted)
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]


def table(header: str, rows: list[str]) -> list[str]:
    separator = "| " + " | ".join("---" for _ in header.strip("| ").split(" | ")) + " |"
    return [header, separator, *rows]


def validate(reference: dict) -> None:
    for header in reference["headers"]:
        if header["surface"] not in SURFACES:
            raise ReferenceError(
                f"{header['name']}: no words for the {header['surface']!r} surface"
            )
        if header["sensitivity"] not in SENSITIVITIES:
            raise ReferenceError(
                f"{header['name']}: no words for {header['sensitivity']!r} values"
            )
        if (
            header["sensitivity"] == "identity"
            and "provider request" in header["direction"]
        ):
            raise ReferenceError(
                f"{header['name']}: the page says no identity header reaches a provider"
            )
    if sorted(reference["debug_modes"]) != sorted(DEBUG_MODES):
        raise ReferenceError(
            f"global.router.headers.debug accepts {reference['debug_modes']}, "
            f"but the page describes {sorted(DEBUG_MODES)}"
        )
    public = {
        header["name"]
        for header in reference["headers"]
        if header["surface"] == "public"
    }
    for name, _ in EXAMPLE_RESPONSE:
        if name not in public:
            raise ReferenceError(f"the example response shows {name}, no public header")
    members = {member["key"] for member in reference["trace_members"]}
    for keys, _ in TRACE_NOTES:
        for key in keys:
            if key not in members:
                raise ReferenceError(f"a trace note names {key}, no trace member")


def render(reference: dict) -> str:
    validate(reference)
    headers = reference["headers"]

    def surface(name: str, sensitivity: str | None = None) -> list[dict]:
        return [
            header
            for header in headers
            if header["surface"] == name
            and (sensitivity is None or header["sensitivity"] == sensitivity)
        ]

    def hops(header: dict) -> str:
        return ", ".join(header["direction"])

    limit = reference["trace_limit_bytes"]
    dropped = set(reference["dropped_from_clients"])
    trusted = [
        header["name"]
        for header in headers
        if header["name"].startswith("x-vsr-")
        and header["surface"] != "control"
        and header["name"] not in dropped
    ]
    lines = [
        "---",
        "title: Router Headers",
        "description: Generated reference for the Router's request and response "
        "headers, the x-vsr-trace members, and who may see them.",
        "toc_max_heading_level: 3",
        "---",
        "",
        "<!-- Generated by tools/docs/generate_header_reference.py. Do not edit directly. -->",
        "",
        "# Router Headers",
        "",
        "The Router reads and writes the headers below: a public set on its "
        "responses, a debug trace, the request headers clients send, and "
        "internal headers no client sees. This reference is generated from the "
        "Router's header registry, `src/semantic-router/pkg/headers`; run "
        "`make headers-generate` after changing it. Header names are "
        "case-insensitive.",
        "",
        "## Who sees which headers",
        "",
        *table(
            "| Surface | Headers | Who sees them |",
            [
                f"| `{name}` | {headers_link} | {who} |"
                for name, (headers_link, who) in SURFACES.items()
            ],
        ),
        "",
        "Hops name where a header travels: `request` (client to Router), "
        "`response` (Router to client), `provider request` (Router to provider) "
        "and `provider response` (provider to Router). A header reaches a "
        "provider only when its hops include `provider request`, as **Reaches "
        "providers** shows for the request headers; the Router removes the "
        "others from every provider-bound request. **Since** is the release the "
        "header first shipped in.",
        "",
        "## Response headers",
        "",
        "Every response of the routing pipeline carries the headers that apply "
        "to it: answers from the upstream model, response-cache hits, and the "
        "responses the Router produces itself, such as a block, a fast response "
        "or a routing error. Responses of the Router's own handlers, such as "
        "`GET /v1/models`, carry `x-vsr-config-version` only.",
        "",
        *table(
            "| Header | Since | Description |",
            [
                f"| `{header['name']}` | {header['version']} | {cell(header['doc'])} |"
                for header in surface("public")
            ],
        ),
        "",
        "A response that answered from the upstream model:",
        "",
        "```http",
        "HTTP/1.1 200 OK",
        "content-type: application/json",
        *(
            f"{name}: {value if value is not None else reference['schema_version']}"
            for name, value in EXAMPLE_RESPONSE
        ),
        "```",
        "",
        f"- Read `x-vsr-schema-version` before parsing the other headers; the "
        f"current value is `{reference['schema_version']}`. Responses of the "
        "Router's own handlers never carry it, so its absence there is expected.",
        "- A request keeps the configuration version it started on, even when a "
        "hot reload activates another one while it runs. `x-vsr-config-version` "
        "counts the activations of one Router, which keeps its own configuration "
        "history; to check that replicas serve the same configuration, compare "
        "the `active_runtime_hash` that `GET /api/v1/config/hash` reports.",
        "- Recipe names scope local signal, projection, decision, cache, replay, "
        "metric, and learning or session identities. Use `x-vsr-selected-recipe` "
        "together with the local decision and signal names when correlating a "
        "response with Insights or metrics.",
        "",
        "## The trace {#the-trace}",
        "",
        *table(
            "| Header | Since | Description |",
            [
                f"| `{header['name']}` | {header['version']} | {cell(header['doc'])} |"
                for header in surface("debug")
            ],
        ),
        "",
        "`x-vsr-trace` carries what else the Router knows about a request, such "
        "as the matched signals, the decision's confidence and the algorithm. It "
        "is an [RFC 8941](https://www.rfc-editor.org/rfc/rfc8941) dictionary of "
        f"at most {limit} bytes, whose [members](#trace-members) are omitted "
        "when they don't apply. A string percent-encodes `%` and every byte "
        "outside printable ASCII, so decode it after unquoting. When the encoded "
        f"trace would exceed {limit} bytes, the Router leaves out members from "
        "the lowest priority up and ends the dictionary with `truncated`.",
        "",
        "```http",
        f"x-vsr-trace: {reference['trace_example']}",
        "```",
        "",
        "### Who may turn on the trace {#who-may-turn-on-the-trace}",
        "",
        "The trace names the signal and guard rules a request matched, the "
        "decision ranking and the Router Learning reasons, so a client can't "
        "turn it on by itself. A client asks for it with `x-vsr-debug: true`, "
        "and `global.router.headers.debug` decides who may; the Router ignores "
        "`x-vsr-debug` from everyone else.",
        "",
        *table(
            "| `debug` | Who gets the trace |",
            [
                f"| {DEBUG_MODES[mode][0]} | {DEBUG_MODES[mode][1]} |"
                for mode in reference["debug_modes"]
            ],
        ),
        "",
        "```yaml",
        "global:",
        "  router:",
        "    headers:",
        "      debug: authenticated",
        "      debug_principals:",
        "        users: [oncall-bot]",
        "        groups: [platform-sre]",
        "```",
        "",
        "`authenticated` trusts the identity headers. A standalone Router keeps "
        "them only on a listener with `identity.trust_headers`, and refuses "
        "`authenticated` when no listener trusts them. Behind an Envoy-based "
        "gateway, the gateway must remove the identity headers that clients "
        "send. The Dashboard Playground, `vllm-sr route probe` and sr-bench ask "
        "for the trace; when the caller isn't allowed, they show the public set "
        "only.",
        "",
        "### Members {#trace-members}",
        "",
        "Highest priority first:",
        "",
        *table(
            "| Member | Owner | Description |",
            [
                f"| `{member['key']}` | {member['owner']} | {cell(member['doc'])} |"
                for member in reference["trace_members"]
            ],
        ),
        "",
        *(f"- {note}" for _, note in TRACE_NOTES),
        "",
        "Projection scores and full projection traces stay in the Router Replay "
        "record rather than the trace. Use `x-vsr-replay-id` to inspect them in "
        "the Dashboard or through the authenticated Router management API; "
        "public inference listeners do not serve replay records.",
        "",
        "## Warnings {#response-warnings}",
        "",
        "`x-vsr-warnings` is an RFC 8941 list of warning codes, omitted when none "
        "applies. The codes of the `response_guard` rules that annotated the "
        "answer come first, in rule order, then the protocol translation losses. "
        "Per-warning detail, such as hallucination spans or jailbreak confidence, "
        "stays in the replay record.",
        "",
        *table(
            "| Code | Meaning |",
            [
                f"| `{code['code']}` | {cell(code['doc'])} |"
                for code in reference["warning_codes"]
            ],
        ),
        "",
        "## Request headers",
        "",
        "Clients send these to steer or identify a request. See "
        "[Session identification](./session-identification) for how the "
        "Router picks the session a request belongs to.",
        "",
        *table(
            "| Header | Reaches providers | Since | Description |",
            [
                f"| `{header['name']}` | "
                f"{'yes' if 'provider request' in header['direction'] else 'no'} | "
                f"{header['version']} | {cell(header['doc'])} |"
                for header in surface("control")
            ],
        ),
        "",
        "## Internal headers",
        "",
        "These pass between the Router, the gateway and authenticator in front "
        "of it, and providers. They are not part of the client contract, and no "
        "response the Router writes carries them. The Router drops a client's "
        "value of every `x-vsr-` header except the request headers above"
        + (
            f" and {code_list(trusted)}, which the gateway in front must strip "
            "from clients"
            if trusted
            else ""
        )
        + ", and drops every `x-vsr-` header a provider returns.",
        "",
        "### Identity {#identity-headers}",
        "",
        "These name an authenticated caller. A listener that doesn't trust "
        "identity headers drops the ones a client sends (see "
        "[Gateway modes](../installation/gateway-modes#identity-headers)), and "
        "no identity header reaches a model provider.",
        "",
        *table(
            "| Header | Since | Description |",
            [
                f"| `{header['name']}` | {header['version']} | {cell(header['doc'])} |"
                for header in surface("internal", "identity")
            ],
        ),
        "",
        "### Credentials",
        "",
        "Credentials, by name only. None reaches a client, and a per-user key "
        "reaches a provider only as its credential.",
        "",
        *table(
            "| Header | Hops |",
            [
                f"| `{header['name']}` | {hops(header)} |"
                for header in surface("internal", "secret")
            ],
        ),
        "",
        "### Routing and transport",
        "",
        *table(
            "| Header | Hops | Since | Description |",
            [
                f"| `{header['name']}` | {hops(header)} | {header['version']} | "
                f"{cell(header['doc'])} |"
                for header in surface("internal", "none")
            ],
        ),
    ]
    page = "\n".join(lines) + "\n"
    registered = {header["name"] for header in headers}
    for name in HEADER_IN_CODE.findall(page):
        if name not in registered:
            raise ReferenceError(f"the page names {name}, which is not registered")
    return page


def sync(output: Path, content: str, *, check: bool) -> bool:
    """Return whether content differs; check mode never creates or modifies files."""
    if output.is_file() and output.read_text(encoding="utf-8") == content:
        return False
    if not check:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail on drift without writing"
    )
    parser.add_argument(
        "--source", type=Path, default=SOURCE, help="header registry export"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT, help="reference Markdown path"
    )
    args = parser.parse_args(argv)
    try:
        content = render(json.loads(args.source.read_text(encoding="utf-8")))
    except ReferenceError as error:
        print(f"Header reference: {error}", file=sys.stderr)
        return 1
    changed = sync(args.output, content, check=args.check)
    if args.check and changed:
        print(
            "Header reference is missing or outdated; run make headers-generate.",
            file=sys.stderr,
        )
        return 1
    print(f"Header reference {'checked' if args.check else 'generated'}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
