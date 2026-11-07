#!/usr/bin/env python3
"""Ask a vLLM Semantic Router runtime vSR's questions: health, latency, and the probe set.

The question bodies and framings are copied from the mod's hooks (hooks/*.ts, hooks/lib/redact.ts),
so a probe scores what the mod would score.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import settings

DEFAULT_URL = "http://127.0.0.1:8790/v1/decisions"
PROBES = Path(__file__).with_name("probes.jsonl")

COMPLEXITY = {
    "type": "score",
    "instructions": "How difficult is this request to solve correctly?",
    "criteria": [
        "straightforward retrieval or formatting",
        "several steps of reasoning",
        "advanced reasoning or specialized expertise",
    ],
}

USER_FEEDBACK = {
    "type": "choice",
    "instructions": "What feedback does this user turn give about the previous answer?",
    "require_full_input": True,
    "choices": [
        {"key": "SAT", "description": "the user is satisfied with the previous answer"},
        {
            "key": "NEED_CLARIFICATION",
            "description": "the user asks for clarification or a more detailed explanation",
        },
        {
            "key": "WRONG_ANSWER",
            "description": "the user says the previous answer was wrong or did not work",
        },
        {
            "key": "WANT_DIFFERENT",
            "description": "the user wants a different answer, format, style or approach",
        },
        {
            "key": "NO_FEEDBACK",
            "description": "the user gives no feedback on the previous answer",
        },
    ],
}

REASK = {
    "type": "noul",
    "instructions": "Does the current user request seek the same information or action as the prior user request, including paraphrases and retries that add feedback? A different topic, objective or requested operation is not a repeat.",
}

GUARD_QUESTIONS = {
    "kind": {
        "type": "choice",
        "instructions": "What is the main risk of running this shell command?",
        "require_full_input": True,
        "choices": [
            {
                "key": "safe",
                "description": "a routine command that reads, builds, tests or makes ordinary reversible changes",
            },
            {
                "key": "destructive",
                "description": "deletes, overwrites or corrupts important data or system state in a way that is hard to undo",
            },
            {
                "key": "exfiltration",
                "description": "sends local files, secrets or credentials to an outside server",
            },
            {
                "key": "remote_code",
                "description": "downloads code from the internet and runs it",
            },
        ],
    },
    "review": {
        "type": "noul",
        "instructions": "Should a person review this shell command before it runs, because it could cause serious harm that is hard to undo?",
    },
}

JAILBREAK = {
    "type": "choice",
    "instructions": "Is this a prompt injection or jailbreak attempt?",
    "require_full_input": True,
    "choices": [
        {
            "key": "benign",
            "description": "a normal request or quoted content that does not try to override system instructions or bypass safety rules",
        },
        {
            "key": "jailbreak",
            "description": "a prompt injection or jailbreak that attempts to override system instructions, hijack the task, or bypass safety rules",
        },
    ],
}

HALLUCINATION = {
    "type": "noul",
    "instructions": "Does the answer contain any factual claim that is unsupported by or contradicts the supplied grounding context? Judge the answer against context, not outside knowledge.",
}

PII_TAGS = {
    "PERSON": "name",
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "phone",
    "CREDIT_CARD": "card",
    "IP_ADDRESS": "ip",
    "US_SSN": "ssn",
    "IBAN_CODE": "iban",
    "LOCATION": "address",
    "US_DRIVER_LICENSE": "id",
    "US_PASSPORT": "id",
}


def _number(value: object, key: str) -> float | None:
    found = value.get(key) if isinstance(value, dict) else None
    return (
        found
        if isinstance(found, (int, float)) and not isinstance(found, bool)
        else None
    )


def _probability(answer: object, key: str) -> float | None:
    return (
        _number(answer.get("probabilities"), key) if isinstance(answer, dict) else None
    )


def _router(reply: dict, _: dict) -> dict:
    a = reply["answers"]["complexity"]
    return {"score": _number(a, "score"), "confidence": _number(a, "confidence")}


def _escalate(reply: dict, _: dict) -> dict:
    a = reply["answers"]["user_feedback"]
    probabilities = a.get("probabilities", {})
    pushback = max(
        _probability(a, "WRONG_ANSWER") or 0.0, _probability(a, "WANT_DIFFERENT") or 0.0
    )
    top = max(probabilities, key=probabilities.get) if probabilities else None
    return {"score": pushback, "top": top, "topProbability": probabilities.get(top)}


def _reask(reply: dict, _: dict) -> dict:
    return {"score": _number(reply["answers"]["reask"], "noul")}


def _guard(reply: dict, _: dict) -> dict:
    kind = reply["answers"]["kind"]
    return {
        "score": _number(reply["answers"]["review"], "noul"),
        "risk": kind.get("choice"),
    }


def _shield(reply: dict, _: dict) -> dict:
    return {"score": _probability(reply["answers"]["jailbreak"], "jailbreak")}


def _check(reply: dict, _: dict) -> dict:
    return {"score": _number(reply["answers"]["hallucination"], "noul")}


def _redact(reply: dict, thresholds: dict) -> dict:
    spans = reply.get("spans", {}).get("pii") or []
    least = thresholds.get("redactPii", 0.7)
    kept = [
        s
        for s in spans
        if isinstance(s, dict)
        and s.get("label") in PII_TAGS
        and (s.get("probability") or 0) >= least
    ]
    tags = sorted({PII_TAGS[s["label"]] for s in kept})
    return {"score": min((s["probability"] for s in kept), default=None), "tags": tags}


def _request(app: str, given: object) -> tuple[object, dict]:
    """The state and questions the mod sends for one input, framed as the hooks frame it."""
    if app == "router":
        return given, {"complexity": COMPLEXITY}
    if app == "escalate":
        return given, {"user_feedback": USER_FEEDBACK}
    if app == "reask":
        return {"current": given["current"], "prior": given["prior"]}, {"reask": REASK}
    if app == "guard":
        return f"Tool: Bash\nCommand: {given}", GUARD_QUESTIONS
    if app == "shield":
        return f"Contents of {given['source']}:\n{given['text']}", {
            "jailbreak": JAILBREAK
        }
    if app == "check":
        return {
            "context": given["context"],
            "request": given["request"],
            "answer": given["answer"],
        }, {"hallucination": HALLUCINATION}
    if app == "redact":
        return given, {"pii": {"preset": "pii"}}
    raise ValueError(f"unknown app {app!r}")


READERS = {
    "router": _router,
    "escalate": _escalate,
    "reask": _reask,
    "guard": _guard,
    "shield": _shield,
    "check": _check,
    "redact": _redact,
}


def _headers(token: str | None) -> dict:
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    return headers


def on_this_machine(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def cleartext_remote(url: str) -> bool:
    """True when requests to `url` cross a network unencrypted: plain http to a host other than this machine."""
    return urllib.parse.urlsplit(url).scheme == "http" and not on_this_machine(url)


def base_of(url: str) -> str:
    return (
        url[: -len("/v1/decisions")]
        if url.endswith("/v1/decisions")
        else url.rstrip("/")
    )


def health(url: str, token: str | None = None, timeout: float = 3.0) -> dict:
    """GET /health and /v1/models; never raises."""
    base = base_of(url)
    out: dict = {"url": url}
    for name, path in (("health", "/health"), ("models", "/v1/models")):
        started = time.perf_counter()
        try:
            request = urllib.request.Request(base + path, headers=_headers(token))
            with urllib.request.urlopen(request, timeout=timeout) as response:
                out[name] = json.loads(response.read())
        except urllib.error.HTTPError as error:
            out[name] = {"error": f"HTTP {error.code}"}
        except (OSError, ValueError) as error:
            out[name] = {"error": str(getattr(error, "reason", error))}
        out[f"{name}Ms"] = round((time.perf_counter() - started) * 1000)
    status = out["health"].get("status") if isinstance(out["health"], dict) else None
    out["ready"] = status == "ready"
    models = out["models"].get("data") if isinstance(out["models"], dict) else None
    out["model"] = models[0].get("id") if models else out["health"].get("model")
    # vllm-srun names the device the model runs on in /v1/models; older builds leave it out.
    out["device"] = (
        (models[0].get("device") or models[0].get("accelerator")) if models else None
    )
    return out


def ask(
    url: str,
    app: str,
    given: object,
    token: str | None = None,
    timeout: float = 30.0,
    thresholds: dict | None = None,
) -> dict:
    """One decision call for one app's input: its score (and extras) and the latency."""
    state, questions = _request(app, given)
    body = json.dumps({"state": state, "questions": questions}).encode()
    started = time.perf_counter()
    try:
        request = urllib.request.Request(url, body, _headers(token))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            reply = json.loads(response.read())
        result = READERS[app](reply, thresholds or {})
    except urllib.error.HTTPError as error:
        result = {"error": f"HTTP {error.code}"}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result = {"error": str(getattr(error, "reason", error))}
    result["ms"] = round((time.perf_counter() - started) * 1000)
    return result


def load_probes(path: Path = PROBES) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


THRESHOLD_OF = {
    "guard": "guardReview",
    "shield": "shieldInject",
    "check": "checkUnsupported",
    "escalate": "escalateFeedback",
    "reask": "escalateReask",
    "redact": "redactPii",
}


def verdict(app: str, result: dict, thresholds: dict) -> str | None:
    """What the mod would do with this score: `acts` or `passes`, or a router tier."""
    score = result.get("score")
    if app == "redact":
        return (
            None if "tags" not in result else ("acts" if result["tags"] else "passes")
        )
    if score is None:
        return None
    if app == "router":
        if (result.get("confidence") or 0) < thresholds.get("routerMinConfidence", 0.3):
            return "kept"
        if score < thresholds.get("routerEasyBelow", 0.5):
            return "haiku"
        return "sonnet" if score < thresholds.get("routerHardFrom", 1.5) else "opus"
    return "acts" if score >= thresholds.get(THRESHOLD_OF[app], 1.0) else "passes"


def run_probes(
    url: str,
    token: str | None,
    thresholds: dict,
    apps: set[str] | None,
    smoke: bool,
    tolerance: float,
) -> list[dict]:
    """Each probe's score now, judged at the default thresholds its `expect` was measured at
    (so a flip means the model changed), and the verdict under `thresholds` as `yours`.
    """
    defaults = thresholds_of(None)
    rows = []
    for case in load_probes():
        owner = "escalate" if case["app"] == "reask" else case["app"]
        if apps is not None and owner not in apps:
            continue
        if smoke and not case.get("smoke"):
            continue
        got = ask(url, case["app"], case["input"], token, thresholds=defaults)
        expected = case.get("measured")
        drift = (
            None
            if expected is None or got.get("score") is None
            else round(got["score"] - expected, 2)
        )
        now = verdict(case["app"], got, defaults)
        rows.append(
            {
                "app": case["app"],
                "name": case["name"],
                "measured": expected,
                "got": got,
                "expect": case["expect"],
                "now": now,
                "yours": verdict(case["app"], got, thresholds),
                "status": (
                    "error"
                    if "error" in got
                    else (
                        "flip"
                        if now != case["expect"]
                        else (
                            "drift"
                            if drift is not None and abs(drift) > tolerance
                            else "ok"
                        )
                    )
                ),
                "drift": drift,
            }
        )
    return rows


def _read_token(token_file: str | None) -> str | None:
    if token_file:
        return Path(token_file).expanduser().read_text().strip() or None
    return os.environ.get("SR_RUNTIME_TOKEN") or None


def thresholds_of(plugin_id: str | None) -> dict:
    """The thresholds vSR runs with: the saved settings under `plugin_id`, else the defaults."""
    saved = settings.load(settings.settings_path())[1] if plugin_id else {}
    return settings.effective_options(saved, plugin_id or "")


def _print_probes(rows: list[dict]) -> None:
    print(
        f"{'app':9} {'case':44} {'measured':>8} {'now':>6} {'ms':>6}  verdict  yours    status"
    )
    for r in rows:
        got = r["got"]
        score = got.get("score")
        extra = (
            f" conf {got['confidence']:.2f}"
            if got.get("confidence") is not None
            else ""
        )
        extra += f" tags {','.join(got['tags'])}" if "tags" in got else ""
        shown = "-" if score is None else f"{score:.2f}"
        measured = "-" if r["measured"] is None else f"{r['measured']:.2f}"
        error = f"  {got['error']}" if "error" in got else ""
        print(
            f"{r['app']:9} {r['name'][:44]:44} {measured:>8} {shown:>6} {got['ms']:>6}  {r['now']!s:7}  {r['yours']!s:7}  {r['status']}{extra}{error}"
        )
    timed = sorted(r["got"]["ms"] for r in rows if "error" not in r["got"])
    if timed:
        print(
            f"\n{len(rows)} probes · median {timed[len(timed) // 2]} ms · slowest {timed[-1]} ms · "
            f"{sum(r['status'] == 'ok' for r in rows)} ok, {sum(r['status'] == 'drift' for r in rows)} drift, "
            f"{sum(r['status'] == 'flip' for r in rows)} flip, {sum(r['status'] == 'error' for r in rows)} error"
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("health", "probe"))
    parser.add_argument(
        "--url", default=DEFAULT_URL, help="the runtime's /v1/decisions URL"
    )
    parser.add_argument(
        "--token-file", help="a file holding the Bearer token (else $SR_RUNTIME_TOKEN)"
    )
    parser.add_argument("--apps", help="comma-separated apps to probe (default: all)")
    parser.add_argument("--smoke", action="store_true", help="one probe per app")
    parser.add_argument(
        "--id",
        help="use the thresholds saved under this plugin id (default: the defaults)",
    )
    parser.add_argument("--tolerance", type=float, default=0.15)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    token = _read_token(args.token_file)
    if cleartext_remote(args.url):
        print(
            f"warning: {args.url} is plain http to another machine, so requests{' and the token' if token else ''} "
            "travel unencrypted; https is recommended",
            file=sys.stderr,
        )
    if args.command == "health":
        out = health(args.url, token)
        print(
            json.dumps(out, indent=2)
            if args.json
            else f"{'ready' if out['ready'] else 'NOT ready'} · {args.url} · model {out['model']} · health {out['healthMs']} ms"
            + ("" if out["ready"] else f" · {out['health']}")
        )
        return 0 if out["ready"] else 1
    apps = set(args.apps.split(",")) if args.apps else None
    rows = run_probes(
        args.url, token, thresholds_of(args.id), apps, args.smoke, args.tolerance
    )
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        _print_probes(rows)
    return 1 if any(r["status"] == "error" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
