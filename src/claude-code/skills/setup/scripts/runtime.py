#!/usr/bin/env python3
"""Ask a vLLM Semantic Router runtime vSR's questions: health, latency, and the probe set.

The questions are the hooks' own, copied into framing.py; test_framing.py fails when the
copy no longer matches hooks/*.ts.
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
from framing import READERS, framed

PROBES = Path(__file__).with_name("probes.jsonl")
# A probe whose score moved more than this from its measured one is reported as drift.
DRIFT_TOLERANCE = 0.15
# Above this median per question, the hooks' 1.5 to 3 s limits are often reached and the apps
# step aside. vela3-lite takes 0.1 to 0.25 s on an x86 CPU with bf16, 0.7 to 2.4 s in float32 on a
# 4-core laptop CPU without it, and about 2 s on an Apple Silicon one.
SLOW_MS = 1000


class _SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    """Follows a redirect, but sends the token only to the scheme, host and port it was given for."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and _origin(newurl) != _origin(req.full_url):
            new.remove_header("Authorization")
        return new


def _origin(url: str) -> tuple:
    parts = urllib.parse.urlsplit(url)
    return parts.scheme, parts.hostname, parts.port


OPENER = urllib.request.build_opener(_SameOriginRedirect)


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
            with OPENER.open(request, timeout=timeout) as response:
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
    state, questions = framed(app, given)
    body = json.dumps({"state": state, "questions": questions}).encode()
    started = time.perf_counter()
    try:
        request = urllib.request.Request(url, body, _headers(token))
        with OPENER.open(request, timeout=timeout) as response:
            reply = json.loads(response.read())
        result = READERS[app](reply, thresholds or settings.defaults())
    except urllib.error.HTTPError as error:
        result = {"error": f"HTTP {error.code}"}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result = {"error": str(getattr(error, "reason", error))}
    result["ms"] = round((time.perf_counter() - started) * 1000)
    return result


def median_ms(results: list[dict]) -> int | None:
    """The median latency of the calls that answered."""
    timed = sorted(r["ms"] for r in results if "error" not in r)
    return timed[len(timed) // 2] if timed else None


def slow_note(median: int | None) -> str | None:
    """Why a runtime this slow is a poor fit, or None when it is fast enough."""
    if median is None or median <= SLOW_MS:
        return None
    return (
        f"slow runtime, a median {median} ms per question: vSR's hooks wait 1.5 to 3 s, "
        "so the apps will often step aside; a GPU or remote runtime is recommended"
    )


def time_question(url: str, token: str | None = None, tries: int = 3) -> int | None:
    """The median time of a short guard question, asked `tries` times; None when it fails."""
    asked = [ask(url, "guard", "git status", token, timeout=10) for _ in range(tries)]
    return median_ms(asked)


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
        if (result.get("confidence") or 0) < thresholds["routerMinConfidence"]:
            return "kept"
        if score < thresholds["routerEasyBelow"]:
            return "haiku"
        return "sonnet" if score < thresholds["routerHardFrom"] else "opus"
    return "acts" if score >= thresholds[THRESHOLD_OF[app]] else "passes"


def run_probes(
    url: str,
    token: str | None,
    thresholds: dict,
    apps: set[str] | None,
    smoke: bool,
    tolerance: float = DRIFT_TOLERANCE,
) -> list[dict]:
    """Each probe's score now, judged at the default thresholds its `expect` was measured at
    (so a flip means the model changed), and the verdict under `thresholds` as `yours`.
    """
    defaults = settings.defaults()
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


def read_token(token_file: str | None) -> str | None:
    """The Bearer token from a file only its owner can read, else from $SR_RUNTIME_TOKEN."""
    if not token_file:
        return os.environ.get("SR_RUNTIME_TOKEN") or None
    path = Path(token_file).expanduser()
    try:
        if path.stat().st_mode & 0o077:
            raise settings.SettingsError(
                f"{path} can be read by other users; a token file must be yours alone "
                f"(chmod 600 {path}), so that a file never meant as a token is not sent"
            )
        return path.read_text().strip() or None
    except OSError as error:
        raise settings.SettingsError(f"cannot read the token file: {error}") from None


def print_probes(rows: list[dict]) -> None:
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
            f"\n{len(rows)} probes · median {median_ms([r['got'] for r in rows])} ms · slowest {timed[-1]} ms · "
            f"{sum(r['status'] == 'ok' for r in rows)} ok, {sum(r['status'] == 'drift' for r in rows)} drift, "
            f"{sum(r['status'] == 'flip' for r in rows)} flip, {sum(r['status'] == 'error' for r in rows)} error"
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("health", "probe"))
    parser.add_argument(
        "--url", help="the runtime's /v1/decisions URL (default: runtimeUrl)"
    )
    parser.add_argument(
        "--token-file", help="a file holding the Bearer token (else $SR_RUNTIME_TOKEN)"
    )
    parser.add_argument("--apps", help="comma-separated apps to probe (default: all)")
    parser.add_argument("--smoke", action="store_true", help="one probe per app")
    parser.add_argument(
        "--id",
        help="use the options saved under this plugin id (default: the defaults)",
    )
    parser.add_argument("--tolerance", type=float, default=DRIFT_TOLERANCE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        token = read_token(args.token_file)
        options = settings.options_for(args.id)
    except settings.SettingsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    url = args.url or options["runtimeUrl"]
    if cleartext_remote(url):
        print(
            f"warning: {url} is plain http to another machine, so requests{' and the token' if token else ''} "
            "travel unencrypted; https is recommended",
            file=sys.stderr,
        )
    if args.command == "health":
        out = health(url, token)
        print(
            json.dumps(out, indent=2)
            if args.json
            else f"{'ready' if out['ready'] else 'NOT ready'} · {url} · model {out['model']} · health {out['healthMs']} ms"
            + ("" if out["ready"] else f" · {out['health']}")
        )
        return 0 if out["ready"] else 1
    apps = set(args.apps.split(",")) if args.apps else None
    rows = run_probes(url, token, options, apps, args.smoke, args.tolerance)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_probes(rows)
    slow = slow_note(median_ms([r["got"] for r in rows]))
    if slow:
        print(f"warning: {slow}", file=sys.stderr)
    return 1 if any(r["status"] == "error" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
