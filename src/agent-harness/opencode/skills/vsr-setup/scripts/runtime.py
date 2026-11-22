#!/usr/bin/env python3
"""Whether a vLLM Semantic Router runtime answers at a URL: its health and the models it lists.

`vsr_check` asks each app's case once vSR runs with the runtime; this only says whether it answers,
so a runtime on another machine is checked before the user connects it.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

LOCAL_URL = "http://127.0.0.1:8790/v1/decisions"
# Above this per question, the hooks' 1.5 to 3 s waits are often reached and the apps step aside.
SLOW_MS = 1000


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


def as_data_listing(listing):
    """A model listing in the `data` form. The vllm-sr/5.0 schema, a Router listener's default,
    lists models under `models`, each with its name, its question types as `features` and the
    vllm-sr/4.0 entry as its `card`."""
    if (
        not isinstance(listing, dict)
        or "data" in listing
        or not isinstance(listing.get("models"), list)
    ):
        return listing
    return {
        "data": [
            (
                m["card"]
                if isinstance(m.get("card"), dict)
                else {"id": m.get("name"), "question_types": m.get("features")}
            )
            for m in listing["models"]
            if isinstance(m, dict)
        ]
    }


def _get(url: str, timeout: float) -> tuple[object, int]:
    started = time.perf_counter()
    try:
        request = urllib.request.Request(url, headers={"accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            found: object = json.loads(response.read())
    except urllib.error.HTTPError as error:
        found = {"error": f"HTTP {error.code}"}
    except (OSError, ValueError) as error:
        found = {"error": str(getattr(error, "reason", error))}
    return found, round((time.perf_counter() - started) * 1000)


def health(url: str, timeout: float = 3.0) -> dict:
    """GET /health and the model listing, without a token; never raises."""
    base = base_of(url)
    out: dict = {"url": url}
    out["health"], out["healthMs"] = _get(base + "/health", timeout)
    # A System One API, a Router listener's or a hosted one, lists its models at /v1/systemone/models;
    # vllm-srun lists its one model at /v1/models.
    listing, _ = _get(base + "/v1/systemone/models", timeout)
    listing = as_data_listing(listing)
    if not (isinstance(listing, dict) and listing.get("data")):
        listing, _ = _get(base + "/v1/models", timeout)
        listing = as_data_listing(listing)
    status = out["health"].get("status") if isinstance(out["health"], dict) else None
    models = listing.get("data") if isinstance(listing, dict) else None
    models = [m for m in models or [] if isinstance(m, dict)]
    # A Router listener's /health says only that it serves, and a hosted System One API has none;
    # the models they list show they are up.
    out["ready"] = status == "ready" or (
        (status == "ok" or out["health"] == {"error": "HTTP 404"}) and bool(models)
    )
    out["loading"] = status == "loading"
    out["needsToken"] = any(
        isinstance(found, dict) and found.get("error") in ("HTTP 401", "HTTP 403")
        for found in (out["health"], listing)
    )
    out["models"] = [
        {
            "id": m.get("id"),
            "presets": m.get("presets") or [],
            "modalities": m.get("modalities") or [],
            "questionTypes": m.get("question_types") or [],
        }
        for m in models
    ]
    out["device"] = (
        (models[0].get("device") or models[0].get("accelerator")) if models else None
    )
    return out


def time_question(url: str, model: str | None = None, tries: int = 3) -> int | None:
    """The median time of one short question, asked `tries` times; None when none is answered."""
    body = json.dumps(
        {
            **({"model": model} if model else {}),
            "state": "git status",
            "questions": {
                "review": {
                    "type": "noul",
                    "instructions": "Could running this shell command cause serious harm?",
                }
            },
        }
    ).encode()
    timed = []
    for _ in range(tries):
        started = time.perf_counter()
        try:
            request = urllib.request.Request(
                url, body, {"content-type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                json.loads(response.read())
        except (OSError, ValueError):
            continue
        timed.append(round((time.perf_counter() - started) * 1000))
    return sorted(timed)[len(timed) // 2] if timed else None


def slow_note(ms: int | None) -> str | None:
    """Why a runtime this slow is a poor fit, or None when it is fast enough."""
    if ms is None or ms <= SLOW_MS:
        return None
    return (
        f"slow runtime, {ms} ms a question: vSR's apps wait 1.5 to 3 s, so they will often "
        "step aside; a GPU or remote runtime is recommended"
    )


def summary(out: dict) -> list[str]:
    ids = [m["id"] for m in out["models"]]
    if out["ready"]:
        state = "ready"
    elif out["loading"]:
        state = "loading (the first start downloads the model)"
    elif out["needsToken"]:
        state = "refuses requests without a token"
    else:
        state = f"NOT ready ({out['health']})"
    lines = [
        f"{state} · {out['url']} · "
        + (
            f"models {', '.join(map(str, ids))}: name one in /connect"
            if len(ids) > 1
            else f"model {ids[0] if ids else 'none listed'}"
        )
        + f" · health {out['healthMs']} ms"
    ]
    for m in out["models"]:
        reads = [x for x in m["modalities"] if x != "text"]
        if len(out["models"]) > 1 and not (m["presets"] or reads):
            continue
        lines.append(
            f"  {m['id']}: presets {', '.join(m['presets']) or 'none'}"
            f" · reads {', '.join(['text', *reads])}"
            + (
                " · no score questions, so the router steps aside"
                if m["questionTypes"] and "score" not in m["questionTypes"]
                else ""
            )
        )
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("health",))
    parser.add_argument(
        "--url", default=LOCAL_URL, help=f"the /v1/decisions URL (default {LOCAL_URL})"
    )
    parser.add_argument(
        "--time", action="store_true", help="also time a short question"
    )
    parser.add_argument("--model", help="the model to time, for a runtime with several")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if cleartext_remote(args.url):
        print(
            f"warning: {args.url} is plain http to another machine, so what the apps check, and the "
            "token if it takes one, travel unencrypted; https is recommended",
            file=sys.stderr,
        )
    out = health(args.url)
    if args.time and out["ready"]:
        # A runtime with several models answers only a request that names one.
        model = args.model or (
            out["models"][0]["id"] if len(out["models"]) > 1 else None
        )
        out["questionMs"] = time_question(args.url, model)
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print("\n".join(summary(out)))
        if out.get("questionMs") is not None:
            print(f"  {out['questionMs']} ms a question")
    slow = slow_note(out.get("questionMs"))
    if slow:
        print(f"warning: {slow}", file=sys.stderr)
    return 0 if out["ready"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
