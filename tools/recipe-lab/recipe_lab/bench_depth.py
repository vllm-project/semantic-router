"""Propose a Speed cutoff from graded development calls and a native Router preview."""

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import yaml

from .deployed_config import config_identity
from .loop import publish_config
from .loop_evidence import check, digest, fit_status, require_isolated, sha
from .semantic_depth import propose_threshold, score_value, threshold_sites
from .srbench import SrbenchCli, SrbenchError, encoded, save_result, subject


def first_calls(client, run, targets):
    """The first subject call precedes any divergence in the two agent trajectories."""
    calls = {}
    cursor = 0
    while True:
        page = client.run_command(
            ["show", run, "--calls", "--after", str(cursor), "--limit", "500"]
        )
        for row in page["calls"]:
            if row["role"] == "subject" and row["target_id"] in targets:
                calls.setdefault((row["target_id"], row["case_id"]), row["id"])
        following = page["next_cursor"]
        if following is None:
            return calls
        check(type(following) is int and following > cursor, "invalid call cursor")
        cursor = following


def result_rows(client, run, target, expected):
    """Every graded result cell of one target, paged from the service."""
    rows = {}
    cursor = 0
    while True:
        page = client.run_command(
            ["show", run, "--results", "--after", str(cursor), "--limit", "500"]
        )
        for row in page["results"]:
            if row["target_id"] != target:
                continue
            case = row["case_id"]
            check(
                case in expected and case not in rows,
                "unexpected or duplicate result cell",
            )
            check(
                row["status"] == "completed" and isinstance(row["correct"], bool),
                f"missing terminal grade for {case}",
            )
            rows[case] = row
        following = page["next_cursor"]
        if following is None:
            break
        check(
            type(following) is int and following > cursor,
            "invalid sr-bench results cursor",
        )
        cursor = following
    check(set(rows) == expected, "missing paired result cells")
    return rows


def preview_request(body):
    # The native preview accepts routing inputs. Sampling and streaming settings
    # still have to agree between the two live calls before reaching this point.
    request = dict(body)
    for field in ("stream", "stream_options", "temperature", "top_p", "seed"):
        request.pop(field, None)
    if type(request.get("n")) is int and request["n"] == 1:
        del request["n"]
    allowed = {
        "messages",
        "tools",
        "functions",
        "tool_choice",
        "function_call",
        "response_format",
        "max_tokens",
        "max_completion_tokens",
        "metadata",
    }
    return request, sorted(request.keys() - allowed)


def preview(args, request):
    request = dict(request)
    request["model"] = args.public_model
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "request.json"
        path.write_bytes(encoded(request))
        run = subprocess.run(
            [
                args.vllm_sr,
                "route",
                "preview",
                "--endpoint",
                args.endpoint,
                "--request-file",
                str(path),
                "--trace",
                "--json",
            ],
            capture_output=True,
            encoding="utf-8",
            timeout=120,
            check=False,
        )
    check(
        run.returncode == 0,
        "native route preview failed; check endpoint and supported request fields",
    )
    return json.loads(run.stdout)


def fast_fit(request, fast_call, card):
    """Rule out known capability or capacity failures before naming a possible move."""
    caps = {c.strip().lower() for c in card.get("capabilities", [])}
    messages = request["messages"]
    tool_history = any(
        m.get("role") in ("tool", "function")
        or m.get("tool_calls")
        or m.get("function_call")
        for m in messages
    )
    offers_tools = bool(request.get("tools") or request.get("functions"))
    if (
        offers_tools
        or tool_history
        or request.get("tool_choice", "none") != "none"
        or request.get("function_call", "none") != "none"
    ) and caps.isdisjoint({"tools", "tool_use"}):
        return "fast backend lacks declared tools capability for this request"
    if request.get("response_format") or any(
        isinstance(m.get("content"), list)
        and any(p.get("type") != "text" for p in m["content"])
        for m in messages
    ):
        return (
            "structured or multimodal request requires native eligibility verification"
        )
    limits = [
        request[k] for k in ("max_tokens", "max_completion_tokens") if k in request
    ]
    if limits and not all(type(v) is int and v > 0 and v == limits[0] for v in limits):
        return "ambiguous or unsupported output reservation"
    prompt = (fast_call.get("raw_usage") or {}).get("prompt_tokens")
    status = fit_status(
        limits[0] if limits else None,
        prompt if type(prompt) is int and prompt >= 0 else None,
        card.get("context_window_size"),
        card.get("max_output_tokens"),
    )
    if status not in ("unknown", "source_token_estimate_within_limits"):
        return status
    return None


def observation(args, site, runtime, calls, marks, case, cards):
    parent, fast = calls
    parent_grade, fast_grade = marks
    for row, target in zip(calls, (args.parent_target, args.fast_target), strict=False):
        check(
            row["target_id"] == target
            and row["case_id"] == case
            and row["role"] == "subject",
            "call detail belongs to a different target or case",
        )
    if not parent_grade["correct"] or any(
        row["status"] != "completed" for row in calls
    ):
        return {
            "case": case,
            "role": "skip",
            "reason": "missing successful parent task or completed opening calls",
        }
    if (
        parent.get("decision") != site["primary"]
        or parent.get("selected_model") != args.strong
    ):
        return {
            "case": case,
            "role": "skip",
            "reason": "opening request did not use the strong reasoning lane",
            "decision": parent.get("decision"),
            "selected": parent.get("selected_model"),
        }
    strong_id = cards[args.strong].get("provider_model_id") or args.strong
    fast_id = cards[args.fast].get("provider_model_id") or args.fast
    if parent.get("model") not in (args.strong, strong_id) or fast.get("model") not in (
        args.fast,
        fast_id,
    ):
        return {
            "case": case,
            "role": "skip",
            "reason": "response model identities do not match bindings",
        }
    check(
        parent.get("config_hash") == runtime,
        "parent call did not acknowledge the frozen runtime",
    )
    bodies = [
        {k: v for k, v in row["request"]["effective_body"].items() if k != "model"}
        for row in calls
    ]
    if bodies[0] != bodies[1]:
        return {
            "case": case,
            "role": "skip",
            "reason": "effective opening requests differ",
        }
    request, unsupported = preview_request(bodies[0])
    if unsupported:
        return {
            "case": case,
            "role": "skip",
            "reason": "opening request uses fields outside the native preview contract",
            "unsupported_fields": unsupported,
        }
    blocked = fast_fit(request, fast, cards[args.fast])
    if blocked:
        return {"case": case, "role": "skip", "reason": blocked}
    result = preview(args, request)
    check(
        result.get("config_hash") == runtime
        and result.get("requested_model") == args.public_model
        and result.get("selection_status") == "selected"
        and result.get("selected_model") == args.strong
        and result.get("decision_result", {}).get("decision_name") == site["primary"],
        "preview configuration or selection differs from the live parent",
    )
    trace = result["eval_trace"]
    traced = {item["decision_name"]: item["matched"] for item in trace}
    check(
        len(traced) == len(trace)
        and set(traced) == set(site["bindings"])
        and all(type(value) is bool for value in traced.values())
        and traced[site["primary"]],
        "missing complete decision trace for the parent",
    )
    key = "systemone:" + site["signal"]
    score = result.get("signal_values", {}).get(key)
    if (
        not score_value(score)
        or score < site["before"]
        or key in (result.get("signal_errors") or {})
    ):
        return {
            "case": case,
            "role": "skip",
            "reason": "no comparable observed reasoning score",
        }
    times = [row.get("latency_s") for row in marks]
    check(
        all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in times),
        "missing comparable whole-task timings",
    )
    move = fast_grade["correct"] and times[1] < times[0] and not traced[site["overlap"]]
    return {
        "case": case,
        "role": "move" if move else "protect",
        "score": score,
        "parent_call": parent["id"],
        "fast_call": fast["id"],
        "request_sha256": digest(encoded(bodies[0])),
        "fast_correct": fast_grade["correct"],
        "parent_finish_reason": parent.get("finish_reason"),
        "fast_finish_reason": fast.get("finish_reason"),
        "requested_output_limit": request.get(
            "max_completion_tokens", request.get("max_tokens")
        ),
        "parent_seconds": times[0],
        "fast_seconds": times[1],
        "overlap": traced[site["overlap"]],
    }


def suggest(args, receipt):
    raw, doc, deployed_hash = config_identity(args.config, args.deployed_config)
    recipe = require_isolated(doc, args.public_model)
    sites = [
        s
        for s in threshold_sites(recipe, args.strong, args.fast)
        if s["signal"] == args.signal
    ]
    check(
        len(sites) == 1,
        "no supported isolated Speed threshold with these model bindings",
    )
    site = sites[0]
    bench = SrbenchCli(args.vllm_sr)
    run = bench.run_command(["show", args.dev_run])
    manifest = run["manifest"]
    cases = manifest["cases"]
    check(
        run["id"] == args.dev_run
        and run["status"] == "completed"
        and manifest["mode"] == "live"
        and manifest["profile"] in ("smoke", "quick")
        and not manifest.get("execution_cells")
        and not manifest.get("recovery")
        and all(c.get("metadata", {}).get("split", "dev") == "dev" for c in cases),
        "expected a completed development run with no holdout cases",
    )
    parent = subject(manifest, args.parent_target)
    direct = subject(manifest, args.fast_target)
    cards = {card["name"]: card for card in doc["providers"]["models"]}
    fast_id = cards[args.fast].get("provider_model_id") or args.fast
    check(
        parent["kind"] == "mom"
        and parent["model"] == args.public_model
        and parent.get("capture_recipe") is True
        and direct["kind"] == "single"
        and direct["model"] in (args.fast, fast_id),
        "expected the captured parent and its direct fast backend",
    )
    report = bench.run_command(["report", args.dev_run])
    check(
        report["run_id"] == args.dev_run and report["status"] == "completed",
        "parent report mismatch",
    )
    snapshot = report["provenance"]["runner"]["recipe_snapshots"][args.parent_target]
    runtime = sha(parent["config_hash"])
    check(
        snapshot["source_config_hash"] == deployed_hash
        and snapshot["active_runtime_hash"] == runtime
        and snapshot["generated_runtime_hash"] == runtime,
        "parent source or active runtime differs",
    )
    targets = (args.parent_target, args.fast_target)
    ids = {c["id"] for c in cases}
    rows = [result_rows(bench, args.dev_run, target, ids) for target in targets]
    calls = first_calls(bench, args.dev_run, targets)
    observations = []
    receipt.update(
        parent_sha256=digest(raw),
        deployed_source_sha256=deployed_hash,
        parent_runtime_hash=runtime,
        output_policy=manifest.get("output_policy", "bounded"),
        observations=observations,
    )
    for case in sorted(ids):
        if any((target, case) not in calls for target in targets):
            observations.append(
                {
                    "case": case,
                    "role": "skip",
                    "reason": "missing paired opening subject calls",
                }
            )
            continue
        pair = [
            bench.run_command(["show", args.dev_run, "--call-id", calls[target, case]])
            for target in targets
        ]
        check(
            all(
                row["id"] == calls[target, case]
                for row, target in zip(pair, targets, strict=False)
            ),
            "call ID mismatch",
        )
        observations.append(
            observation(
                args,
                site,
                runtime,
                pair,
                [arm[case] for arm in rows],
                case,
                cards,
            )
        )
    movable = [r["score"] for r in observations if r["role"] == "move"]
    protected = [r["score"] for r in observations if r["role"] == "protect"]
    proposal = propose_threshold(doc, args.public_model, site, movable, protected)
    if not proposal:
        receipt.update(
            status="keep-parent",
            reason="no observed boundary separates a faster successful fast case from a protected case",
        )
        return
    draft, predicate = proposal
    candidate_sha = publish_config(
        args.out, yaml.safe_dump(draft, sort_keys=False, allow_unicode=True)
    )
    receipt.update(
        status="proposed",
        before=site["before"],
        after=predicate["after"],
        candidate=str(args.out.resolve()),
        candidate_sha256=candidate_sha,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="recipe-lab suggest-depth", description=__doc__
    )
    for option in (
        "public-model",
        "strong",
        "fast",
        "dev-run",
        "parent-target",
        "fast-target",
        "endpoint",
    ):
        parser.add_argument("--" + option, required=True)
    for name in ("config", "workspace", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--deployed-config",
        type=Path,
        help="materialized parent YAML with its native serve provenance",
    )
    parser.add_argument("--signal", default="depth")
    parser.add_argument("--vllm-sr", default="vllm-sr")
    args = parser.parse_args(argv)
    receipt = {
        "schema": "recipe-lab-sr-bench-depth-v1",
        "status": "inconclusive",
        "dev_run": args.dev_run,
        "parent_target": args.parent_target,
        "fast_target": args.fast_target,
        "signal": args.signal,
    }
    try:
        suggest(args, receipt)
    except yaml.YAMLError:
        receipt["reason"] = "invalid configuration YAML"
    except (
        SrbenchError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        receipt["reason"] = str(exc)
    try:
        save_result(args.workspace, receipt)
    except OSError as exc:
        receipt.update(
            status="inconclusive", reason=f"could not save proposal receipt: {exc}"
        )
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))
    return 2 if receipt["status"] == "inconclusive" or "report_error" in receipt else 0
