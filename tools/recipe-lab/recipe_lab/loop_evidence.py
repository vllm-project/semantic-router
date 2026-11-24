"""Read a frozen DSH development run and pin the evidence behind each observation."""

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import yaml

# The replay receipts' HTTP convention and the near-stream-cutoff timing
# window (seconds) the evidence reader reports as incomplete-at-cutoff.
HTTP_OK = 200
NEAR_CUTOFF_LOW_S = 295
NEAR_CUTOFF_HIGH_S = 305


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def is_number(value):
    """A finite int or float - the shared bounded-numeric predicate."""
    return type(value) in (int, float) and math.isfinite(value)


def fit_status(requested, prompt, context_limit, output_limit):
    """Classify one request against the fast backend's token budget.

    `requested` is the caller's single unambiguous output reservation (or
    None), `prompt` the observed prompt-token estimate (or None); the two
    limits come from the backend card. Statuses are honest about unknown
    inputs: any missing piece yields `unknown`, never a rejection.
    """
    if requested is not None and output_limit is not None and requested > output_limit:
        return "requested_output_exceeds_output_limit"
    if (
        requested is not None
        and context_limit is not None
        and requested > context_limit
    ):
        return "requested_output_exceeds_context"
    if None in (requested, prompt, context_limit, output_limit):
        return "unknown"
    if prompt + requested > context_limit:
        return "source_token_estimate_exceeds_context"
    return "source_token_estimate_within_limits"


class InvalidExecutionError(ValueError):
    """An infrastructure failure cannot establish task quality."""


class ModelEndpointError(InvalidExecutionError):
    """The agent could not finish because its model endpoint stopped serving."""


class FrozenSplitMismatchError(ValueError):
    """The measurement cannot be shared in the requested dataset split."""


def packet_split(packet, requested):
    if "split" not in packet:
        # Historical packets relied on the importer's declared split.
        return requested or "dev"
    declared = packet["split"]
    if declared not in ("dev", "holdout"):
        raise FrozenSplitMismatchError("unrecognized split in frozen task packet")
    if requested is not None and requested != declared:
        raise FrozenSplitMismatchError(
            "requested split conflicts with frozen task packet"
        )
    return declared


def check_termination(termination, case):
    if termination == "model_endpoint_error":
        raise ModelEndpointError(
            f"{case}: model_endpoint_error invalidates the measurement"
        )
    if termination in {
        "runner_error",
        "grader_error",
        "judge_error",
        "official_verifier_timeout",
    }:
        raise InvalidExecutionError(
            f"{case}: {termination} invalidates the measurement"
        )


def check_runtime_result(result, case):
    check_termination(result["termination"], case)
    if result.get("runner_error") or result.get("judge_error"):
        raise InvalidExecutionError(
            f"{case}: runner or grader reported an execution error"
        )


def sha(value):
    check(
        isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value),
        "expected a SHA-256 fingerprint",
    )
    return value


def component(value):
    check(
        isinstance(value, str) and re.fullmatch("[a-zA-Z0-9_][a-zA-Z0-9_.-]*", value),
        f"invalid task or arm name: {value!r}",
    )
    return value


def target(doc, public):
    check(doc.get("version") == "v0.5", "expected a v0.5 configuration")
    entries = [e for e in doc["entrypoints"] if public in e.get("model_names", [])]
    check(len(entries) == 1, "public model must resolve to exactly one entrypoint")
    name = entries[0]["recipe"]
    matches = [r for r in doc["recipes"] if r["name"] == name]
    check(len(matches) == 1, f"expected exactly one recipe named {name!r}")
    return matches[0]


def require_isolated(doc, public):
    recipe = target(doc, public)
    names = [
        name
        for e in doc["entrypoints"]
        if e["recipe"] == recipe["name"]
        for name in e["model_names"]
    ]
    check(names == [public], "target recipe is shared with other public names")
    return recipe


class Files:
    def __init__(self):
        self.refs = {}

    def read(self, path):
        path = path.resolve(strict=True)
        data = path.read_bytes()
        self.refs[str(path)] = digest(data)
        return data

    def json(self, path):
        return json.loads(self.read(path))


def grade(score, expected):
    criteria = score["criteria"]
    if isinstance(criteria, list):
        pairs = [(item["name"], item["passed"]) for item in criteria]
    else:
        pairs = [(key, item["passed"]) for key, item in criteria.items()]
    actual = dict(pairs)
    check(
        expected
        and len(expected) == len(set(expected)) == len(pairs)
        and set(actual) == set(expected),
        "incomplete or duplicate grader criteria",
    )
    check(
        all(type(v) is bool for v in actual.values())
        and score["correct"] is all(actual.values()),
        "invalid strict grade",
    )
    return actual


def traffic(raw, result, public, cards, strong):
    dispatch, responses = {}, {}
    for line in raw.splitlines():
        row = json.loads(line)
        if row["event"] not in ("dispatch", "response"):
            continue
        table = dispatch if row["event"] == "dispatch" else responses
        call = row["call"]
        check(call not in table, f"duplicate {row['event']} for call {call}")
        table[call] = row
    check(
        dispatch
        and set(dispatch) == set(responses)
        and len(dispatch) == result["model_requests"],
        "incomplete request receipts",
    )
    selected, returned = Counter(), Counter()
    tools = strong_tools = near_cutoff = incomplete = 0
    for call, request in dispatch.items():
        reply = responses[call]
        check(request["model"] == public, "public request model mismatch")
        model, upstream = reply.get("selected_model"), reply.get("returned_model")
        if reply.get("http") == HTTP_OK and reply.get("done") is True:
            check(
                model in cards and upstream,
                "complete stream lacks selected or returned model",
            )
        if model and upstream:
            check(
                model in cards and upstream in {model, cards[model]},
                "returned model differs from selection",
            )
        selected[model or "(missing)"] += 1
        returned[upstream or "(missing)"] += 1
        if request["last_role"] == "tool":
            tools += 1
            strong_tools += model == strong
        if reply.get("done") is not True:
            incomplete += 1
            if (
                reply.get("http") == HTTP_OK
                and NEAR_CUTOFF_LOW_S <= reply["elapsed_s"] <= NEAR_CUTOFF_HIGH_S
            ):
                near_cutoff += 1
    return {
        "selected_models": dict(selected),
        "returned_models": dict(returned),
        "tool_results": tools,
        "strong_tool_results": strong_tools,
        "near_300s_incomplete": near_cutoff,
        "incomplete_responses": incomplete,
    }


def check_client(folder, public, window):
    # Read only the active model's effective fields. This file can contain
    # deployment-specific connection settings unrelated to a quality verdict.
    plugins = json.loads((folder / "provider.json").read_bytes())

    def config(name):
        entries = [x["config"] for x in plugins if x["id"] == name]
        check(len(entries) == 1, f"ambiguous or missing DSH plugin {name}")
        return entries[0]

    default = config("agent-default-model")
    check(default["model"] == public, "DSH default public model mismatch")
    provider = config("llm-pi-ai")["providers"][default["provider"]]
    matches = [m for m in provider["models"] if m["id"] == public]
    check(
        len(matches) == 1
        and type(matches[0].get("contextWindow")) is int
        and matches[0]["contextWindow"] == window,
        "DSH model window differs from gateway binding",
    )


def episode(files, root, case, spec, arm, public, config_hash, info, cards, strong):
    name = component(case)
    folder = root / "runs" / name / arm / "main"
    result = files.json(folder / "result.json")
    check_runtime_result(result, name)
    score = files.json(folder / "grade/score.json")
    protocol = files.json(folder / "protocol.json")
    binding = files.json(root / "bindings" / name / (arm + ".json"))
    check(
        type(result["judge_exit"]) is int
        and result["judge_exit"] == 0
        and result["grade"] == score,
        "missing or conflicting independent grade",
    )
    checks = grade(score, spec["expected_criteria"])
    check(
        binding["model"] == result["model"] == protocol["model"] == public
        and arm == result["arm"] == protocol["arm"],
        "binding/episode arm mismatch",
    )
    check(
        binding["source_sha256"] == config_hash, "wrong deployed source configuration"
    )
    active = sha(binding["active_runtime_hash"])
    window = binding["client_window"]
    check(type(window) is int and window > 0, "missing positive client window")
    check_client(folder, public, window)
    check(
        sha(protocol["runner_sha256"]) == info["runner_sha256"]
        and sha(protocol["verified_task_packet_sha256"])
        == files.refs[str((root / "frozen.json").resolve())],
        "runner or frozen packet identity mismatch",
    )
    check(
        sha(result["prompt_sha256"])
        == spec["instruction_sha256"]
        == protocol["prompt_sha256"]
        and sha(result["judge_sha256"]) == spec["pinned_files"][spec["judge"][-1]],
        "task or judge fingerprint mismatch",
    )
    check(protocol["image"] == spec["image"], "task image mismatch")
    for key, declared in (
        ("official_agent_timeout_s", "author_agent_timeout_s"),
        ("official_verifier_timeout_s", "author_verifier_timeout_s"),
    ):
        check(
            protocol[key] == result[key] == spec[declared],
            "task timeout differs from frozen packet",
        )
    check(
        isinstance(result["agent_seconds"], (float, int))
        and not isinstance(result["agent_seconds"], bool)
        and math.isfinite(result["agent_seconds"])
        and result["agent_seconds"] > 0,
        "invalid task duration",
    )
    controls = {
        k: protocol[k]
        for k in (
            "dsh_version",
            "provider_impl_sha256",
            "runner_sha256",
            "request_limit",
            "configured_output_cap",
            "manual_retries",
            "prompt_mutations",
            "docker_network",
            "framework",
            "declared_environment",
            "applied_cpus",
            "applied_memory_mb",
            "storage",
        )
    }
    controls.update(client_window=window, case=spec)
    analysis = traffic(
        files.read(folder / "requests.jsonl"), result, public, cards, strong
    )
    return dict(
        active_sha256=active,
        controls=controls,
        correct=score["correct"],
        passed=sum(checks.values()),
        total=len(checks),
        failed=[k for k, ok in checks.items() if not ok],
        seconds=result["agent_seconds"],
        termination=result["termination"],
        **analysis,
    )


def read_run(args):
    root = Path(args.run).resolve() / "dsh"
    arm = component(args.arm)
    files = Files()
    config = Path(args.config).resolve()
    doc = yaml.safe_load(files.read(config))
    recipe = target(doc, args.public_model)
    cards = {
        m["name"]: m.get("provider_model_id") or m["name"]
        for m in doc["providers"]["models"]
    }
    check(args.strong in cards, "strong model missing from providers.models")
    info = files.json(root / "frozen.json")
    split = packet_split(info, args.split)
    check(
        info["replicates"] == 1 and info["cases"],
        "expected a frozen single replicate with tasks",
    )
    config_hash = files.refs[str(config)]
    cases = {
        case: episode(
            files,
            root,
            case,
            spec,
            arm,
            args.public_model,
            config_hash,
            info,
            cards,
            args.strong,
        )
        for case, spec in sorted(info["cases"].items())
    }
    check(
        len({x["active_sha256"] for x in cases.values()}) == 1,
        "active configuration changed between tasks",
    )
    return {
        "kind": "measurement",
        "arm": arm,
        "split": split,
        "hypothesis": args.hypothesis,
        "based_on": args.based_on,
        "public_model": args.public_model,
        "strong_model": args.strong,
        "recipe": recipe["name"],
        "config": str(config),
        "config_sha256": config_hash,
        "cases": cases,
        "evidence": files.refs,
    }
