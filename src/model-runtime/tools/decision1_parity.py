"""Parity of the decision1 family with the Decision 1.0 packages' bundled runtime, on the same requests.

    python3 tools/decision1_parity.py reference --package DIR --repo REPO --device cpu|cuda:0 \\
        --panel NAME:REQUESTS.jsonl:COUNT ... --answers REFERENCE.jsonl
    python3 tools/decision1_parity.py native --model REPO --revision REV --cache-dir DIR \\
        --device cpu|rocm:0 --panel NAME:REQUESTS.jsonl:COUNT ... --answers NATIVE.jsonl [--profile P] \\
        [--also REPO ...] [--concurrent N]
    python3 tools/decision1_parity.py compare REFERENCE.jsonl NATIVE.jsonl --output parity.json

``reference`` loads the package with Transformers remote code (``system_one``), the
runtime the packages ship; it is a separate tool process, so the runtime itself
never imports package code. On a GPU both sides run the built-in table's FLA
kernel choices, so they run the same kernels. ``native``
serves the package through ``vllm_srun`` (verification, readiness, the
scheduler) on one profile; ``--also`` serves more built-in models in the same
process on the same device, loaded first, as one GPU process of the default
layout does, and the panel still asks ``--model``; ``--concurrent N`` sends N
prompts at a time, to ``--model`` and to every ``--also`` model at once, so
their batches (and graph captures) meet, and counts the other models' failed
requests. ``compare`` reports, per panel, the prompts whose
answers are byte-identical (canonical JSON; a structured Score legend of the
reference is compared as the canonical JSON the API contract returns), decision
changes, error mismatches and the largest absolute difference of any number,
and lists every differing prompt by ID (no request text).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vllm_srun.accel.autotune import KernelChoices  # noqa: E402
from vllm_srun.registry import builtin  # noqa: E402
from vllm_srun.registry.artifacts import (  # noqa: E402
    canonical_json as canonical,
)

DEVICE_CLASS = "rocm:gfx942"


def panels(specs: list[str]) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for spec in specs:
        name, path, count = spec.split(":")
        with open(path, encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index >= int(count):
                    break
                rows.append((name, json.loads(line)))
    return rows


def pin(repo: str | None, device: str) -> bool:
    """Run this thread's FLA kernels on the built-in table's choices for a GPU run."""
    if device == "cpu" or repo is None:
        return False
    known = builtin.lookup(repo)
    recorded = known.kernel_choices.get(DEVICE_CLASS) if known else None
    if not recorded:
        return False
    choices = KernelChoices(recorded)
    if choices.install() is not None:
        return False
    choices.pin_thread()
    return True


def write(
    sink, panel: str, prompt: dict[str, Any], answers: Any, seconds: float
) -> None:
    record = {
        "panel": panel,
        "id": prompt["id"],
        "answers": answers,
        "seconds": seconds,
    }
    sink.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_reference(args: argparse.Namespace) -> int:
    pinned = pin(args.repo, args.device)
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    from transformers import AutoModel

    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        args.package, trust_remote_code=True, device=args.device
    )
    receipt = {"load_seconds": time.perf_counter() - started, "fla_pinned": pinned}
    with open(args.answers, "x", encoding="utf-8") as sink:
        for panel, prompt in panels(args.panel):
            started = time.perf_counter()
            try:
                answers = model.system_one(
                    state=prompt["state"], questions=prompt["questions"]
                )["answers"]
            except Exception as exc:
                answers = {"request_error": type(exc).__name__}
            write(sink, panel, prompt, answers, time.perf_counter() - started)
    print(json.dumps(receipt))
    return 0


def run_native(args: argparse.Namespace) -> int:
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.runtime import Runtime

    others = tuple(
        ModelConfig(model=repo, device=args.device, profile=args.profile)
        for repo in args.also
    )
    config = ServeConfig(
        models=(
            *others,
            ModelConfig(
                model=args.model,
                revision=args.revision,
                device=args.device,
                profile=args.profile,
            ),
        ),
        cache_dir=args.cache_dir,
        offline=True,
        threads=args.threads,
        exit_on_device_error=False,
    )
    started = time.perf_counter()
    runtime = Runtime(config)
    runtime.start(background=False)
    served = runtime.lookup(args.model)
    receipt = {
        "load_seconds": time.perf_counter() - started,
        "health": served.health.state,
        "golden": served.health.golden.describe(),
        "also": {
            other.label: other.health.golden.describe()
            for other in runtime.served
            if other is not served
        },
    }

    async def ask(model: str, prompt: dict[str, Any]) -> tuple[int, Any, float]:
        body = {
            "model": model,
            "state": prompt["state"],
            "questions": prompt["questions"],
            "options": {"return_meta": False, "profile": args.profile},
        }
        started = time.perf_counter()
        size = len(json.dumps(body, separators=(",", ":")).encode("utf-8"))
        status, response = await runtime.call("decisions", body, size)
        answers = (
            response.get("answers") if status == 200 else {"request_error": status}
        )
        return status, answers, time.perf_counter() - started

    async def wave(batch: list[tuple[str, dict[str, Any]]]) -> list:
        calls = [ask(args.model, prompt) for _, prompt in batch]
        if args.concurrent:
            calls += [ask(repo, prompt) for _, prompt in batch for repo in args.also]
        return await asyncio.gather(*calls)

    rows = panels(args.panel)
    size = max(args.concurrent, 1)
    others_failed = 0
    loop = asyncio.new_event_loop()
    try:
        with open(args.answers, "x", encoding="utf-8") as sink:
            for start in range(0, len(rows), size):
                batch = rows[start : start + size]
                results = loop.run_until_complete(wave(batch))
                mine = results[: len(batch)]
                for (panel, prompt), (_, answers, seconds) in zip(
                    batch, mine, strict=True
                ):
                    write(sink, panel, prompt, answers, seconds)
                others_failed += sum(
                    status != 200 for status, _, _ in results[len(batch) :]
                )
        receipt["engine"] = served.model.engine_model.receipt()
        receipt["concurrent"] = args.concurrent
        receipt["also_failed_requests"] = others_failed
        receipt["health_after"] = {s.label: s.health.state for s in runtime.served}
    finally:
        loop.close()
        runtime.stop()
    print(json.dumps(receipt))
    return 0


def normalized(answers: Any) -> Any:
    """The reference's answers in the API's form: a structured Score legend becomes canonical JSON."""
    if not isinstance(answers, dict):
        return answers
    out = {}
    for question, answer in answers.items():
        out[question] = answer
        if isinstance(answer, dict) and isinstance(answer.get("legend"), dict):
            legend = {
                key: value if isinstance(value, str) else canonical(value)
                for key, value in answer["legend"].items()
            }
            out[question] = {**answer, "legend": legend}
    return out


def category(answer: Any) -> Any:
    if not isinstance(answer, dict) or "error" in answer:
        return ("error", answer.get("error") if isinstance(answer, dict) else None)
    if answer.get("type") == "noul":
        return ("noul", answer["noul"] > 0.5)
    if answer.get("type") == "choice":
        return ("choice", answer["choice"])
    probabilities = answer.get("probabilities") or {}
    return (
        "score",
        max(probabilities, key=probabilities.get) if probabilities else None,
    )


def numbers(answer: Any) -> dict[str, float]:
    if not isinstance(answer, dict):
        return {}
    out = {
        key: float(answer[key])
        for key in ("noul", "score", "confidence")
        if key in answer
    }
    out.update(
        {f"p/{k}": float(v) for k, v in (answer.get("probabilities") or {}).items()}
    )
    return out


def run_compare(args: argparse.Namespace) -> int:
    def load(path: str) -> dict[tuple[str, str], dict[str, Any]]:
        with open(path, encoding="utf-8") as stream:
            return {(row["panel"], row["id"]): row for row in map(json.loads, stream)}

    reference, native = load(args.reference), load(args.native)
    summary: dict[str, dict[str, Any]] = {}
    differing: list[dict[str, Any]] = []
    for key in sorted(reference):
        panel = summary.setdefault(
            key[0],
            {"prompts": 0, "identical": 0, "questions": 0, "decision_changes": 0,
             "error_mismatches": 0, "missing": 0, "max_abs_diff": 0.0},
        )  # fmt: skip
        panel["prompts"] += 1
        if key not in native:
            panel["missing"] += 1
            continue
        left, right = normalized(reference[key]["answers"]), native[key]["answers"]
        if canonical(left) == canonical(right):
            panel["identical"] += 1
            panel["questions"] += len(left) if isinstance(left, dict) else 0
            continue
        worst, changes = 0.0, 0
        both = isinstance(left, dict) and isinstance(right, dict)
        for question in (set(left) | set(right)) if both else ():
            a, b = left.get(question), right.get(question)
            panel["questions"] += 1
            if category(a) != category(b):
                changes += 1
                if ("error" in (a or {})) != ("error" in (b or {})):
                    panel["error_mismatches"] += 1
            na, nb = numbers(a), numbers(b)
            worst = max([worst, *(abs(na[k] - nb[k]) for k in set(na) & set(nb))])
        panel["decision_changes"] += changes
        panel["max_abs_diff"] = max(panel["max_abs_diff"], worst)
        differing.append(
            {
                "panel": key[0],
                "id": key[1],
                "decision_changes": changes,
                "max_abs_diff": worst,
            }
        )
    passed = all(p["identical"] == p["prompts"] for p in summary.values())
    result = {
        "schema": "decision1-parity/1",
        "panels": summary,
        "differing": differing,
        "identical": passed,
    }
    Path(args.output).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"panels": summary, "identical": passed}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("reference")
    reference.add_argument("--package", required=True)
    reference.add_argument(
        "--repo", help="built-in repository ID, for its pinned kernel choices"
    )
    native = commands.add_parser("native")
    native.add_argument("--model", required=True)
    native.add_argument("--revision")
    native.add_argument("--cache-dir")
    native.add_argument("--profile", default="exact")
    native.add_argument(
        "--also",
        action="append",
        default=[],
        metavar="REPO",
        help="another built-in model served in the same process, loaded first",
    )
    native.add_argument(
        "--concurrent",
        type=int,
        default=0,
        metavar="N",
        help="ask N prompts at once, of --model and of every --also model together",
    )
    for command in (reference, native):
        command.add_argument("--device", default="cpu")
        command.add_argument("--panel", action="append", required=True)
        command.add_argument("--answers", required=True)
        command.add_argument("--threads", type=int)
    compare = commands.add_parser("compare")
    compare.add_argument("reference")
    compare.add_argument("native")
    compare.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return {"reference": run_reference, "native": run_native, "compare": run_compare}[
        args.command
    ](args)


if __name__ == "__main__":
    sys.exit(main())
