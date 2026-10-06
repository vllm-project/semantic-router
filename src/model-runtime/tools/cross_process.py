"""Answers that repeat across GPU processes: the same prompts answered by separate processes (records).

    python3 tools/cross_process.py answer MODEL [--revision REV] [--device rocm:0] [--autotune-cache DIR]
        --panel NAME:PROMPTS.jsonl:COUNT ... --answers OUT.jsonl --receipt OUT.json
        [--cache-dir DIR] [--base-path DIR] [--offline]
    python3 tools/cross_process.py compare REFERENCE.jsonl OTHER.jsonl ... --output OUT.json

answer   loads MODEL as ``vllm-srun serve`` does (resolution, verification, placement, the golden check that
         gates readiness, ``--autotune-cache``) and answers every prompt as one request on the exact profile through
         the scheduler. The receipt records the golden check, the device, the library versions and the autotune
         entries in the cache before and after the run (no new entry: every choice was reused).
compare  every other answers file against the first, prompt by prompt: identical prompts (canonical JSON), category
         changes, values that differ and the largest absolute difference of any probability, Noul or Score value.
         Exits 1 unless every prompt is identical.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu_parity import canonical, category, numbers
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.runtime import Runtime


def autotune_entries(directory: str | None) -> int | None:
    if not directory or not Path(directory).is_dir():
        return None
    return sum(1 for _ in Path(directory).rglob("*.autotune.json"))


def versions() -> dict[str, str | None]:
    found = {}
    for name in ("torch", "triton", "fla"):
        module = sys.modules.get(name)
        found[name] = getattr(module, "__version__", None) if module else None
    return found


def panels(specs: list[str]) -> list[tuple[str, list[dict[str, Any]]]]:
    loaded = []
    for spec in specs:
        name, path, count = spec.split(":")
        with open(path, encoding="utf-8") as stream:
            prompts = [json.loads(line) for line in stream][: int(count)]
        loaded.append((name, prompts))
    return loaded


def answer(args: argparse.Namespace) -> int:
    before = autotune_entries(args.autotune_cache)
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(
                    model=args.model, revision=args.revision, device=args.device
                ),
            ),
            cache_dir=args.cache_dir,
            base_path=args.base_path,
            offline=args.offline,
            autotune_cache=args.autotune_cache,
        )
    )
    started = time.perf_counter()
    try:
        runtime.load()
        load_seconds = time.perf_counter() - started
        served = runtime.lookup(None)
        assert served.model is not None and served.placement is not None
        count = 0
        started = time.perf_counter()
        with args.answers.open("x", encoding="utf-8") as sink:
            for name, prompts in panels(args.panel):
                for prompt in prompts:
                    prepared = runtime.prepare(
                        "decisions",
                        {
                            "state": prompt["state"],
                            "questions": prompt["questions"],
                            "options": {"profile": "exact", "return_meta": False},
                        },
                    )
                    results = served.submit_items(
                        prepared.plan.items, None, "exact"
                    ).result()
                    answers = runtime.finish(prepared, results, 0.0, 0.0)
                    row = {
                        "panel": name,
                        "id": prompt["id"],
                        "answers": answers["answers"],
                    }
                    sink.write(json.dumps(row) + "\n")
                    count += 1
        info, engine = served.model.info, served.model.engine_model
        receipt = {
            "schema": "model-runtime-cross-process/1",
            "model": info.id,
            "repo": info.repo,
            "revision": info.revision,
            "model_sha256": info.model_sha256,
            "device": served.placement.device.label,
            "device_name": served.placement.device.name,
            "golden": served.health.golden.describe(),
            "autotune_cache": args.autotune_cache,
            "autotune_entries": {
                "before": before,
                "after": autotune_entries(args.autotune_cache),
            },
            "versions": versions(),
            "prompts": count,
            "load_seconds": load_seconds,
            "answer_seconds": time.perf_counter() - started,
            "fast_path": engine.receipt() if hasattr(engine, "receipt") else None,
        }
    finally:
        runtime.stop()
    with args.receipt.open("x", encoding="utf-8") as sink:
        json.dump(receipt, sink, indent=1)
    print(json.dumps({key: receipt[key] for key in ("model", "golden", "prompts")}))
    return 0


def read_answers(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    return {(row["panel"], row["id"]): row["answers"] for row in rows}


def compare_one(
    reference: dict[tuple[str, str], dict[str, Any]],
    other: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "prompts": len(reference),
        "identical_prompts": 0,
        "category_changes": 0,
        "missing": 0,
        "differing_values": 0,
        "max_abs_drift": 0.0,
    }
    drifts = []
    for key, expected in reference.items():
        answers = other.get(key)
        if answers is None:
            totals["missing"] += 1
            continue
        totals["identical_prompts"] += canonical(answers) == canonical(expected)
        for question_id in set(answers) | set(expected):
            if question_id not in answers or question_id not in expected:
                totals["missing"] += 1
                continue
            totals["category_changes"] += category(answers[question_id]) != category(
                expected[question_id]
            )
            left, right = numbers(answers[question_id]), numbers(expected[question_id])
            for name in set(left) & set(right):
                drift = abs(left[name] - right[name])
                if drift:
                    drifts.append(drift)
                    totals["max_abs_drift"] = max(totals["max_abs_drift"], drift)
    totals["missing"] += sum(1 for key in other if key not in reference)
    totals["differing_values"] = len(drifts)
    totals["median_abs_drift"] = statistics.median(drifts) if drifts else 0.0
    return totals


def compare(args: argparse.Namespace) -> int:
    reference = read_answers(args.files[0])
    others = {
        str(path): compare_one(reference, read_answers(path)) for path in args.files[1:]
    }
    passed = all(
        entry["identical_prompts"] == entry["prompts"] and not entry["missing"]
        for entry in others.values()
    )
    result = {
        "schema": "model-runtime-cross-process-compare/1",
        "reference": str(args.files[0]),
        "others": others,
        "passed": passed,
    }
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump(result, sink, indent=1)
    print(json.dumps(result))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("answer")
    run.add_argument("model")
    run.add_argument("--revision")
    run.add_argument("--device", default="rocm:0")
    run.add_argument("--cache-dir")
    run.add_argument("--base-path")
    run.add_argument("--offline", action="store_true")
    run.add_argument("--autotune-cache")
    run.add_argument("--panel", action="append", required=True)
    run.add_argument("--answers", type=Path, required=True)
    run.add_argument("--receipt", type=Path, required=True)
    check = commands.add_parser("compare")
    check.add_argument("files", type=Path, nargs="+")
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare" and len(args.files) < 2:
        parser.error("compare needs a reference and at least one other answers file")
    return answer(args) if args.command == "answer" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
