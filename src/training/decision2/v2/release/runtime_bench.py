"""Single-request latency and GPU memory of one package's own runtime, in-process on one GPU.

Run like ``examples.py`` (isolated interpreter, package read-only, no network):

    python -I -B v2/release/runtime_bench.py run --package PKG --prompts PROMPTS.jsonl \
        --count 400 --warmup 20 --output OUT.json [--base-path B] [--site DIR] [--require-kernels]
    python3 v2/release/runtime_bench.py compare OLD.json NEW.json --output CMP.json

``run`` loads the package with its vendored runtime on cuda:0 (so an old and a
new revision each measure their own runtime), answers the first ``warmup``
prompts ``--warmup-passes`` times untimed, then times the first ``count`` prompts one request at a time (all of a
prompt's questions in one call, as the release parity does) with
``torch.cuda.synchronize()`` around each call. It records p50 / p95 / mean
latency, items per second, the GPU memory allocated after loading, the peak
while loading and the peak during the timed requests (``max_memory_allocated``
after a reset), loaded elements by dtype, and the answers in
``OUT.answers.jsonl`` beside the receipt (gold-free prompts; stays on the node)
with their digest. ``compare`` checks two runs over the same prompts for
identical answers (release-parity definitions) and reports the latency and
memory change.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "dev2-runtime-bench/1"
GIB = 2**30


def examples_module():
    path = Path(__file__).resolve().with_name("examples.py")
    spec = importlib.util.spec_from_file_location("dev2_release_examples", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def run(args: argparse.Namespace, ex: Any) -> dict[str, Any]:
    import torch

    prompts = ex.read_jsonl(args.prompts)[: args.count]
    if len(prompts) != args.count or args.warmup > args.count:
        raise ValueError("the prompts file has fewer prompts than --count")
    torch.cuda.reset_peak_memory_stats()
    model, load_seconds = ex.load_package(
        args.package, "cuda:0", args.threads, args.base_path, args.fp32_master
    )
    kernels = ex.kernel_runtime(args.require_kernels)
    torch.cuda.synchronize()
    after_load = torch.cuda.memory_allocated()
    load_peak = torch.cuda.max_memory_allocated()
    for _ in range(args.warmup_passes):
        for prompt in prompts[: args.warmup]:
            model.system_one(state=prompt["state"], questions=prompt["questions"])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    latencies, rows, tokens = [], [], 0
    for prompt in prompts:
        torch.cuda.synchronize()
        started = time.perf_counter()
        response = model.system_one(
            state=prompt["state"], questions=prompt["questions"]
        )
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - started)
        tokens += response["usage"]["input_tokens"]
        rows.append({"id": prompt["id"], "answers": response["answers"]})
    request_peak = torch.cuda.max_memory_allocated()
    answers = args.output.with_name(
        args.output.name.removesuffix(".json") + ".answers.jsonl"
    )
    with answers.open("x", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    module = model.backend.model
    fast = getattr(model.backend, "fast", None)
    milliseconds = [1000 * value for value in latencies]
    return {
        "schema": SCHEMA,
        "mode": "run",
        "package_manifest_sha256": ex.sha_file(args.package / "MODEL_MANIFEST.json"),
        "model_name": model.model_name,
        "profile": model.manifest["profile"],
        "device_name": torch.cuda.get_device_name(0),
        "fp32_master": args.fp32_master,
        "runtime": {
            **ex.runtime_versions(),
            "sites": args.site,
            "kernels": kernels,
            "residency": getattr(model.backend, "residency", None),
            "parameter_dtypes": ex.parameter_dtypes(model.backend),
            "fast_path": fast.receipt() if fast is not None else None,
        },
        "parameter_bytes": sum(
            p.numel() * p.element_size() for p in module.parameters()
        ),
        "load_seconds": load_seconds,
        "memory_gib": {
            "after_load": after_load / GIB,
            "load_peak": load_peak / GIB,
            "request_peak": request_peak / GIB,
        },
        "prompts_sha256": ex.sha_file(args.prompts),
        "ids_sha256": ex.digest([p["id"] for p in prompts]),
        "count": len(prompts),
        "warmup": args.warmup,
        "warmup_passes": args.warmup_passes,
        "input_tokens_mean": tokens / len(prompts),
        "latency_ms": {
            "p50": percentile(milliseconds, 0.50),
            "p95": percentile(milliseconds, 0.95),
            "mean": sum(milliseconds) / len(milliseconds),
            "min": min(milliseconds),
            "max": max(milliseconds),
        },
        "items_per_s": len(prompts) / sum(latencies),
        "answers_file": answers.name,
        "answers_sha256": ex.digest(rows),
        "passed": True,
    }


def compare(args: argparse.Namespace, ex: Any) -> dict[str, Any]:
    sides = {}
    for name, path in (("old", args.old), ("new", args.new)):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        rows = ex.read_jsonl(path.with_name(receipt["answers_file"]))
        if ex.digest(rows) != receipt["answers_sha256"]:
            raise ValueError(f"{path}: answers differ from the receipt digest")
        sides[name] = (receipt, rows)
    (old, old_rows), (new, new_rows) = sides["old"], sides["new"]
    totals = {"slots": 0, "category_changes": 0, "missing": 0, "max_abs_drift": 0.0}
    identical = 0
    for left, right in zip(old_rows, new_rows):
        if left["id"] != right["id"]:
            raise ValueError("the two runs answered different prompts")
        result = ex.compare_answers(left["answers"], right["answers"])
        identical += ex.canonical(left["answers"]) == ex.canonical(right["answers"])
        for key in ("slots", "category_changes", "missing"):
            totals[key] += result[key]
        totals["max_abs_drift"] = max(totals["max_abs_drift"], result["max_abs_drift"])

    def change(group: str, key: str) -> dict[str, float]:
        a, b = old[group][key], new[group][key]
        return {"old": a, "new": b, "delta": b - a, "ratio": b / a if a else None}

    same = old["ids_sha256"] == new["ids_sha256"] and len(old_rows) == len(new_rows)
    return {
        "schema": SCHEMA,
        "mode": "compare",
        "old": {
            "sha256": ex.sha_file(args.old),
            "manifest": old["package_manifest_sha256"],
        },
        "new": {
            "sha256": ex.sha_file(args.new),
            "manifest": new["package_manifest_sha256"],
        },
        "same_prompts": same,
        "same_device": old["device_name"] == new["device_name"],
        "device_name": new["device_name"],
        "items": len(old_rows),
        "bit_identical_items": identical,
        "totals": totals,
        "latency_ms": {
            key: change("latency_ms", key) for key in ("p50", "p95", "mean")
        },
        "memory_gib": {
            key: change("memory_gib", key)
            for key in ("after_load", "load_peak", "request_peak")
        },
        "items_per_s": {"old": old["items_per_s"], "new": new["items_per_s"]},
        "residency": {
            "old": old["runtime"]["residency"],
            "new": new["runtime"]["residency"],
        },
        "passed": same
        and totals["category_changes"] == 0
        and totals["missing"] == 0
        and identical == len(old_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--count", type=int, default=400)
    p.add_argument("--warmup", type=int, default=20)
    # The fast path captures a shape's HIP graph on its second use; two passes measure replays.
    p.add_argument("--warmup-passes", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int)
    p.add_argument("--base-path")
    p.add_argument("--site", action="append", default=[])
    p.add_argument("--require-kernels", action="store_true")
    p.add_argument("--fp32-master", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("old", type=Path)
    c.add_argument("new", type=Path)
    c.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for site in reversed(getattr(args, "site", [])):
        if not Path(site).is_absolute() or not Path(site).is_dir():
            raise ValueError(f"--site needs an existing absolute directory: {site}")
        sys.path.insert(0, site)
    ex = examples_module()
    result = (run if args.command == "run" else compare)(args, ex)
    ex.write_exclusive(args.output, result)
    print(json.dumps({"mode": args.command, "passed": result["passed"]}))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
