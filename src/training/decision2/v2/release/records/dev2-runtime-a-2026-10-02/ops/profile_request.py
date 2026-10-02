"""Where one System One request's time goes, for a package's own runtime on cuda:0 (timings only).

Run like ``runtime_bench.py`` (isolated interpreter, package read-only, no network):

    python -I -B profile_request.py --package PKG --prompts PROMPTS.jsonl --count 400 --output OUT.json
        [--site DIR] [--base-path B]

Answers the first ``count`` prompts twice untimed, then once more with the
runtime's steps wrapped by synchronized timers: question encoding
(``encode``), ``collate``, the host-to-device copies (``Tensor.to`` on CPU
tensors), the model call and,
inside it, the backbone call; the rest of ``system_one`` is answer
post-processing. Writes per-step p50 / mean milliseconds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path


def examples_module():
    path = Path(__file__).resolve().parents[3] / "examples.py"
    spec = importlib.util.spec_from_file_location("dev2_release_examples", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-path")
    parser.add_argument("--site", action="append", default=[])
    parser.add_argument("--require-kernels", action="store_true")
    parser.add_argument(
        "--trace",
        type=int,
        default=50,
        help="then trace this many requests eagerly (graphs off) for a per-kernel GPU time table",
    )
    args = parser.parse_args()
    for site in reversed(args.site):
        sys.path.insert(0, site)
    ex = examples_module()
    import torch

    model, _ = ex.load_package(args.package, "cuda:0", 4, args.base_path)
    ex.kernel_runtime(args.require_kernels)
    backend = model.backend
    prompts = ex.read_jsonl(args.prompts)[: args.count]
    for _ in range(2):
        for prompt in prompts:
            model.system_one(state=prompt["state"], questions=prompt["questions"])
    torch.cuda.synchronize()

    spent: dict[str, float] = {}

    def timed(name, function):
        def wrapper(*a, **k):
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                return function(*a, **k)
            finally:
                torch.cuda.synchronize()
                spent[name] = spent.get(name, 0.0) + time.perf_counter() - started

        return wrapper

    package = type(backend).__module__.rsplit(".", 1)[0]
    decision_model = sys.modules[f"{package}._vendor.dev2model.decision_model"]
    decision_model.collate = timed("collate", decision_model.collate)
    if hasattr(backend, "encode_fn"):
        backend.encode_fn = timed("encode", backend.encode_fn or decision_model.encode)
    else:
        decision_model.encode = timed("encode", decision_model.encode)
    backend.model.forward = timed("model", backend.model.forward)
    backend.model.backbone.forward = timed("backbone", backend.model.backbone.forward)
    to = torch.Tensor.to
    host_to_device = timed("to_device", to)
    torch.Tensor.to = lambda self, *a, **k: (
        to(self, *a, **k) if self.is_cuda else host_to_device(self, *a, **k)
    )
    rows = []
    try:
        for prompt in prompts:
            spent.clear()
            torch.cuda.synchronize()
            started = time.perf_counter()
            model.system_one(state=prompt["state"], questions=prompt["questions"])
            torch.cuda.synchronize()
            total = time.perf_counter() - started
            step = {k: 1000 * v for k, v in spent.items()}
            step["head_and_rest_of_model"] = step.get("model", 0) - step.get(
                "backbone", 0
            )
            step["total"] = 1000 * total
            step["outside_model"] = step["total"] - step.get("model", 0)
            step["questions"] = len(prompt["questions"])
            rows.append(step)
    finally:
        torch.Tensor.to = to
    names = sorted({k for row in rows for k in row})
    summary = {
        name: {
            "p50": percentile([r.get(name, 0.0) for r in rows], 0.5),
            "mean": sum(r.get(name, 0.0) for r in rows) / len(rows),
        }
        for name in names
    }
    kernels = None
    if args.trace:
        from torch.profiler import ProfilerActivity, profile

        graphs = getattr(getattr(backend, "fast", None), "graphs", None)
        if graphs is not None:
            graphs.max_tokens = 0
            graphs.graphs.clear()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            for prompt in prompts[: args.trace]:
                model.system_one(state=prompt["state"], questions=prompt["questions"])
            torch.cuda.synchronize()
        table: dict[str, list[float]] = {}
        for event in trace.events():
            if event.device_type.name != "CUDA":
                continue
            entry = table.setdefault(event.name[:120], [0, 0.0])
            entry[0] += 1
            entry[1] += event.device_time_total
        total = sum(v[1] for v in table.values())
        kernels = {
            "requests": args.trace,
            "gpu_ms_per_request": total / 1000 / args.trace,
            "launches_per_request": sum(v[0] for v in table.values()) / args.trace,
            "top": [
                {
                    "kernel": name,
                    "launches_per_request": count / args.trace,
                    "gpu_ms_per_request": us / 1000 / args.trace,
                }
                for name, (count, us) in sorted(
                    table.items(), key=lambda kv: -kv[1][1]
                )[:40]
            ],
        }
    fast = getattr(backend, "fast", None)
    result = {
        "kernels": kernels,
        "schema": "dev2-runtime-a-profile/1",
        "package_manifest_sha256": ex.sha_file(args.package / "MODEL_MANIFEST.json"),
        "count": len(rows),
        "ms": summary,
        "fast_path": fast.receipt() if fast is not None else None,
    }
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump(result, sink, indent=1, sort_keys=True)
    print(
        json.dumps({k: round(v["p50"], 3) for k, v in summary.items()}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
