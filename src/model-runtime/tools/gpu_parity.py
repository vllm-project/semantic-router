"""Parity of the native engine with the released packages' runtime on the four scored panels (GPU records).

    python3 tools/gpu_parity.py --package DIR --released ANSWERS.jsonl --panel NAME:PROMPTS:COUNT ...
        --output OUT.json [--answers NATIVE.jsonl] [--device rocm:0] [--base-path DIR]
        [--profile exact|shared_context] [--no-graphs] [--no-fused]

``--panel NAME:REQUESTS.jsonl:COUNT`` takes any JSONL of ``{"id", "state", "questions"}`` requests; with
``--profile shared_context`` the released answers are the released runtime's with ``share_context=True``.

The package is verified and loaded through the decision2 family and the native engine; no package code runs.
Every prompt is answered as one request with the profile's batching (``exact``: the request's questions in one
padded batch, split only by the forward token budget) and compared with the released runtime's answers for the
same panel and prompt ID (the ``--answers`` file of ``v2/release/examples.py parity``, produced on the same device
class with the same autotune cache; a built-in model runs with its pinned kernel choices, the released runtime's).
Per panel: prompts, identical prompts (canonical JSON equality), category changes, missing answers and the largest
absolute difference of any probability, Noul or Score value. Exits 1 unless every prompt is identical.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vllm_srun.accel.autotune import KernelChoices  # noqa: E402
from vllm_srun.accel.cpu import CPUAccelerator  # noqa: E402
from vllm_srun.accel.cuda import CUDAAccelerator  # noqa: E402
from vllm_srun.accel.npu import NPUAccelerator  # noqa: E402
from vllm_srun.accel.rocm import ROCmAccelerator  # noqa: E402
from vllm_srun.engines.native.engine import NativeEngine  # noqa: E402
from vllm_srun.families.decision2.family import Decision2Family  # noqa: E402
from vllm_srun.plugins.base import (  # noqa: E402
    EngineOptions,
    Job,
    PackageRef,
    RegistryOptions,
)
from vllm_srun.profiles.exact import ExactProfile  # noqa: E402
from vllm_srun.profiles.shared_context import (  # noqa: E402
    SharedContextProfile,
    SharePolicy,
)
from vllm_srun.registry import builtin  # noqa: E402

ACCELERATORS = {
    "cpu": CPUAccelerator,
    "cuda": CUDAAccelerator,
    "npu": NPUAccelerator,
    "rocm": ROCmAccelerator,
}
PROFILES = {"exact": ExactProfile, "shared_context": SharedContextProfile}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def category(answer: Any) -> tuple:
    if not isinstance(answer, dict) or "error" in answer:
        return ("invalid", answer.get("error") if isinstance(answer, dict) else None)
    if answer.get("type") == "noul" or "noul" in answer:
        value = answer.get("noul")
        return ("noul", None if value is None or value == 0.5 else value > 0.5)
    if "choice" in answer:
        return ("choice", answer["choice"])
    probabilities = answer.get("probabilities") or {}
    if probabilities:
        best = max(probabilities.values())
        winners = sorted(k for k, v in probabilities.items() if abs(v - best) <= 1e-8)
        return ("score", winners[0] if len(winners) == 1 else None)
    return ("score", None)


def numbers(answer: Any) -> dict[str, float]:
    if not isinstance(answer, dict):
        return {}
    out = {}
    for key in ("noul", "score", "confidence"):
        if isinstance(answer.get(key), (int, float)):
            out[key] = float(answer[key])
    for key, value in (answer.get("probabilities") or {}).items():
        if isinstance(value, (int, float)):
            out[f"p/{key}"] = float(value)
    return out


def load(args: argparse.Namespace):
    family = Decision2Family(RegistryOptions(base_path=args.base_path, offline=True))
    package = family.verify(PackageRef(root=Path(args.package)))
    spec = family.describe(package)
    kind, _, index = args.device.partition(":")
    accelerator = ACCELERATORS[kind]()
    devices = accelerator.devices()
    device = devices[int(index or 0)] if kind != "cpu" else devices[0]
    recorded = builtin.kernel_choices(
        package.model_sha256, device.accelerator, device.arch
    ) or family.kernel_choices(package, device)
    if recorded:
        choices = KernelChoices(recorded)
        if choices.install() is None:
            choices.pin_thread()
    options = EngineOptions(graphs=not args.no_graphs, fused_kernels=not args.no_fused)
    engine_model = NativeEngine().load(spec, accelerator, device, options)
    return family.load(package, spec, engine_model)


def system_one(
    model: Any, profile: Any, state: Any, questions: dict[str, Any]
) -> dict[str, Any]:
    plan = model.plan(state, questions)
    job = Job(
        items=plan.items, deadline=None, enqueued=time.monotonic(), profile=profile.name
    )
    logits: dict[int, list[float] | None] = {}
    for batch in profile.plan([job], model.forward_token_budget()):
        indices = [index for _, part in batch.parts for index in part]
        items = [plan.items[i] for i in indices]
        if batch.shared_prefix:
            values = model.run_shared(items, batch.shared_prefix)
        else:
            values = model.run(items)
        for index, value in zip(indices, values, strict=True):
            logits[index] = value
    answers = dict(plan.errors)
    for index, item in enumerate(plan.items):
        answers[item.question_id] = model.answer(item, logits.get(index))
    return {
        "answers": {qid: answers[qid] for qid in plan.question_ids},
        "usage": {"input_tokens": plan.input_tokens, "output_tokens": 0},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", required=True)
    parser.add_argument("--released", type=Path, required=True)
    parser.add_argument("--panel", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--device", default="rocm:0")
    parser.add_argument("--base-path")
    parser.add_argument("--profile", default="exact", choices=sorted(PROFILES))
    parser.add_argument(
        "--share-policy", default="{}", help="shared_context SharePolicy fields as JSON"
    )
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--no-fused", action="store_true")
    args = parser.parse_args()
    released = {}
    with args.released.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            released[(row["panel"], row["id"])] = row["answers"]
    started = time.perf_counter()
    model = load(args)
    if args.profile == "shared_context":
        profile = SharedContextProfile(SharePolicy(**json.loads(args.share_policy)))
    else:
        profile = PROFILES[args.profile]()
    unavailable = profile.available(model)
    if unavailable:
        raise SystemExit(f"profile {args.profile} is unavailable: {unavailable}")
    profile.bind(model)
    load_seconds = time.perf_counter() - started
    sink = args.answers.open("x", encoding="utf-8") if args.answers else None
    panels = {}
    for spec in args.panel:
        name, prompts_path, count = spec.split(":")
        prompts = [json.loads(line) for line in open(prompts_path, encoding="utf-8")][
            : int(count)
        ]
        totals = {
            "prompts": 0,
            "identical_prompts": 0,
            "category_changes": 0,
            "missing": 0,
            "max_abs_drift": 0.0,
        }
        for prompt in prompts:
            response = system_one(model, profile, prompt["state"], prompt["questions"])
            answers = response["answers"]
            if sink is not None:
                sink.write(
                    json.dumps({"panel": name, "id": prompt["id"], "answers": answers})
                    + "\n"
                )
            reference = released.get((name, prompt["id"]))
            totals["prompts"] += 1
            if reference is None:
                totals["missing"] += 1
                continue
            totals["identical_prompts"] += canonical(answers) == canonical(reference)
            for qid in set(answers) | set(reference):
                if qid not in answers or qid not in reference:
                    totals["missing"] += 1
                    continue
                totals["category_changes"] += category(answers[qid]) != category(
                    reference[qid]
                )
                a, b = numbers(answers[qid]), numbers(reference[qid])
                for key in set(a) & set(b):
                    totals["max_abs_drift"] = max(
                        totals["max_abs_drift"], abs(a[key] - b[key])
                    )
        panels[name] = totals
        print(json.dumps({name: totals}), flush=True)
    if sink is not None:
        sink.close()
    engine = model.engine_model
    passed = all(
        p["prompts"] == p["identical_prompts"] and not p["missing"]
        for p in panels.values()
    )
    result = {
        "schema": "model-runtime-gpu-parity/1",
        "package": str(args.package),
        "model": model.info.id,
        "model_sha256": model.info.model_sha256,
        "device": engine.device_info.name,
        "kernels": engine.kernels.describe(),
        "fast_path": engine.receipt() if hasattr(engine, "receipt") else None,
        "options": {"graphs": not args.no_graphs, "fused_kernels": not args.no_fused},
        "load_seconds": load_seconds,
        "panels": panels,
        "passed": passed,
    }
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump(result, sink, indent=1)
    print(json.dumps({"passed": passed}))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
