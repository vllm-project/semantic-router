"""Latency of multi-question decision requests through the served path (performance records).

    python3 tools/state_once_bench.py --package DIR --output OUT.json [--attention-mode MODE]
        [--device rocm:0] [--questions 1,4,8,16,64] [--runs 10] [--warmup 3] [--image-questions 8]
        [--concurrency 16] [--load-rounds 5]

Starts the runtime in-process with one model and the default engine arguments, then times
``Runtime.call("decisions", ...)``: validation, rendering, scheduling, the forwards and the answers, without
HTTP. Requests: the public many-question ticket (``many_questions.py``, a state block of about 400 tokens) and
the same ticket followed by the store's earlier replies (about 2,000 tokens), each with every count of
``--questions``, and one request with a new 1.59 MP image (1456 x 1092, a fresh image every run, so the tower
always runs) and ``--image-questions`` questions. A Decision 3.0 package answers each question as its own row
(the reference); with ``--attention-mode`` the same weights are served as a state-once model, whose requests
read their state once. Reported per request: p50, mean, p80, min and max milliseconds of ``--runs`` timed
calls after ``--warmup`` untimed ones, and the input tokens the runtime counted. With ``--concurrency N``, each
request then also runs ``--load-rounds`` rounds of N concurrent copies (after one untimed round; image requests
get a new image per copy): requests and questions per second, and the p50 / p80 latency of a copy under load.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import many_questions  # noqa: E402
from state_once_parity import HISTORY, convert  # noqa: E402
from vllm_srun.config import ModelConfig, ServeConfig  # noqa: E402
from vllm_srun.families.decision3 import package as pkg  # noqa: E402
from vllm_srun.runtime import Runtime  # noqa: E402

IMAGE_SIZE = (1456, 1092)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def image_url(seed: int) -> str:
    """A deterministic 1.59 MP photo-like PNG (noise over gradients), new pixels for every seed."""
    import numpy as np
    from PIL import Image

    width, height = IMAGE_SIZE
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 96, size=(height, width, 3), dtype=np.uint8)
    pixels += np.linspace(0, 159, width, dtype=np.uint8)[None, :, None]
    pixels[:, :, 1] += np.linspace(0, 64, height, dtype=np.uint8)[:, None]
    buffer = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def requests(counts: list[int], image_questions: int) -> list[tuple[str, Any]]:
    """(name, body or body factory) of every timed request."""
    states = {
        "ticket": many_questions.TICKET,
        "thread": many_questions.TICKET + " " + HISTORY * 4,
    }
    out: list[tuple[str, Any]] = [
        (
            f"{name}-{count}q",
            {"state": state, "questions": many_questions.questions(count)},
        )
        for name, state in states.items()
        for count in counts
    ]
    if image_questions:
        out.append(
            (
                f"image-{image_questions}q",
                lambda seed: {
                    "state": "Describe what the attached photo shows.",
                    "questions": many_questions.questions(image_questions),
                    "images": [image_url(seed)],
                },
            )
        )
    return out


def load(
    runtime: Runtime,
    name: str,
    body: Any,
    seed: int,
    concurrency: int,
    rounds: int,
    warmup: int = 2,
) -> tuple[int, dict[str, Any]]:
    """Throughput of ``concurrency`` concurrent copies of a request: (next seed, figures).

    ``warmup`` untimed rounds come first: merged batches take shapes a request
    alone never does, and kernels tune for each new shape once.
    """

    async def timed(request: Any) -> tuple[float, int, dict[str, Any]]:
        begin = time.perf_counter()
        status, answer = await runtime.call("decisions", request)
        return (time.perf_counter() - begin) * 1000, status, answer

    async def one_round(seed: int) -> list[tuple[float, int, dict[str, Any]]]:
        copies = [
            body(seed + i) if callable(body) else body for i in range(concurrency)
        ]
        return await asyncio.gather(*(timed(copy) for copy in copies))

    latencies: list[float] = []
    elapsed = 0.0
    questions = 0
    for round_index in range(warmup + rounds):
        begin = time.perf_counter()
        outcomes = asyncio.run(one_round(seed))
        seconds = time.perf_counter() - begin
        seed += concurrency
        for _, status, answer in outcomes:
            if status != 200 or any("error" in a for a in answer["answers"].values()):
                raise SystemExit(f"{name} under load: {status}")
        if round_index >= warmup:
            elapsed += seconds
            latencies += [ms for ms, _, _ in outcomes]
            questions += sum(len(answer["answers"]) for _, _, answer in outcomes)
    return seed, {
        "concurrency": concurrency,
        "rounds": rounds,
        "requests_per_s": concurrency * rounds / elapsed,
        "questions_per_s": questions / elapsed,
        "p50_ms": percentile(latencies, 0.5),
        "p80_ms": percentile(latencies, 0.8),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-mode", choices=pkg.STATE_ONCE_MODES)
    parser.add_argument("--device", default="rocm:0")
    parser.add_argument("--questions", default="1,4,8,16,64")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--image-questions", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=0)
    parser.add_argument("--load-rounds", type=int, default=5)
    parser.add_argument("--load-warmup", type=int, default=2)
    args = parser.parse_args()
    package = args.package
    if args.attention_mode:
        target = package.parent / f"{package.name}-{args.attention_mode}"
        if not (target / pkg.MANIFEST_NAME).is_file():
            convert(package, target, args.attention_mode)
        package = target
    started = time.perf_counter()
    runtime = Runtime(
        ServeConfig(models=(ModelConfig(model=str(package), device=args.device),))
    )
    runtime.start(background=False)
    ready = time.perf_counter() - started
    served = runtime.lookup(None)
    engine = getattr(served.model, "engine_model", None)

    def graphs() -> dict[str, Any]:
        """The engine's graph statistics (captures, replays, eager runs) so far."""
        receipt = engine.receipt() if hasattr(engine, "receipt") else {}
        return {key: value for key, value in receipt.items() if key.endswith("graphs")}

    results: dict[str, Any] = {}
    seed = 0
    try:
        for name, body in requests(
            [int(c) for c in args.questions.split(",")], args.image_questions
        ):
            times, tokens = [], None
            for run in range(args.warmup + args.runs):
                seed += 1
                request = body(seed) if callable(body) else body
                begin = time.perf_counter()
                status, answer = asyncio.run(runtime.call("decisions", request))
                elapsed = (time.perf_counter() - begin) * 1000
                if status != 200:
                    raise SystemExit(f"{name}: {status} {answer}")
                errors = [a for a in answer["answers"].values() if "error" in a]
                if errors:
                    raise SystemExit(f"{name}: {errors[:2]}")
                tokens = answer["usage"]["input_tokens"]
                if run >= args.warmup:
                    times.append(elapsed)
            results[name] = {
                "p50_ms": percentile(times, 0.5),
                "mean_ms": statistics.fmean(times),
                "p80_ms": percentile(times, 0.8),
                "min_ms": min(times),
                "max_ms": max(times),
                "input_tokens": tokens,
            }
            if args.concurrency:
                seed, results[name]["load"] = load(
                    runtime,
                    name,
                    body,
                    seed,
                    args.concurrency,
                    args.load_rounds,
                    args.load_warmup,
                )
            results[name]["graphs"] = graphs()
            print(name, json.dumps(results[name]), flush=True)
        card = served.card([])
    finally:
        runtime.stop()
    report = {
        "package": str(args.package),
        "attention_mode": json.loads(
            (package / "decision_config.json").read_text(encoding="utf-8")
        )["attention_mode"],
        "device": card["device"],
        "engine_args": card["engine_args"],
        "ready_seconds": ready,
        "runs": args.runs,
        "warmup": args.warmup,
        "requests": results,
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
