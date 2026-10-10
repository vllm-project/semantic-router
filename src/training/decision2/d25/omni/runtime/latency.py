"""Latency of image requests through the image-capable runtime (one request at a time, after warm-up).

Each scenario: ``--warmup`` untimed then ``--timed`` timed requests; the time is the device-synchronized wall
time of one request through ``prepare`` + ``run`` + ``respond`` (what ``system_one`` does: image decoding,
prompt rendering and token planning, then image preprocessing, forward pass and readout), also reported per
phase. Images go in as data URLs, as a server or the kit engine would receive them, so decoding is timed.

- ``one_image``: one suite image of at least 1.6 MP per request (read at the 1,638,400-pixel cap), the
  image's own question;
- ``four_images``: four such images per request, one question;
- ``suite``: natural suite rows (one or two images at their own size), one question each.

    python -m d25.omni.runtime.latency --package PKG --suite SUITE --out latency.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from d25.omni.runtime.common import (
    data_url,
    latency_stats,
    load_runtime,
    read_jsonl,
    sha_key,
    stratified,
    write_json,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--suite", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--timed", type=int, default=200)
    ap.add_argument("--timed-four", type=int, default=100)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from PIL import Image

    rt = load_runtime(args.package)
    rows = read_jsonl(args.suite / "rows.jsonl.gz")
    large = []
    for row in sorted(rows, key=lambda r: sha_key(r["id"])):
        if len(row["images"]) != 1:
            continue
        path = args.suite / row["images"][0]
        with Image.open(path) as image:
            width, height = image.size
        if width * height >= rt.IMAGE_MAX_PIXELS and max(width, height) <= 20 * min(
            width, height
        ):
            large.append((row, path))
        if len(large) >= args.warmup + args.timed:
            break
    if len(large) < 8:
        raise SystemExit(f"only {len(large)} suite images of at least 1.6 MP")

    def one(i):
        row, path = large[i % len(large)]
        return row.get("state"), row["questions"], [data_url(path)]

    def four(i):
        row, _ = large[i % len(large)]
        paths = [large[(i + k) % len(large)][1] for k in range(4)]
        return row.get("state"), row["questions"], [data_url(p) for p in paths]

    natural = stratified(rows, args.warmup + args.timed)

    def suite(i):
        row = natural[i % len(natural)]
        return (
            row.get("state"),
            row["questions"],
            [data_url(args.suite / r) for r in row["images"]],
        )

    started = time.time()
    # The release package has no input limit (max_length null).
    model = rt.Decision25.from_pretrained(
        args.package, device=args.device, max_length=1 << 20
    )
    load_seconds = time.time() - started
    warmup_seconds = model.warmup()
    report = {
        "what": "image-request latency, image-capable runtime, one request at a time after warm-up",
        "package": str(args.package),
        "load_seconds": round(load_seconds, 1),
        "warmup_seconds": round(warmup_seconds, 1),
        "scenarios": {},
    }
    for name, make, timed in (
        ("one_image", one, args.timed),
        ("four_images", four, args.timed_four),
        ("suite", suite, args.timed),
    ):
        requests = [make(i) for i in range(args.warmup + timed)]
        times, prepare_ms, run_ms, processor_ms, tokens, errors = [], [], [], [], [], 0
        for i, (state, questions, images) in enumerate(requests):
            model.synchronize()
            begin = time.perf_counter()
            prepared = model.prepare(state, questions, images)
            middle = time.perf_counter()
            probabilities, used = model.run(prepared)
            response = model.respond(prepared, probabilities, used)
            model.synchronize()
            end = time.perf_counter()
            errors += any("error" in a for a in response["answers"].values())
            if i >= args.warmup:
                times.append((end - begin) * 1000)
                prepare_ms.append((middle - begin) * 1000)
                run_ms.append((end - middle) * 1000)
                tokens.append(response["usage"]["input_tokens"])
                keys = prepared.runnable[: model.batch_size]
                again = time.perf_counter()
                model.processor(
                    text=[prepared.texts[k] for k in keys],
                    images=[image for _ in keys for image in prepared.images],
                    padding=True,
                    return_tensors="pt",
                )
                processor_ms.append((time.perf_counter() - again) * 1000)
        tokens.sort()
        median = lambda values: round(sorted(values)[len(values) // 2], 1)  # noqa: E731
        report["scenarios"][name] = {
            **latency_stats(times),
            "prepare_median_ms": median(prepare_ms),
            "run_median_ms": median(run_ms),
            "processor_median_ms_untimed": median(processor_ms),
            "warmup_requests": args.warmup,
            "images_per_request": (
                len(requests[0][2]) if name != "suite" else "1-2 (natural)"
            ),
            "input_tokens_median": tokens[len(tokens) // 2],
            "input_tokens_max": tokens[-1],
            "answer_errors": errors,
        }
        print(name, json.dumps(report["scenarios"][name]), flush=True)
    report["runtime"] = model.runtime_info()
    report["kernels"] = model.kernels
    report["image_contract"] = model.image_contract()
    write_json(args.out, report)


if __name__ == "__main__":
    main()
