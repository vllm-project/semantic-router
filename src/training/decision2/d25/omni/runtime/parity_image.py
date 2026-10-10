"""Image parity: the image-capable runtime against ``VisionCodeReadoutModel`` (the engine behind the vision numbers).

One process, one device; the engine loads the assembled evaluation checkpoint exactly as the zero-shot run
did (``run_suite`` settings), the runtime loads the package. The engine pass runs first (per-row tensors are
kept as SHA-256 digests), then the runtime pass; ``--sequential`` frees the engine in between (one 27B copy
at a time). Rows: a deterministic stratified sample of the public vision suite. Per row:

- rendering: the runtime's prompt text and planned tokens against the engine's plan, and the processor
  tensors the runtime feeds the backbone against the engine's ``_prepare`` output (exact equality);
- matched batching: the engine scores the row on its own, which is the runtime's batch for a one-question
  request; probabilities are compared exactly;
- evaluation batching: the engine scores all sample rows in one ``score`` call (token budget 65,536, up to
  64 rows per batch, as in the zero-shot run), plus the zero-shot run's recorded probabilities when present;
  max |dp| and argmax agreement against the runtime;
- input kinds: the first rows again with the images as data URLs, PIL images and http URLs (served from the
  pod), which must give identical probabilities.

    python -m d25.omni.runtime.parity_image --engine-ckpt GRAFT --package PKG --suite SUITE --out parity-image.json
"""

from __future__ import annotations

import argparse
import collections
import functools
import http.server
import json
import threading
import time
from pathlib import Path

from d25.omni.runtime.common import (
    compare_probs,
    data_url,
    load_runtime,
    read_jsonl,
    stratified,
    write_json,
)


def correct(question: dict, expected, probs: list[float]) -> bool:
    from d25.omni.common import vision_format

    if question["type"] == "noul":
        truth = (
            expected
            if isinstance(expected, bool)
            else str(expected).lower() in ("true", "yes", "1")
        )
        return (probs[1] >= 0.5) == truth
    keys, _ = vision_format.options(question)
    return keys[max(range(len(probs)), key=probs.__getitem__)] == expected


def flips(rows: list[dict], pairs: list[tuple[list[float], list[float]]]) -> dict:
    """Accuracy of reference and runtime on the same rows, and the rows whose argmax differs."""
    ref_ok = run_ok = 0
    changed = []
    for row, (want, have) in zip(rows, pairs):
        ((qid, question),) = row["questions"].items()
        expected = (row.get("expected") or {}).get(qid)
        a, b = correct(question, expected, want), correct(question, expected, have)
        ref_ok, run_ok = ref_ok + a, run_ok + b
        if max(range(len(want)), key=want.__getitem__) != max(
            range(len(have)), key=have.__getitem__
        ):
            top = sorted(want, reverse=True)
            changed.append(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "reference_margin": top[0] - top[1],
                    "reference_correct": a,
                    "runtime_correct": b,
                }
            )
    return {
        "rows": len(pairs),
        "reference_correct": ref_ok,
        "runtime_correct": run_ok,
        "argmax_changes": changed,
    }


def zero_shot(directory: Path | None) -> dict[str, dict]:
    if directory is None or not directory.exists():
        return {}
    files = (
        [directory / "results.jsonl"] if (directory / "results.jsonl").exists() else []
    )
    files = files or sorted((directory / "shards").glob("results-*.jsonl"))
    records = {}
    for path in files:
        for record in read_jsonl(path):
            if record.get("status") == "ok":
                records[record["id"]] = record
    return records


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


def serve(root: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    handler = functools.partial(QuietHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine-ckpt", required=True, type=Path)
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--suite", required=True, type=Path)
    ap.add_argument("--zs-results", type=Path)
    ap.add_argument("--rows", type=int, default=360)
    ap.add_argument("--min-multi", type=int, default=40)
    ap.add_argument("--kinds-rows", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--sequential",
        action="store_true",
        help="free the engine's model before loading the runtime (one 27B copy at a time, for 96 GB GPUs)",
    )
    ap.add_argument(
        "--save-probs", type=Path, help="write the runtime's probabilities per row id"
    )
    ap.add_argument(
        "--reference",
        type=Path,
        help="probabilities per row id from another device (a --save-probs file), compared and reported",
    )
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import gc

    import torch
    from PIL import Image

    from d25.omni.common import vision_format
    from d25.omni.eval.engine import VisionCodeReadoutModel
    from d25.omni.model.checkpoint import tensor_digest

    rt = load_runtime(args.package)
    rows = stratified(
        read_jsonl(args.suite / "rows.jsonl.gz"), args.rows, args.min_multi
    )
    seconds = {}

    # Engine pass: the row on its own (matched batching), then all rows in one score call.
    started = time.time()
    engine = VisionCodeReadoutModel(
        args.engine_ckpt,
        device=args.device,
        max_pixels=vision_format.MAX_PIXELS,
        token_budget=65_536,
        max_batch_size=64,
        readout_dtype="float32",
    )
    seconds["engine_load"] = time.time() - started
    started = time.time()
    reference: dict[str, dict] = {}
    for row in rows:
        question = next(iter(row["questions"].values()))
        request = {
            "state": row.get("state"),
            "question": question,
            "images": row["images"],
        }
        plan = engine.plan([request], root=args.suite)[0]
        entry = {"status": plan.status, "reason": plan.reason}
        if plan.status == "ok":
            encoded = engine._prepare([plan])
            entry.update(
                text=plan.text,
                tokens=plan.tokens,
                digests={k: tensor_digest(v) for k, v in encoded.items()},
                probs=engine._probabilities(encoded, [plan.n_options])[0],
            )
        reference[row["id"]] = entry
    supported = [r for r in rows if reference[r["id"]]["status"] == "ok"]
    outcomes = engine.score(
        [
            {
                "state": r.get("state"),
                "question": next(iter(r["questions"].values())),
                "images": r["images"],
            }
            for r in supported
        ],
        root=args.suite,
    )
    batched = {
        r["id"]: o.probabilities
        for r, o in zip(supported, outcomes)
        if o.status == "ok"
    }
    seconds["engine_rows"] = time.time() - started
    # The evaluation checkpoint allows 16,384 input tokens; the runtime takes the engine's limit here.
    limit = engine.max_length
    if args.sequential:
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    # Runtime pass.
    started = time.time()
    runtime = rt.Decision25.from_pretrained(
        args.package, device=args.device, max_length=limit
    )
    seconds["runtime_load"] = time.time() - started
    counts = collections.Counter()
    tensor_keys = collections.Counter()
    matched, mine, examples = [], {}, []
    started = time.time()
    for row in rows:
        qid = next(iter(row["questions"]))
        paths = [str(args.suite / ref) for ref in row["images"]]
        prepared = runtime.prepare(row.get("state"), row["questions"], paths)
        want = reference[row["id"]]
        counts["rows"] += 1
        counts[f"images_{len(paths)}"] += 1
        runtime_ok = qid in prepared.texts
        if want["status"] != "ok" or not runtime_ok:
            both = want["status"] != "ok" and not runtime_ok
            counts["unsupported_both" if both else "unsupported_mismatch"] += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "id": row["id"],
                        "engine": want["reason"],
                        "runtime": prepared.errors.get(qid),
                    }
                )
            continue
        counts["text_mismatch"] += want["text"] != prepared.texts[qid]
        counts["token_mismatch"] += want["tokens"] != prepared.lengths[qid]
        encoded = runtime.processor(
            text=[prepared.texts[qid]],
            images=prepared.images,
            padding=True,
            return_tensors="pt",
        )
        digests = {k: tensor_digest(v) for k, v in encoded.items()}
        bad = [
            k
            for k in sorted(set(digests) | set(want["digests"]))
            if digests.get(k) != want["digests"].get(k)
        ]
        for key in bad:
            tensor_keys[key] += 1
        counts["tensor_mismatch_rows"] += bool(bad)
        got, _ = runtime.run(prepared)
        mine[row["id"]] = got[qid]
        matched.append((want["probs"], got[qid]))
        if want["probs"] != got[qid] and len(examples) < 10:
            examples.append(
                {"id": row["id"], "engine": want["probs"], "runtime": got[qid]}
            )
    seconds["runtime_rows"] = time.time() - started

    scored = [r for r in rows if r["id"] in mine]
    evaluated = [r for r in scored if r["id"] in batched]
    evaluation = compare_probs((batched[r["id"]], mine[r["id"]]) for r in evaluated)
    evaluation["detail"] = flips(
        evaluated, [(batched[r["id"]], mine[r["id"]]) for r in evaluated]
    )
    recorded = zero_shot(args.zs_results)
    zs_rows = [r for r in scored if r["id"] in recorded]
    zs_pairs = [
        (recorded[r["id"]]["probabilities"][next(iter(r["questions"]))], mine[r["id"]])
        for r in zs_rows
    ]
    zs = compare_probs(zs_pairs) if zs_pairs else {"questions": 0}
    if zs_pairs:
        zs["detail"] = flips(zs_rows, zs_pairs)

    started = time.time()
    server, base = serve(args.suite)
    kinds = collections.Counter()
    try:
        for row in scored[: args.kinds_rows]:
            paths = [args.suite / ref for ref in row["images"]]
            variants = {
                "data_url": [data_url(p) for p in paths],
                "pil": [Image.open(p) for p in paths],
                "http_url": [f"{base}/{ref}" for ref in row["images"]],
            }
            for name, images in variants.items():
                prepared = runtime.prepare(row.get("state"), row["questions"], images)
                got, _ = runtime.run(prepared)
                kinds[f"{name}_rows"] += 1
                kinds[f"{name}_identical"] += list(got.values()) == [mine[row["id"]]]
    finally:
        server.shutdown()
    seconds["input_kinds"] = time.time() - started

    matched_stats = compare_probs(matched)
    if args.save_probs:
        write_json(
            args.save_probs, {"device": runtime.runtime_info(), "probabilities": mine}
        )
    cross = None
    if args.reference:
        stored = json.loads(args.reference.read_text())
        pairs = [
            (stored["probabilities"][k], v)
            for k, v in mine.items()
            if k in stored["probabilities"]
        ]
        cross = {"reference_device": stored.get("device"), **compare_probs(pairs)}
    report = {
        "what": "image parity: image-capable runtime vs VisionCodeReadoutModel on the public vision suite 0.3.1",
        "engine_ckpt": str(args.engine_ckpt),
        "package": str(args.package),
        "suite": str(args.suite),
        "sample": {
            "rows": len(rows),
            "families": dict(collections.Counter(r["family"] for r in rows)),
            "multi_image_rows": sum(len(r["images"]) > 1 for r in rows),
        },
        "counts": dict(counts),
        "tensor_mismatch_keys": dict(tensor_keys),
        "matched_batching": matched_stats,
        "evaluation_batching": evaluation,
        "zero_shot_run": zs,
        "reference": cross,
        "input_kinds": dict(kinds),
        "seconds": {k: round(v, 1) for k, v in seconds.items()},
        "runtime": runtime.runtime_info(),
        "kernels": runtime.kernels,
        "image_contract": runtime.image_contract(),
        "examples": examples,
    }
    report["pass"] = (
        counts["unsupported_mismatch"] == 0
        and counts["text_mismatch"] == 0
        and counts["token_mismatch"] == 0
        and counts["tensor_mismatch_rows"] == 0
        and matched_stats["exact"] == matched_stats["questions"]
        and all(
            kinds[f"{n}_identical"] == kinds[f"{n}_rows"]
            for n in ("data_url", "pil", "http_url")
        )
    )
    write_json(args.out, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "sample",
                    "counts",
                    "matched_batching",
                    "evaluation_batching",
                    "zero_shot_run",
                    "input_kinds",
                    "pass",
                )
            }
        )
    )
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
