"""Long multi-question request probe for a released package (private repro of runtime failures).

    python3 -m v2.eval.ix1.long_probe shape --package <dir> --rows <rows.jsonl.gz>
    python3 -m v2.eval.ix1.long_probe synth --words N --questions Q --out <rows.jsonl.gz> [--seed S]
    python3 -m v2.eval.ix1.long_probe probe --package <dir> --rows <rows.jsonl.gz> --out <log.jsonl> \
        [--counts 1,2,4,...] [--trace] [--base-path <snapshot>]

``shape`` prints, per request, the question count and token lengths of the padded batch the
runtime builds (counts only). ``synth`` writes requests of non-benchmark text: one question whose
candidate text is N words long (``shape`` gives the tokens) plus Q - 1 short ones, all ``choice`` questions. ``probe``
loads the package once and runs ``system_one`` on growing subsets of each request (the longest
question plus the first k - 1 others); every attempt is logged and fsynced before it starts, so
a GPU memory fault still leaves the failing shape in the log. ``--trace`` synchronizes after every
decoder sub-module and records the first module whose output is not finite (and, on a fault, the
last module entered); ``--alone`` then asks every question on its own. The log holds answers and
stays private.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from pathlib import Path

WORDS = (
    "amber basin cedar delta ember fjord granite harbor island juniper kestrel lantern meadow "
    "nectar orchard pebble quarry river summit timber upland valley willow yarrow zephyr "
    "anchor bridge canyon dune estuary forest glacier hollow inlet jetty knoll lagoon marsh"
).split()


def load_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def import_runtime(package: Path):
    sys.path.insert(0, str(package))
    from decision2 import Decision2

    if (
        Path(sys.modules["decision2"].__file__).resolve().parent
        != package / "decision2"
    ):
        raise SystemExit("decision2 was imported from outside the package")
    return Decision2


def request_id(row: dict, index: int) -> str:
    return row.get("_evaluation", {}).get("run_id", f"row-{index}")


def encoded_lengths(backend, row: dict) -> dict[str, int]:
    from decision2._vendor.dev2model.decision_model import encode
    from decision2._vendor.dev2model.infer import question_to_row

    item = {"id": "request", "state": row["state"]}
    lengths = {}
    for qid, question in row["questions"].items():
        encoded = encode(question_to_row(item, qid, question), backend.tokenizer, 10**9)
        lengths[qid] = len(encoded["ids"])
    return lengths


def batch_shape(lengths: list[int]) -> dict[str, int]:
    padded = math.ceil(max(lengths) / 8) * 8
    return {
        "questions": len(lengths),
        "max_tokens": max(lengths),
        "sum_tokens": sum(lengths),
        "padded_length": padded,
        "padded_tokens": padded * len(lengths),
    }


class Tokenizer:
    """Just enough of the backend for ``encoded_lengths`` without loading weights."""

    def __init__(self, package: Path):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(package, local_files_only=True)


def shape(args) -> None:
    import_runtime(args.package.resolve(strict=True))
    backend = Tokenizer(args.package.resolve(strict=True))
    for index, row in enumerate(load_rows(args.rows)):
        lengths = encoded_lengths(backend, row)
        print(json.dumps({"request": index, **batch_shape(list(lengths.values()))}))


def synth(args) -> None:
    rng = random.Random(args.seed)

    def text(words: int) -> str:
        return " ".join(rng.choice(WORDS) for _ in range(words))

    rows = []
    for index in range(args.requests):
        questions = {}
        for q in range(args.questions):
            words = args.words if q == 0 else args.short_words
            questions[f"q{q}"] = {
                "type": "choice",
                "instructions": {"task": text(20), "candidate": text(words)},
                "criteria": {"no": text(6), "yes": text(6)},
            }
        rows.append(
            {
                "_evaluation": {
                    "run_id": f"synthetic:{args.words}x{args.questions}:{index}"
                },
                "state": {"note": text(40)},
                "questions": questions,
            }
        )
    with gzip.open(args.out, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def attach_trace(model, torch, log):
    """Synchronize after every decoder sub-module; record the first non-finite output."""
    state = {"first_nonfinite": None}
    targets = []
    for name, module in model.backbone.named_modules():
        parts = name.split(".")
        if "layers" in parts and parts[-1] in {
            "linear_attn",
            "self_attn",
            "mlp",
            "input_layernorm",
        }:
            targets.append((name, module))
        if parts[-1] == "norm" and "layers" not in parts:
            targets.append((name, module))

    def before(name):
        def hook(module, inputs):
            log({"event": "enter", "module": name})

        return hook

    def after(name):
        def hook(module, inputs, output):
            torch.cuda.synchronize()
            tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor) and state["first_nonfinite"] is None:
                bad = int((~torch.isfinite(tensor)).sum())
                if bad:
                    state["first_nonfinite"] = {
                        "module": name,
                        "nonfinite": bad,
                        "numel": tensor.numel(),
                    }

        return hook

    for name, module in targets:
        module.register_forward_pre_hook(before(name))
        module.register_forward_hook(after(name))
    return state


def probe(args) -> None:
    package = args.package.resolve(strict=True)
    Decision2 = import_runtime(package)
    out = open(args.out, "a", encoding="utf-8")

    def log(record):
        out.write(json.dumps({"t": round(time.time(), 3), **record}) + "\n")
        out.flush()
        if record["event"] == "attempt":
            os.fsync(out.fileno())

    started = time.perf_counter()
    model = Decision2.from_pretrained(
        package, device=args.device, base_path=args.base_path
    )
    backend = model.backend
    torch = backend.torch
    log({"event": "loaded", "seconds": round(time.perf_counter() - started, 1)})
    trace = attach_trace(backend.model, torch, log) if args.trace else None
    counts = [int(c) for c in args.counts.split(",")] if args.counts else None
    for index, row in enumerate(load_rows(args.rows)):
        lengths = encoded_lengths(backend, row)
        order = sorted(lengths, key=lambda q: -lengths[q])[:1]
        order += [q for q in row["questions"] if q not in order]
        for k in counts or [len(order)]:
            if k > len(order):
                continue
            subset = {q: row["questions"][q] for q in order[:k]}
            shape_ = batch_shape([lengths[q] for q in subset])
            log({"event": "attempt", "request": request_id(row, index), **shape_})
            if trace:
                trace["first_nonfinite"] = None
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            try:
                answers, _ = backend.system_one(row["state"], subset)
                torch.cuda.synchronize()
                errors = sorted({a.get("error") for a in answers.values()} - {None})
                record = {"status": "error" if errors else "ok", "errors": errors}
                if args.answers:
                    record["answers"] = answers
            except Exception as exc:
                record = {
                    "status": "exception",
                    "errors": [f"{type(exc).__name__}: {exc}"[:400]],
                }
            record.update(
                event="result",
                request=request_id(row, index),
                questions=k,
                wall_ms=round((time.perf_counter() - t) * 1000, 1),
                peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
            )
            if trace:
                record["first_nonfinite"] = trace["first_nonfinite"]
            log(record)
        if args.alone:
            for qid, question in row["questions"].items():
                answers, _ = backend.system_one(row["state"], {qid: question})
                log(
                    {
                        "event": "alone",
                        "request": request_id(row, index),
                        "question": qid,
                        "answer": answers[qid],
                    }
                )
    log({"event": "done"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("shape")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p = sub.add_parser("synth")
    p.add_argument("--words", type=int, required=True)
    p.add_argument("--questions", type=int, required=True)
    p.add_argument("--short-words", type=int, default=120)
    p.add_argument("--requests", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("probe")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--counts")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--answers", action="store_true")
    p.add_argument("--alone", action="store_true")
    p.add_argument("--base-path", type=Path)
    p.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    {"shape": shape, "synth": synth, "probe": probe}[args.mode](args)


if __name__ == "__main__":
    main()
