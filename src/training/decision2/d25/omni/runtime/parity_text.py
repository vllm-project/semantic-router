"""Text-only parity: the image-capable runtime against the package's original runtime.

Both runtimes load in one process on one device (same kernels, same Triton autotune state). For every
request: token ids, per-question errors, probabilities (exact equality) and the full ``system_one``
response are compared. An image request goes through the new runtime after every ``--image-every`` text
requests, and the first ``--rerun`` requests are run again at the end: state left behind by the image path
would show up as a difference.

    python -m d25.omni.runtime.parity_text --original ORIG --package PKG --rows parity-600.jsonl.gz --out parity-text.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from d25.omni.runtime.common import (
    argmax,
    compare_probs,
    load_runtime,
    read_jsonl,
    write_json,
)


def dumps(value) -> str:
    return json.dumps(value, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--original", required=True, type=Path)
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--rows", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image-every", type=int, default=50)
    ap.add_argument("--rerun", type=int, default=50)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from PIL import Image

    original_rt = load_runtime(args.original, "decision25_runtime_original")
    new_rt = load_runtime(args.package, "decision25_runtime")
    rows = read_jsonl(args.rows)[: args.limit]
    seconds = {}
    started = time.time()
    old = original_rt.Decision25.from_pretrained(args.original, device=args.device)
    new = new_rt.Decision25.from_pretrained(args.package, device=args.device)
    seconds["load"] = time.time() - started
    started = time.time()
    seconds["warmup_new"] = new.warmup()
    image_request = {
        "state": "A photo attached by the customer.",
        "questions": {
            "blue": {"type": "noul", "instructions": "Is the picture mostly blue?"},
            "shape": {
                "type": "choice",
                "criteria": {"square": None, "wide": None, "tall": None},
            },
        },
        "images": [Image.new("RGB", (640, 480), (30, 60, 200))],
    }
    stats = {
        "requests": 0,
        "questions": 0,
        "answered": 0,
        "token_mismatch": 0,
        "error_mismatch": 0,
        "response_mismatch": 0,
        "argmax_changes": 0,
        "noul_flips": 0,
        "image_requests_interleaved": 0,
    }
    pairs, examples, first = [], [], []
    started = time.time()
    for number, row in enumerate(rows):
        state, questions = row["state"], row["questions"]
        a, b = old.prepare(state, questions), new.prepare(state, questions)
        stats["requests"] += 1
        stats["questions"] += len(questions)
        if a.keys != b.keys or a.sequences != b.sequences:
            stats["token_mismatch"] += 1
        if a.errors != b.errors:
            stats["error_mismatch"] += 1
        got_a, tokens_a = old.run(a)
        got_b, tokens_b = new.run(b)
        response_a = old.respond(a, got_a, tokens_a)
        response_b = new.respond(b, got_b, tokens_b)
        if dumps(response_a) != dumps(response_b):
            stats["response_mismatch"] += 1
            if len(examples) < 10:
                examples.append(
                    {"id": row["id"], "original": response_a, "new": response_b}
                )
        for key, want in got_a.items():
            have = got_b.get(key)
            if have is None:
                continue
            stats["answered"] += 1
            pairs.append((want, have))
            if a.questions[key].kind == "noul":
                stats["noul_flips"] += (want[1] >= 0.5) != (have[1] >= 0.5)
            else:
                stats["argmax_changes"] += argmax(want) != argmax(have)
        if number < args.rerun:
            first.append((row, response_b))
        if (number + 1) % args.image_every == 0:
            new.system_one(**image_request)
            stats["image_requests_interleaved"] += 1
    seconds["rows"] = time.time() - started
    rerun_identical = sum(
        dumps(new.system_one(state=row["state"], questions=row["questions"]))
        == dumps(response)
        for row, response in first
    )
    probabilities = compare_probs(pairs)
    stats.update(
        {
            "probabilities": probabilities,
            "rerun": {"requests": len(first), "identical": rerun_identical},
        }
    )
    stats["pass"] = (
        stats["token_mismatch"] == 0
        and stats["error_mismatch"] == 0
        and stats["response_mismatch"] == 0
        and stats["argmax_changes"] == 0
        and stats["noul_flips"] == 0
        and probabilities["exact"] == probabilities["questions"]
        and probabilities["max_abs_dp"] == 0.0
        and rerun_identical == len(first)
    )
    report = {
        "what": "text-only parity: image-capable runtime vs the package's original runtime",
        "original": str(args.original),
        "package": str(args.package),
        "rows_file": str(args.rows),
        **stats,
        "seconds": {k: round(v, 1) for k, v in seconds.items()},
        "runtime": new.runtime_info(),
        "kernels": new.kernels,
        "image_contract": new.image_contract(),
        "examples": examples,
    }
    write_json(args.out, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "requests",
                    "questions",
                    "answered",
                    "token_mismatch",
                    "error_mismatch",
                    "response_mismatch",
                    "argmax_changes",
                    "noul_flips",
                    "probabilities",
                    "rerun",
                    "pass",
                )
            }
        )
    )
    raise SystemExit(0 if stats["pass"] else 1)


if __name__ == "__main__":
    main()
