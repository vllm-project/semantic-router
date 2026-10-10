"""Parity of a built package's runtime with ws-eval's ``engine.py`` on suite requests.

Both run in one process on one device (same kernels, same Triton autotune state): first the engine
through its official ``KitEngine`` request path (questions in request order, ``batch_size`` per forward
pass) over every row, then the package through ``decision25_runtime.Decision25.prepare/run``. Token ids,
unsupported decisions, probabilities and the kit answers are compared request by request.
``--sequential`` frees the engine's model before loading the package (two 27B copies need ~110 GB).

    python -m d25.vega.release.parity --package <package dir> --rows parity-600.jsonl.gz --out parity.json \
        [--engine-ckpt <dir>] [--device cuda:0] [--readout-dtype float32] [--sequential]

Pass: identical token ids, identical unsupported requests, zero argmax changes, zero noul flips at 0.5 and
``max_abs_dp_decision`` (the engine's chosen option, and p(true) of noul questions) <= 1e-5.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

from d25.vega.release.sample import read_rows

TOLERANCE = 1e-5


def load_package_runtime(package: Path):
    sys.path.insert(0, str(package))
    if (package / "d3_runtime.py").is_file():
        import d3_runtime

        d3_runtime.Decision25 = d3_runtime.D3
        return d3_runtime
    import decision25_runtime

    return decision25_runtime


def engine_pass(engine, rows: list[dict], batch_size: int) -> list[dict]:
    from d25.vega.eval.engine import option_count

    out = []
    for row in rows:
        state, questions = row["state"], row["questions"]
        keys = list(questions)
        sequences = engine.codec.encode(
            [{"state": state, "question": questions[k]} for k in keys]
        )
        record = {
            "keys": keys,
            "sequences": sequences,
            "over": engine.codec.over_limit(max(map(len, sequences))),
        }
        if not record["over"]:
            counts = [option_count(questions[k]) for k in keys]
            probs = []
            for start in range(0, len(keys), batch_size):
                probs += engine.probabilities(
                    sequences[start : start + batch_size],
                    counts[start : start + batch_size],
                )
            record["probs"] = probs
        out.append(record)
    engine.synchronize()
    return out


def compare_pass(package, rows: list[dict], expected: list[dict]) -> tuple[dict, list]:
    from d25.vega.common import decision_format as df

    stats = {
        "requests": 0,
        "questions": 0,
        "unsupported_both": 0,
        "unsupported_mismatch": 0,
        "token_mismatch": 0,
        "invalid": 0,
        "argmax_changes": 0,
        "noul_flips": 0,
        "answer_mismatch": 0,
        "max_abs_dp_all": 0.0,
        "max_abs_dp_decision": 0.0,
    }
    examples = []
    for row, want_row in zip(rows, expected):
        questions = row["questions"]
        keys = want_row["keys"]
        stats["requests"] += 1
        stats["questions"] += len(keys)
        prepared = package.prepare(row["state"], questions)
        if any(e["error"] == "invalid_question" for e in prepared.errors.values()):
            stats["invalid"] += 1
            examples.append(
                {"run_id": row["_evaluation"]["run_id"], "invalid": prepared.errors}
            )
            continue
        package_over = any(
            e["error"] == "max_length_exceeded" for e in prepared.errors.values()
        )
        for key, sequence in zip(keys, want_row["sequences"]):
            if key in prepared.sequences and prepared.sequences[key] != sequence:
                stats["token_mismatch"] += 1
        if want_row["over"] or package_over:
            stats[
                (
                    "unsupported_both"
                    if want_row["over"] == package_over
                    else "unsupported_mismatch"
                )
            ] += 1
            continue
        got, _ = package.run(prepared)
        response = package.respond(prepared, got, 0)
        for key, want in zip(keys, want_row["probs"]):
            have = got[key]
            stats["max_abs_dp_all"] = max(
                stats["max_abs_dp_all"], max(abs(a - b) for a, b in zip(want, have))
            )
            question = questions[key]
            best = max(range(len(want)), key=want.__getitem__)
            if question["type"] == "noul":
                decision_diff = abs(want[1] - have[1])
                stats["noul_flips"] += (want[1] >= 0.5) != (have[1] >= 0.5)
            else:
                decision_diff = abs(want[best] - have[best])
                stats["argmax_changes"] += (
                    max(range(len(have)), key=have.__getitem__) != best
                )
            stats["max_abs_dp_decision"] = max(
                stats["max_abs_dp_decision"], decision_diff
            )
            kit = df.to_answer(question, want)
            mine = {
                k: v for k, v in response["answers"][key].items() if k != "confidence"
            }
            if kit.get("choice") != mine.get("choice") or (
                question["type"] == "noul"
                and abs(kit["noul"] - mine["noul"]) > TOLERANCE
            ):
                stats["answer_mismatch"] += 1
                if len(examples) < 20:
                    examples.append(
                        {
                            "run_id": row["_evaluation"]["run_id"],
                            "question": key,
                            "engine": kit,
                            "package": mine,
                        }
                    )
    package.synchronize()
    return stats, examples


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--rows", required=True, nargs="+")
    ap.add_argument(
        "--engine-ckpt",
        type=Path,
        help="checkpoint for engine.py (default: the package dir)",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--readout-dtype", choices=("float32", "bfloat16"))
    ap.add_argument(
        "--sequential",
        action="store_true",
        help="free the engine model before loading the package",
    )
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    import torch

    from d25.vega.eval.engine import CodeReadoutModel

    rt = load_package_runtime(args.package)
    rows = [r for path in args.rows for r in read_rows(path)][: args.limit]
    seconds = {}
    started = time.time()
    engine = CodeReadoutModel(
        str(args.engine_ckpt or args.package),
        device=args.device,
        **({"readout_dtype": args.readout_dtype} if args.readout_dtype else {}),
    )
    seconds["engine_load"] = time.time() - started
    started = time.time()
    # The package skips cuDNN attention on CUDA builds; the engine runs under the same SDPA backends.
    with rt.sdpa_backends(args.device):
        expected = engine_pass(engine, rows, args.batch_size)
    seconds["engine_rows"] = time.time() - started
    engine_provenance = engine.provenance()
    if args.sequential:
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    started = time.time()
    package = rt.Decision25.from_pretrained(
        args.package,
        device=args.device,
        batch_size=args.batch_size,
        readout_dtype=args.readout_dtype,
    )
    seconds["package_load"] = time.time() - started
    started = time.time()
    stats, examples = compare_pass(package, rows, expected)
    seconds["package_rows"] = time.time() - started
    stats["pass"] = (
        stats["token_mismatch"] == 0
        and stats["unsupported_mismatch"] == 0
        and stats["invalid"] == 0
        and stats["argmax_changes"] == 0
        and stats["noul_flips"] == 0
        and stats["answer_mismatch"] == 0
        and stats["max_abs_dp_decision"] <= TOLERANCE
    )
    report = {
        "package": str(args.package),
        "engine_ckpt": str(args.engine_ckpt or args.package),
        "device": args.device,
        "batch_size": args.batch_size,
        "sequential": args.sequential,
        "rows": [str(p) for p in args.rows],
        "tolerance": TOLERANCE,
        **stats,
        "seconds": {k: round(v, 1) for k, v in seconds.items()},
        "engine_provenance": engine_provenance,
        "package_provenance": package.provenance(),
        "runtime": package.runtime_info(),
        "examples": examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "requests",
                    "questions",
                    "unsupported_both",
                    "unsupported_mismatch",
                    "token_mismatch",
                    "invalid",
                    "argmax_changes",
                    "noul_flips",
                    "answer_mismatch",
                    "max_abs_dp_all",
                    "max_abs_dp_decision",
                    "pass",
                )
            }
        )
    )
    return 0 if stats["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
