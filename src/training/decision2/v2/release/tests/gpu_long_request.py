"""GPU regression: a long multi-question request gives valid, batch-independent answers.

Run in the pinned image on one leased GPU, with PYTHONPATH at the mirror's
src/training/decision2 (the image's FLA kernels on the path):

    python3 -m v2.release.tests.gpu_long_request --package DIR [--base-path DIR] \
        --out RESULT.json [--tokens 14224] [--questions 32]

Builds a synthetic request of generated (non-benchmark) text: one question whose
prompt is about ``--tokens`` tokens long and ``--questions - 1`` short ones, Choice
and Noul alternating. With 32 questions of a 14,224-token prompt the padded batch
holds 455,168 tokens; on a 48-value-head Qwen3.5 backbone (Decision-2.0-Vega-27B) that is
2.8e9 elements per gated-delta q / k / v tensor, past the kernels' 32-bit offsets,
which before the forward token budget gave non-finite logits or a GPU memory fault
(the budget, 174,762 padded tokens on that backbone, splits it into two forwards).
Passes if every answer is valid, every forward stays within the runtime's budget,
and each question's answer equals the same question asked alone (same choice or
Noul side; probabilities within ``--tolerance``). With ``--tie-margin M``, a side
flip within ``--tolerance`` whose alone answer lies within M of its decision
boundary (Noul |P(true) - 0.5| or Choice top-two margin) is listed as a tie instead
of a mismatch. Writes the result JSON (counts and shapes only); exits non-zero on
any failure.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

WORDS = (
    "amber basin cedar delta ember fjord granite harbor island juniper kestrel lantern meadow "
    "nectar orchard pebble quarry river summit timber upland valley willow yarrow zephyr"
).split()


def synthetic(backend, tokens: int, questions: int, seed: int = 0) -> dict:
    from decision2._vendor.dev2model.decision_model import encode
    from decision2._vendor.dev2model.infer import question_to_row

    rng = random.Random(seed)

    def text(n: int) -> str:
        return " ".join(rng.choice(WORDS) for _ in range(n))

    def question(index: int, words: int) -> dict:
        if index % 2:
            criteria = {"a": text(4), "b": text(4), "c": text(4)}
            kind = "choice"
        else:
            criteria = {"false": text(3), "true": text(3)}
            kind = "noul"
        return {
            "type": kind,
            "instructions": {"task": text(12), "candidate": text(words)},
            "criteria": criteria,
        }

    state = {"note": text(30)}

    def length(q: dict) -> int:
        row = question_to_row({"id": "synthetic", "state": state}, "q", q)
        return len(encode(row, backend.tokenizer, 10**9)["ids"])

    words = tokens
    for _ in range(4):
        long = question(0, words)
        got = length(long)
        if abs(got - tokens) <= 8:
            break
        words = max(1, round(words * tokens / got))
    rows = {"q0": long}
    for index in range(1, questions):
        rows[f"q{index}"] = question(index, 40 + 7 * index)
    return {"state": state, "questions": rows}


def same(a: dict, b: dict, tolerance: float) -> tuple[bool, float]:
    if a.get("error") or b.get("error") or a["type"] != b["type"]:
        return False, float("inf")
    if a["type"] == "noul":
        drift = abs(a["noul"] - b["noul"])
        return (a["noul"] >= 0.5) == (b["noul"] >= 0.5) and drift <= tolerance, drift
    drift = max(
        abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"]
    )
    return a["choice"] == b["choice"] and drift <= tolerance, drift


def margin(answer: dict) -> float:
    if answer["type"] == "noul":
        return abs(answer["noul"] - 0.5)
    top = sorted(answer["probabilities"].values(), reverse=True)
    return top[0] - top[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=14_224)
    parser.add_argument("--questions", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--tie-margin", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--budget", type=int, help="override the runtime's token budget"
    )
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    sys.path.insert(0, str(package))
    from decision2 import Decision2

    model = Decision2.from_pretrained(
        package, device=args.device, base_path=args.base_path
    )
    backend = model.backend
    torch = backend.torch
    if args.budget:
        backend.batch_tokens = args.budget
    shapes: list[list[int]] = []
    forward = backend.model.forward

    def recording(*a, **kw):
        shapes.append(list(kw["input_ids"].shape))
        print(json.dumps({"forward": shapes[-1]}), file=sys.stderr, flush=True)
        return forward(*a, **kw)

    backend.model.forward = recording
    row = synthetic(backend, args.tokens, args.questions)
    started = time.perf_counter()
    together = model.system_one(state=row["state"], questions=row["questions"])[
        "answers"
    ]
    torch.cuda.synchronize()
    together_ms = (time.perf_counter() - started) * 1000
    batched = [list(s) for s in shapes]
    alone, worst, mismatched, ties = {}, 0.0, [], []
    for qid, question in row["questions"].items():
        alone[qid] = model.system_one(state=row["state"], questions={qid: question})[
            "answers"
        ][qid]
        ok, drift = same(together[qid], alone[qid], args.tolerance)
        worst = max(worst, drift)
        if not ok:
            tie = (
                drift <= args.tolerance
                and not alone[qid].get("error")
                and margin(alone[qid]) <= args.tie_margin
            )
            (ties if tie else mismatched).append(qid)
    budget = getattr(backend, "batch_tokens", None)
    invalid = sorted(q for q, a in together.items() if a.get("error"))
    within = budget is None or all(rows * width <= budget for rows, width in batched)
    result = {
        "package_model": model.model_name,
        "tokens_long_question": args.tokens,
        "questions": args.questions,
        "budget_tokens": budget,
        "forwards": batched,
        "invalid_answers": invalid,
        "mismatched_vs_alone": mismatched,
        "tie_margin": args.tie_margin,
        "ties_vs_alone": {q: margin(alone[q]) for q in ties},
        "max_drift_vs_alone": worst,
        "together_ms": round(together_ms, 1),
        "within_budget": within,
        "pass": not invalid and not mismatched and within,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
