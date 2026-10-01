"""Diagnose a ``gpu_long_request`` mismatch: the same synthetic request, asked in other batch shapes.

    python3 -m v2.release.tests.gpu_long_request_diag --package DIR [--base-path DIR] \
        --tokens N --questions Q --question q24 --out RESULT.json

Rebuilds the regression's synthetic request (same seed) and answers the named question
(a) in the forward-budget split as the regression ran it, (b) alone, and (c) within the request's
short questions only (one batch of the same padded width as the split's short group, the
single-batch path that the budget leaves unchanged). Records each answer's choice / Noul side and
top-two margin (generated text only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from v2.release.tests.gpu_long_request import same, synthetic


def summary(answer: dict) -> dict:
    if answer.get("error"):
        return {"error": answer["error"]}
    if answer["type"] == "noul":
        return {"noul": answer["noul"], "margin": abs(2 * answer["noul"] - 1)}
    p = sorted(answer["probabilities"].values(), reverse=True)
    return {
        "choice": answer["choice"],
        "probabilities": answer["probabilities"],
        "margin": p[0] - p[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--questions", type=int, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="skip (a): a runtime without the budget would run the whole request as one batch",
    )
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    sys.path.insert(0, str(package))
    from decision2 import Decision2

    model = Decision2.from_pretrained(package, base_path=args.base_path)
    row = synthetic(model.backend, args.tokens, args.questions)
    qs, q = row["questions"], args.question

    def ask(questions: dict) -> dict:
        return model.system_one(state=row["state"], questions=questions)["answers"][q]

    split = None if args.skip_split else ask(qs)
    alone = ask({q: qs[q]})
    short = ask({k: v for k, v in qs.items() if k != "q0"})
    result = {
        "package_model": model.model_name,
        "question": q,
        "split": summary(split) if split else None,
        "alone": summary(alone),
        "short_only_single_batch": summary(short),
        "split_vs_alone_same": same(split, alone, 0.02)[0] if split else None,
        "short_only_vs_alone_same": same(short, alone, 0.02)[0],
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
