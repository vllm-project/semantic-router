"""Answer parity of two prediction files of the same panel (standard library only).

For every question: the decision (Choice key, Noul side of 0.5, Score argmax level) and the largest absolute
probability difference. Errors on either side count as differences.

usage: m10_compare.py --pair NAME=LEFT,RIGHT [--pair ...] --output OUT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def probabilities(answer: dict[str, Any]) -> dict[str, float] | None:
    if "error" in answer:
        return None
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1 - answer["noul"]}
    return answer["probabilities"]


def decision(answer: dict[str, Any]) -> Any:
    if "error" in answer:
        return ("error", answer["error"])
    if answer["type"] == "noul":
        return answer["noul"] > 0.5
    if answer["type"] == "choice":
        return answer["choice"]
    probs = answer["probabilities"]
    return max(probs, key=probs.get)


def compare(left: Path, right: Path) -> dict[str, Any]:
    def load(path: Path) -> dict[str, dict]:
        return {
            json.loads(line)["id"]: json.loads(line)["answers"]
            for line in path.open(encoding="utf-8")
        }

    a, b = load(left), load(right)
    if set(a) != set(b):
        raise ValueError(f"{left} and {right} cover different items")
    questions = differ = 0
    drift = 0.0
    for item, answers in a.items():
        for qid, answer in answers.items():
            other = b[item][qid]
            questions += 1
            differ += decision(answer) != decision(other)
            p, q = probabilities(answer), probabilities(other)
            if p is not None and q is not None:
                drift = max(drift, max(abs(p[k] - q[k]) for k in p))
    return {
        "items": len(a),
        "questions": questions,
        "decisions_differ": differ,
        "max_probability_drift": drift,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for spec in args.pair:
        name, paths = spec.split("=", 1)
        left, right = paths.split(",", 1)
        result[name] = {
            "left": left,
            "right": right,
            **compare(Path(left), Path(right)),
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: (v["decisions_differ"], v["max_probability_drift"])
                for k, v in result.items()
            }
        )
    )


if __name__ == "__main__":
    main()
