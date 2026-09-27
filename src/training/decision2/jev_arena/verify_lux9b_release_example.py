"""Build gold-free prompts and check Lux 1.0 against its own release example."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from inference.run import file_digest

EXAMPLE_SHA256 = "12d68a696b50851b3614c1ed9d5e73347780ca5ad487ee2ad35e746ab40bbb18"
MAX_DRIFT = 0.02


def _example(model_path: Path) -> dict[str, Any]:
    path = model_path / "model-card-example.json"
    if file_digest(path) != EXAMPLE_SHA256:
        raise ValueError("Published Lux example differs from the fixed release")
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value["requests"]) != {"model_card", "usage"} or set(
        value["actual_responses"]
    ) != {"model_card", "usage"}:
        raise ValueError("Published example has unexpected cases")
    return value


def make_prompts(model_path: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    example = _example(model_path)
    with output.open("x", encoding="utf-8") as target:
        for name, payload in example["requests"].items():
            target.write(
                json.dumps(
                    {"id": f"lux-release-example:{name}", **payload},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _finite_drift(actual: Any, expected: Any) -> float:
    actual = float(actual)
    expected = float(expected)
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError("Nonfinite native or reference probability")
    return abs(actual - expected)


def verify(model_path: Path, predictions: Path) -> dict[str, Any]:
    example = _example(model_path)
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != 2 or {row["id"] for row in rows} != {
        "lux-release-example:model_card",
        "lux-release-example:usage",
    }:
        raise ValueError("Native release example must contain both cases exactly once")
    maximum_drift = 0.0
    answer_count = 0
    for row in rows:
        if row.get("runtime_matches_validated") is not True or not row.get(
            "revision_attested"
        ):
            raise ValueError("Native release profile or model revision is unverified")
        name = row["id"].split(":", 1)[1]
        expected = example["actual_responses"][name]["answers"]
        if set(row["answers"]) != set(expected):
            raise ValueError("Release example answer IDs differ")
        for key, wanted in expected.items():
            got = row["answers"][key]
            kind = wanted["type"]
            if got.get("type") != kind:
                raise ValueError(f"{name}/{key}: answer type differs")
            if kind == "choice" and got.get("choice") != wanted["choice"]:
                raise ValueError(f"{name}/{key}: choice differs")
            if kind in {"noul", "score"}:
                maximum_drift = max(
                    maximum_drift, _finite_drift(got[kind], wanted[kind])
                )
            reference_probs = wanted.get("probabilities")
            if reference_probs is not None:
                probabilities = got.get("probabilities")
                if not isinstance(probabilities, dict) or set(probabilities) != set(
                    reference_probs
                ):
                    raise ValueError(f"{name}/{key}: probability support differs")
                maximum_drift = max(
                    maximum_drift,
                    *(
                        _finite_drift(probabilities[label], reference_probs[label])
                        for label in reference_probs
                    ),
                )
            answer_count += 1
    if answer_count != 5 or maximum_drift > MAX_DRIFT:
        raise ValueError(
            f"Published release example mismatch: {answer_count=} {maximum_drift=}"
        )
    return {
        "items": len(rows),
        "answer_slots": answer_count,
        "max_absolute_probability_or_score_drift": maximum_drift,
        "model_example_sha256": EXAMPLE_SHA256,
        "predictions_sha256": file_digest(predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prompts", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--model-path", type=Path, required=True)
        command.add_argument(
            "--output" if name == "prompts" else "--predictions",
            type=Path,
            required=True,
        )
    args = parser.parse_args()
    if args.command == "prompts":
        make_prompts(args.model_path, args.output)
    else:
        print(json.dumps(verify(args.model_path, args.predictions), sort_keys=True))


if __name__ == "__main__":
    main()
