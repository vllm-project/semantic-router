"""Gold-free reproducibility gate for the pinned current Lux 1.0 package.

The released example's *requests* are usable fixed inputs. Its recorded answers
belong to an earlier bundle and are deliberately never read by this gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from inference.run import completed_rows, file_digest, load_prompts, local_revision

MODEL_ID = "llm-semantic-router/Decision-1.0-Lux-9B"
REVISION = "bd45a30aee8c84032791c245c70f86dee5389cc8"
BUNDLE_SHA256 = "985ade73c509399291d60b5f98e8bbbbe99c0ee0efe611a5f84604f71420e0fd"
RELEASE_SHA256 = "f786ce80a159c716f22b7d9c652af765b06f9db083bf43a8b08e67c3be3e69c5"
EXAMPLE_SHA256 = "12d68a696b50851b3614c1ed9d5e73347780ca5ad487ee2ad35e746ab40bbb18"
MAX_REPRO_DRIFT = 1e-6
EXPECTED_QUESTIONS = {"model_card": 2, "usage": 3}


def verify_package(model_path: Path) -> None:
    if file_digest(model_path / "bundle-manifest.json") != BUNDLE_SHA256:
        raise ValueError("Lux bundle differs from pinned current package")
    if file_digest(model_path / "release-manifest.json") != RELEASE_SHA256:
        raise ValueError("Lux release manifest differs from pinned current package")
    if file_digest(model_path / "model-card-example.json") != EXAMPLE_SHA256:
        raise ValueError("Lux released example requests differ")
    if not local_revision(model_path, REVISION):
        raise ValueError("Lux downloaded file revision is not attested")
    release = json.loads((model_path / "release-manifest.json").read_text())
    if release.get("bundle_manifest_sha256") != BUNDLE_SHA256:
        raise ValueError("Lux release manifest does not bind current bundle")


def make_prompts(model_path: Path, output: Path) -> dict[str, Any]:
    verify_package(model_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    example = json.loads((model_path / "model-card-example.json").read_text())
    requests = example.get("requests")
    if not isinstance(requests, dict) or set(requests) != set(EXPECTED_QUESTIONS):
        raise ValueError("Lux released requests have unexpected names")
    with output.open("x", encoding="utf-8") as stream:
        for name in EXPECTED_QUESTIONS:
            request = requests[name]
            if (
                not isinstance(request, dict)
                or set(request) != {"state", "questions"}
                or not isinstance(request["questions"], dict)
                or len(request["questions"]) != EXPECTED_QUESTIONS[name]
            ):
                raise ValueError("Lux released request shape differs")
            stream.write(
                json.dumps(
                    {"id": f"lux-current-gate:{name}", **request},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    return {"items": 2, "answer_slots": 5, "prompts_sha256": file_digest(output)}


def _max_numeric_drift(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise ValueError("Native answer field or probability support differs")
        return max((_max_numeric_drift(left[k], right[k]) for k in left), default=0.0)
    if isinstance(left, bool) or isinstance(right, bool):
        if left is not right:
            raise ValueError("Native boolean answer field differs")
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("Nonfinite native answer")
        return abs(float(left) - float(right))
    if left != right:
        raise ValueError("Native answer category or field differs")
    return 0.0


def verify_pair(
    model_path: Path, prompts: Path, first: Path, second: Path
) -> dict[str, Any]:
    verify_package(model_path)
    rows = load_prompts(prompts)
    if (
        len(rows) != 2
        or {r["id"] for r in rows}
        != {f"lux-current-gate:{name}" for name in EXPECTED_QUESTIONS}
        or sum(len(r["questions"]) for r in rows) != 5
    ):
        raise ValueError("Current-package prompts have unexpected shape")
    by_id = {r["id"]: r for r in rows}
    results = []
    for path in (first, second):
        completed = completed_rows(
            path, rows, "lux", REVISION, BUNDLE_SHA256, MODEL_ID, True
        )
        if len(completed) != 2:
            raise ValueError("Native process did not complete both prompts")
        predictions = {
            r["id"]: r
            for r in (json.loads(line) for line in path.read_text().splitlines())
        }
        for item_id, row in predictions.items():
            if row.get("runtime_matches_validated") is not True or row.get(
                "runtime_differences"
            ) not in ({}, None):
                raise ValueError("Native process runtime is not release qualified")
            for name, answer in row["answers"].items():
                if not isinstance(answer, dict) or answer.get("type") != by_id[item_id][
                    "questions"
                ][name].get("type"):
                    raise ValueError("Native answer type differs from question")
        results.append(predictions)
    drift = max(
        _max_numeric_drift(
            results[0][item_id]["answers"], results[1][item_id]["answers"]
        )
        for item_id in by_id
    )
    if drift > MAX_REPRO_DRIFT:
        raise ValueError(f"Current Lux native process drift exceeds gate: {drift}")
    return {
        "status": "PASS",
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "bundle_sha256": BUNDLE_SHA256,
        "input_items": 2,
        "answer_slots": 5,
        "max_answer_numeric_drift": drift,
        "max_allowed_drift": MAX_REPRO_DRIFT,
        "prompts_sha256": file_digest(prompts),
        "first_sha256": file_digest(first),
        "second_sha256": file_digest(second),
        "compared_with_stale_published_answers": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prompts_command = sub.add_parser("prompts")
    prompts_command.add_argument("--model-path", type=Path, required=True)
    prompts_command.add_argument("--output", type=Path, required=True)
    verify_command = sub.add_parser("verify")
    verify_command.add_argument("--model-path", type=Path, required=True)
    verify_command.add_argument("--prompts", type=Path, required=True)
    verify_command.add_argument("--first", type=Path, required=True)
    verify_command.add_argument("--second", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prompts":
        result = make_prompts(args.model_path, args.output)
    else:
        result = verify_pair(args.model_path, args.prompts, args.first, args.second)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
