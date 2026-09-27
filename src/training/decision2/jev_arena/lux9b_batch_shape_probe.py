"""Gold-free, one-request diagnostic for Lux's published Score example.

This changes the number of questions in a request, so it is never a release
example or JevArena result. The published and prior native outputs are fixed
references. The native collector and its numerical threshold stay unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from inference.run import digest, file_digest

EXAMPLE_SHA256 = "12d68a696b50851b3614c1ed9d5e73347780ca5ad487ee2ad35e746ab40bbb18"
COMBINED_SHA256 = "923536bbf4d0f0a971e62e4aca4a9ae74828817e0b1ebd8aeeddae5155111a1c"
MODEL_REVISION = "bd45a30aee8c84032791c245c70f86dee5389cc8"
MODEL_CONFIG_SHA256 = "985ade73c509399291d60b5f98e8bbbbe99c0ee0efe611a5f84604f71420e0fd"
PROBE_ID = "lux-batch-shape:severity-only"


def _read_one(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("Expected exactly one native prediction")
    return rows[0]


def _references(
    model_path: Path, combined_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    example_path = model_path / "model-card-example.json"
    if (
        file_digest(example_path) != EXAMPLE_SHA256
        or file_digest(combined_path) != COMBINED_SHA256
    ):
        raise ValueError("Published example or prior native prediction bytes changed")
    example = json.loads(example_path.read_text(encoding="utf-8"))
    combined = _read_one(combined_path)
    request = example["requests"]["usage"]
    if set(request) != {"state", "questions"} or set(request["questions"]) != {
        "owner",
        "active",
        "severity",
    }:
        raise ValueError("Published usage request shape changed")
    if (
        combined.get("source_input_sha256") != digest(request)
        or combined.get("model_revision") != MODEL_REVISION
        or combined.get("model_config_sha256") != MODEL_CONFIG_SHA256
        or combined.get("runtime_matches_validated") is not True
        or set(combined.get("answers", {})) != set(request["questions"])
    ):
        raise ValueError("Prior native request, model or runtime differs")
    return example, combined


def prepare(model_path: Path, combined_path: Path, output: Path) -> dict[str, Any]:
    example, _ = _references(model_path, combined_path)
    request = example["requests"]["usage"]
    prompt = {
        "id": PROBE_ID,
        "state": request["state"],
        "questions": {"severity": request["questions"]["severity"]},
    }
    with output.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(prompt, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
    return {
        "probe_prompt_sha256": file_digest(output),
        "input_sha256": digest(
            {"state": prompt["state"], "questions": prompt["questions"]}
        ),
    }


def analyze(model_path: Path, combined_path: Path, probe_path: Path) -> dict[str, Any]:
    example, combined = _references(model_path, combined_path)
    probe = _read_one(probe_path)
    request = example["requests"]["usage"]
    subset = {
        "state": request["state"],
        "questions": {"severity": request["questions"]["severity"]},
    }
    if (
        probe.get("id") != PROBE_ID
        or probe.get("source_input_sha256") != digest(subset)
        or probe.get("model_revision") != MODEL_REVISION
        or probe.get("model_config_sha256") != MODEL_CONFIG_SHA256
        or probe.get("runtime_matches_validated") is not True
        or set(probe.get("answers", {})) != {"severity"}
    ):
        raise ValueError("Probe request, model, or native runtime differs")
    published = example["actual_responses"]["usage"]["answers"]["severity"]
    original = combined["answers"]["severity"]
    current = probe["answers"]["severity"]
    support = {"0", "1", "2"}
    if any(
        answer.get("type") != "score"
        or set(answer.get("probabilities", {})) != support
        or not math.isfinite(float(answer.get("score", float("nan"))))
        for answer in (published, original, current)
    ):
        raise ValueError("Score answer support or numerical value differs")

    def maximum(left: dict[str, Any], right: dict[str, Any]) -> float:
        values = [abs(float(left["score"]) - float(right["score"]))]
        values += [
            abs(float(left["probabilities"][key]) - float(right["probabilities"][key]))
            for key in sorted(support)
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite Score drift")
        return max(values)

    return {
        "diagnostic_only": True,
        "model_revision": MODEL_REVISION,
        "published_example_sha256": EXAMPLE_SHA256,
        "prior_combined_sha256": COMBINED_SHA256,
        "probe_predictions_sha256": file_digest(probe_path),
        "prior_combined_vs_published_max": maximum(original, published),
        "score_only_vs_combined_max": maximum(current, original),
        "score_only_vs_published_max": maximum(current, published),
        "frozen_release_example_ceiling": 0.02,
        "published_gate_still_applies_to_original_request": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "analyze"):
        command = sub.add_parser(name)
        command.add_argument("--model-path", required=True, type=Path)
        command.add_argument("--combined", required=True, type=Path)
        command.add_argument(
            "--output" if name == "prepare" else "--probe", required=True, type=Path
        )
    args = parser.parse_args()
    result = (
        prepare(args.model_path, args.combined, args.output)
        if args.command == "prepare"
        else analyze(args.model_path, args.combined, args.probe)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
