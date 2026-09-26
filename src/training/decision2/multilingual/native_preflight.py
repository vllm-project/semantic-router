"""Verify every multilingual prompt against the pinned Eikos option parser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from inference.run import load_prompts
from multilingual.audit import sha256


def check(panel: Path, eikos_source: Path) -> dict:
    sys.path.insert(0, str(eikos_source.resolve(strict=True)))
    import decision_core

    prompts = load_prompts(panel / "prompts.jsonl")
    targets = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (panel / "targets.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    if len(prompts) != len(targets):
        raise ValueError("Prompt/target counts differ")
    counts = {"choice": 0, "noul": 0, "score": 0}
    max_options = 0
    for row in prompts:
        target = targets[row["id"]]
        question = row["questions"]["decision"]
        options = decision_core.options_of(question)
        max_options = max(max_options, len(options))
        if target["task_type"] == "choice":
            expected = list(question["criteria"])
            if [name for name, _ in options] != expected or target[
                "gold"
            ] not in expected:
                raise ValueError(f"{row['id']}: Choice option labels differ")
        elif target["task_type"] == "noul":
            if [name for name, _ in options] != ["yes", "no"]:
                raise ValueError(f"{row['id']}: native Noul order differs")
        elif target["task_type"] == "score":
            if [name for name, _ in options] != ["0", "1", "2", "3"]:
                raise ValueError(f"{row['id']}: native Score levels differ")
        else:
            raise ValueError("Unknown task type")
        counts[target["task_type"]] += 1
    return {
        "prompts_sha256": sha256(panel / "prompts.jsonl"),
        "eikos_decision_core_sha256": sha256(eikos_source / "decision_core.py"),
        "items": len(prompts),
        "by_type": counts,
        "native_max_options": max_options,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--eikos-source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.panel, args.eikos_source), sort_keys=True))


if __name__ == "__main__":
    main()
