"""Check frozen multilingual hard prompts against a pinned native option parser."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(panel: Path, source: Path) -> dict:
    manifest = json.loads((panel / "manifest.json").read_text(encoding="utf-8"))
    prompt_path = panel / "prompts.jsonl"
    target_path = panel / "targets.private.jsonl"
    if sha256(prompt_path) != manifest["files_sha256"]["prompts.jsonl"]:
        raise ValueError("Prompt SHA mismatch")
    if sha256(target_path) != manifest["private_targets_sha256"]:
        raise ValueError("Target SHA mismatch")
    core_path = source / "decision_core.py"
    core_hash = sha256(core_path)
    sys.path.insert(0, str(source.resolve(strict=True)))
    import decision_core

    prompts = [
        json.loads(line)
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
    ]
    targets = [
        json.loads(line)
        for line in target_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(prompts) != 72 or len(targets) != 72:
        raise ValueError("Unexpected panel length")
    counts = {"choice": 0, "noul": 0, "score": 0}
    for prompt, target in zip(prompts, targets, strict=True):
        if prompt["id"] != target["id"]:
            raise ValueError("Prompt/target order mismatch")
        question = prompt["questions"]["decision"]
        options = decision_core.options_of(question)
        kind = target["task_type"]
        if kind == "choice":
            expected = list(question["criteria"])
            if [name for name, _description in options] != expected:
                raise ValueError(f"{prompt['id']}: Choice native order mismatch")
            if target["gold"] not in expected:
                raise ValueError(f"{prompt['id']}: missing gold Choice label")
        elif kind == "noul":
            if [name for name, _description in options] != ["yes", "no"]:
                raise ValueError(f"{prompt['id']}: Noul native order mismatch")
        elif kind == "score":
            if [name for name, _description in options] != ["0", "1", "2", "3"]:
                raise ValueError(f"{prompt['id']}: Score native order mismatch")
        else:
            raise ValueError(f"Unknown type: {kind}")
        counts[kind] += 1
    if counts != {"choice": 24, "noul": 24, "score": 24}:
        raise ValueError("Task type counts differ")
    return {
        "manifest_sha256": sha256(panel / "manifest.json"),
        "native_core_sha256": core_hash,
        "preflight_source_sha256": sha256(Path(__file__)),
        "prompt_rows_checked": len(prompts),
        "by_type": counts,
        "result": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = check(args.panel, args.source)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
