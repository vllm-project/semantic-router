"""Run one C1 converter over its pinned snapshot: valid candidates + count-only receipt.

    python3 -m v2.eval.sealed.candidates --source <key> --root <snapshot dir> \
        --output <private candidates.jsonl> --receipt <receipt.json>

The candidates file holds item text and gold and stays in private storage. The receipt
holds the source spec, a manifest of the snapshot files, and counts only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

from v2.eval.sealed.schema import LONG_INPUT_CHARS, input_chars, validate

SCHEMA = "dev2-sealed-c1-candidates/1"


def load(key: str) -> ModuleType:
    module = importlib.import_module(f"v2.eval.sealed.sources.{key}")
    if module.SPEC.key != key:
        raise ValueError(f"{key}: SPEC.key is {module.SPEC.key!r}")
    return module


def snapshot_manifest(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".cache" not in path.relative_to(root).parts:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(block)
            files[str(path.relative_to(root))] = digest.hexdigest()
    joined = "".join(f"{k}\0{v}\n" for k, v in files.items()).encode()
    return {"files": files, "manifest_sha256": hashlib.sha256(joined).hexdigest()}


def run(key: str, root: Path, output: Path | None) -> dict[str, Any]:
    module = load(key)
    spec = module.SPEC
    counts: dict[str, Counter] = defaultdict(Counter)
    invalid: Counter = Counter()
    groups: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()
    lines = []
    for candidate in module.candidates(root):
        problems = validate(candidate, spec)
        identity = (candidate.task, candidate.source_item_id)
        if identity in seen:
            problems.append("duplicate task/source_item_id")
        if problems:
            invalid.update(problems)
            continue
        seen.add(identity)
        size = input_chars(candidate.state, candidate.question)
        task = counts[candidate.task]
        task["candidates"] += 1
        task[f"type:{candidate.question['type']}"] += 1
        task[f"label:{candidate.balance_label}"] += 1
        task[f"language:{candidate.language}"] += 1
        task["long_input"] += size >= LONG_INPUT_CHARS
        groups[candidate.task].add(candidate.group_id)
        lines.append(
            json.dumps(candidate.to_json(), ensure_ascii=False, sort_keys=True)
        )
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "spec": asdict(spec),
        "snapshot": snapshot_manifest(root),
        "tasks": {
            task: {**dict(sorted(value.items())), "groups": len(groups[task])}
            for task, value in sorted(counts.items())
        },
        "invalid": dict(invalid),
        "valid": len(lines),
    }
    if output is not None:
        data = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        receipt["candidates_sha256"] = hashlib.sha256(data).hexdigest()
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run(args.source, args.root, args.output)
    text = json.dumps(receipt, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(text, encoding="utf-8")
    summary = {
        task: {k: v for k, v in value.items() if not k.startswith("language:")}
        for task, value in receipt["tasks"].items()
    }
    print(
        json.dumps(
            {
                "valid": receipt["valid"],
                "invalid": receipt["invalid"],
                "tasks": summary,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
