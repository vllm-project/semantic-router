"""Answers of a Decision 2.0 package through its native runtime or through AutoModel (one process each).

  verify_dev2route1.py answer --mode native|automodel --package DIR --prompts JSONL:N... --output OUT.jsonl
  verify_dev2route1.py compare A.jsonl B.jsonl

The native mode imports the package's own ``decision2`` (from a link-free directory, as the runtime
requires); the automodel mode loads ``AutoModel.from_pretrained(DIR, trust_remote_code=True)``.
``compare`` prints identical / changed / maximum drift with the release ``compare_answers``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def link_free(package: Path, target: Path) -> Path:
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if path.is_dir() or relative.parts[0] == ".cache":
            continue
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        os.link(path.resolve(), target / relative)
    return target


def answer(args: argparse.Namespace) -> None:
    package = args.package
    if any(p.is_symlink() for p in package.rglob("*")):
        package = link_free(package, args.output.with_suffix(".view"))
    if args.mode == "native":
        sys.path.insert(0, str(package))
        from decision2 import Decision2

        model = Decision2.from_pretrained(package, device="cpu")
    else:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            str(package), trust_remote_code=True, device_map="cpu"
        )
    with args.output.open("x", encoding="utf-8") as out:
        for spec in args.prompts:
            path, count = spec.rsplit(":", 1)
            with open(path, encoding="utf-8") as stream:
                rows = [json.loads(line) for line, _ in zip(stream, range(int(count)))]
            for row in rows:
                response = model.system_one(
                    state=row["state"], questions=row["questions"]
                )
                out.write(
                    json.dumps({"id": row["id"], "answers": response["answers"]}) + "\n"
                )


def compare(left: Path, right: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "_examples", HERE.parent / "examples.py"
    )
    examples = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(examples)
    a = [json.loads(line) for line in left.open()]
    b = {row["id"]: row for row in (json.loads(line) for line in right.open())}
    changed = drift = 0
    for row in a:
        result = examples.compare_answers(row["answers"], b[row["id"]]["answers"])
        changed += result["category_changes"] + result["missing"]
        drift = max(drift, result["max_abs_drift"])
    identical = left.read_text() == right.read_text()
    print(
        json.dumps(
            {
                "prompts": len(a),
                "identical": identical,
                "answer_changes": changed,
                "max_abs_drift": drift,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="step", required=True)
    run = sub.add_parser("answer")
    run.add_argument("--mode", choices=("native", "automodel"), required=True)
    run.add_argument("--package", type=Path, required=True)
    run.add_argument("--prompts", action="append", required=True)
    run.add_argument("--output", type=Path, required=True)
    both = sub.add_parser("compare")
    both.add_argument("left", type=Path)
    both.add_argument("right", type=Path)
    args = parser.parse_args()
    if args.step == "answer":
        answer(args)
    else:
        compare(args.left, args.right)


if __name__ == "__main__":
    main()
