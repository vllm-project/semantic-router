"""Score a frozen multilingual hard DEV pilot with paired base-case units."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from multilingual.audit import sha256
from multilingual.hard_pilot import LANGUAGES, VERSION
from multilingual.score import _interpret


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def score(panel: Path, predictions: Path) -> dict:
    manifest_path = panel / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VERSION:
        raise ValueError("Unknown multilingual hard panel version")
    prompt_path, target_path = panel / "prompts.jsonl", panel / "targets.private.jsonl"
    if sha256(prompt_path) != manifest["files_sha256"]["prompts.jsonl"]:
        raise ValueError("Prompt SHA changed after freeze")
    if sha256(target_path) != manifest["private_targets_sha256"]:
        raise ValueError("Private target SHA changed after freeze")
    prompts, targets, answers = (
        _rows(prompt_path),
        _rows(target_path),
        _rows(predictions),
    )
    by_prompt = {row["id"]: row for row in prompts}
    by_target = {row["id"]: row for row in targets}
    by_answer = {row["id"]: row for row in answers}
    if (
        len(by_prompt) != 72
        or len(by_target) != 72
        or len(by_prompt) != len(prompts)
        or len(by_target) != len(targets)
        or len(by_answer) != len(answers)
        or set(by_prompt) != set(by_target)
        or not set(by_answer) <= set(by_prompt)
    ):
        raise ValueError("Missing/extra/duplicate panel identity")
    keys = (
        "backend",
        "model_id",
        "model_revision",
        "model_sha256",
        "calibration_sha256",
    )
    identity = {key: answers[0].get(key) for key in keys} if answers else {}
    if any({key: row.get(key) for key in keys} != identity for row in answers):
        raise ValueError("Model identity changed within predictions")
    per_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
    invalid = Counter()
    results = {}
    for target in targets:
        item_id = target["id"]
        prediction = by_answer.get(item_id)
        semantic = None
        valid = False
        if prediction is not None:
            if prediction.get("source_input_sha256") != target["source_input_sha256"]:
                raise ValueError(f"{item_id}: prediction input SHA mismatch")
            native = prediction.get("answers")
            if isinstance(native, dict) and set(native) == {"decision"}:
                semantic, valid = _interpret(native["decision"], target)
        if not valid:
            invalid[target["language"]] += 1
        correct = int(valid and semantic == target["semantic_gold"])
        key = (target["base_id"], target["language"])
        if key in results:
            raise ValueError("Duplicate base/language pair")
        results[key] = {"correct": correct, "valid": valid, "semantic": semantic}
        per_cell[(target["language"], target["task_type"])].append(correct)
    bases = sorted({base for base, _language in results})
    if len(bases) != 18 or any(
        (base, language) not in results for base in bases for language in LANGUAGES
    ):
        raise ValueError("Incomplete frozen base/language matrix")
    by_language = {}
    for language in LANGUAGES:
        rows = [results[(base, language)] for base in bases]
        english = [results[(base, "en")] for base in bases]
        by_language[language] = {
            "correct": sum(row["correct"] for row in rows),
            "items": len(rows),
            "invalid_or_missing": invalid[language],
            "english_paired_both_correct": sum(
                row["correct"] and en["correct"]
                for row, en in zip(rows, english, strict=True)
            ),
            "semantic_prediction_agreement_with_english": sum(
                row["valid"] and en["valid"] and row["semantic"] == en["semantic"]
                for row, en in zip(rows, english, strict=True)
            ),
            "by_type": {
                task: {
                    "correct": sum(per_cell[(language, task)]),
                    "items": len(per_cell[(language, task)]),
                }
                for task in ("choice", "noul", "score")
            },
        }
    return {
        "schema_version": VERSION + "-score/1",
        "scope": "private DEV diagnostic; not a release leaderboard",
        "manifest_sha256": sha256(manifest_path),
        "prompts_sha256": sha256(prompt_path),
        "targets_sha256": sha256(target_path),
        "predictions_sha256": sha256(predictions),
        "model": identity,
        "independent_base_cases": len(bases),
        "prompt_rows": len(targets),
        "all_four_languages_correct_bases": sum(
            all(results[(base, language)]["correct"] for language in LANGUAGES)
            for base in bases
        ),
        "by_language": by_language,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = score(args.panel, args.predictions)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report_sha256": sha256(args.output),
                "manifest_sha256": report["manifest_sha256"],
                "independent_base_cases": report["independent_base_cases"],
                "all_four_languages_correct_bases": report[
                    "all_four_languages_correct_bases"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
