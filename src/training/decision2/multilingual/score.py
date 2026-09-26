"""Score a native model on the paired multilingual DEV slice by base case.

Reported language gaps use 18 independent semantic groups. They never count
translations or option perturbations as additional independent questions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from multilingual.audit import sha256
from multilingual.pilot import VERSION


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _interpret(answer: dict, target: dict) -> tuple[object | None, bool]:
    if (
        not isinstance(answer, dict)
        or "error" in answer
        or answer.get("type") != target["task_type"]
    ):
        return None, False
    if target["task_type"] == "choice":
        label = answer.get("choice")
        if label not in target["semantic_by_label"]:
            return None, False
        return target["semantic_by_label"][label], True
    if target["task_type"] == "noul":
        probability = answer.get("noul", answer.get("probability"))
        if (
            type(probability) not in (int, float)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            return None, False
        return probability >= 0.5, True
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != {
        "0",
        "1",
        "2",
        "3",
    }:
        return None, False
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in probabilities.values()
    ):
        return None, False
    if not 0.99 <= sum(probabilities.values()) <= 1.01:
        return None, False
    return int(max(probabilities, key=probabilities.get)), True


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def score(panel: Path, predictions: Path) -> dict:
    manifest_path = panel / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VERSION:
        raise ValueError("Unknown multilingual panel")
    prompt_path, target_path = panel / "prompts.jsonl", panel / "targets.jsonl"
    for name, path in (("prompts.jsonl", prompt_path), ("targets.jsonl", target_path)):
        if sha256(path) != manifest["files"][name]:
            raise ValueError(f"{name} differs from frozen panel manifest")
    prompts, targets, answers = (
        _load_jsonl(path) for path in (prompt_path, target_path, predictions)
    )
    if len(prompts) != len(targets) or len(prompts) != len(answers):
        raise ValueError("Incomplete prediction panel")
    prompt_by_id = {row["id"]: row for row in prompts}
    target_by_id = {row["id"]: row for row in targets}
    answer_by_id = {row["id"]: row for row in answers}
    if any(
        len(mapping) != len(prompts)
        for mapping in (prompt_by_id, target_by_id, answer_by_id)
    ):
        raise ValueError("Duplicate ID")
    if set(prompt_by_id) != set(target_by_id) or set(prompt_by_id) != set(answer_by_id):
        raise ValueError("Prompt/target/prediction IDs differ")
    identity_keys = (
        "backend",
        "model_id",
        "model_revision",
        "model_sha256",
        "calibration_sha256",
    )
    identity = {key: answers[0].get(key) for key in identity_keys}
    if any(
        {key: answer.get(key) for key in identity_keys} != identity
        for answer in answers
    ):
        raise ValueError("Prediction model identity changed within file")
    rows = {}
    per_language = defaultdict(list)
    per_language_type = defaultdict(list)
    per_base_language = defaultdict(list)
    invalid = Counter()
    for target in targets:
        row_id = target["id"]
        prediction = answer_by_id[row_id]
        if prediction.get("source_input_sha256") != target["source_input_sha256"]:
            raise ValueError(f"{row_id}: prediction refers to different prompt")
        native_answers = prediction.get("answers")
        if not isinstance(native_answers, dict) or set(native_answers) != set(
            prompt_by_id[row_id]["questions"]
        ):
            raise ValueError(f"{row_id}: incomplete native answers")
        semantic, valid = _interpret(native_answers["decision"], target)
        correct = int(valid and semantic == target["semantic_gold"])
        if not valid:
            invalid[target["language"]] += 1
        row = {
            "base_id": target["base_id"],
            "language": target["language"],
            "variant": target["variant"],
            "task_type": target["task_type"],
            "semantic_prediction": semantic,
            "valid": valid,
            "correct": correct,
        }
        rows[(target["base_id"], target["language"], target["variant"])] = row
        per_language[target["language"]].append(correct)
        per_language_type[(target["language"], target["task_type"])].append(correct)
        per_base_language[(target["base_id"], target["language"])].append(correct)
    base_ids = sorted({row["base_id"] for row in rows.values()})
    if len(base_ids) != 18:
        raise ValueError("Panel must have 18 independent base cases")
    by_language = {}
    by_type = {}
    paired = {}
    for language in manifest["counts"]["by_language"]:
        base_scores = [_mean(per_base_language[(base, language)]) for base in base_ids]
        english_scores = [_mean(per_base_language[(base, "en")]) for base in base_ids]
        variant_pairs = [
            (rows[row], rows[(row[0], "en", row[2])])
            for row in rows
            if row[1] == language
        ]
        by_language[language] = {
            "prompts": len(per_language[language]),
            "item_accuracy": _mean(per_language[language]),
            "base_macro_accuracy": _mean(base_scores),
            "invalid": invalid[language],
        }
        paired[language] = {
            "independent_base_cases": len(base_ids),
            "base_macro_accuracy_delta_vs_en": _mean(
                x - y for x, y in zip(base_scores, english_scores)
            ),
            "semantic_prediction_agreement_vs_en": _mean(
                row["semantic_prediction"] == en["semantic_prediction"]
                and row["valid"]
                and en["valid"]
                for row, en in variant_pairs
            ),
            "both_correct_prompt_fraction": _mean(
                row["correct"] and en["correct"] for row, en in variant_pairs
            ),
        }
        by_type[language] = {
            task_type: {"prompts": len(values), "accuracy": _mean(values)}
            for (lang, task_type), values in per_language_type.items()
            if lang == language
        }
    order = {}
    for language in manifest["counts"]["by_language"]:
        order[language] = {}
        for task_type, ids in (
            ("choice", [case for case in base_ids if case.startswith("choice:")]),
            ("noul", [case for case in base_ids if case.startswith("noul:")]),
        ):
            changed = sum(
                rows[(case, language, "base")]["semantic_prediction"]
                != rows[
                    (
                        case,
                        language,
                        "order_label" if task_type == "choice" else "label_style",
                    )
                ]["semantic_prediction"]
                for case in ids
            )
            order[language][task_type] = {
                "base_cases": len(ids),
                "semantic_prediction_flip_cases": changed,
            }
    return {
        "schema_version": "decision2-multilingual-paired-dev-score/2",
        "scope": "self-authored multilingual DEV diagnostic; not final or SOTA evidence",
        "prompts_sha256": sha256(prompt_path),
        "targets_sha256": sha256(target_path),
        "panel_manifest_sha256": sha256(manifest_path),
        "predictions_sha256": sha256(predictions),
        "model": identity,
        "independence_unit": manifest["independence_unit"],
        "by_language": by_language,
        "by_type": by_type,
        "paired_vs_en": paired,
        "perturbation_robustness": order,
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
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model": report["model"],
                "by_language": report["by_language"],
                "report_sha256": sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
