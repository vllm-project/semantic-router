"""Score human-translated XNLI/PAWS-X DEV with English-paired units."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from math import comb
from pathlib import Path

from multilingual.audit import sha256
from multilingual.parallel import VERSION
from multilingual.score import _interpret, _load_jsonl


def _exact_mcnemar_p(b: int, c: int) -> float:
    discordant = b + c
    if not discordant:
        return 1.0
    return min(
        1.0, 2 * sum(comb(discordant, i) for i in range(min(b, c) + 1)) / 2**discordant
    )


def score(panel: Path, predictions: Path) -> dict:
    manifest_path = panel / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VERSION:
        raise ValueError("Unknown multilingual parallel panel")
    prompt_path, target_path = panel / "prompts.jsonl", panel / "targets.jsonl"
    for name, path in (("prompts.jsonl", prompt_path), ("targets.jsonl", target_path)):
        if sha256(path) != manifest["files"][name]:
            raise ValueError(f"{name}: frozen panel hash mismatch")
    prompts, targets, predictions_rows = (
        _load_jsonl(path) for path in (prompt_path, target_path, predictions)
    )
    if len(prompts) != 600 or len(targets) != 600 or len(predictions_rows) != 600:
        raise ValueError("Parallel pilot requires exactly 600 predictions")
    prompt_by_id = {row["id"]: row for row in prompts}
    target_by_id = {row["id"]: row for row in targets}
    prediction_by_id = {row["id"]: row for row in predictions_rows}
    if (
        any(
            len(mapping) != 600
            for mapping in (prompt_by_id, target_by_id, prediction_by_id)
        )
        or set(prompt_by_id) != set(target_by_id)
        or set(prompt_by_id) != set(prediction_by_id)
    ):
        raise ValueError("Prompt/target/prediction IDs differ")
    identity_keys = (
        "backend",
        "model_id",
        "model_revision",
        "model_sha256",
        "calibration_sha256",
    )
    identity = {key: predictions_rows[0].get(key) for key in identity_keys}
    if any(
        {key: row.get(key) for key in identity_keys} != identity
        for row in predictions_rows
    ):
        raise ValueError("Prediction identity varies across rows")
    results = {}
    invalid = Counter()
    by_gold = defaultdict(list)
    for target in targets:
        prompt = prompt_by_id[target["id"]]
        predicted = prediction_by_id[target["id"]]
        if predicted.get("source_input_sha256") != target["source_input_sha256"]:
            raise ValueError("Prediction input hash differs from target")
        answers = predicted.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(prompt["questions"]):
            raise ValueError("Incomplete native answers")
        semantic, valid = _interpret(answers["decision"], target)
        correct = int(valid and semantic == target["semantic_gold"])
        key = (target["corpus"], target["base_id"], target["language"])
        if key in results:
            raise ValueError("Duplicated corpus/base/language")
        results[key] = {
            "correct": correct,
            "valid": valid,
            "semantic": semantic,
            "gold": target["semantic_gold"],
        }
        if not valid:
            invalid[(target["corpus"], target["language"])] += 1
        by_gold[
            (target["corpus"], target["language"], str(target["semantic_gold"]))
        ].append(correct)
    out = {}
    for corpus, expected, languages in (
        ("xnli", 60, ("en", "ar", "de", "es", "fr", "zh")),
        ("pawsx", 40, ("en", "de", "es", "fr", "ja", "zh")),
    ):
        bases = sorted(
            {base for name, base, lang in results if name == corpus and lang == "en"}
        )
        if len(bases) != expected:
            raise ValueError(f"{corpus}: expected {expected} base IDs")
        corpus_result = {}
        for language in languages:
            pairs = [
                (results[(corpus, base, "en")], results[(corpus, base, language)])
                for base in bases
            ]
            en_to_wrong = sum(
                en["correct"] and not translated["correct"] for en, translated in pairs
            )
            wrong_to_correct = sum(
                not en["correct"] and translated["correct"] for en, translated in pairs
            )
            corpus_result[language] = {
                "independent_base_cases": len(bases),
                "correct": sum(translated["correct"] for _, translated in pairs),
                "accuracy": sum(translated["correct"] for _, translated in pairs)
                / len(bases),
                "invalid": invalid[(corpus, language)],
                "paired_accuracy_delta_vs_en": (wrong_to_correct - en_to_wrong)
                / len(bases),
                "en_correct_target_wrong": en_to_wrong,
                "en_wrong_target_correct": wrong_to_correct,
                "paired_exact_mcnemar_p": _exact_mcnemar_p(
                    en_to_wrong, wrong_to_correct
                ),
                "semantic_prediction_agreement_vs_en": sum(
                    en["valid"]
                    and translated["valid"]
                    and en["semantic"] == translated["semantic"]
                    for en, translated in pairs
                )
                / len(bases),
                "by_gold_label": {
                    gold: {"items": len(values), "accuracy": sum(values) / len(values)}
                    for (family, lang, gold), values in by_gold.items()
                    if family == corpus and lang == language
                },
            }
        out[corpus] = corpus_result
    return {
        "schema_version": "decision2-multilingual-parallel-dev-score/1",
        "scope": "public human-translated validation development diagnostic; no final gold",
        "manifest_sha256": sha256(manifest_path),
        "prompts_sha256": sha256(prompt_path),
        "targets_sha256": sha256(target_path),
        "predictions_sha256": sha256(predictions),
        "model": identity,
        "independence_unit": manifest["independence_unit"],
        "by_corpus_language": out,
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
                "summary": {
                    corpus: {
                        language: {
                            key: data[key]
                            for key in ("correct", "independent_base_cases", "invalid")
                        }
                        for language, data in languages.items()
                    }
                    for corpus, languages in report["by_corpus_language"].items()
                },
                "report_sha256": sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
