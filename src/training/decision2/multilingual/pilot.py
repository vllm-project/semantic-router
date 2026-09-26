"""Build a paired, gold-free multilingual typed-decision development slice.

The authored English and six translated versions of one case are one unit of
analysis. Gold lives in a separate target file; no sealed panel is read.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from inference.run import digest
from multilingual import content
from multilingual.audit import script_of, sha256

VERSION = "decision2-multilingual-paired-dev/2"
SEMANTIC_OPTIONS = ("track", "cancel", "address", "refund")
CHOICE_ORDERS = {"base": (0, 1, 2, 3), "order_label": (3, 2, 0, 1)}
NOUL_VARIANTS = ("base", "label_style")


def digits(text: str) -> list[str]:
    return re.findall(r"[0-9]+", text)


def _check_translations() -> dict[str, int]:
    languages = set(content.LANGUAGES)
    dictionaries = (
        content.CHOICE_INSTRUCTION,
        content.CHOICE_OPTIONS,
        content.NOUL_LEAD,
        content.NOUL_OPTIONS,
        content.NOUL_ALTERNATE_OPTIONS,
        content.SCORE_STATE,
        content.SCORE_INSTRUCTION,
        content.SCORE_CRITERIA,
    )
    if any(set(mapping) != languages for mapping in dictionaries):
        raise ValueError("Translation dictionaries omit a language")
    counters = Counter()
    for _case_id, _answer, states in content.CHOICE_CASES:
        if set(states) != languages:
            raise ValueError("Choice case omits a language")
        for language, state in states.items():
            if not state.strip():
                raise ValueError("Empty Choice translation")
            if digits(state) != digits(states["en"]):
                raise ValueError("Choice numerical facts changed")
            counters["choice_numeric_checked"] += 1
            counters["choice_author_reviewed"] += 1
    for case_id, _answer, translations in content.NOUL_CASES:
        if set(translations) != languages:
            raise ValueError("Noul case omits a language")
        reference = translations["en"]
        for language, pair in translations.items():
            if len(pair) != 2 or not all(part.strip() for part in pair):
                raise ValueError("Malformed Noul translation")
            if digits(" ".join(pair)) != digits(" ".join(reference)):
                raise ValueError(
                    f"Noul numerical facts changed in {case_id}/{language}"
                )
            if case_id == "apple_count" and "Mina" not in " ".join(pair):
                raise ValueError("Protected entity Mina was translated")
            counters["noul_numeric_entity_checked"] += 1
            counters["noul_author_reviewed"] += 1
    for language in content.LANGUAGES:
        if (
            len(content.CHOICE_OPTIONS[language]) != 4
            or len(content.SCORE_CRITERIA[language]) != 4
        ):
            raise ValueError("Choice or Score rubric length changed")
        rubric = " ".join(content.SCORE_CRITERIA[language])
        if digits(rubric) != digits(" ".join(content.SCORE_CRITERIA["en"])):
            raise ValueError("Score boundaries changed")
        for failed in range(6):
            state = content.SCORE_STATE[language].format(n=failed)
            if digits(state) != digits(content.SCORE_STATE["en"].format(n=failed)):
                raise ValueError("Score count changed")
            counters["score_numeric_rubric_checked"] += 1
        if language in ("zh", "ja", "ar"):
            expected = {"zh": "han", "ja": "kana", "ar": "arabic"}[language]
            all_text = " ".join(
                [
                    *(states[language] for _, _, states in content.CHOICE_CASES),
                    *(
                        " ".join(states[language])
                        for _, _, states in content.NOUL_CASES
                    ),
                    content.SCORE_STATE[language],
                    content.CHOICE_INSTRUCTION[language],
                ]
            )
            if sum(script_of(char) == expected for char in all_text) < 20:
                raise ValueError(f"{language}: target script missing")
            counters["nonlatin_script_checked"] += 1
    return dict(counters)


def make_rows() -> tuple[list[dict], list[dict], dict]:
    checks = _check_translations()
    prompts, targets = [], []
    for case_id, semantic_index, states in content.CHOICE_CASES:
        for language in content.LANGUAGES:
            for variant, ordering in CHOICE_ORDERS.items():
                criteria = {
                    chr(97 + position): content.CHOICE_OPTIONS[language][index]
                    for position, index in enumerate(ordering)
                }
                gold = chr(97 + ordering.index(semantic_index))
                question = {
                    "type": "choice",
                    "instructions": content.CHOICE_INSTRUCTION[language],
                    "criteria": criteria,
                }
                prompts.append(
                    {
                        "id": f"mldev-choice-{case_id}-{language}-{variant}",
                        "state": states[language],
                        "questions": {"decision": question},
                    }
                )
                targets.append(
                    {
                        "id": prompts[-1]["id"],
                        "base_id": f"choice:{case_id}",
                        "language": language,
                        "variant": variant,
                        "task_type": "choice",
                        "gold": gold,
                        "semantic_gold": SEMANTIC_OPTIONS[semantic_index],
                        "semantic_by_label": {
                            chr(97 + pos): SEMANTIC_OPTIONS[index]
                            for pos, index in enumerate(ordering)
                        },
                        "source_input_sha256": digest(
                            {
                                "state": prompts[-1]["state"],
                                "questions": prompts[-1]["questions"],
                            }
                        ),
                    }
                )
    for case_id, gold, translations in content.NOUL_CASES:
        for language in content.LANGUAGES:
            state, question_text = translations[language]
            for variant in NOUL_VARIANTS:
                labels = (
                    content.NOUL_OPTIONS
                    if variant == "base"
                    else content.NOUL_ALTERNATE_OPTIONS
                )[language]
                question = {
                    "type": "noul",
                    "instructions": f"{content.NOUL_LEAD[language]} {question_text}",
                    "criteria": {key: labels[key] for key in ("false", "true")},
                }
                prompts.append(
                    {
                        "id": f"mldev-noul-{case_id}-{language}-{variant}",
                        "state": state,
                        "questions": {"decision": question},
                    }
                )
                targets.append(
                    {
                        "id": prompts[-1]["id"],
                        "base_id": f"noul:{case_id}",
                        "language": language,
                        "variant": variant,
                        "task_type": "noul",
                        "gold": gold,
                        "semantic_gold": gold,
                        "source_input_sha256": digest(
                            {"state": state, "questions": prompts[-1]["questions"]}
                        ),
                    }
                )
    for failed in range(6):
        level = 0 if failed == 0 else 1 if failed == 1 else 2 if failed <= 3 else 3
        for language in content.LANGUAGES:
            state = content.SCORE_STATE[language].format(n=failed)
            question = {
                "type": "score",
                "instructions": content.SCORE_INSTRUCTION[language],
                "criteria": list(content.SCORE_CRITERIA[language]),
            }
            prompts.append(
                {
                    "id": f"mldev-score-failed{failed}-{language}-base",
                    "state": state,
                    "questions": {"decision": question},
                }
            )
            targets.append(
                {
                    "id": prompts[-1]["id"],
                    "base_id": f"score:failed{failed}",
                    "language": language,
                    "variant": "base",
                    "task_type": "score",
                    "gold": level,
                    "semantic_gold": level,
                    "source_input_sha256": digest(
                        {"state": state, "questions": prompts[-1]["questions"]}
                    ),
                }
            )
    ids = [row["id"] for row in prompts]
    if (
        len(ids) != 210
        or len(ids) != len(set(ids))
        or ids != [row["id"] for row in targets]
    ):
        raise ValueError("Expected 18 semantic groups, 210 unique native prompts")
    if any("gold" in row for row in prompts):
        raise ValueError("Gold leaked into prompts")
    counts = {
        "prompts": len(prompts),
        "semantic_groups": 18,
        "languages": len(content.LANGUAGES),
        "by_type": dict(Counter(row["task_type"] for row in targets)),
        "by_language": dict(Counter(row["language"] for row in targets)),
        "translation_checks": checks,
    }
    return prompts, targets, counts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )


def build(output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    prompts, targets, counts = make_rows()
    output_dir.mkdir(parents=True)
    prompt_path = output_dir / "prompts.jsonl"
    target_path = output_dir / "targets.jsonl"
    write_jsonl(prompt_path, prompts)
    write_jsonl(target_path, targets)
    manifest = {
        "schema_version": VERSION,
        "phase": "dev",
        "scope": "self-authored public pilot; no sealed final",
        "translation_review": "Single-author semantic review of controlled facts; exact numeric/entity and target-script checks; no independent native-speaker review",
        "independence_unit": "18 English base cases; translations, Choice order/key rotation and Noul label paraphrases are paired observations",
        "content_sha256": sha256(Path(content.__file__)),
        "builder_sha256": sha256(Path(__file__)),
        "files": {
            "prompts.jsonl": sha256(prompt_path),
            "targets.jsonl": sha256(target_path),
        },
        "counts": counts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.output_dir)
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "files": report["files"],
                "manifest_sha256": sha256(args.output_dir / "manifest.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
