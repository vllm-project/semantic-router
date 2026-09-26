"""Build a paired typed-decision DEV panel from pinned XNLI and PAWS-X validation.

The builder copies no upstream example text into source control. The prompt
JSONL is a local development artifact; only counts/hashes should enter public
research notes. It never reads a sealed final panel.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from inference.run import digest
from multilingual.audit import sha256
from multilingual.pilot import write_jsonl

VERSION = "decision2-multilingual-parallel-dev/1"
XNLI_REPO = "facebook/xnli"
XNLI_REVISION = "b8dd5d7af51114dbda02c0e3f6133f332186418e"
PAWSX_REPO = "google-research-datasets/paws-x"
PAWSX_REVISION = "4cd8187c404bda33cb1f62b49b001115862acf37"
XNLI_LANGUAGES = ("en", "ar", "de", "es", "fr", "zh")
PAWSX_LANGUAGES = ("en", "de", "es", "fr", "ja", "zh")
SALT = "decision2-parallel-dev-20260927-v1"

NLI_WORDS = {
    "en": (
        "Premise",
        "Hypothesis",
        "What is the logical relation of the hypothesis to the premise?",
        (
            "Entailed by the premise",
            "Neither entailed nor contradicted",
            "Contradicts the premise",
        ),
    ),
    "ar": (
        "المقدمة",
        "الفرضية",
        "ما العلاقة المنطقية بين الفرضية والمقدمة؟",
        ("تستنتج بالضرورة من المقدمة", "لا تستنتج ولا تتناقض", "تتناقض مع المقدمة"),
    ),
    "de": (
        "Prämisse",
        "Hypothese",
        "In welcher logischen Beziehung steht die Hypothese zur Prämisse?",
        (
            "Folgt aus der Prämisse",
            "Weder gefolgert noch widersprochen",
            "Widerspricht der Prämisse",
        ),
    ),
    "es": (
        "Premisa",
        "Hipótesis",
        "¿Qué relación lógica tiene la hipótesis con la premisa?",
        (
            "Se deduce de la premisa",
            "Ni se deduce ni se contradice",
            "Contradice la premisa",
        ),
    ),
    "fr": (
        "Prémisse",
        "Hypothèse",
        "Quel est le rapport logique entre l'hypothèse et la prémisse ?",
        (
            "Découle de la prémisse",
            "Ni impliquée ni contredite",
            "Contredit la prémisse",
        ),
    ),
    "zh": (
        "前提",
        "假设",
        "假设与前提之间是什么逻辑关系？",
        ("可由前提必然推出", "既无法推出也不矛盾", "与前提矛盾"),
    ),
}
NLI_SEMANTICS = ("entailment", "neutral", "contradiction")

PAWSX_WORDS = {
    "en": (
        "Sentence A",
        "Sentence B",
        "Do the two sentences have the same meaning?",
        {
            "false": "No, their meanings differ",
            "true": "Yes, they express the same meaning",
        },
    ),
    "de": (
        "Satz A",
        "Satz B",
        "Haben die beiden Sätze dieselbe Bedeutung?",
        {
            "false": "Nein, ihre Bedeutungen unterscheiden sich",
            "true": "Ja, sie drücken dasselbe aus",
        },
    ),
    "es": (
        "Oración A",
        "Oración B",
        "¿Tienen las dos oraciones el mismo significado?",
        {
            "false": "No, sus significados son distintos",
            "true": "Sí, expresan lo mismo",
        },
    ),
    "fr": (
        "Phrase A",
        "Phrase B",
        "Les deux phrases ont-elles le même sens ?",
        {
            "false": "Non, leurs sens diffèrent",
            "true": "Oui, elles expriment la même chose",
        },
    ),
    "ja": (
        "文 A",
        "文 B",
        "二つの文の意味は同じですか。",
        {"false": "いいえ、意味が異なります", "true": "はい、同じ意味です"},
    ),
    "zh": (
        "句子 A",
        "句子 B",
        "这两个句子的意思相同吗？",
        {"false": "否，意思不同", "true": "是，表达相同的意思"},
    ),
}


def _read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _select(candidates: list[int], *, corpus: str, label: int, n: int) -> list[int]:
    if len(candidates) < n:
        raise ValueError("Not enough validation examples for deterministic stratum")
    return sorted(
        candidates,
        key=lambda item: hashlib.sha256(
            f"{SALT}:{corpus}:{label}:{item}".encode()
        ).digest(),
    )[:n]


def _source_row_digest(row: dict) -> str:
    return digest(row)


def build(xnli_root: Path, pawsx_root: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    xnli_file = xnli_root / "all_languages/validation-00000-of-00001.parquet"
    paws_files = {
        language: pawsx_root / language / "validation-00000-of-00001.parquet"
        for language in PAWSX_LANGUAGES
    }
    xnli = _read_parquet(xnli_file)
    if len(xnli) != 2490 or Counter(row["label"] for row in xnli) != {
        0: 830,
        1: 830,
        2: 830,
    }:
        raise ValueError("Pinned XNLI validation roster differs")
    for row in xnli:
        if (
            not isinstance(row["premise"], dict)
            or len(row["hypothesis"]["language"]) != 15
            or len(set(row["hypothesis"]["language"])) != 15
            or set(row["hypothesis"]["language"]) != set(row["premise"])
        ):
            raise ValueError("XNLI row is not a one-to-one parallel set")
    paws = {
        language: {row["id"]: row for row in _read_parquet(path)}
        for language, path in paws_files.items()
    }
    if any(len(rows) != 2000 or set(rows) != set(paws["en"]) for rows in paws.values()):
        raise ValueError("PAWS-X validation IDs differ across languages")
    disagreement = []
    for row_id in sorted(paws["en"]):
        labels = {
            language: paws[language][row_id]["label"] for language in PAWSX_LANGUAGES
        }
        if len(set(labels.values())) != 1:
            disagreement.append({"id": row_id, "labels": labels})
    if len(disagreement) != 59:
        raise ValueError("PAWS-X cross-language label-mismatch count changed")
    safe_ids = sorted(set(paws["en"]) - {row["id"] for row in disagreement})
    xnli_selected = sorted(
        index
        for label in range(3)
        for index in _select(
            [i for i, row in enumerate(xnli) if row["label"] == label],
            corpus="xnli",
            label=label,
            n=20,
        )
    )
    paws_selected = sorted(
        row_id
        for label in range(2)
        for row_id in _select(
            [item for item in safe_ids if paws["en"][item]["label"] == label],
            corpus="pawsx",
            label=label,
            n=20,
        )
    )
    prompts, targets = [], []
    for index in xnli_selected:
        row = xnli[index]
        hypothesis = dict(
            zip(row["hypothesis"]["language"], row["hypothesis"]["translation"])
        )
        label = int(row["label"])
        order = tuple((position + index % 3) % 3 for position in range(3))
        for language in XNLI_LANGUAGES:
            premise, hypothesis_label, instruction, options = NLI_WORDS[language]
            state = (
                f"{premise}: {row['premise'][language]}\n"
                f"{hypothesis_label}: {hypothesis[language]}"
            )
            if not row["premise"][language].strip() or not hypothesis[language].strip():
                raise ValueError("XNLI empty translated pair")
            criteria = {
                chr(97 + position): options[semantic]
                for position, semantic in enumerate(order)
            }
            question = {
                "type": "choice",
                "instructions": instruction,
                "criteria": criteria,
            }
            prompt = {
                "id": f"parallel-xnli-{index}-{language}",
                "state": state,
                "questions": {"decision": question},
            }
            gold = chr(97 + order.index(label))
            target = {
                "id": prompt["id"],
                "base_id": f"xnli:{index}",
                "corpus": "xnli",
                "language": language,
                "task_type": "choice",
                "gold": gold,
                "semantic_gold": NLI_SEMANTICS[label],
                "semantic_by_label": {
                    chr(97 + position): NLI_SEMANTICS[semantic]
                    for position, semantic in enumerate(order)
                },
                "source_row_sha256": _source_row_digest(row),
                "source_input_sha256": digest(
                    {"state": state, "questions": prompt["questions"]}
                ),
            }
            prompts.append(prompt)
            targets.append(target)
    for row_id in paws_selected:
        label = int(paws["en"][row_id]["label"])
        for language in PAWSX_LANGUAGES:
            row = paws[language][row_id]
            if (
                row["label"] != label
                or not row["sentence1"].strip()
                or not row["sentence2"].strip()
            ):
                raise ValueError("PAWS-X selected translation or label invalid")
            first, second, instruction, options = PAWSX_WORDS[language]
            state = f"{first}: {row['sentence1']}\n{second}: {row['sentence2']}"
            question = {
                "type": "noul",
                "instructions": instruction,
                "criteria": options,
            }
            prompt = {
                "id": f"parallel-pawsx-{row_id}-{language}",
                "state": state,
                "questions": {"decision": question},
            }
            target = {
                "id": prompt["id"],
                "base_id": f"pawsx:{row_id}",
                "corpus": "pawsx",
                "language": language,
                "task_type": "noul",
                "gold": bool(label),
                "semantic_gold": bool(label),
                "source_row_sha256": _source_row_digest(row),
                "source_input_sha256": digest(
                    {"state": state, "questions": prompt["questions"]}
                ),
            }
            prompts.append(prompt)
            targets.append(target)
    if (
        len(prompts) != 600
        or len({row["id"] for row in prompts}) != 600
        or [row["id"] for row in prompts] != [row["id"] for row in targets]
        or any("gold" in prompt for prompt in prompts)
    ):
        raise ValueError("Parallel DEV roster is incomplete or leaks targets")
    output_dir.mkdir(parents=True)
    prompt_path, target_path = (
        output_dir / "prompts.jsonl",
        output_dir / "targets.jsonl",
    )
    write_jsonl(prompt_path, prompts)
    write_jsonl(target_path, targets)
    quarantine_path = output_dir / "pawsx-label-mismatch-quarantine.json"
    quarantine_path.write_text(
        json.dumps(
            {
                "reason": "same upstream ID has differing labels across languages",
                "count": len(disagreement),
                "rows": disagreement,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": VERSION,
        "phase": "dev",
        "scope": "public, parallel translated validation diagnostic; no sealed final or training",
        "independence_unit": "100 source validation IDs; each translated language is paired to English",
        "source": {
            "xnli": {
                "repo": XNLI_REPO,
                "revision": XNLI_REVISION,
                "files": {
                    "all_languages/validation-00000-of-00001.parquet": sha256(xnli_file)
                },
                "rows": len(xnli),
                "selected_by_label": {str(label): 20 for label in range(3)},
            },
            "pawsx": {
                "repo": PAWSX_REPO,
                "revision": PAWSX_REVISION,
                "files": {
                    str(path.relative_to(pawsx_root)): sha256(path)
                    for path in paws_files.values()
                },
                "rows_per_language": 2000,
                "label_mismatch_quarantined": 59,
                "selected_by_label": {str(label): 20 for label in range(2)},
            },
        },
        "selection": "SHA256 rank within source validation label stratum, fixed salt; no model output used",
        "selection_salt": SALT,
        "translation_check": "upstream human-translated validation; one-to-one XNLI language map; PAWS-X matched ID and equal label; no independent manual reannotation",
        "builder_sha256": sha256(Path(__file__)),
        "files": {
            "prompts.jsonl": sha256(prompt_path),
            "targets.jsonl": sha256(target_path),
            "pawsx-label-mismatch-quarantine.json": sha256(quarantine_path),
        },
        "counts": {
            "base_cases": 100,
            "prompts": 600,
            "by_corpus": {"xnli": 360, "pawsx": 240},
            "by_language": dict(Counter(row["language"] for row in targets)),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xnli-root", type=Path, required=True)
    parser.add_argument("--pawsx-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.xnli_root, args.pawsx_root, args.output_dir)
    print(
        json.dumps(
            {
                "counts": result["counts"],
                "files": result["files"],
                "manifest_sha256": sha256(args.output_dir / "manifest.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
