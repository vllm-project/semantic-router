"""Verification-need data with explicit source roles and transformation negatives."""

# ruff: noqa: PLC0415, RUF001
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from ..sequence_repair.corpus import group_split, write_corpus

MIN_SOURCE_CHARACTERS = 20
MAX_SOURCE_CHARACTERS = 1200
NATURAL_TRAIN_CAP = 250
LABELS = {"NO_FACT_CHECK_NEEDED": 0, "FACT_CHECK_NEEDED": 1}
LANGUAGE_CODES = {
    "eng": "en",
    "zho": "zh",
    "spa": "es",
    "fra": "fr",
    "deu": "de",
    "jpn": "ja",
}
TRANSFORMS = {
    "en": [
        "Translate the following question into French; do not answer or verify it:\n{}",
        "Write a fictional dialogue where a character asks this question; do not supply factual answers:\n{}",
        "Proofread the wording of this question, keeping its meaning and leaving it unanswered:\n{}",
        "Restate this question in simpler words. This is a wording task; do not investigate its answer:\n{}",
        "Use the question below as a line spoken in an imaginary play. Produce only the fictional scene:\n{}",
        "Convert the following question to uppercase. Output only the converted text:\n{}",
    ],
    "zh": [
        "请把以下问题翻译成英文，不要回答或核实它：\n{}",
        "创作一个虚构对话，让角色提出以下问题，不要提供事实答案：\n{}",
        "只校对以下问题的措辞，不要回答问题：\n{}",
        "请把这个问题改写得更通俗；这只是文字改写，不需要查找答案：\n{}",
        "把以下问题当作虚构剧本中的一句台词，只输出想象的场景：\n{}",
        "把下列问题里的换行改为空格，保持原文内容，不要解答：\n{}",
    ],
    "es": [
        "Traduce esta pregunta al inglés; no la respondas ni la verifiques:\n{}",
        "Escribe un diálogo ficticio en el que un personaje haga esta pregunta, sin responderla:\n{}",
        "Corrige solo la redacción de esta pregunta; no busques la respuesta:\n{}",
        "Reformula esta pregunta con palabras más sencillas, sin investigar su respuesta:\n{}",
        "Usa esta pregunta como una frase de una obra imaginaria; entrega solo la escena ficticia:\n{}",
        "Convierte la siguiente pregunta a mayúsculas y entrega únicamente el texto convertido:\n{}",
    ],
    "fr": [
        "Traduis cette question en anglais sans y répondre ni la vérifier :\n{}",
        "Écris un dialogue fictif où un personnage pose cette question, sans donner de réponse factuelle :\n{}",
        "Corrige uniquement la formulation de cette question, sans y répondre :\n{}",
        "Reformule cette question avec des mots plus simples sans rechercher sa réponse :\n{}",
        "Utilise cette question comme une réplique dans une pièce imaginaire ; fournis seulement la scène fictive :\n{}",
        "Mets cette question en majuscules et renvoie uniquement le texte transformé :\n{}",
    ],
    "de": [
        "Übersetze diese Frage ins Englische, ohne sie zu beantworten oder zu überprüfen:\n{}",  # codespell:ignore oder
        "Schreibe einen fiktiven Dialog, in dem eine Figur diese Frage stellt, ohne sachliche Antworten zu geben:\n{}",
        "Korrigiere nur die Formulierung dieser Frage, ohne sie zu beantworten:\n{}",
        "Formuliere diese Frage einfacher, ohne nach der Antwort zu recherchieren:\n{}",
        "Verwende diese Frage als Dialogzeile in einem erfundenen Theaterstück; gib nur die fiktive Szene aus:\n{}",  # codespell:ignore als,szene
        "Wandle die folgende Frage in Großbuchstaben um und gib nur den umgewandelten Text aus:\n{}",
    ],
    "ja": [
        "次の質問を英語に翻訳してください。質問に回答したり事実確認したりしないでください：\n{}",
        "登場人物が次の質問をする架空の会話を書いてください。事実に基づく回答は書かないでください：\n{}",
        "次の質問の文章表現だけを校正し、質問には回答しないでください：\n{}",
        "この質問を簡単な表現に言い換えてください。答えを調べる必要はありません：\n{}",
        "次の質問を架空の劇の台詞として使い、想像上の場面だけを書いてください：\n{}",
        "次の質問の改行を空白に置き換え、質問には答えず変換した文章だけを返してください：\n{}",
    ],
}
NEGATIVE_PATTERNS = {
    "eng": r"^(?:please\s+)?(?:translate|rewrite|proofread|summari[sz]e|paraphrase|compose a poem|write a poem|write a fictional|write a short story)",
    "zho": r"^(?:请|請)?(?:翻译|翻譯|改写|改寫|给以下.*译文|把以下.*现代汉字|总结以下|总结下面|寫一首|写一首|创作一个虚构)",
    "spa": r"^(?:por favor[, ]*)?(?:traduce|reescribe|corrige|resume|escribe un poema|escribe un cuento)",
    "fra": r"^(?:s'il te plaît[, ]*)?(?:traduis|réécris|reformule|corrige|résume|écris un poème|raconte une blague)",
    "deu": r"^(?:bitte\s+)?(?:übersetze|formuliere.*um|korrigiere|fasse.*zusammen|schreibe ein gedicht|erzähle.*witz)",
    "jpn": r"^(?:次の文章を読んで、その内容に適したタイトル|次の.*(?:翻訳|要約)|この.*(?:言い換え|校正)|文中の主な言葉を見つけて|文章から主要な単語を特定し)",
}


def read_rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()
    ]


def main():
    from pyarrow import parquet

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-corpus", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits = {split: [] for split in ["train", "dev", "test"]}
    for split in ("train", "dev", "test"):
        counts = Counter()
        for row in read_rows(args.domain_corpus / f"{split}.jsonl"):
            if row["source"] != "global-mmlu-domain" or row["label"] in {
                "math",
                "computer science",
                "engineering",
                "philosophy",
                "physics",
            }:
                continue
            lang = row["language"]
            if counts[lang] >= {"train": 250, "dev": 40, "test": 60}[split]:
                continue
            counts[lang] += 1
            base = {
                **row,
                "label": "FACT_CHECK_NEEDED",
                "source": "global-mmlu-verification-supervision",
                "id": "fact-positive-" + row["id"],
            }
            splits[split].append(base)
            families = {"train": [0, 1, 2], "dev": [3], "test": [4, 5]}[split]
            family = families[counts[lang] % len(families)]
            splits[split].append(
                {
                    **base,
                    "id": "fact-transform-" + row["id"],
                    "text": TRANSFORMS[lang][family].format(row["text"]),
                    "label": "NO_FACT_CHECK_NEEDED",
                    "source": "authored-transformation",
                    "template_family": f"fact-transform-{split}-{family}",
                }
            )
    for row in read_rows(args.annotations):
        label = row["task_labels"]["factcheck"]
        if label is None:
            continue
        split = group_split(row["group_id"])
        splits[split].append(
            {
                **row,
                "id": "fact-" + row["id"],
                "label": label,
                "split": split,
                "length_bucket": "short",
            }
        )
    rows = parquet.read_table(
        args.sources / "CohereLabs--aya_dataset/data/train-00000-of-00001.parquet",
        columns=["inputs", "language_code", "user_id"],
    ).to_pylist()
    counts = Counter()
    for row in sorted(
        rows, key=lambda item: hashlib.sha256(item["inputs"].encode()).hexdigest()
    ):
        language = row["language_code"]
        if (
            language not in NEGATIVE_PATTERNS
            or not row["user_id"]
            or not MIN_SOURCE_CHARACTERS <= len(row["inputs"]) <= MAX_SOURCE_CHARACTERS
        ):
            continue
        group = "aya-author-" + hashlib.sha256(row["user_id"].encode()).hexdigest()
        if group_split(group) != "train" or counts[language] >= NATURAL_TRAIN_CAP:
            continue
        if not re.search(
            NEGATIVE_PATTERNS[language], row["inputs"].strip(), re.IGNORECASE
        ):
            continue
        counts[language] += 1
        splits["train"].append(
            {
                "id": "aya-fact-negative-"
                + hashlib.sha256(row["inputs"].encode()).hexdigest(),
                "text": row["inputs"],
                "label": "NO_FACT_CHECK_NEEDED",
                "group_id": group,
                "language": LANGUAGE_CODES[language],
                "source": "aya-rule-supervision",
                "split": "train",
                "length_bucket": "short",
            }
        )
    contract = {
        "label2id": LABELS,
        "id2label": {index: label for label, index in LABELS.items()},
        "problem_type": "single_label_classification",
    }
    provenance = {
        "aya_revision": "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
        "licenses": ["Apache-2.0"],
        "domain_parent_manifest_sha256": hashlib.sha256(
            (args.domain_corpus / "manifest.json").read_bytes()
        ).hexdigest(),
        "task": "Need for external factual verification, not truth of an answer",
        "annotation_roles": {
            "global-mmlu-verification-supervision": "Academic factual-knowledge questions, excluding computational/logic-heavy subjects; weak task supervision",
            "authored-transformation": "Explicit translate/edit/fiction task, question is quoted material and must not be answered",
            "aya-rule-supervision": "Train-only high-precision lexical task selection, not human fact-check labels",
            "aya-original-human": "Human-authored source prompts, task labels independently assistant-reviewed before model predictions",
        },
    }
    print(
        json.dumps(write_corpus(args.output, splits, contract, provenance)), flush=True
    )


if __name__ == "__main__":
    main()
