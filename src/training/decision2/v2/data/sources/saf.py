"""SAF short-answer feedback (en, de) projection: A6h verdict Score rows (L=3).

The expert ``verification_feedback`` verdict is the label; any value other
than the three known verdicts fails the build. ``answer_feedback`` and
``score`` never reach the state.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

from v2.data.sources import klue
from v2.data.sources.common import make_row, score_options
from v2.data.textnorm import normalize

FILE = Path("data/train.jsonl")
FAMILY = "saf"
VERDICTS = ("Incorrect", "Partially correct", "Correct")
SOURCES = {"en": "saf_en_train", "de": "saf_de_train"}
TEXT = {
    "en": {
        "state": "Question: {question}\nReference answer: {reference}\n"
        "Student answer: {answer}",
        "instructions": "Grade the student answer against the reference answer. "
        "Wording may differ; what matters is whether the answer contains what the "
        "reference answer requires.",
        "levels": (
            "Incorrect: the answer is wrong or misses the point of the question.",
            "Partially correct: the answer contains some of what the reference answer "
            "requires but is incomplete or partly wrong.",
            "Correct: the answer contains what the reference answer requires, with no "
            "significant errors.",
        ),
    },
    "de": {
        "state": "Frage: {question}\nMusterlösung: {reference}\n"
        "Antwort des Lernenden: {answer}",
        "instructions": "Bewerten Sie die Antwort des Lernenden anhand der "
        "Musterlösung. Der Wortlaut darf abweichen; entscheidend ist, ob die Antwort "
        "die Inhalte enthält, die die Musterlösung verlangt.",
        "levels": (
            "Falsch: Die Antwort ist unzutreffend oder geht an der Frage vorbei.",
            "Teilweise richtig: Die Antwort enthält einen Teil dessen, was die "
            "Musterlösung verlangt, ist aber unvollständig oder teilweise falsch.",
            "Richtig: Die Antwort enthält, was die Musterlösung verlangt, und hat "
            "keine wesentlichen Fehler.",
        ),
    },
}


def verdict_level(value: Any, where: str) -> int:
    if value not in VERDICTS:
        raise ValueError(f"{where}: unknown verification_feedback {value!r}")
    return VERDICTS.index(value)


def saf(root: Path, language: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    texts, source = TEXT[language], SOURCES[language]
    rows, verdicts = [], collections.Counter()
    for where, record in klue.read_jsonl(root / FILE):
        level = verdict_level(record.get("verification_feedback"), where)
        verdicts[VERDICTS[level]] += 1
        question = klue.text(record, "question", where)
        answer = klue.text(record, "provided_answer", where)
        rows.append(
            make_row(
                arm=klue.A6H,
                source=source,
                family=FAMILY,
                task_type="score",
                language=language,
                group_key=f"{normalize(question)}|{normalize(answer)}",
                local_id=klue.ident(record, "id", where),
                state=texts["state"].format(
                    question=question,
                    reference=klue.text(record, "reference_answer", where),
                    answer=answer,
                ),
                instructions=texts["instructions"],
                options=score_options(texts["levels"]),
                label=level,
                render_template=f"saf_reference_grading_{language}_v1",
                audit={"verdict": VERDICTS[level], "score": record.get("score")},
            )
        )
    return rows, {
        "input": klue.input_receipt(root, FILE, len(rows)),
        "level_counts": [len(VERDICTS)],
        "verdict_levels": {verdict: level for level, verdict in enumerate(VERDICTS)},
        "train_class_histogram": {verdict: verdicts[verdict] for verdict in VERDICTS},
    }
