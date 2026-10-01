"""Answer extraction and scoring regressions for the standalone MCQ judge."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.training.model_eval import mcq_judge


class ExtractLetterTest(unittest.TestCase):
    def test_rejects_word_prefixes(self):
        for text in (
            "Answer: APPLE",
            "Answer: AB",
            "Answer: A1",
            "Answer: A_name",
            "Answer: (APPLE)",
            "答案: BANANA",
        ):
            with self.subTest(text=text):
                self.assertIsNone(mcq_judge.extract_letter(text))

    def test_accepts_explicit_answer_formats(self):
        for prefix in ("Answer", "answer", "ANSWER", "答案"):
            for colon in (":", "\uff1a"):
                for letter in "ABCDEFGHIJ":
                    text = f"{prefix} {colon} ({letter})."
                    with self.subTest(text=text):
                        self.assertEqual(mcq_judge.extract_letter(text), letter)

    def test_accepts_adjacent_chinese_explanations(self):
        cases = (
            ("答案\uff1aA因为前提成立", "A"),
            ("Answer: A是正确的", "A"),
            ("Answer: A\n答案\uff1aB因为前提成立", "B"),
            ("答案\uff1aB因为前提成立\nAnswer: APPLE", "B"),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(mcq_judge.extract_letter(text), expected)

    def test_uses_last_valid_explicit_answer(self):
        cases = (
            ("Answer: A\nAnswer: C", "C"),
            ("Answer: B\nAnswer: APPLE", "B"),
            ("Answer: B\nAnswer: (APPLE)", "B"),
            ("Answer: A\n答案\uff1a B", "B"),
            ("答案\uff1a B\nAnswer: AB", "B"),
            ("Answer: B\nC", "B"),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(mcq_judge.extract_letter(text), expected)

    def test_falls_back_to_last_standalone_line(self):
        cases = (
            ("Reasoning\n A \n (C) ", "C"),
            ("Answer: APPLE\n(B)", "B"),
            ("A\nAnswer: BANANA", "A"),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(mcq_judge.extract_letter(text), expected)

    def test_returns_no_answer_without_a_valid_letter(self):
        for text in (
            None,
            "",
            "No option selected.",
            "Answer: K",
            "Answer: a",
            "Answer: \uff22",
        ):
            with self.subTest(text=text):
                self.assertIsNone(mcq_judge.extract_letter(text))


class JudgeCliTest(unittest.TestCase):
    def test_scores_only_valid_answers(self):
        cases = (
            ("word-prefix", "A", "Answer: APPLE"),
            ("fullwidth-colon", "B", "答案\uff1aB"),
            ("last-valid", "B", "答案\uff1aB因为前提成立\nAnswer: APPLE"),
            ("standalone", "C", "Reasoning\n(C)"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks.jsonl"
            records = root / "records.jsonl"
            judged = root / "judged.jsonl"
            summary = root / "summary.json"
            analysis = root / "analysis.json"
            tasks.write_text(
                "".join(
                    json.dumps({"task_id": task_id, "answer": answer}) + "\n"
                    for task_id, answer, _ in cases
                ),
                encoding="utf-8",
            )
            records.write_text(
                "".join(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "completion": completion,
                            "category": "example",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                    for task_id, _, completion in cases
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    mcq_judge.__file__,
                    "--tasks",
                    str(tasks),
                    "--records",
                    str(records),
                    "--out",
                    str(judged),
                    "--summary",
                    str(summary),
                    "--analysis",
                    str(analysis),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            rows = [
                json.loads(line)
                for line in judged.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [
                    (row["answer_extracted"], row["is_correct"], row["no_answer"])
                    for row in rows
                ],
                [
                    (None, False, True),
                    ("B", True, False),
                    ("B", True, False),
                    ("C", True, False),
                ],
            )
            stats = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(stats["n_total"], 4)
            self.assertEqual(stats["accuracy"], 75.0)
            self.assertEqual(stats["no_answer_count"], 1)
            self.assertEqual(stats["by_category"]["example"]["accuracy"], 75.0)
            config_analysis = json.loads(analysis.read_text(encoding="utf-8"))
            self.assertEqual(config_analysis["overall_accuracy"], 0.75)
            self.assertEqual(config_analysis["category_accuracy"], {"example": 0.75})


if __name__ == "__main__":
    unittest.main()
