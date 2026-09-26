import json
from pathlib import Path
import tempfile
import unittest

from inference.run import digest
from multilingual.audit import sha256
from multilingual.parallel import VERSION, _select
from multilingual.parallel_score import _exact_mcnemar_p, score


class ParallelScoreTest(unittest.TestCase):
    def test_selection_is_deterministic(self):
        self.assertEqual(
            _select(list(range(100)), corpus="xnli", label=1, n=10),
            _select(list(reversed(range(100))), corpus="xnli", label=1, n=10),
        )

    def test_grouped_score_does_not_count_languages_as_independent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            panel = root / "panel"
            panel.mkdir()
            prompts, targets, predictions = [], [], []
            for corpus, n, languages in (
                ("xnli", 60, ("en", "ar", "de", "es", "fr", "zh")),
                ("pawsx", 40, ("en", "de", "es", "fr", "ja", "zh")),
            ):
                for base in range(n):
                    for language in languages:
                        row_id = f"{corpus}-{base}-{language}"
                        qtype = "choice" if corpus == "xnli" else "noul"
                        question = {"type": qtype}
                        prompt = {
                            "id": row_id,
                            "state": "fixture",
                            "questions": {"decision": question},
                        }
                        target = {
                            "id": row_id,
                            "base_id": f"{corpus}:{base}",
                            "corpus": corpus,
                            "language": language,
                            "task_type": qtype,
                            "semantic_gold": "entailment" if corpus == "xnli" else True,
                            "semantic_by_label": {"a": "entailment", "b": "neutral"},
                            "source_input_sha256": digest(
                                {"state": "fixture", "questions": prompt["questions"]}
                            ),
                        }
                        wrong = corpus == "xnli" and base == 0 and language == "ar"
                        answer = (
                            {"type": "choice", "choice": "b" if wrong else "a"}
                            if corpus == "xnli"
                            else {"type": "noul", "noul": 0.9}
                        )
                        prediction = {
                            "id": row_id,
                            "answers": {"decision": answer},
                            "source_input_sha256": target["source_input_sha256"],
                            "backend": "fixture",
                            "model_id": "fixture",
                            "model_revision": "fixture",
                        }
                        prompts.append(prompt)
                        targets.append(target)
                        predictions.append(prediction)
            for name, rows in (("prompts.jsonl", prompts), ("targets.jsonl", targets)):
                (panel / name).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
            pred_path = root / "predictions.jsonl"
            pred_path.write_text("".join(json.dumps(row) + "\n" for row in predictions))
            (panel / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": VERSION,
                        "independence_unit": "100 source IDs",
                        "files": {
                            name: sha256(panel / name)
                            for name in ("prompts.jsonl", "targets.jsonl")
                        },
                    }
                )
            )
            report = score(panel, pred_path)
            self.assertEqual(report["by_corpus_language"]["xnli"]["ar"]["correct"], 59)
            self.assertEqual(
                report["by_corpus_language"]["xnli"]["ar"]["independent_base_cases"], 60
            )
            self.assertEqual(
                report["by_corpus_language"]["xnli"]["ar"]["en_correct_target_wrong"], 1
            )
            self.assertEqual(report["by_corpus_language"]["pawsx"]["ja"]["correct"], 40)
            self.assertEqual(_exact_mcnemar_p(4, 0), 0.125)


if __name__ == "__main__":
    unittest.main()
