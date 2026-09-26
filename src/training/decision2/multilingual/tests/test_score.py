import json
from pathlib import Path
import tempfile
import unittest

from multilingual.pilot import build
from multilingual.score import score


class ScoreTest(unittest.TestCase):
    def test_perfect_native_predictions_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            panel = base / "panel"
            build(panel)
            targets = [
                json.loads(line)
                for line in (panel / "targets.jsonl").read_text().splitlines()
            ]
            predictions = base / "predictions.jsonl"
            with predictions.open("w") as stream:
                for target in targets:
                    if target["task_type"] == "choice":
                        answer = {"type": "choice", "choice": target["gold"]}
                    elif target["task_type"] == "noul":
                        answer = {
                            "type": "noul",
                            "noul": 0.9 if target["gold"] else 0.1,
                        }
                    else:
                        answer = {
                            "type": "score",
                            "probabilities": {
                                str(index): 1.0 if index == target["gold"] else 0.0
                                for index in range(4)
                            },
                        }
                    stream.write(
                        json.dumps(
                            {
                                "id": target["id"],
                                "answers": {"decision": answer},
                                "backend": "test",
                                "model_id": "fixture",
                                "model_revision": "fixture",
                                "source_input_sha256": target["source_input_sha256"],
                            }
                        )
                        + "\n"
                    )
            report = score(panel, predictions)
            self.assertEqual(report["by_language"]["ar"]["base_macro_accuracy"], 1.0)
            self.assertEqual(
                report["paired_vs_en"]["ja"]["semantic_prediction_agreement_vs_en"], 1.0
            )
            self.assertEqual(
                report["perturbation_robustness"]["zh"]["choice"][
                    "semantic_prediction_flip_cases"
                ],
                0,
            )


if __name__ == "__main__":
    unittest.main()
