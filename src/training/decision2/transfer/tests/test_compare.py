"""Paired bootstrap checks on a compact complete 15-task evaluation panel."""

import json
import tempfile
import unittest
from pathlib import Path

from transfer.build import EVALUATION_TASKS, PANEL_VERSION, sha_file
from transfer.compare import _sample_pair, compare


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PairedBootstrapTest(unittest.TestCase):
    def test_single_joint_draw_uses_the_same_ids_and_counts_invalid(self):
        class DrawBoth:
            positions = iter((0, 1))

            def randrange(self, _count):
                return next(self.positions)

        a_f1, b_f1, a_accuracy, b_accuracy = _sample_pair(
            [0, 1],
            [0, 1],
            [0, -1],
            2,
            DrawBoth(),
        )
        self.assertEqual((a_f1, b_f1, a_accuracy, b_accuracy), (1.0, 0.5, 1.0, 0.5))

    def fixture(self, root: Path) -> tuple[Path, Path, Path]:
        gold_rows = []
        perfect = []
        partial = []
        for task in EVALUATION_TASKS:
            for gold_label in ("a", "b"):
                item_id = f"css/{task}/{gold_label}"
                payload_hash = f"frozen-{task}-{gold_label}"
                gold_rows.append(
                    {
                        "id": item_id,
                        "panel_version": PANEL_VERSION,
                        "task": task,
                        "role": "evaluation",
                        "gold": gold_label,
                        "labels": ["a", "b"],
                        "input_sha256": payload_hash,
                    }
                )
                perfect.append(
                    {
                        "id": item_id,
                        "source_input_sha256": payload_hash,
                        "answers": {
                            "label": {
                                "type": "choice",
                                "choice": gold_label,
                                "probabilities": {
                                    "a": float(gold_label == "a"),
                                    "b": float(gold_label == "b"),
                                },
                            }
                        },
                    }
                )
                if gold_label == "a":
                    partial.append(perfect[-1])
        gold_path, a_path, b_path = (
            root / name for name in ("gold.jsonl", "a.jsonl", "b.jsonl")
        )
        write_jsonl(gold_path, gold_rows)
        write_jsonl(a_path, perfect)
        write_jsonl(b_path, partial)
        return gold_path, a_path, b_path

    def test_identical_predictions_have_exact_zero_paired_difference(self):
        with tempfile.TemporaryDirectory() as temporary:
            gold, a, _ = self.fixture(Path(temporary))
            report = compare(
                gold, a, a, model_a="A", model_b="A-clone", replicates=300, seed=81
            )
            self.assertEqual(
                report["evaluation_median_over_15_tasks"]["task_count"], 15
            )
            for metric in ("macro_f1_all", "accuracy_all"):
                headline = report["evaluation_median_over_15_tasks"][metric]
                self.assertEqual(headline["difference_a_minus_b"], 0)
                self.assertEqual(
                    headline["difference_interval95"], {"low": 0.0, "high": 0.0}
                )
                self.assertTrue(
                    all(
                        task[metric]["difference_interval95"]
                        == {"low": 0.0, "high": 0.0}
                        for task in report["tasks"].values()
                    )
                )

    def test_invalid_or_missing_is_a_miss_and_seed_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            gold, a, b = self.fixture(Path(temporary))
            first = compare(
                gold, a, b, model_a="A", model_b="B", replicates=300, seed=91
            )
            second = compare(
                gold, a, b, model_a="A", model_b="B", replicates=300, seed=91
            )
            self.assertEqual(first, second)
            self.assertEqual(first["gold_sha256"], sha_file(gold))
            self.assertEqual(first["predictions_a_sha256"], sha_file(a))
            self.assertEqual(first["predictions_b_sha256"], sha_file(b))
            for metric in ("macro_f1_all", "accuracy_all"):
                headline = first["evaluation_median_over_15_tasks"][metric]
                self.assertEqual(headline["median_a"], 1.0)
                self.assertEqual(headline["median_b"], 0.5)
                self.assertEqual(headline["difference_a_minus_b"], 0.5)
                self.assertTrue(
                    0
                    <= headline["difference_interval95"]["low"]
                    <= headline["difference_interval95"]["high"]
                    <= 0.5
                )
            self.assertTrue(
                all(
                    task["invalid_or_missing_b"] == 1
                    for task in first["tasks"].values()
                )
            )

    def test_incomplete_evaluation_panel_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            gold, a, b = self.fixture(Path(temporary))
            missing_task = EVALUATION_TASKS[-1]
            gold.write_text(
                "".join(
                    line
                    for line in gold.read_text().splitlines(keepends=True)
                    if f'"task": "{missing_task}"' not in line
                )
            )
            # Prediction files may not name IDs outside the selected gold file.
            for path in (a, b):
                path.write_text(
                    "".join(
                        line
                        for line in path.read_text().splitlines(keepends=True)
                        if f"css/{missing_task}/" not in line
                    )
                )
            with self.assertRaisesRegex(ValueError, "Expected all 15"):
                compare(gold, a, b, model_a="A", model_b="B", replicates=100)


if __name__ == "__main__":
    unittest.main()
