"""Meaningful publication checks with small schema-consistent scorer fixtures."""

from __future__ import annotations

import json
import statistics
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from benchmark.generate import FINAL_FAMILIES
from publication.generate import ARTIFACTS, generate
from transfer.build import EVALUATION_TASKS, PANEL_VERSION, sha_file

GOLD = "a" * 64
CSS_GOLD = "b" * 64


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def summary(n: int, correct: int, invalid: int = 0) -> dict:
    return {
        "n": n,
        "valid_n": n - invalid,
        "invalid_or_missing_n": invalid,
        "correct_n": correct,
        "accuracy_all": correct / n,
        "brier": None,
        "probability_n": 0,
    }


def benchmark(
    correct_by_family: tuple[int, int, int, int], *, identity: str, gold: str = GOLD
) -> dict:
    total = sum(correct_by_family)
    first, second, third = total // 3, total // 3, total - 2 * (total // 3)
    return {
        "schema_version": "typed-decision-report/2",
        "split": "final",
        "gold_sha256": gold,
        "predictions_sha256": ("1" if identity == "new" else "2") * 64,
        "model": {
            "id": identity,
            "revision": "abc123" if identity == "new" else "def456",
            "backend": "native",
        },
        "items": 12,
        "predicted_items": 12,
        "overall": summary(48, total, 2),
        "by_family": {
            name: summary(12, correct, 1 if index < 2 else 0)
            for index, (name, correct) in enumerate(
                zip(FINAL_FAMILIES, correct_by_family)
            )
        },
        "by_type": {
            name: summary(16, correct, 1 if index < 2 else 0)
            for index, (name, correct) in enumerate(
                zip(("choice", "noul", "score"), (first, second, third))
            )
        },
        "macro_family_accuracy": statistics.mean(
            correct / 12 for correct in correct_by_family
        ),
    }


def css_report(pred_sha: str, offset: float) -> dict:
    tasks = {
        name: {
            "role": "evaluation",
            "macro_f1_all": 0.3 + i * 0.02 + offset,
            "n": 100,
            "valid_n": 100,
            "invalid_or_missing_n": 0,
            "correct_n": round((0.4 + i * 0.02 + offset) * 100),
            "accuracy_all": 0.4 + i * 0.02 + offset,
        }
        for i, name in enumerate(EVALUATION_TASKS)
    }
    return {
        "score_schema_version": "css-transfer-score/2",
        "panel_version": PANEL_VERSION,
        "gold_sha256": CSS_GOLD,
        "predictions_sha256": pred_sha,
        "tasks": tasks,
        "roles": {
            "evaluation": {
                "tasks": 15,
                "items": sum(row["n"] for row in tasks.values()),
                "valid_items": sum(row["valid_n"] for row in tasks.values()),
                "micro_accuracy_all": sum(row["correct_n"] for row in tasks.values())
                / sum(row["n"] for row in tasks.values()),
                "median_task_macro_f1_all": statistics.median(
                    row["macro_f1_all"] for row in tasks.values()
                ),
                "median_task_accuracy_all": statistics.median(
                    row["accuracy_all"] for row in tasks.values()
                ),
            }
        },
    }


class PublicationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_json(self.root / "new.json", benchmark((9, 8, 7, 6), identity="new"))
        write_json(self.root / "old.json", benchmark((8, 7, 6, 5), identity="old"))
        self.config = self.root / "config.json"
        self.value = {
            "models": [
                {
                    "key": "old",
                    "label": "Decision 1.0",
                    "group": "decision1",
                    "size": "4B",
                    "benchmark_report": "old.json",
                },
                {
                    "key": "new",
                    "label": "Decision 2.0",
                    "group": "decision2",
                    "size": "4B",
                    "benchmark_report": "new.json",
                },
            ],
            "comparison_pairs": [{"new": "new", "old": "old"}],
        }
        write_json(self.config, self.value)

    def test_rank_matrix_provenance_and_determinism(self) -> None:
        out = self.root / "out"
        manifest = generate(self.config, out)
        markdown = (out / "score-table.md").read_text(encoding="utf-8")
        self.assertIn(
            "| 1 | Decision 2.0 | decision2 | 4B | 62.50% | 62.50% | 2/48 | — |",
            markdown,
        )
        self.assertIn("| Decision 2.0 | Decision 1.0 | +8.33 pp | +8.33 pp |", markdown)
        self.assertIn("75.00%", markdown)  # 9/12 in the first family
        self.assertIn("no synthetic-benchmark interval is shown", markdown)
        for name in ("ranking.svg", "matrix.svg"):
            ET.parse(out / name)
        self.assertIn("Decision 2.0", (out / "ranking.svg").read_text())
        self.assertIn("75.00%", (out / "matrix.svg").read_text())
        self.assertEqual(manifest["frozen_benchmark"]["gold_sha256"], GOLD)
        self.assertEqual(
            manifest["models"][0]["benchmark"]["report_sha256"],
            sha_file(self.root / "old.json"),
        )
        for name in ARTIFACTS:
            self.assertEqual(manifest["artifacts_sha256"][name], sha_file(out / name))
        self.assertNotIn(str(self.root), (out / "manifest.json").read_text())
        with self.assertRaises(FileExistsError):
            generate(self.config, out)
        generate(self.config, self.root / "out2")
        for name in (*ARTIFACTS, "manifest.json"):
            self.assertEqual(
                (out / name).read_bytes(), (self.root / "out2" / name).read_bytes()
            )

    def test_rejects_mismatched_or_inconsistent_score_reports(self) -> None:
        bad = benchmark((8, 7, 6, 5), identity="old", gold="c" * 64)
        write_json(self.root / "old.json", bad)
        with self.assertRaisesRegex(ValueError, "same frozen gold"):
            generate(self.config, self.root / "out")
        self.assertFalse((self.root / "out").exists())
        bad["gold_sha256"] = GOLD
        bad["macro_family_accuracy"] = 0.99
        write_json(self.root / "old.json", bad)
        with self.assertRaisesRegex(ValueError, "macro_family_accuracy disagrees"):
            generate(self.config, self.root / "out")

    def test_css_pair_interval_requires_matching_prediction_hashes(self) -> None:
        new_css = css_report("3" * 64, 0.05)
        old_css = css_report("4" * 64, 0.0)
        write_json(self.root / "new-css.json", new_css)
        write_json(self.root / "old-css.json", old_css)
        self.value["models"][0]["css_report"] = "old-css.json"
        self.value["models"][1]["css_report"] = "new-css.json"
        compare = {
            "comparison_version": "css-paired-item-bootstrap/1",
            "gold_sha256": CSS_GOLD,
            "predictions_a_sha256": new_css["predictions_sha256"],
            "predictions_b_sha256": old_css["predictions_sha256"],
            "bootstrap": {"replicates": 100, "seed": 7},
            "evaluation_median_over_15_tasks": {"task_count": 15},
        }
        for key, css_key in (
            ("macro_f1_all", "median_task_macro_f1_all"),
            ("accuracy_all", "median_task_accuracy_all"),
        ):
            a = new_css["roles"]["evaluation"][css_key]
            b = old_css["roles"]["evaluation"][css_key]
            compare["evaluation_median_over_15_tasks"][key] = {
                "median_a": a,
                "median_b": b,
                "difference_a_minus_b": a - b,
                "difference_interval95": {"low": 0.02, "high": 0.08},
            }
        write_json(self.root / "pair.json", compare)
        self.value["comparison_pairs"][0]["css_comparison_report"] = "pair.json"
        write_json(self.config, self.value)
        generate(self.config, self.root / "out")
        markdown = (self.root / "out" / "score-table.md").read_text()
        self.assertIn("+5.00 pp [+2.00 pp, +8.00 pp]", markdown)
        self.assertIn("CSS human-label transfer", markdown)
        compare["predictions_a_sha256"] = "f" * 64
        write_json(self.root / "pair.json", compare)
        with self.assertRaisesRegex(ValueError, "does not match CSS prediction files"):
            generate(self.config, self.root / "out-bad")
        self.assertFalse((self.root / "out-bad").exists())

    def test_public_artifacts_reject_credentials_and_host_paths(self) -> None:
        self.value["title"] = "Private run /home/person/experiments/final"
        write_json(self.config, self.value)
        with self.assertRaisesRegex(ValueError, "absolute host path"):
            generate(self.config, self.root / "out")
        self.assertFalse((self.root / "out").exists())
        self.value["title"] = "Decision 2.0 benchmark"
        self.value["models"][1]["label"] = "hf_" + "a" * 24
        write_json(self.config, self.value)
        with self.assertRaisesRegex(ValueError, "credential"):
            generate(self.config, self.root / "out")
        self.value["models"][1]["label"] = "Decision 2.0"
        write_json(self.config, self.value)
        leaked = benchmark((9, 8, 7, 6), identity="new")
        leaked["model"]["backend"] = "/work/private/backend"
        write_json(self.root / "new.json", leaked)
        with self.assertRaisesRegex(ValueError, "absolute host path"):
            generate(self.config, self.root / "out")

    def test_css_report_rejects_inconsistent_denominators(self) -> None:
        css = css_report("3" * 64, 0.0)
        css["tasks"][EVALUATION_TASKS[0]]["n"] -= 1
        write_json(self.root / "new-css.json", css)
        self.value["models"][1]["css_report"] = "new-css.json"
        write_json(self.config, self.value)
        with self.assertRaisesRegex(ValueError, "inconsistent task counts"):
            generate(self.config, self.root / "out")
        css = css_report("3" * 64, 0.0)
        css["roles"]["evaluation"]["items"] -= 1
        write_json(self.root / "new-css.json", css)
        with self.assertRaisesRegex(ValueError, "role counts disagree"):
            generate(self.config, self.root / "out")
        self.assertFalse((self.root / "out").exists())


if __name__ == "__main__":
    unittest.main()
