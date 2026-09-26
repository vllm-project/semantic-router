"""Release artwork requires matched panels and bound paired comparisons."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from publication.generate_arena import FIGURES, generate, sha_file


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class ArenaPublicationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.hashes = {
            "synthetic_gold": "a" * 64,
            "css_gold": "b" * 64,
            "jevbench_prompts": "c" * 64,
            "jevbench_targets": "d" * 64,
            "dbv4_prompts": "e" * 64,
            "dbv4_targets": "f" * 64,
            "authored_prompts": "1" * 64,
            "authored_targets": "2" * 64,
        }
        self.coverage = {
            "synthetic_items": 1600,
            "css_items": 6547,
            "jevbench_public_items": 231,
            "decision_bench_v4_eligible": 1041,
            "decision_bench_v4_ineligible_ne": 30,
            "sealed_authored_items": 1200,
            "sealed_authored_independent_groups": 1200,
            "effective_text_answers": 10619,
        }
        self.entries = (
            ("new", "Decision 2.0", "decision2", "org/dev-2.0-0.8b", 0.8, 0.6),
            ("old", "Decision 1.0", "decision1", "org/Decision-1.0-Eos", 0.75, 0.5),
            ("peer", "Open peer", "other", "peer/model", 2.0, 0.4),
        )
        self.arena_rows, self.public_rows = [], []
        for rank, (key, label, group, model_id, size, point) in enumerate(
            self.entries, 1
        ):
            typed_report = {"predictions_sha256": str(rank) * 64}
            transfer_report = {"predictions_sha256": str(rank + 3) * 64}
            write(self.root / f"{key}-typed.json", typed_report)
            write(self.root / f"{key}-transfer.json", transfer_report)
            task_scores = {
                "typed": {name: point for name in ("choice", "noul", "score")},
                "transfer": {f"task_{index:02d}": point for index in range(15)},
            }
            self.arena_rows.append(
                {
                    "key": key,
                    "label": label,
                    "group": group,
                    "model_id": model_id,
                    "revision": f"revision-{key}",
                    "size_b": size,
                    "axes": {
                        name: point
                        for name in (
                            "typed",
                            "transfer",
                            "jevbench_public",
                            "decision_bench_v4",
                            "sealed_authored",
                            "robustness",
                        )
                    },
                    "score": 100 * math.prod([point] * 6) ** (1 / 6),
                    "rank": rank,
                    "pareto_frontier": True,
                    "task_scores": task_scores,
                    "coverage": self.coverage,
                    "report_sha256": {
                        "synthetic": sha_file(self.root / f"{key}-typed.json"),
                        "css": sha_file(self.root / f"{key}-transfer.json"),
                        "public": str(rank + 6) * 64,
                    },
                }
            )
            self.public_rows.append(
                {
                    "key": key,
                    "label": label,
                    "group": group,
                    "model_id": model_id,
                    "revision": f"revision-{key}",
                    "size_b": size,
                    "score": 100 * point,
                    "accuracy_all": point,
                    "tier_macro_accuracy": point,
                    "rank": rank,
                    "pareto_frontier": True,
                    "report_sha256": str(rank + 6) * 64,
                }
            )
        self.arena = {
            "schema_version": "jevarena-ranking/2",
            "phase": "release",
            "panel_sha256": self.hashes,
            "models": self.arena_rows,
        }
        self.public = {
            "schema_version": "jevarena-jevbench-public-rank/1",
            "items": 231,
            "panel_sha256": {
                "prompts_sha256": self.hashes["jevbench_prompts"],
                "targets_sha256": self.hashes["jevbench_targets"],
            },
            "models": self.public_rows,
        }
        write(self.root / "arena.json", self.arena)
        write(self.root / "public.json", self.public)
        write(
            self.root / "typed-compare.json",
            {
                "schema_version": "typed-decision-comparison/1",
                "split": "final",
                "models": {"left": self.entries[0][3], "right": self.entries[1][3]},
                "gold_sha256": self.hashes["synthetic_gold"],
                "left_sha256": "1" * 64,
                "right_sha256": "2" * 64,
                "iterations": 500,
                "family_macro": {
                    "left": 0.6,
                    "right": 0.5,
                    "delta_left_minus_right": 0.1,
                    "delta_ci95": [0.02, 0.18],
                },
            },
        )
        write(
            self.root / "transfer-compare.json",
            {
                "comparison_version": "css-paired-item-bootstrap/1",
                "model_a": self.entries[0][3],
                "model_b": self.entries[1][3],
                "gold_sha256": self.hashes["css_gold"],
                "predictions_a_sha256": "4" * 64,
                "predictions_b_sha256": "5" * 64,
                "bootstrap": {"replicates": 500},
                "evaluation_median_over_15_tasks": {
                    "macro_f1_all": {
                        "median_a": 0.6,
                        "median_b": 0.5,
                        "difference_a_minus_b": 0.1,
                        "difference_interval95": {"low": 0.01, "high": 0.17},
                    }
                },
            },
        )
        self.config = {
            "arena_rank": "arena.json",
            "jevbench_public_rank": "public.json",
            "comparison_pairs": [
                {
                    "new": "new",
                    "old": "old",
                    "typed_comparison": "typed-compare.json",
                    "transfer_comparison": "transfer-compare.json",
                    "new_typed_report": "new-typed.json",
                    "old_typed_report": "old-typed.json",
                    "new_transfer_report": "new-transfer.json",
                    "old_transfer_report": "old-transfer.json",
                }
            ],
        }
        write(self.root / "config.json", self.config)

    def test_six_figures_table_hashes_and_no_overwrite(self) -> None:
        output = self.root / "artifacts"
        manifest = generate(self.root / "config.json", output)
        self.assertEqual(manifest["coverage"]["effective_text_answers"], 10619)
        table = (output / "score-table.md").read_text()
        self.assertIn("JevArena", table)
        self.assertIn("+10.00 pp [+2.00 pp, +18.00 pp]", table)
        self.assertIn("JevBench public rank", table)
        for name in FIGURES:
            ET.parse(output / name)
            self.assertEqual(
                manifest["artifacts_sha256"][name], sha_file(output / name)
            )
        with self.assertRaises(FileExistsError):
            generate(self.root / "config.json", output)

    def test_rejects_panel_or_comparison_mismatch(self) -> None:
        self.public["panel_sha256"]["targets_sha256"] = "9" * 64
        write(self.root / "public.json", self.public)
        with self.assertRaisesRegex(ValueError, "panels differ"):
            generate(self.root / "config.json", self.root / "bad")
        self.assertFalse((self.root / "bad").exists())
        self.public["panel_sha256"]["targets_sha256"] = self.hashes["jevbench_targets"]
        write(self.root / "public.json", self.public)
        wrong = json.loads((self.root / "typed-compare.json").read_text())
        wrong["left_sha256"] = "9" * 64
        write(self.root / "typed-compare.json", wrong)
        with self.assertRaisesRegex(ValueError, "not bound to ranked predictions"):
            generate(self.root / "config.json", self.root / "bad")

    def test_rejects_private_text_in_model_label(self) -> None:
        self.arena_rows[2]["label"] = "peer 192.168.1.10"
        self.public_rows[2]["label"] = "peer 192.168.1.10"
        write(self.root / "arena.json", self.arena)
        write(self.root / "public.json", self.public)
        with self.assertRaisesRegex(ValueError, "private infrastructure"):
            generate(self.root / "config.json", self.root / "unsafe")
        self.assertFalse((self.root / "unsafe").exists())


if __name__ == "__main__":
    unittest.main()
