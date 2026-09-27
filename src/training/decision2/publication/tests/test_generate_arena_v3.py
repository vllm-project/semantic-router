"""CPU contracts for the separate v3 sealed-core and public231 card figures."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from jev_arena.arena_v3 import SCORER_SOURCE_PATHS
from publication.generate_arena_v3 import FIGURES, generate, sha_file


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class ArenaV3ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.entries = (
            ("new", "Decision 2.0", "decision2", "org/dev-2.0-0.8b", 0.8, 0.6),
            ("old", "Decision 1.0", "decision1", "org/Decision-1.0-Eos", 0.75, 0.5),
        )
        rows, public_rows = [], []
        for rank, (key, label, group, model_id, size, point) in enumerate(
            self.entries, 1
        ):
            typed = {"predictions_sha256": str(rank) * 64}
            css = {"predictions_sha256": str(rank + 3) * 64}
            write(self.root / f"{key}-typed.json", typed)
            write(self.root / f"{key}-css.json", css)
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "group": group,
                    "model_id": model_id,
                    "revision": f"revision-{key}",
                    "size_b": size,
                    "native_model_sha256": "a" * 64 if group == "decision2" else None,
                    "adapter_sha256": "b" * 64,
                    "calibration_sha256": "c" * 64,
                    "axes": {"typed": point, "transfer": point},
                    "score": 100 * math.sqrt(point * point),
                    "rank": rank,
                    "pareto_frontier": True,
                    "task_scores": {
                        "typed": dict.fromkeys(("choice", "noul", "score"), point),
                        "transfer": {f"task_{i:02d}": point for i in range(15)},
                    },
                    "coverage": {
                        "typed_items": 1600,
                        "css_items": 6547,
                        "sealed_core_items": 8147,
                    },
                    "report_sha256": {
                        "typed": sha_file(self.root / f"{key}-typed.json"),
                        "css": sha_file(self.root / f"{key}-css.json"),
                    },
                }
            )
            public_rows.append(
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
                    "tiers": dict.fromkeys(("easy", "standard", "hard"), point),
                    "valid": 231,
                    "items": 231,
                    "rank": rank,
                    "pareto_frontier": True,
                    "report_sha256": str(rank + 6) * 64,
                }
            )
        self.arena = {
            "schema_version": "jevarena-ranking/3",
            "phase": "release",
            "status": "scored_pending_independent_release_audit",
            "manifest_sha256": "d" * 64,
            "freeze_sha256": "e" * 64,
            "panel_sha256": {
                "typed_gold_sha256": "f" * 64,
                "css_gold_sha256": "0" * 64,
            },
            "models": rows,
        }
        self.public = {
            "schema_version": "jevarena-jevbench-public-rank/1",
            "items": 231,
            "panel_sha256": {
                "prompts_sha256": "1" * 64,
                "targets_sha256": "2" * 64,
                "panel_manifest_sha256": "3" * 64,
            },
            "models": public_rows,
        }
        write(self.root / "arena.json", self.arena)
        write(self.root / "public.json", self.public)
        write(
            self.root / "typed-pair.json",
            {
                "schema_version": "typed-decision-comparison/1",
                "split": "final",
                "models": {"left": self.entries[0][3], "right": self.entries[1][3]},
                "gold_sha256": "f" * 64,
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
            self.root / "transfer-pair.json",
            {
                "comparison_version": "css-paired-item-bootstrap/1",
                "model_a": self.entries[0][3],
                "model_b": self.entries[1][3],
                "gold_sha256": "0" * 64,
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
        panel_hashes = self.arena["panel_sha256"]
        write(
            self.root / "joint-pair.json",
            {
                "schema_version": "jevarena-v3-paired-aggregate/1",
                "models": {"left": self.entries[0][3], "right": self.entries[1][3]},
                "typed_gold_sha256": panel_hashes["typed_gold_sha256"],
                "css_gold_sha256": panel_hashes["css_gold_sha256"],
                "panel_sha256": hashlib.sha256(
                    json.dumps(
                        panel_hashes, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "predictions_sha256": {
                    "left": {"typed": "1" * 64, "css": "4" * 64},
                    "right": {"typed": "2" * 64, "css": "5" * 64},
                },
                "replicates": 5000,
                "seed": 20260927,
                "source_sha256": {
                    name: sha_file(path) for name, path in SCORER_SOURCE_PATHS.items()
                },
                "point": {
                    "left": {"T": 0.6, "H": 0.6, "score": 60.0},
                    "right": {"T": 0.5, "H": 0.5, "score": 50.0},
                    "delta": {"T": 0.1, "H": 0.1, "score": 10.0},
                },
                "ci95": {"low": 2.0, "high": 17.0},
            },
        )
        self.config = {
            "arena_rank": "arena.json",
            "jevbench_public_rank": "public.json",
            "comparison_pairs": [
                {
                    "new": "new",
                    "old": "old",
                    "typed_comparison": "typed-pair.json",
                    "transfer_comparison": "transfer-pair.json",
                    "joint_comparison": "joint-pair.json",
                    "new_typed_report": "new-typed.json",
                    "old_typed_report": "old-typed.json",
                    "new_transfer_report": "new-css.json",
                    "old_transfer_report": "old-css.json",
                }
            ],
        }
        write(self.root / "config.json", self.config)

    def test_generates_separate_v3_and_public_figures(self) -> None:
        output = self.root / "artifacts"
        manifest = generate(self.root / "config.json", output)
        self.assertEqual(manifest["coverage"]["sealed_core_items"], 8147)
        self.assertEqual(manifest["coverage"]["public_items_in_arena_score"], 0)
        self.assertEqual(manifest["coverage"]["authored_items_in_arena_score"], 0)
        table = (output / "score-table.md").read_text()
        self.assertIn("8,147 sealed-core", table)
        self.assertIn("JevBench public", table)
        self.assertNotIn("six-axis", table)
        self.assertIn("+10.00 pp [+2.00 pp, +18.00 pp]", table)
        self.assertIn("+10.00 [+2.00, +17.00] points", table)
        for name in FIGURES:
            ET.parse(output / name)
            self.assertEqual(
                manifest["artifacts_sha256"][name], sha_file(output / name)
            )
        self.assertFalse(list(output.glob("*pareto*")))
        with self.assertRaises(FileExistsError):
            generate(self.root / "config.json", output)

    def test_rejects_v2_or_public_in_sealed_score(self) -> None:
        self.arena["schema_version"] = "jevarena-ranking/2"
        write(self.root / "arena.json", self.arena)
        with self.assertRaisesRegex(ValueError, "v3 sealed-core"):
            generate(self.root / "config.json", self.root / "bad")
        self.arena["schema_version"] = "jevarena-ranking/3"
        self.arena["models"][0]["axes"]["jevbench_public"] = 0.99
        write(self.root / "arena.json", self.arena)
        with self.assertRaisesRegex(ValueError, "only two sealed axes"):
            generate(self.root / "config.json", self.root / "bad")

    def test_rejects_mismatched_public_roster_and_pair(self) -> None:
        self.public["models"][0]["revision"] = "another"
        write(self.root / "public.json", self.public)
        with self.assertRaisesRegex(ValueError, "identity"):
            generate(self.root / "config.json", self.root / "bad")
        self.public["models"][0]["revision"] = "revision-new"
        write(self.root / "public.json", self.public)
        paired = json.loads((self.root / "typed-pair.json").read_text())
        paired["left_sha256"] = "9" * 64
        write(self.root / "typed-pair.json", paired)
        with self.assertRaisesRegex(ValueError, "not bound"):
            generate(self.root / "config.json", self.root / "bad")

    def test_rejects_private_text(self) -> None:
        self.arena["models"][0]["label"] = "private 192.168.1.10"
        write(self.root / "arena.json", self.arena)
        with self.assertRaisesRegex(ValueError, "private infrastructure"):
            generate(self.root / "config.json", self.root / "bad")

    def test_rejects_unbound_joint_interval(self) -> None:
        joint = json.loads((self.root / "joint-pair.json").read_text())
        joint["predictions_sha256"]["left"]["css"] = "9" * 64
        write(self.root / "joint-pair.json", joint)
        with self.assertRaisesRegex(ValueError, "Joint v3 comparison"):
            generate(self.root / "config.json", self.root / "bad")
        joint["predictions_sha256"]["left"]["css"] = "4" * 64
        joint["ci95"]["low"] = 18.0
        write(self.root / "joint-pair.json", joint)
        with self.assertRaisesRegex(
            ValueError, "joint v3 score interval|Joint v3 score interval"
        ):
            generate(self.root / "config.json", self.root / "bad")

    def test_rejects_fabricated_rank_or_pareto_label(self) -> None:
        self.arena["models"][0]["rank"] = 2
        write(self.root / "arena.json", self.arena)
        with self.assertRaisesRegex(ValueError, "rank number"):
            generate(self.root / "config.json", self.root / "bad")
        self.arena["models"][0]["rank"] = 1
        self.public["models"][0]["pareto_frontier"] = False
        write(self.root / "arena.json", self.arena)
        write(self.root / "public.json", self.public)
        with self.assertRaisesRegex(ValueError, "Pareto label"):
            generate(self.root / "config.json", self.root / "bad")


if __name__ == "__main__":
    unittest.main()
