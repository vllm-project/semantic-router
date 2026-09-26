"""The six-axis release gate rejects missing panels and unaudited authored data."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from decision_bench_v4.bench import SCORE_VERSION as DBV4_VERSION

from jev_arena.arena_v2 import rank
from jev_arena.jevbench_public import SCORE_VERSION as PUBLIC_VERSION


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class ArenaV2Test(unittest.TestCase):
    def test_release_requires_all_six_axes_and_matching_panels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roster = []
            for key, size, quality in (("small", 0.6, 0.6), ("large", 4.0, 0.8)):
                model_id, revision = f"test/{key}", f"rev-{key}"
                synthetic = root / f"{key}-synthetic.json"
                css = root / f"{key}-css.json"
                public = root / f"{key}-public.json"
                dbv4 = root / f"{key}-dbv4.json"
                authored = root / f"{key}-authored.json"
                write(
                    synthetic,
                    {
                        "schema_version": "typed-decision-report/2",
                        "split": "final",
                        "items": 1600,
                        "gold_sha256": "a" * 64,
                        "model": {"id": model_id, "revision": revision},
                        "macro_family_accuracy": quality,
                        "by_type": {
                            kind: {"accuracy_all": quality}
                            for kind in ("choice", "noul", "score")
                        },
                        "pairs": {
                            name: {"joint_accuracy_all": quality}
                            for name in (
                                "counterfactual",
                                "order_invariance",
                                "label_invariance",
                            )
                        },
                    },
                )
                write(
                    css,
                    {
                        "gold_sha256": "b" * 64,
                        "tasks": {
                            f"transfer_{index:02d}": {
                                "role": "evaluation",
                                "macro_f1_all": quality,
                            }
                            for index in range(15)
                        },
                        "roles": {
                            "evaluation": {
                                "items": 6547,
                                "tasks": 15,
                                "median_task_macro_f1_all": quality,
                            }
                        },
                    },
                )
                write(
                    public,
                    {
                        "score_version": PUBLIC_VERSION,
                        "items": 231,
                        "model_id": model_id,
                        "model_revision": revision,
                        "prompts_sha256": "c" * 64,
                        "targets_sha256": "d" * 64,
                        "tier_macro_accuracy": quality,
                    },
                )
                write(
                    dbv4,
                    {
                        "score_version": DBV4_VERSION,
                        "eligible_items": 1041,
                        "ineligible_items": 30,
                        "model_id": model_id,
                        "model_revision": revision,
                        "prompts_sha256": "e" * 64,
                        "targets_sha256": "f" * 64,
                        "task_macro_accuracy": quality,
                    },
                )
                write(
                    authored,
                    {
                        "score_version": "jevarena-authored-score/1",
                        "phase": "release",
                        "quality_gate": {"status": "passed"},
                        "items": 1296,
                        "independent_groups": 1296,
                        "model_id": model_id,
                        "model_revision": revision,
                        "prompts_sha256": "1" * 64,
                        "targets_sha256": "2" * 64,
                        "macro_family_accuracy": quality,
                    },
                )
                roster.append(
                    {
                        "key": key,
                        "label": key,
                        "group": "open",
                        "model_id": model_id,
                        "revision": revision,
                        "size_b": size,
                        "synthetic_report": str(synthetic),
                        "css_report": str(css),
                        "jevbench_public_report": str(public),
                        "decision_bench_v4_report": str(dbv4),
                        "sealed_authored_report": str(authored),
                    }
                )
            manifest = root / "manifest.json"
            write(manifest, {"phase": "release", "models": roster})
            result = rank(manifest, "release")
            self.assertEqual(
                [row["key"] for row in result["models"]], ["large", "small"]
            )
            self.assertAlmostEqual(result["models"][0]["score"], 80)
            self.assertEqual(
                result["models"][0]["coverage"]["effective_text_answers"], 10715
            )
            self.assertTrue(all(row["pareto_frontier"] for row in result["models"]))
            self.assertEqual(len(result["models"][0]["task_scores"]["transfer"]), 15)
            changed = json.loads(Path(roster[1]["sealed_authored_report"]).read_text())
            changed["quality_gate"]["status"] = "blocked"
            write(Path(roster[1]["sealed_authored_report"]), changed)
            with self.assertRaisesRegex(ValueError, "quality gate"):
                rank(manifest, "release")
            changed["quality_gate"]["status"] = "passed"
            changed["targets_sha256"] = "3" * 64
            write(Path(roster[1]["sealed_authored_report"]), changed)
            with self.assertRaisesRegex(ValueError, "panel digest differs"):
                rank(manifest, "release")


if __name__ == "__main__":
    unittest.main()
