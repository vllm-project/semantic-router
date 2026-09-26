"""Cross-panel identity and ranking contract tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jev_arena.arena import rank
from jev_arena.jevbench_public import SCORE_VERSION


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class ArenaTest(unittest.TestCase):
    def test_rank_and_pareto_require_same_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roster = []
            for key, size, quality in (("small", 0.5, 0.6), ("large", 4, 0.8)):
                model_id, revision = f"test/{key}", f"rev-{key}"
                synthetic = root / f"{key}.synthetic.json"
                css = root / f"{key}.css.json"
                public = root / f"{key}.public.json"
                _write(
                    synthetic,
                    {
                        "schema_version": "typed-decision-report/2",
                        "split": "final",
                        "items": 1600,
                        "gold_sha256": "a" * 64,
                        "model": {"id": model_id, "revision": revision},
                        "macro_family_accuracy": quality,
                        "pairs": {
                            relation: {"joint_accuracy_all": quality}
                            for relation in (
                                "counterfactual",
                                "order_invariance",
                                "label_invariance",
                            )
                        },
                    },
                )
                _write(
                    css,
                    {
                        "gold_sha256": "b" * 64,
                        "roles": {
                            "evaluation": {
                                "items": 6547,
                                "tasks": 15,
                                "median_task_macro_f1_all": quality,
                            }
                        },
                    },
                )
                _write(
                    public,
                    {
                        "score_version": SCORE_VERSION,
                        "items": 231,
                        "model_id": model_id,
                        "model_revision": revision,
                        "prompts_sha256": "c" * 64,
                        "targets_sha256": "d" * 64,
                        "tier_macro_accuracy": quality,
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
                    }
                )
            manifest = root / "manifest.json"
            _write(manifest, {"phase": "release", "models": roster})
            result = rank(manifest, "release")
            self.assertEqual(
                [row["key"] for row in result["models"]], ["large", "small"]
            )
            self.assertAlmostEqual(result["models"][0]["score"], 80)
            self.assertTrue(all(row["pareto_frontier"] for row in result["models"]))
            changed = json.loads(Path(roster[1]["jevbench_public_report"]).read_text())
            changed["prompts_sha256"] = "e" * 64
            _write(Path(roster[1]["jevbench_public_report"]), changed)
            with self.assertRaisesRegex(ValueError, "panel digest differs"):
                rank(manifest, "release")


if __name__ == "__main__":
    unittest.main()
