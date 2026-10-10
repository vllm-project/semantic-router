"""Stdlib tests of the release tooling: python -m unittest d25.vega.release.tests.test_release (from src/training/decision2)."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "package"))

from d25.vega.common import decision_format as df  # noqa: E402
from d25.vega.release import build, compare, sample  # noqa: E402

try:
    import decision25_runtime as rt  # noqa: E402  (needs decision25_format next to it: copied in setUpModule)
except ImportError:
    rt = None


def setUpModule():
    global rt
    target = HERE / "package" / "decision25_format.py"
    if not target.exists():
        target.write_bytes(build.FORMAT_SOURCE.read_bytes())
        TEMPORARY.append(target)
    import decision25_runtime

    rt = decision25_runtime


def tearDownModule():
    for path in TEMPORARY:
        path.unlink(missing_ok=True)


TEMPORARY: list[Path] = []


class Questions(unittest.TestCase):
    def test_choice_and_noul_render_like_the_training_contract(self):
        choice = {
            "type": "choice",
            "instructions": "Pick.",
            "criteria": {"a": None, "b": "second"},
        }
        q = rt.normalize_question(choice)
        self.assertEqual(df.options(q.rendered), df.options(choice))
        noul = {"type": "noul", "instructions": "Yes?"}
        self.assertEqual(
            df.options(rt.normalize_question(noul).rendered), df.options(noul)
        )

    def test_score_levels_are_options_without_descriptions(self):
        q = rt.normalize_question(
            {
                "type": "score",
                "instructions": "How?",
                "criteria": ["Low", "High", {"x": 1}],
            }
        )
        self.assertEqual(q.keys, ["0", "1", "2"])
        self.assertEqual(df.options(q.rendered)[1], ["Low", "High", '{"x": 1}'])

    def test_invalid_questions(self):
        for bad in (
            {"type": "choice", "criteria": {}},
            {"type": "score", "criteria": ["one"]},
            {"type": "score", "criteria": ["a", "a"]},
            {"type": "noul", "criteria": {"maybe": "x"}},
            {"type": "rank"},
            "text",
            {"type": "choice", "criteria": {"": "empty key"}},
        ):
            with self.assertRaises(ValueError):
                rt.normalize_question(bad)

    def test_answers_keep_the_kit_values_and_add_2_0_fields(self):
        choice = rt.normalize_question(
            {"type": "choice", "criteria": {"a": None, "b": None, "c": None}}
        )
        answer = rt.product_answer(choice, [0.2, 0.5, 0.3])
        kit = df.to_answer(choice.rendered, [0.2, 0.5, 0.3])
        self.assertEqual({k: answer[k] for k in kit}, kit)
        self.assertTrue(0 <= answer["confidence"] <= 1)
        noul = rt.normalize_question({"type": "noul"})
        self.assertEqual(
            rt.product_answer(noul, [0.25, 0.75]), {"type": "noul", "noul": 0.75}
        )
        score = rt.normalize_question(
            {"type": "score", "criteria": ["Routine", "Soon", "Today"]}
        )
        answer = rt.product_answer(score, [0.1, 0.2, 0.7])
        self.assertAlmostEqual(answer["score"], 0.2 + 1.4)
        self.assertEqual(answer["legend"], {"0": "Routine", "1": "Soon", "2": "Today"})
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0)
        uniform = rt.product_answer(score, [1 / 3] * 3)
        self.assertAlmostEqual(uniform["confidence"], 0.0)


class Build(unittest.TestCase):
    def test_sanitize_drops_paths_and_machines(self):
        cleaned = build.sanitize(
            {
                "init": {"path": "/data/d25/shared/models/x"},
                "host": "gpu-node-04.example",
                "keep": "Qwen/Qwen3.8-27B",
                "list": ["/root/a/b"],
            }
        )
        self.assertEqual(cleaned["init"]["path"], "local:x")
        self.assertNotIn("node-04", json.dumps(cleaned))
        self.assertEqual(cleaned["keep"], "Qwen/Qwen3.8-27B")
        self.assertEqual(cleaned["list"], ["local:b"])
        keyed = build.sanitize(
            {"data_sha256": {"/data/d25/vega/train/smoke/smoke4k.jsonl": "ab"}}
        )
        self.assertEqual(keyed, {"data_sha256": {"local:smoke4k.jsonl": "ab"}})
        self.assertFalse(build.lint_text("x.md", json.dumps(cleaned)))

    def test_lint_finds_leaks(self):
        for text in (
            "see 10.0.0.12",
            "/data/d25/vega",
            "token hf_" + "a" * 30,
            "node 04 GPU",
        ):
            self.assertTrue(build.lint_text("x.md", text), text)
        self.assertFalse(
            build.lint_text("x.md", "transformers==5.17.0 on Qwen/Qwen3.8-27B")
        )

    def test_site_patterns_come_from_a_private_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patterns.txt"
            path.write_text("# private\nexample-cluster-\\d+\n")
            os.environ[build.PRIVATE_PATTERNS_ENV] = str(path)
            try:
                self.assertTrue(build.lint_text("x.md", "ran on example-cluster-7"))
            finally:
                del os.environ[build.PRIVATE_PATTERNS_ENV]
        self.assertFalse(build.lint_text("x.md", "ran on example-cluster-7"))

    def test_package_code_is_lint_clean(self):
        for name in build.CODE_FILES:
            path = build.PACKAGE_SOURCE / name
            self.assertFalse(build.lint_text(name, path.read_text()), name)

    def test_tensor_counts_reads_safetensors_headers(self):
        header = json.dumps(
            {
                "language_model.w": {
                    "dtype": "BF16",
                    "shape": [3, 4],
                    "data_offsets": [0, 24],
                },
                "visual.v": {"dtype": "BF16", "shape": [5], "data_offsets": [24, 34]},
            }
        ).encode()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.safetensors"
            path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 34)
            self.assertEqual(build.tensor_counts(path), {"text": 12, "vision": 5})


class Sampling(unittest.TestCase):
    def rows(self, n=1000):
        return [
            {
                "state": "x" * (i % 37),
                "questions": {"q": {"type": "noul"}},
                "_evaluation": {"run_id": f"r{i}", "catalog_id": i % 5 + 1},
            }
            for i in range(n)
        ]

    def test_systematic_sample_is_deterministic_and_proportional(self):
        a = sample.draw(self.rows(), 100, 10, 20260926)
        b = sample.draw(self.rows(), 100, 10, 20260926)
        self.assertEqual(a[1]["run_ids"], b[1]["run_ids"])
        self.assertEqual(len(set(a[1]["run_ids"])), 100)
        self.assertEqual(a[1]["warmup_run_ids"], a[1]["run_ids"][:10])
        counts = list(a[1]["per_catalog"].values())
        self.assertLessEqual(max(counts) - min(counts), 3)


class Compare(unittest.TestCase):
    def test_percentile_matches_linear_interpolation(self):
        self.assertEqual(compare.percentile([1, 2, 3, 4, 5], 80), 4.2)
        self.assertEqual(compare.percentile([10], 80), 10)

    def test_latency_gate_and_answers(self):
        rows = {
            f"r{i}": {
                "run_id": f"r{i}",
                "status": "ok",
                "catalog_id": 1,
                "total_wall_ms": float(i),
            }
            for i in range(1, 101)
        }
        rows["w"] = {
            "run_id": "w",
            "status": "ok",
            "catalog_id": 1,
            "total_wall_ms": 99999.0,
        }
        stats = compare.latency(rows, {"w"})
        self.assertEqual(
            (stats["ok"], stats["median_ms"], stats["p80_ms"]), (100, 50.5, 80.2)
        )
        self.assertTrue(stats["gate_pass"])

        def run(p):
            return {
                "r": {
                    "run_id": "r",
                    "status": "ok",
                    "response": {
                        "answers": {
                            "c": {
                                "type": "choice",
                                "choice": "a" if p > 0.5 else "b",
                                "probabilities": {"a": p, "b": 1 - p},
                            },
                            "n": {"type": "noul", "noul": p},
                        }
                    },
                }
            }

        same = compare.answers(run(0.7), run(0.7))
        self.assertEqual(
            (same["argmax_changes"], same["noul_flips"], same["max_abs_dp"]),
            (0, 0, 0.0),
        )
        tie = compare.answers(run(0.505), run(0.495))
        self.assertEqual(
            (tie["argmax_changes"], tie["noul_flips"], tie["flips_near_tie"]), (1, 1, 2)
        )
        self.assertTrue(tie["all_flips_near_ties"])
        self.assertTrue(math.isclose(tie["max_abs_dp"], 0.01, abs_tol=1e-9))


class CardValuesTest(unittest.TestCase):
    def test_pending_card_states_no_values(self):
        from d25.vega.release import card

        own = {k: None for k in ("full", "public", "same_skill", "new_domain")}
        data = {
            "model_name": "Decision-2.5-Vega-27B",
            "base_model": "Qwen/Qwen3.8-27B",
            "index": {
                "status": "pending",
                "edition": "0.3",
                "snapshot": "2026-10-07",
                "own": own,
                "previous": {"name": "Decision-2.0-Vega-27B", "full": 55.9},
                "peers": [],
            },
        }
        text = (
            " ".join(card.highlights(data))
            + card.footnote(data)
            + card.chart_note(data)
        )
        self.assertIn("evaluation of this checkpoint is pending", text)
        self.assertNotRegex(text, r"\d\d\.\d")

    def test_own_values_come_from_measurements(self):
        from d25.vega.release import card_own

        with self.assertRaisesRegex(ValueError, "public index"):
            card_own.own_values(
                {"public": None, "proxy": {"O_proxy": 66.6}}, 63.0, 0.8, 62.0, 57.0
            )
        areas = {
            "knowledge": 50.1,
            "language": 60.2,
            "retrieval": 61.3,
            "tools": 75.4,
            "arts": 44.5,
        }
        per = {str(i): {"coverage": 1.0} for i in range(37)}
        result = {
            "arm": "w1-t2-nc",
            "step": 5135,
            "public": {"index": 0.6123, "areas": areas, "per_benchmark": per},
        }
        values = card_own.own_values(result, 63.04, 0.81, 62.0, 57.0)
        self.assertEqual(
            (values["public"], values["full"], values["uncertainty"]),
            (61.23, 63.04, 0.81),
        )
        self.assertEqual(values["areas"], areas)
        with self.assertRaisesRegex(ValueError, "areas"):
            card_own.own_values(
                {"public": {"index": 61.2, "areas": {}, "per_benchmark": per}},
                63.0,
                0.8,
                62.0,
                57.0,
            )


if __name__ == "__main__":
    unittest.main()
