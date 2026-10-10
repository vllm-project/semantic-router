"""Board reproduction and per-row scoring rules of ``d25.omni.suite.score``."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from d25.omni.suite import score

BOARD = json.loads((Path(__file__).parent / "data" / "vision-board.json").read_text())


def choice_row(rid, keys, gold, benchmark="CV-Bench", extra_questions=None):
    questions = {
        "q1": {"type": "choice", "instructions": "?", "criteria": {k: k for k in keys}}
    }
    expected = {"q1": gold}
    for qid, (q, g) in (extra_questions or {}).items():
        questions[qid], expected[qid] = q, g
    return {
        "id": rid,
        "family": benchmark,
        "questions": questions,
        "expected": expected,
        "metadata": {"benchmark": benchmark},
    }


def answer(choice, keys):
    return {
        "type": "choice",
        "choice": choice,
        "probabilities": {k: float(k == choice) for k in keys},
    }


class BoardReproductionTest(unittest.TestCase):
    def test_constants_match_every_snapshot(self):
        for snap in BOARD["snapshots"]:
            self.assertEqual(snap["bench_weights"], score.WEIGHTS)
            self.assertEqual(snap["bench_rows"], score.PUBLIC_ROWS)
            self.assertEqual(tuple(snap["private_sets"]), score.PRIVATE_SETS)
            self.assertEqual(snap["weights"], score.FULL_WEIGHTS)
        self.assertEqual(score.PUBLIC_WEIGHT_SUM, 9.75)
        self.assertEqual(sum(score.PUBLIC_ROWS.values()), 12210)

    def test_every_entry_reproduces_public_private_full(self):
        worst = 0.0
        for snap in BOARD["snapshots"]:
            for entry in snap["entries"]:
                got = score.board_entry(entry["bench"])
                for key in ("pub", "priv", "full"):
                    error = abs(got[key] - entry[key])
                    worst = max(worst, error)
                    self.assertLessEqual(
                        error, 0.011, (snap["generated_utc"], entry["engine"], key)
                    )
        self.assertGreater(worst, 0.0)

    def test_zero_floor_is_required(self):
        """Without the floor, entrants with a negative public benchmark miss their published public."""
        snap = BOARD["snapshots"][-1]
        negatives = [
            e
            for e in snap["entries"]
            if any((v["pub"] or 0) < 0 for v in e["bench"].values())
        ]
        self.assertTrue(negatives)
        for entry in negatives:
            raw = (
                sum(
                    score.WEIGHTS[b] * (entry["bench"][b]["pub"] or 0)
                    for b in score.BENCHMARKS
                )
                / 9.75
            )
            floored = score.board_entry(entry["bench"])["pub"]
            self.assertLess(
                abs(floored - entry["pub"]),
                abs(raw - entry["pub"]) + 1e-9,
                entry["engine"],
            )

    def test_single_image_entrants_floor_blink_private(self):
        snap = BOARD["snapshots"][-1]
        single = [e for e in snap["entries"] if e["single_image"]]
        self.assertEqual(len(single), 2)
        for entry in single:
            self.assertAlmostEqual(entry["bench"]["BLINK"]["private"], -33.33, places=2)

    def test_strict_ranks_reproduce_rank_full(self):
        for snap in BOARD["snapshots"]:
            entrants = [
                e
                for e in snap["entries"]
                if not e.get("ref") and e.get("rank_full") is not None
            ]
            if not entrants:
                continue
            ranks = score.rank({e["engine"]: e["full"] for e in entrants})
            for entry in entrants:
                self.assertEqual(
                    ranks[entry["engine"]],
                    entry["rank_full"],
                    (snap["generated_utc"], entry["engine"]),
                )


class RowScoringTest(unittest.TestCase):
    def test_choice_row_rules(self):
        keys = ["A", "B", "C", "D"]
        row = choice_row("r", keys, "C")
        self.assertTrue(score.row_correct(row, {"q1": answer("C", keys)}))
        self.assertFalse(score.row_correct(row, {"q1": answer("B", keys)}))
        self.assertFalse(
            score.row_correct(row, {"q1": {"type": "choice", "choice": "E"}})
        )
        self.assertFalse(score.row_correct(row, {}))
        self.assertFalse(score.row_correct(row, None))
        self.assertAlmostEqual(score.row_chance(row), 0.25)

    def test_every_question_must_be_right_and_chance_multiplies(self):
        noul = {"type": "noul", "instructions": "?"}
        row = choice_row("r", ["A", "B"], "A", extra_questions={"q2": (noul, True)})
        self.assertAlmostEqual(score.row_chance(row), 0.25)
        self.assertTrue(
            score.row_correct(
                row, {"q1": answer("A", "AB"), "q2": {"type": "noul", "noul": 0.5}}
            )
        )
        self.assertFalse(
            score.row_correct(
                row, {"q1": answer("A", "AB"), "q2": {"type": "noul", "noul": 0.49}}
            )
        )
        self.assertFalse(score.row_correct(row, {"q1": answer("A", "AB")}))

    def test_skill_matches_the_lattice_and_is_unclipped(self):
        rows = [
            choice_row(f"r{i}", ["A", "B"] if i % 2 else ["A", "B", "C", "D"], "A")
            for i in range(10)
        ]
        chance = (5 * 0.5 + 5 * 0.25) / 10
        for k in range(11):
            answers = {
                f"r{i}": {"q1": answer("A" if i < k else "B", "ABCD")}
                for i in range(10)
            }
            got = score.score_benchmark(rows, answers)
            self.assertAlmostEqual(got["chance"], chance)
            self.assertAlmostEqual(got["skill"], (k / 10 - chance) / (1 - chance) * 100)
        self.assertLess(score.score_benchmark(rows, {})["skill"], 0)

    def test_unknown_benchmark_is_rejected(self):
        with self.assertRaises(ValueError):
            score.score_suite([choice_row("r", "AB", "A", benchmark="Nope")], {})

    def test_results_file_status_and_resume(self):
        records = [
            {
                "run_id": "a",
                "status": "ok",
                "response": {"answers": {"q1": answer("A", "AB")}},
            },
            {"run_id": "b", "status": "unsupported", "error": "too many images"},
            {"id": "c", "answers": {"q1": answer("B", "AB")}},
            {
                "run_id": "d",
                "status": "ok",
                "response": {"answers": {"q1": answer("B", "AB")}},
            },
            {
                "run_id": "d",
                "status": "ok",
                "response": {"answers": {"q1": answer("A", "AB")}},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl.gz"
            with gzip.open(path, "wt") as f:
                f.write("\n".join(json.dumps(r) for r in records) + "\n")
            answers = score.load_answers(path)
        self.assertEqual(sorted(answers), ["a", "c", "d"])
        rows = [choice_row(rid, "AB", "A") for rid in "abcd"]
        got = score.score_suite(rows, answers)["benchmarks"]["CV-Bench"]
        self.assertEqual((got["rows"], got["answered"], got["correct"]), (4, 3, 2))
        self.assertAlmostEqual(got["skill"], 0.0)

    def test_public_private_full_aggregation(self):
        skills = {b: 50.0 for b in score.BENCHMARKS} | {"MMMU-Pro vision": -10.0}
        self.assertAlmostEqual(score.public_score(skills), (9 * 50 + 0.5 * 50) / 9.75)
        self.assertAlmostEqual(score.private_score(skills), 50.0)
        self.assertAlmostEqual(score.full_score(60.0, 40.0), 50.0)
        self.assertAlmostEqual(score.public_score({}), 0.0)


if __name__ == "__main__":
    unittest.main()
