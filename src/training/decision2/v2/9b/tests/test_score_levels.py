import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

from lux9b import score_levels  # noqa: E402
from v2.eval.same_panel import sha_file  # noqa: E402

SCORE_Q = {"type": "score", "instructions": "x", "criteria": ["0", "1", "2", "3", "4"]}
NOUL_Q = {"type": "noul", "instructions": "x", "criteria": {"true": "T", "false": "F"}}

# (gold level, predicted score or None for no answer)
SLOTS = [
    (0, 0),
    (0, 1),
    (0, 0),
    (1, 1),
    (2, 2),
    (2, 2.4),
    (3, 4),
    (3, 3.5),
    (4, None),
    (4, 4),
]


def panel():
    gold, preds = [], {}
    for k, (level, answer) in enumerate(SLOTS):
        item = f"i{k}"
        gold.append(
            {
                "id": item,
                "questions": {"s": SCORE_Q, "n": NOUL_Q},
                "gold": {"s": {"value": level}, "n": {"value": True}},
            }
        )
        answers = {"n": {"noul": 0.9}}
        if answer is not None:
            answers["s"] = {"score": answer}
        preds[item] = {"id": item, "answers": answers}
    return gold, preds


class ScoreLevelsTest(unittest.TestCase):
    def test_histograms_and_recall(self):
        gold, preds = panel()
        out = score_levels.score_levels(gold, preds)
        # 3.5 is equidistant from 3 and 4 -> no point (invalid)
        self.assertEqual(
            out["predicted_levels"], {"0": 2, "1": 2, "2": 2, "3": 0, "4": 2}
        )
        self.assertEqual(out["invalid"], 2)
        self.assertEqual(out["gold_levels"], {"0": 3, "1": 1, "2": 2, "3": 2, "4": 2})
        self.assertAlmostEqual(out["level0_recall"], 2 / 3)
        self.assertEqual(out["recall_by_level"]["3"], 0)
        self.assertEqual(out["recall_by_level"]["4"], 0.5)
        self.assertEqual(out["levels_used"], 4)
        self.assertEqual(out["slots"], 10)
        self.assertAlmostEqual(out["largest_answer_share"], 0.2)
        self.assertAlmostEqual(out["accuracy"], 0.6)

    def test_single_answer_share(self):
        gold, preds = panel()
        for row in preds.values():
            row["answers"]["s"] = {"score": 2}
        out = score_levels.score_levels(gold, preds)
        self.assertEqual(out["largest_answer"], "2")
        self.assertEqual(out["largest_answer_share"], 1.0)
        self.assertEqual(out["level0_recall"], 0)

    def test_cli_checks_the_seal(self):
        gold, preds = panel()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold_path = root / "gold.jsonl"
            gold_path.write_text("".join(json.dumps(g) + "\n" for g in gold))
            run = root / "run"
            (run / "output").mkdir(parents=True)
            pred = run / "output" / "typed-final.predictions.jsonl"
            pred.write_text("".join(json.dumps(p) + "\n" for p in preds.values()))
            seal = {"panels": {"typed-final": {"predictions_sha256": sha_file(pred)}}}
            (run / "SEAL.json").write_text(json.dumps(seal))
            argv = ["--run-dir", str(run), "--output", str(root / "out.json")]
            with mock.patch.object(score_levels.panels, "verify"), mock.patch.object(
                score_levels.panels, "path", return_value=gold_path
            ), contextlib.redirect_stdout(io.StringIO()):
                score_levels.main(argv)
                out = json.loads((root / "out.json").read_text())
                self.assertEqual(out["score"]["gold_levels"]["0"], 3)
                pred.write_text(pred.read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "after the seal"):
                    score_levels.main(argv + ["--output", str(root / "out2.json")])


if __name__ == "__main__":
    unittest.main()
