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

from lux9b import mlx_paired  # noqa: E402
from v2.eval import multilingual_panel  # noqa: E402
from v2.eval.same_panel import sha_file  # noqa: E402

CHOICE_Q = {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B"}}
NOUL_Q = {"type": "noul", "instructions": "x", "criteria": {"true": "T", "false": "F"}}
SCORE_Q = {"type": "score", "instructions": "x", "criteria": ["0", "1", "2"]}
CELLS = [
    ("choice", "en", CHOICE_Q),
    ("choice", "de", CHOICE_Q),
    ("noul", "en", NOUL_Q),
    ("noul", "ko", NOUL_Q),
    ("score", "en", SCORE_Q),
    ("score", "de", SCORE_Q),
]
PER_CELL = 4


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def build_panel(root):
    gold, prompts = [], []
    for kind, lang, q in CELLS:
        for k in range(PER_CELL):
            item = f"{kind}-{lang}-{k}"
            value = {"choice": "a", "noul": True, "score": 1}[kind]
            gold.append(
                {
                    "id": item,
                    "source": kind,
                    "source_id": str(k),
                    "language": lang,
                    "type": kind,
                    "value": value,
                    "input_sha256": f"h-{item}",
                }
            )
            prompts.append({"id": item, "state": "s", "questions": {"q": q}})
    root.mkdir()
    write_jsonl(root / "gold.jsonl", gold)
    write_jsonl(root / "prompts.jsonl", prompts)
    return gold


def make_run(root, panel, gold, wrong):
    """A run that answers every item correctly except the ids in `wrong`."""
    bad = {"choice": {"choice": "b"}, "noul": {"noul": 0.1}, "score": {"score": 2}}
    good = {"choice": {"choice": "a"}, "noul": {"noul": 0.9}, "score": {"score": 1}}
    rows = [
        {
            "id": g["id"],
            "source_input_sha256": g["input_sha256"],
            "answers": {"q": (bad if g["id"] in wrong else good)[g["type"]]},
        }
        for g in gold
    ]
    (root / "output").mkdir(parents=True)
    pred = root / "output" / "mlx-diag.predictions.jsonl"
    write_jsonl(pred, rows)
    score = multilingual_panel.score(panel, pred)
    (root / "mlx-diag.score.json").write_text(json.dumps(score))
    return root


class MlxPairedTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.panel = self.root / "panel"
        self.gold = build_panel(self.panel)
        self.patch = mock.patch.object(
            mlx_paired, "FROZEN_GOLD_SHA256", sha_file(self.panel / "gold.jsonl")
        )
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def run_pair(self, left, right, name="out.json"):
        out = self.root / name
        with contextlib.redirect_stdout(io.StringIO()):
            mlx_paired.main(
                [
                    "--left",
                    str(left),
                    "--right",
                    str(right),
                    "--panel",
                    str(self.panel),
                    "--left-name",
                    "L",
                    "--right-name",
                    "R",
                    "--output",
                    str(out),
                    "--draws",
                    "300",
                ]
            )
        return json.loads(out.read_text())

    def test_self_pair_is_zero(self):
        run = make_run(
            self.root / "a", self.panel, self.gold, {"choice-de-0", "noul-ko-1"}
        )
        out = self.run_pair(run, run)
        self.assertEqual(out["overall"]["delta"], 0)
        self.assertEqual(out["overall"]["ci95"], {"low": 0, "high": 0})
        self.assertTrue(out["not_significantly_below"])
        self.assertEqual(
            out["left"]["predictions_sha256"], out["right"]["predictions_sha256"]
        )

    def test_delta_excludes_score_and_macros_over_languages(self):
        left = make_run(self.root / "l", self.panel, self.gold, {"score-en-0"})
        # right: choice/de 2 of 4 wrong, noul/ko 4 of 4 wrong; score fully right
        wrong = {"choice-de-0", "choice-de-1", *(f"noul-ko-{k}" for k in range(4))}
        right = make_run(self.root / "r", self.panel, self.gold, wrong)
        out = self.run_pair(left, right)
        # right overall = mean(mean(1, .5), mean(1, 0)) = .625; left = 1
        self.assertAlmostEqual(out["overall"]["delta"], 0.375)
        self.assertAlmostEqual(out["by_type"]["choice"]["delta"], 0.25)
        self.assertAlmostEqual(out["by_type"]["noul"]["delta"], 0.5)
        self.assertAlmostEqual(out["by_language"]["ko"]["delta"], 1.0)
        self.assertAlmostEqual(out["by_language"]["en"]["delta"], 0.0)
        self.assertAlmostEqual(out["non_english"]["choice"]["delta"], 0.5)
        self.assertEqual(out["non_english"]["noul"]["items"], PER_CELL)
        self.assertEqual(
            out["units"], {"choice/de": 4, "choice/en": 4, "noul/en": 4, "noul/ko": 4}
        )
        self.assertTrue(out["not_significantly_below"])
        reverse = self.run_pair(right, left, "rev.json")
        self.assertAlmostEqual(reverse["overall"]["delta"], -0.375)
        self.assertLess(reverse["overall"]["ci95"]["high"], 0)
        self.assertFalse(reverse["not_significantly_below"])

    def test_self_check_fails_loudly(self):
        run = make_run(self.root / "a", self.panel, self.gold, set())
        stored = json.loads((run / "mlx-diag.score.json").read_text())
        stored["by_type"]["noul"]["languages"]["ko"]["correct"] -= 1
        (run / "mlx-diag.score.json").write_text(json.dumps(stored))
        with self.assertRaisesRegex(ValueError, "noul"):
            self.run_pair(run, run)

    def test_changed_predictions_fail(self):
        run = make_run(self.root / "a", self.panel, self.gold, set())
        pred = run / "output" / "mlx-diag.predictions.jsonl"
        pred.write_text(pred.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "differ from the stored score"):
            self.run_pair(run, run)

    def test_unfrozen_gold_is_rejected(self):
        run = make_run(self.root / "a", self.panel, self.gold, set())
        with mock.patch.object(mlx_paired, "FROZEN_GOLD_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "frozen"):
                self.run_pair(run, run)


if __name__ == "__main__":
    unittest.main()
