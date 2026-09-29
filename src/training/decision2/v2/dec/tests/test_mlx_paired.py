"""CPU tests for v2.dec.mlx_paired (synthetic mlx-diag panels)."""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from v2.dec import mlx_paired  # noqa: E402
from v2.eval.multilingual_panel import CHOICE_Q, NOUL_Q, SCORE_Q  # noqa: E402

LANGS = ("en", "de", "ja")
QUESTION = {"choice": CHOICE_Q, "noul": NOUL_Q, "score": SCORE_Q}


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def gold_value(kind: str, i: int):
    if kind == "choice":
        return sorted(CHOICE_Q["criteria"])[i % 4]
    if kind == "noul":
        return i % 2 == 0
    return i % 3


def answer(kind: str, value, right: bool):
    if kind == "choice":
        labels = sorted(CHOICE_Q["criteria"])
        return {
            "choice": (
                value if right else labels[(labels.index(value) + 1) % len(labels)]
            )
        }
    if kind == "noul":
        return {"noul": 0.9 if value == right else 0.1}
    return {"score": value if right else (value + 1) % 3}


def make_panel(root: Path, per_cell: int = 12) -> list[dict]:
    gold, prompts = [], []
    for kind in ("choice", "noul", "score"):
        for lang in LANGS:
            for i in range(per_cell):
                item = f"mlx-{kind}-{lang}-{i}"
                gold.append(
                    {
                        "id": item,
                        "type": kind,
                        "language": lang,
                        "value": gold_value(kind, i),
                        "input_sha256": f"sha-{item}",
                        "source": kind,
                        "source_id": str(i),
                    }
                )
                prompts.append({"id": item, "questions": {"q": QUESTION[kind]}})
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "gold.jsonl", gold)
    write_jsonl(root / "prompts.jsonl", prompts)
    return gold


def predictions(
    path: Path, gold: list[dict], p_right: dict[str, float], seed: int
) -> None:
    rng = random.Random(seed)
    rows = []
    for g in gold:
        right = rng.random() < p_right[g["type"]]
        rows.append(
            {
                "id": g["id"],
                "source_input_sha256": g["input_sha256"],
                "answers": {"q": answer(g["type"], g["value"], right)},
            }
        )
    write_jsonl(path, rows)


class MlxPairedTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.panel = self.root / "panel"
        self.gold = make_panel(self.panel)

    def tearDown(self):
        self.tmp.cleanup()

    def test_identical_files_give_zero_delta_and_pass(self):
        p = self.root / "a.jsonl"
        predictions(p, self.gold, {"choice": 0.7, "noul": 0.6, "score": 0.5}, 1)
        out = mlx_paired.compare(self.panel, p, p, replicates=200)
        for key in ("card_eligible", "full"):
            self.assertEqual(out[key]["delta"], 0.0)
            self.assertEqual(out[key]["ci95"], {"low": 0.0, "high": 0.0})
        self.assertTrue(out["rule4_pass"])
        self.assertEqual(out["problems"], [])
        self.assertTrue(
            all(c["equal"] for c in out["reproduces_frozen_score"].values())
        )
        self.assertFalse(out["frozen_panel"])

    def test_points_match_frozen_scorer_and_strata(self):
        a, b = self.root / "a.jsonl", self.root / "b.jsonl"
        predictions(a, self.gold, {"choice": 0.9, "noul": 0.8, "score": 0.3}, 2)
        predictions(b, self.gold, {"choice": 0.5, "noul": 0.5, "score": 0.9}, 3)
        out = mlx_paired.compare(self.panel, a, b, replicates=300)
        fa = mlx_paired.frozen_score(self.panel, a)
        fb = mlx_paired.frozen_score(self.panel, b)
        self.assertAlmostEqual(
            out["full"]["candidate"], fa["type_macro_accuracy"], places=12
        )
        self.assertAlmostEqual(
            out["full"]["reference"], fb["type_macro_accuracy"], places=12
        )
        card_a = (
            sum(
                sum(fa["by_type"][t]["languages"][lang]["accuracy"] for lang in LANGS)
                / 3
                for t in ("choice", "noul")
            )
            / 2
        )
        self.assertAlmostEqual(out["card_eligible"]["candidate"], card_a, places=12)
        self.assertEqual(out["card_eligible"]["strata"], 6)
        self.assertEqual(out["card_eligible"]["items"], 72)
        self.assertEqual(out["full"]["strata"], 9)
        self.assertEqual(set(out["per_type_language"]), {"choice", "noul", "score"})
        cell = out["per_type_language"]["noul"]["de"]
        self.assertEqual(cell["n"], 12)
        self.assertAlmostEqual(
            cell["candidate"],
            fa["by_type"]["noul"]["languages"]["de"]["accuracy"],
            places=12,
        )
        # The candidate is much better on Choice / Noul and much worse on Score.
        self.assertGreater(out["card_eligible"]["ci95"]["low"], 0)
        self.assertLess(out["per_type"]["score"]["ci95"]["high"], 0)
        self.assertTrue(out["rule4_pass"])
        low, high = (
            out["card_eligible"]["ci95"]["low"],
            out["card_eligible"]["ci95"]["high"],
        )
        self.assertLessEqual(low, out["card_eligible"]["delta"])
        self.assertGreaterEqual(high, out["card_eligible"]["delta"])

    def test_worse_candidate_fails_rule4_and_is_deterministic(self):
        a, b = self.root / "a.jsonl", self.root / "b.jsonl"
        predictions(a, self.gold, {"choice": 0.2, "noul": 0.2, "score": 0.9}, 4)
        predictions(b, self.gold, {"choice": 0.9, "noul": 0.9, "score": 0.2}, 5)
        one = mlx_paired.compare(self.panel, a, b, replicates=300)
        two = mlx_paired.compare(self.panel, a, b, replicates=300)
        self.assertLess(one["card_eligible"]["ci95"]["high"], 0)
        self.assertFalse(one["rule4_pass"])
        self.assertEqual(one["card_eligible"], two["card_eligible"])
        self.assertGreater(one["full"]["delta"], one["card_eligible"]["delta"])

    def test_missing_prediction_counts_as_wrong(self):
        a, b = self.root / "a.jsonl", self.root / "b.jsonl"
        predictions(b, self.gold, {"choice": 1.0, "noul": 1.0, "score": 1.0}, 6)
        rows = [json.loads(line) for line in b.read_text().splitlines()]
        write_jsonl(a, [r for r in rows if r["id"] != "mlx-choice-en-0"])
        out = mlx_paired.compare(self.panel, a, b, replicates=100)
        self.assertAlmostEqual(
            out["per_type_language"]["choice"]["en"]["delta"], -1 / 12
        )
        self.assertEqual(out["problems"], [])

    def test_cli_writes_exclusive_output(self):
        a = self.root / "a.jsonl"
        predictions(a, self.gold, {"choice": 0.7, "noul": 0.6, "score": 0.5}, 7)
        out = self.root / "out.json"
        args = [
            "--panel",
            str(self.panel),
            "--candidate",
            str(a),
            "--reference",
            str(a),
            "--replicates",
            "50",
            "--output",
            str(out),
        ]
        self.assertEqual(mlx_paired.main(args), 0)
        doc = json.loads(out.read_text())
        self.assertEqual(doc["schema"], mlx_paired.SCHEMA)
        self.assertEqual(doc["bootstrap"]["replicates"], 50)
        self.assertNotIn("mlx-choice-en-0", out.read_text())
        with self.assertRaises(FileExistsError):
            mlx_paired.main(args)

    def test_defaults_are_the_preregistered_ones(self):
        self.assertEqual(mlx_paired.PAIRED_REPLICATES, 5000)
        self.assertEqual(mlx_paired.PAIRED_SEED, 20260927)
        self.assertEqual(mlx_paired.CARD_TYPES, ("choice", "noul"))


if __name__ == "__main__":
    unittest.main()
