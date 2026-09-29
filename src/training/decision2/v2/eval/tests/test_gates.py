from __future__ import annotations

import contextlib
import hashlib
import io
import json
import random
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from v2.eval import gates, panels
from v2.eval.sealed import score as c1score
from v2.eval.sealed.tests.test_score import CHOICE, LEVELS, NOUL
from v2.eval.sealed.tests.test_score import item as c1_item
from v2.eval.sealed.tests.test_score import prediction as c1_prediction


def item(i: int, kind: str, value) -> dict:
    if kind == "score":
        question = {"type": "score", "instructions": "?", "criteria": ["a", "b", "c"]}
        truth = {"type": "score", "value": value, "semantic_value": value}
    else:
        question = {
            "type": "noul",
            "instructions": "?",
            "criteria": {"true": "t", "false": "f"},
        }
        truth = {"type": "noul", "value": value, "semantic_value": value}
    return {
        "id": f"i{i}",
        "questions": {"decision": question},
        "gold": {"decision": truth},
    }


class GatesTest(unittest.TestCase):
    def test_constant_score_is_collapsed_and_good_noul_is_ok(self):
        gold = [item(i, "score", i % 3) for i in range(60)] + [
            item(100 + i, "noul", bool(i % 2)) for i in range(60)
        ]
        preds = {}
        for row in gold:
            q = row["questions"]["decision"]
            if q["type"] == "score":
                answer = {"type": "score", "score": 0}
            else:
                answer = {
                    "type": "noul",
                    "noul": 0.9 if row["gold"]["decision"]["value"] else 0.1,
                }
            preds[row["id"]] = {"answers": {"decision": answer}}
        out = gates.type_summary(gold, preds)
        self.assertTrue(out["score"]["verdict"].startswith("COLLAPSED"))
        self.assertEqual(
            out["score"]["recall_by_level"], {"0": 1.0, "1": 0.0, "2": 0.0}
        )
        self.assertEqual(out["noul"]["verdict"], "OK")
        self.assertAlmostEqual(out["noul"]["accuracy"], 1.0)

    def test_wilson(self):
        low, high = gates.wilson(50, 100)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_mcnemar_exact(self):
        self.assertEqual(gates.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(gates.mcnemar_exact(3, 3), 1.0)
        self.assertAlmostEqual(gates.mcnemar_exact(2, 7), 92 / 512)
        self.assertAlmostEqual(gates.mcnemar_exact(7, 2), 92 / 512)
        self.assertLess(gates.mcnemar_exact(6, 27), 0.001)

    def test_public_guard_flags_only_significant_losses(self):
        targets = {
            f"p{i}": {
                "tier": "hard" if i < 40 else "easy",
                "family": "long" if i < 10 else "short",
                "task_type": "choice",
            }
            for i in range(60)
        }

        def outcomes(wrong: set[int]) -> dict:
            return {
                key: {"tier": row["tier"], "correct": int(key[1:]) not in wrong}
                for key, row in targets.items()
            }

        base = outcomes(set())
        big = gates.public_guard(outcomes(set(range(12))), base, targets, 200, 1)
        self.assertEqual(big["delta"], -12)
        self.assertEqual(big["discordant"], {"left_only": 0, "right_only": 12})
        self.assertEqual(big["verdict"], "REGRESSION")
        self.assertEqual(big["tiers"]["hard"], {"items": 40, "left": 28, "right": 40})
        self.assertEqual(big["families"]["long"]["left"], 0)
        small = gates.public_guard(outcomes({0, 1}), base, targets, 200, 1)
        self.assertEqual(small["verdict"], "OK")
        gain = gates.public_guard(base, outcomes(set(range(12))), targets, 200, 1)
        self.assertEqual(gain["verdict"], "OK")
        with self.assertRaises(ValueError):
            gates.public_guard(base, {"p0": base["p0"]}, targets, 200, 1)


def wrong_value(kind: str, truth):
    if kind == "choice":
        return {"x": "y", "y": "x"}[truth]
    if kind == "noul":
        return not truth
    return (truth + 1) % 3


def c1_gold() -> list[dict]:
    return (
        [c1_item(i, "a/choice", CHOICE, "x" if i % 2 else "y") for i in range(40)]
        + [
            c1_item(i, "b/noul", NOUL, bool(i % 2), "hallutruthqa", "ar")
            for i in range(40)
        ]
        + [c1_item(i, "c/score", LEVELS, i % 3) for i in range(42)]
    )


def c1_answers(gold: list[dict], wrong: set[int]) -> dict[str, dict]:
    """Predictions that are right except on the items at the given positions."""
    out = {}
    for index, row in enumerate(gold):
        truth = row["gold"]["decision"]["value"]
        if index in wrong:
            truth = wrong_value(row["questions"]["decision"]["type"], truth)
        out[row["id"]] = c1_prediction(row, truth)
    return out


class C1GuardTest(unittest.TestCase):
    def test_bootstrap_p_is_the_interval_inverted(self):
        rng = random.Random(3)
        for n in (5000, 400, 200):
            for shift in (-0.3, -0.18, -0.05, 0.0, 0.05, 0.2, 0.35):
                draws = [rng.gauss(shift, 0.1) for _ in range(n)]
                p = gates.bootstrap_p(draws)
                low, high = c1score.interval(draws)
                with self.subTest(n=n, shift=shift):
                    self.assertEqual(p < 0.05, high < 0 or low > 0)
                    self.assertGreaterEqual(p, 0.0)
                    self.assertLessEqual(p, 1.0)
        self.assertEqual(gates.bootstrap_p([0.0] * 50), 1.0)
        self.assertEqual(gates.bootstrap_p([-1.0] * 50), 0.0)
        self.assertEqual(gates.bootstrap_p([2.0] * 50), 0.0)
        self.assertAlmostEqual(gates.bootstrap_p([-1.0] * 25 + [1.0] * 25), 48 / 49)

    def test_guard_reproduces_the_paired_comparison(self):
        gold = c1_gold()
        left = c1_answers(gold, set(range(0, 122, 3)))
        right = c1_answers(gold, set(range(1, 122, 4)))
        guard = gates.c1_guard(gold, left, right, 300, c1score.SEED)
        event = c1score.paired(gold, left, right, 300, c1score.SEED)
        self.assertEqual(guard["delta"], event["delta"])
        self.assertEqual(guard["ci95"], event["ci95"])
        for kind, cell in event["by_type"].items():
            self.assertEqual(guard["by_type"][kind]["delta"], cell["delta"])
            self.assertEqual(guard["by_type"][kind]["ci95"], cell["ci95"])
        self.assertEqual(guard["left"]["c1"] - guard["right"]["c1"], guard["delta"])
        self.assertEqual(guard["left"]["items"], 122)

    def test_guard_flags_only_significant_losses(self):
        gold = c1_gold()
        perfect = c1_answers(gold, set())
        big = gates.c1_guard(
            gold, c1_answers(gold, set(range(0, 122, 2))), perfect, 400, 1
        )
        self.assertLess(big["delta"], 0)
        self.assertLess(big["ci95"][1], 0)
        self.assertLess(big["p"], 0.05)
        self.assertEqual(big["verdict"], "REGRESSION")
        small = gates.c1_guard(gold, c1_answers(gold, {5, 50}), perfect, 400, 1)
        self.assertLess(small["delta"], 0)
        self.assertGreaterEqual(small["p"], 0.05)
        self.assertEqual(small["verdict"], "PASS")
        gain = gates.c1_guard(
            gold, perfect, c1_answers(gold, set(range(0, 122, 2))), 400, 1
        )
        self.assertGreater(gain["delta"], 0)
        self.assertLess(gain["p"], 0.05)
        self.assertEqual(gain["verdict"], "PASS")
        self.assertEqual(sorted(big["by_type"]), ["choice", "noul", "score"])

    def write(self, path: Path, rows) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def c1_case(self, root: Path) -> dict:
        gold = c1_gold()
        paths = {
            "gold": self.write(root / "gold.jsonl", gold),
            "prompts": self.write(
                root / "prompts.jsonl",
                [
                    {"id": g["id"], "state": g["state"], "questions": g["questions"]}
                    for g in gold
                ],
            ),
            "retired": root / "retired.json",
        }
        paths["retired"].write_text(
            json.dumps(
                {
                    "schema": "dev2-c1-retired/1",
                    "version": "v1.2",
                    "candidates": ["a/choice|0", "a/choice|1"],
                    "protected_rows": [],
                }
            )
        )
        for name, wrong in (("succ", set(range(0, 122, 2))), ("curr", set())):
            run = root / name
            predictions = self.write(
                run / "output" / "sealed-c1.predictions.jsonl",
                c1_answers(gold, wrong).values(),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                c1score.seal(
                    Namespace(
                        prompts=paths["prompts"],
                        predictions=predictions,
                        output=run / "SEAL-C1.json",
                        post_key=name == "succ",
                    )
                )
            paths[name] = run
        return paths

    @contextlib.contextmanager
    def pinned(self, paths: dict):
        with (
            mock.patch.dict(
                panels.SEALED["sealed-c1"],
                {
                    "prompts_sha256": gates.sha_file(paths["prompts"]),
                    "gold_sha256": gates.sha_file(paths["gold"]),
                },
            ),
            mock.patch.object(
                c1score, "POSTKEY_RETIRED_SHA256", gates.sha_file(paths["retired"])
            ),
            mock.patch.object(gates, "PAIRED_REPLICATES", 300),
        ):
            yield

    def c1_argv(self, paths: dict, output: Path, left="succ", right="curr") -> list:
        return [
            "c1",
            "--left",
            paths[left],
            "--right",
            paths[right],
            "--left-name",
            "successor",
            "--right-name",
            "current",
            "--gold",
            paths["gold"],
            "--retired",
            paths["retired"],
            "--retired-sha",
            gates.sha_file(paths["retired"]),
            "--output",
            output,
        ]

    def test_cli_verdicts_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.c1_case(root)
            with self.pinned(paths):
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    gates.main(
                        [str(a) for a in self.c1_argv(paths, root / "gate.json")]
                    )
                printed = json.loads(out.getvalue())
                with contextlib.redirect_stdout(io.StringIO()):
                    gates.main(
                        [
                            str(a)
                            for a in self.c1_argv(
                                paths, root / "reverse.json", "curr", "succ"
                            )
                        ]
                    )
            result = json.loads((root / "gate.json").read_text())
            self.assertEqual(list(printed), ["delta", "ci95", "p", "verdict"])
            self.assertEqual(printed["verdict"], "REGRESSION")
            self.assertEqual(result["verdict"], "REGRESSION")
            self.assertEqual(result["schema"], "dev2-gate-c1/1")
            self.assertEqual(
                result["label"],
                "JevArena-C1 v1.2, post-key (not an independent validation)",
            )
            self.assertIn("never a selection criterion", result["use"])
            self.assertEqual(result["item_set"]["dropped_items"], 2)
            self.assertEqual(result["left"]["items"], 120)
            self.assertEqual(
                result["runs"]["left"]["seal_label"], c1score.POSTKEY_LABEL
            )
            self.assertEqual(result["runs"]["right"]["seal_label"], c1score.LABEL)
            self.assertEqual(
                result["runs"]["right"]["seal_sha256"],
                hashlib.sha256(
                    (paths["curr"] / "SEAL-C1.json").read_bytes()
                ).hexdigest(),
            )
            self.assertNotIn("tasks", result["left"])
            reverse = json.loads((root / "reverse.json").read_text())
            self.assertEqual(reverse["verdict"], "PASS")
            self.assertAlmostEqual(reverse["delta"], -result["delta"])

    def test_cli_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.c1_case(root)
            output = root / "gate.json"
            with self.pinned(paths):
                with self.assertRaises(ValueError):
                    gates.main(
                        [str(a) for a in self.c1_argv(paths, paths["succ"] / "g.json")]
                    )
                argv = self.c1_argv(paths, output)
                with mock.patch.dict(
                    panels.SEALED["sealed-c1"], {"gold_sha256": "0" * 64}
                ):
                    with self.assertRaises(ValueError):
                        gates.main([str(a) for a in argv])
                with mock.patch.dict(
                    panels.SEALED["sealed-c1"], {"prompts_sha256": "0" * 64}
                ):
                    with self.assertRaises(ValueError):
                        gates.main([str(a) for a in argv])
                with mock.patch.object(c1score, "POSTKEY_RETIRED_SHA256", "0" * 64):
                    with self.assertRaises(ValueError):
                        gates.main([str(a) for a in argv])
                seal = paths["curr"] / "SEAL-C1.json"
                original = seal.read_text()
                seal.write_text(json.dumps({**json.loads(original), "missing": 1}))
                with self.assertRaises(ValueError):
                    gates.main([str(a) for a in argv])
                seal.write_text(original)
                predictions = paths["curr"] / "output" / "sealed-c1.predictions.jsonl"
                predictions.write_text(predictions.read_text() + "\n")
                with self.assertRaises(ValueError):
                    gates.main([str(a) for a in argv])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
