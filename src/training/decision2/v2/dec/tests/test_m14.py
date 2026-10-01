"""CPU tests for decoder M14: the per-row loss weights of the upweighted arms (m14_weights)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, HERE / "ops" / "m14" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def row(rid: str, kind: str) -> bytes:
    return json.dumps({"id": rid, "task_type": kind, "family": "f"}).encode() + b"\n"


def ids(rid: str, block: str) -> bytes:
    return json.dumps({"id": rid, "block": block, "tokens": 3}).encode() + b"\n"


class WeightsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.w = load("m14_weights")
        self.train = [
            row("a", "choice"),
            row("b", "score"),
            row("ib1", "noul"),
            row("ib1~c2", "noul"),
        ]
        self.ids = [
            ids("a", "base"),
            ids("b", "base"),
            ids("ib1", "ib1"),
            ids("ib1~c2", "ib1"),
        ]
        self.released_sha = hashlib.sha256(b"".join(self.train[:2])).hexdigest()
        self.typed = self.w.parse_weights("choice=1.5,noul=1.5,score=2")

    def test_released_rows_take_their_type_weight(self) -> None:
        records, report = self.w.build(
            self.train, self.ids, 2, self.released_sha, self.typed, 1.0
        )
        self.assertEqual([r["weight"] for r in records], [1.5, 2.0, 1.0, 1.0])
        self.assertEqual([r["id"] for r in records], ["a", "b", "ib1", "ib1~c2"])
        self.assertAlmostEqual(report["released_weight_share"], 3.5 / 5.5)
        self.assertAlmostEqual(report["released_row_share"], 0.5)
        self.assertAlmostEqual(report["type_weight_share"]["score"], 2.0 / 5.5)
        self.assertEqual(
            report["by_block_type"]["ib"]["noul"], {"rows": 2, "weight": 2.0}
        )

    def test_rejects_inconsistent_inputs(self) -> None:
        with self.assertRaises(ValueError):
            self.w.build(self.train, self.ids, 3, self.released_sha, self.typed, 1.0)
        with self.assertRaises(ValueError):
            self.w.build(self.train, self.ids, 2, "0" * 64, self.typed, 1.0)
        with self.assertRaises(ValueError):
            self.w.build(
                self.train, self.ids[:3], 2, self.released_sha, self.typed, 1.0
            )
        swapped = [self.ids[0], self.ids[2], self.ids[1], self.ids[3]]
        with self.assertRaises(ValueError):
            self.w.build(self.train, swapped, 2, self.released_sha, self.typed, 1.0)
        late = self.ids[:2] + [ids("ib1", "ib1"), ids("ib1~c2", "base")]
        with self.assertRaises(ValueError):
            self.w.build(self.train, late, 2, self.released_sha, self.typed, 1.0)
        for spec in (
            "choice=1.5,noul=1.5",
            "choice=1,noul=1,score=0",
            "choice=1,noul=1,score=1,x=2",
        ):
            with self.assertRaises(ValueError):
                self.w.parse_weights(spec)


def paired(delta, lo, hi, right=60.0):
    return {
        "point": {
            "delta": {"score": delta, "T": 0.01, "H": 0.01},
            "left": {"score": right + delta},
            "right": {"score": right},
        },
        "ci95": {"low": lo, "high": hi},
        "axis_ci95": {
            "H": {"delta": {"low": -0.02, "high": 0.03}},
            "T": {"delta": {"low": 0, "high": 0.02}},
        },
    }


def reduced_pair(lo):
    return {
        "v3": {
            "reduced": {
                "ci95": {"low": lo, "high": lo + 3},
                "axis_ci95": {"H": {"delta": {"low": -0.05, "high": 0.03}}},
            }
        }
    }


def mlx(hi):
    return {
        "reference_name": "ref",
        "card_eligible": {"delta": 0.002, "ci95": {"low": -0.004, "high": hi}},
        "full": {"delta": 0.0, "ci95": {"low": -0.01, "high": 0.01}},
        "frozen_panel": True,
        "problems": [],
    }


class TwoBarSuccessorTest(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self.succ = load("m14_successor")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make(self, bar_b=(1.5, 0.3, 3.0), mlx_b_hi=0.01, pub_b_right="bar-b"):
        run = self.root / "m14-4b-LHA10UP"
        run.mkdir()
        (run / "REPORT.json").write_text(
            json.dumps(
                {
                    "v3": {"score": 68.8, "T": 0.8, "H": 0.5},
                    "panels": {"public231": {"correct": 172}},
                }
            )
        )
        (run / "M6-RECEIPT.json").write_text(
            json.dumps(
                {
                    "name": run.name,
                    "point": "4b-LHA10UP",
                    "revision": "cd" * 32,
                    "selection": {"slot": 1},
                }
            )
        )
        for name, args, right in (
            ("bar-lh", (1.6, 0.4, 3.1), 67.345),
            ("bar-b", bar_b, 67.3),
            ("adopted-1.0", (12.0, 6.0, 15.0), 56.5),
            ("decider4b", (8.0, 3.0, 12.0), 60.0),
            ("jet62", (9.0, 4.0, 13.0), 59.0),
        ):
            (run / f"PAIRED-vs-{name}.json").write_text(
                json.dumps(paired(*args, right=right))
            )
        overlap = {
            "pairs": {
                f"{run.name} - bar-lh": reduced_pair(0.3),
                f"{run.name} - bar-b": reduced_pair(0.2),
                f"{run.name} - adopted-1.0": reduced_pair(4.0),
                f"{run.name} - decider4b": reduced_pair(2.0),
                f"{run.name} - jet62": reduced_pair(2.0),
            },
            "models": {run.name: {"reduced": {"v3": 68.0}}},
            "reproduction": [{"match": True}],
            "problems": [],
        }

        exposure = self.root / "exposure.json"
        exposure.write_text(json.dumps({"groups": []}))

        def public(right):
            return {
                "runs": {"left": str(run), "right": "/runs/bar"},
                "names": {"left": run.name, "right": right},
                "verdict": "OK",
                "delta": 1,
                "mcnemar_exact_p": 0.8,
                "left_correct": 172,
                "right_correct": 171,
            }

        inputs = {
            "types": {
                "types": {t: {"verdict": "OK"} for t in ("choice", "noul", "score")}
            },
            "mlx": {"bar-lh": mlx(0.01), "bar-b": mlx(mlx_b_hi)},
            "overlap": overlap,
            "exposures": [(str(exposure), {"groups": []})],
            "public": {"bar-lh": public("bar-lh"), "bar-b": public(pub_b_right)},
            "c1": None,
        }
        return run, inputs

    def test_both_bars_pass_and_bar_restored(self) -> None:
        run, inputs = self.make()
        out = self.succ.two_bar("4b", ["bar-lh", "bar-b"], run, inputs)
        self.assertEqual(out["status_1_7"], "PASS")
        self.assertEqual(out["status"], "INCOMPLETE")
        self.assertEqual(out["items"]["1_v3_vs_bar"]["bar-lh"]["right_v3"], 67.345)
        self.assertEqual(self.succ.m6.BAR, "bar-t1")

    def test_parity_bar_binds(self) -> None:
        run, inputs = self.make(bar_b=(0.9, -0.2, 2.5))
        out = self.succ.two_bar("4b", ["bar-lh", "bar-b"], run, inputs)
        self.assertIs(out["items"]["1_v3_vs_bar"]["pass"], False)
        self.assertEqual(out["per_bar"]["bar-lh"]["status_1_7"], "PASS")
        self.assertEqual(out["status_1_7"], "FAIL")

    def test_parity_mlx_binds(self) -> None:
        run, inputs = self.make(mlx_b_hi=-0.001)
        out = self.succ.two_bar("4b", ["bar-lh", "bar-b"], run, inputs)
        self.assertIs(out["items"]["4_mlx_card_eligible"]["pass"], False)

    def test_public_guard_must_pair_with_its_bar(self) -> None:
        run, inputs = self.make(pub_b_right="bar-lh")
        out = self.succ.two_bar("4b", ["bar-lh", "bar-b"], run, inputs)
        self.assertIsNone(out["items"]["7_jevbench_public231"]["pass"])

    def test_cli_requires_two_bars(self) -> None:
        with self.assertRaises(SystemExit):
            self.succ.main(
                [
                    "--tier",
                    "4b",
                    "--run",
                    "r",
                    "--types",
                    "t",
                    "--bar",
                    "bar-lh=a,b",
                    "--overlap",
                    "o",
                    "--output",
                    "x",
                ]
            )


if __name__ == "__main__":
    unittest.main()
