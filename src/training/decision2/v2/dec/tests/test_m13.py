"""CPU tests for decoder M13: the family-upweighted TRAIN (m13_data)."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, HERE / "ops" / "m13" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def line(rid: str, family: str) -> bytes:
    return json.dumps({"id": rid, "family": family, "label": "1"}).encode() + b"\n"


class UpweightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = load("m13_data")
        self.arm = [
            line("a1", "gate"),
            line("a2", "other"),
            line("a3", "gate"),
            line("ib1~c2", "ib_family"),
        ]
        self.released = {"a1", "a2", "a3"}

    def test_keeps_arm_then_appends_copies(self) -> None:
        lines, info = self.data.upweighted(self.arm, self.released, {"gate"}, 3)
        self.assertEqual(lines[:4], self.arm)
        ids = [json.loads(x)["id"] for x in lines[4:]]
        self.assertEqual(ids, ["a1~a2", "a3~a2", "a1~a3", "a3~a3"])
        self.assertEqual(info["upweighted_rows"], 2)
        self.assertEqual(info["added_rows"], 4)
        copy, original = json.loads(lines[4]), json.loads(self.arm[0])
        self.assertEqual(copy.pop("id"), "a1~a2")
        original.pop("id")
        self.assertEqual(copy, original)

    def test_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            self.data.upweighted(self.arm, self.released, {"gate"}, 1)
        with self.assertRaises(ValueError):
            self.data.upweighted(self.arm, self.released, {"gate", "absent"}, 2)
        with self.assertRaises(ValueError):
            self.data.upweighted(self.arm, {"a1", "a2"}, {"gate"}, 2)
        with self.assertRaises(ValueError):
            self.data.upweighted(self.arm + [self.arm[0]], self.released, {"gate"}, 2)


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

        self.succ = load("m13_successor")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make(self, bar_f=(1.5, 0.3, 3.0), mlx_f_hi=0.01, pub_f_right="bar-f"):
        run = self.root / "m13-4b-LHA10SD"
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
                    "point": "4b-LHA10SD",
                    "revision": "cd" * 32,
                    "selection": {"slot": 1},
                }
            )
        )
        for name, args, right in (
            ("bar-lh", (1.6, 0.4, 3.1), 67.345),
            ("bar-f", bar_f, 67.3),
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
                f"{run.name} - bar-f": reduced_pair(0.2),
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
            "mlx": {"bar-lh": mlx(0.01), "bar-f": mlx(mlx_f_hi)},
            "overlap": overlap,
            "exposures": [(str(exposure), {"groups": []})],
            "public": {"bar-lh": public("bar-lh"), "bar-f": public(pub_f_right)},
            "c1": None,
        }
        return run, inputs

    def test_both_bars_pass_and_bar_restored(self) -> None:
        run, inputs = self.make()
        out = self.succ.two_bar("4b", ["bar-lh", "bar-f"], run, inputs)
        self.assertEqual(out["status_1_7"], "PASS")
        self.assertEqual(out["status"], "INCOMPLETE")
        self.assertEqual(out["items"]["1_v3_vs_bar"]["bar-lh"]["right_v3"], 67.345)
        self.assertEqual(self.succ.m6.BAR, "bar-t1")

    def test_parity_bar_binds(self) -> None:
        run, inputs = self.make(bar_f=(0.9, -0.2, 2.5))
        out = self.succ.two_bar("4b", ["bar-lh", "bar-f"], run, inputs)
        self.assertIs(out["items"]["1_v3_vs_bar"]["pass"], False)
        self.assertEqual(out["per_bar"]["bar-lh"]["status_1_7"], "PASS")
        self.assertEqual(out["status_1_7"], "FAIL")

    def test_parity_mlx_binds(self) -> None:
        run, inputs = self.make(mlx_f_hi=-0.001)
        out = self.succ.two_bar("4b", ["bar-lh", "bar-f"], run, inputs)
        self.assertIs(out["items"]["4_mlx_card_eligible"]["pass"], False)

    def test_public_guard_must_pair_with_its_bar(self) -> None:
        run, inputs = self.make(pub_f_right="bar-lh")
        out = self.succ.two_bar("4b", ["bar-lh", "bar-f"], run, inputs)
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
