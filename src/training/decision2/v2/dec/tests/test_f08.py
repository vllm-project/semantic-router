"""CPU tests for the decoder 0.8B fast track: ops/f08/f08_successor.py (two bars) and the shell wrappers' syntax."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

OPS = Path(__file__).resolve().parents[1] / "ops" / "f08"
_spec = importlib.util.spec_from_file_location(
    "f08_successor", OPS / "f08_successor.py"
)
f08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f08)


def paired(delta, lo, hi, h_hi=0.03, right=50.0):
    return {
        "point": {
            "delta": {"score": delta, "T": 0.01, "H": 0.01},
            "left": {"score": right + delta},
            "right": {"score": right},
        },
        "ci95": {"low": lo, "high": hi},
        "axis_ci95": {
            "H": {"delta": {"low": -0.02, "high": h_hi}},
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


class TwoBarTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make(
        self,
        bar_e=(1.5, 0.3, 3.0),
        bar_t1=(1.6, 0.4, 3.1),
        mlx_e_hi=0.01,
        red_e=0.2,
        pub_e_right="bar-e",
    ):
        run = self.root / "f08-08b-RA"
        run.mkdir()
        (run / "REPORT.json").write_text(
            json.dumps(
                {
                    "v3": {"score": 51.8, "T": 0.6, "H": 0.45},
                    "panels": {"public231": {"correct": 160}},
                }
            )
        )
        (run / "M6-RECEIPT.json").write_text(
            json.dumps(
                {
                    "name": "f08-08b-RA",
                    "point": "08b-RA",
                    "revision": "cd" * 32,
                    "selection": {"slot": 1},
                }
            )
        )
        (run / "PAIRED-vs-bar-t1.json").write_text(
            json.dumps(paired(*bar_t1, right=50.236))
        )
        (run / "PAIRED-vs-bar-e.json").write_text(
            json.dumps(paired(*bar_e, right=50.263))
        )
        (run / "PAIRED-vs-adopted-1.0.json").write_text(
            json.dumps(paired(9.0, 5.0, 13.0, right=42.547))
        )
        for i, n in enumerate(("intern", "kev")):
            (run / f"PAIRED-vs-{n}.json").write_text(
                json.dumps(paired(8.0, 3.0, 12.0, right=43.5 - i))
            )
        name = run.name
        overlap = {
            "pairs": {
                f"{name} - bar-t1": reduced_pair(0.3),
                f"{name} - bar-e": reduced_pair(red_e),
                f"{name} - adopted-1.0": reduced_pair(4.0),
            },
            "models": {name: {"reduced": {"v3": 51.0}}},
            "reproduction": [{"match": True}],
            "problems": [],
        }
        exposure = self.root / "exposure.json"
        exposure.write_text(json.dumps({"groups": []}))

        def public(right):
            return {
                "runs": {"left": str(run), "right": "/runs/bar"},
                "names": {"left": name, "right": right},
                "verdict": "OK",
                "delta": 3,
                "mcnemar_exact_p": 0.4,
                "left_correct": 160,
                "right_correct": 157,
            }

        types = {"types": {t: {"verdict": "OK"} for t in ("choice", "noul", "score")}}
        inputs = {
            "types": types,
            "mlx": {"bar-t1": mlx(0.01), "bar-e": mlx(mlx_e_hi)},
            "overlap": overlap,
            "exposures": [(str(exposure), {"groups": []})],
            "public": {"bar-t1": public("bar-t1"), "bar-e": public(pub_e_right)},
            "c1": None,
        }
        return run, inputs

    def test_both_bars_pass(self):
        run, inputs = self.make()
        out = f08.two_bar(run, inputs)
        self.assertEqual(out["status_1_7"], "PASS")
        self.assertEqual(out["status"], "INCOMPLETE")  # item 8 pending
        self.assertEqual(out["missing"], [])
        self.assertEqual(out["items"]["1_v3_vs_bar"]["bar-e"]["right_v3"], 50.263)
        self.assertEqual(f08.m6.BAR, "bar-t1")

    def test_one_bar_failing_fails_the_item(self):
        run, inputs = self.make(bar_e=(0.9, -0.2, 2.5))
        out = f08.two_bar(run, inputs)
        self.assertIs(out["items"]["1_v3_vs_bar"]["pass"], False)
        self.assertIs(out["items"]["1_v3_vs_bar"]["bar-t1"]["pass"], True)
        self.assertEqual(out["status_1_7"], "FAIL")
        self.assertEqual(out["per_bar"]["bar-t1"]["status_1_7"], "PASS")

    def test_bar_e_mlx_binds(self):
        run, inputs = self.make(mlx_e_hi=-0.001)
        self.assertIs(
            f08.two_bar(run, inputs)["items"]["4_mlx_card_eligible"]["pass"], False
        )

    def test_reduced_panel_bar_e(self):
        run, inputs = self.make(red_e=-0.1)
        out = f08.two_bar(run, inputs)
        self.assertIs(out["items"]["6b_reduced_panels"]["pass"], False)
        self.assertIs(out["items"]["6b_reduced_panels"]["bar-t1"]["pass"], True)

    def test_public_guard_must_pair_with_its_bar(self):
        run, inputs = self.make(pub_e_right="bar-t1")
        out = f08.two_bar(run, inputs)
        self.assertIsNone(out["items"]["7_jevbench_public231"]["pass"])
        self.assertEqual(out["status_1_7"], "INCOMPLETE")

    def test_cli_writes_json_and_md(self):
        run, inputs = self.make()
        paths = {}
        for key, doc in (
            ("types", inputs["types"]),
            ("mlx", inputs["mlx"]["bar-t1"]),
            ("mlx_e", inputs["mlx"]["bar-e"]),
            ("overlap", inputs["overlap"]),
            ("pub", inputs["public"]["bar-t1"]),
            ("pub_e", inputs["public"]["bar-e"]),
        ):
            paths[key] = self.root / f"{key}.json"
            paths[key].write_text(json.dumps(doc))
        prefix = self.root / "succ"
        rc = f08.main(
            [
                "--run",
                str(run),
                "--types",
                str(paths["types"]),
                "--mlx-paired",
                str(paths["mlx"]),
                "--mlx-paired-e",
                str(paths["mlx_e"]),
                "--overlap",
                str(paths["overlap"]),
                "--exposure",
                inputs["exposures"][0][0],
                "--public231",
                str(paths["pub"]),
                "--public231-e",
                str(paths["pub_e"]),
                "--output",
                str(prefix),
            ]
        )
        self.assertEqual(rc, 0)
        out = json.loads(Path(f"{prefix}.json").read_text())
        self.assertEqual(out["status_1_7"], "PASS")
        self.assertIn("two bars", Path(f"{prefix}.md").read_text())


class ShellTest(unittest.TestCase):
    def test_syntax(self):
        for script in ("f08-formal.sh", "f08-score.sh"):
            subprocess.run(["bash", "-n", str(OPS / script)], check=True)

    def test_formal_refuses_foreign_gpus(self):
        for gpu in ("0", "3", "4", "5"):
            proc = subprocess.run(
                [
                    "bash",
                    str(OPS / "f08-formal.sh"),
                    "run",
                    "x",
                    gpu,
                    "points",
                    "08b-RA",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("not a fast-track GPU", proc.stderr)


if __name__ == "__main__":
    unittest.main()
