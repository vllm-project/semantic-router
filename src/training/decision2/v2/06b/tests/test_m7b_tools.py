import importlib
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from training.model.data import file_sha256

m6_select = importlib.import_module("v2.06b.m6_select")
m7_select = importlib.import_module("v2.06b.m7_select")
m7_contrast = importlib.import_module("v2.06b.m7_contrast")
make = importlib.import_module("v2.06b.tests.test_m6_select").make


def quiet(fn, *args):
    with redirect_stdout(io.StringIO()):
        return fn(*args)


def devcheck(root: Path, name: str, *, top: float = 0.4, lower: float = 0.05) -> None:
    path = root / name / "devcheck.json"
    path.parent.mkdir(parents=True)
    checks = {
        "top_share_le_0.90": top <= 0.9,
        "delta_vs_majority_lower_bound_gt_0": lower > 0,
    }
    path.write_text(
        json.dumps(
            {
                "label": name,
                "state_sha256": "s-" + name,
                "chk": {
                    "L5": {
                        "before": {
                            "n": 100,
                            "accuracy": 0.3,
                            "majority_level": 3,
                            "majority_accuracy": 0.2,
                            "delta_vs_majority": 0.1,
                            "delta_vs_majority_ci95": [lower, 0.2],
                            "top_value": "0",
                            "top_share": top,
                            "predicted_distribution": {"0": 40, "4": 60},
                        }
                    }
                },
                "B-D1": {"checks": checks, "pass": all(checks.values())},
            }
        )
    )


def q_of(t: float, tasks: tuple[float, float, float]) -> float:
    return 100 * math.sqrt(t * sum(tasks) / 3)


class SelectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "arms"
        self.checks = Path(self.tmp.name) / "chk"
        self.checks.mkdir()
        m6_select.SCORE_GUARD = "modal90"

    def tearDown(self):
        self.tmp.cleanup()

    def family(self, fam: str, seed_t: tuple[float, ...], soup_t: float) -> list[str]:
        seeds = [f"{fam}-s{k}" for k in range(1, len(seed_t) + 1)]
        for seed, t in zip(seeds, seed_t):
            make(self.root, seed, t=t, tasks=(0.35, 0.35, 0.5))
            devcheck(self.checks, seed)
        make(self.root, f"{fam}-soup", t=soup_t, tasks=(0.35, 0.35, 0.5), soup=seeds)
        devcheck(self.checks, f"{fam}-soup")
        return [f"{fam}-soup", *seeds]

    def run_select(self, families, cross="m7-mxcx-soup"):
        return m7_select.select(families, cross, self.checks, self.root)

    def test_cross_first_then_best_other_within_band(self):
        mx = self.family("m7-mx", (0.45, 0.50, 0.48), 0.55)
        cx = self.family("m7-cx", (0.40, 0.44, 0.42), 0.50)
        make(
            self.root,
            "m7-mxcx-soup",
            t=0.52,
            tasks=(0.35, 0.35, 0.5),
            soup=mx[1:] + cx[1:],
        )
        devcheck(self.checks, "m7-mxcx-soup")
        result = self.run_select([mx, cx])
        self.assertEqual(
            [f["name"] for f in result["finalists"]], ["m7-mxcx-soup", "m7-mx-soup"]
        )
        self.assertEqual(result["families"][0]["artifact"], "m7-mx-soup")
        self.assertAlmostEqual(result["best_eligible_Q"], q_of(0.55, (0.35, 0.35, 0.5)))

    def test_cross_failing_bd1_leaves_one_other(self):
        mx = self.family("m7-mx", (0.45, 0.50, 0.48), 0.55)
        cx = self.family("m7-cx", (0.40, 0.44, 0.42), 0.50)
        make(self.root, "m7-mxcx-soup", t=0.60, tasks=(0.35, 0.35, 0.5), soup=mx[1:])
        devcheck(self.checks, "m7-mxcx-soup", lower=-0.01)
        result = self.run_select([mx, cx])
        cross = next(c for c in result["candidates"] if c["name"] == "m7-mxcx-soup")
        self.assertTrue(cross["guards_pass"])
        self.assertFalse(cross["eligible"])
        self.assertEqual([f["name"] for f in result["finalists"]], ["m7-mx-soup"])

    def test_other_outside_band_is_not_selected(self):
        mx = self.family("m7-mx", (0.10, 0.12, 0.11), 0.13)
        make(self.root, "m7-mxcx-soup", t=0.60, tasks=(0.35, 0.35, 0.5), soup=mx[1:])
        devcheck(self.checks, "m7-mxcx-soup")
        result = self.run_select([mx])
        self.assertEqual([f["name"] for f in result["finalists"]], ["m7-mxcx-soup"])
        self.assertEqual(result["not_selected"]["name"], "m7-mx-soup")

    def test_median_seed_artifact_lacks_cal_noul(self):
        mx = self.family("m7-mx", (0.45, 0.50, 0.48), 0.40)
        result = self.run_select([mx], cross=None)
        self.assertEqual(result["families"][0]["artifact"], "m7-mx-s3")
        seed = next(c for c in result["candidates"] if c["name"] == "m7-mx-s3")
        self.assertEqual(seed["roles"], ["family artifact"])
        self.assertIn("cal_noul_ge_085", seed["guards_failed"])
        self.assertEqual([f["name"] for f in result["finalists"]], ["m7-mx-soup"])

    def test_modal_share_and_missing_devcheck_fail(self):
        mx = self.family("m7-mx", (0.45, 0.50, 0.48), 0.55)
        (self.checks / "m7-mx-soup" / "devcheck.json").unlink()
        make(
            self.root,
            "m7-mxcx-soup",
            t=0.6,
            tasks=(0.35, 0.35, 0.5),
            soup=mx[1:],
            levels={"0": 5, "2": 395},
        )
        devcheck(self.checks, "m7-mxcx-soup")
        result = self.run_select([mx])
        self.assertEqual(result["finalists"], [])
        self.assertIn("no eligible", result["verdict"])
        by = {c["name"]: c for c in result["candidates"]}
        self.assertEqual(by["m7-mx-soup"]["B-D1"]["reason"], "no devcheck")
        self.assertIn("score_modal_le_090", by["m7-mxcx-soup"]["guards_failed"])

    def test_missing_soup_readout_and_cli(self):
        mx = self.family("m7-mx", (0.45, 0.50), 0.55)
        cx = ["m7-cx-soup", "m7-cx-s1", "m7-cx-s2"]
        out = Path(self.tmp.name) / "select.json"
        args = ["--root", str(self.root), "--devchecks", str(self.checks)]
        args += ["--family", *mx, "--family", *cx, "--cross", "m7-mxcx-soup"]
        quiet(m7_select.main, [*args, "--output", str(out)])
        result = json.loads(out.read_text())
        self.assertIsNone(result["families"][1]["artifact"])
        self.assertIn("no readout", result["cross_note"])
        self.assertEqual([f["name"] for f in result["finalists"]], ["m7-mx-soup"])
        with self.assertRaises(FileExistsError):
            quiet(m7_select.main, [*args, "--output", str(out)])


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class ChkDeltaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.aho = self.dir / "aho"
        self.aho.mkdir()
        rows, index = [], []
        for i in range(40):
            levels = 5 if i < 30 else 3
            rows.append(
                {
                    "id": f"r{i}",
                    "task_type": "score",
                    "options": [{"key": str(k)} for k in range(levels)],
                    "label": i % levels,
                }
            )
            index.append({"id": f"r{i}", "arm": "A7q" if i % 2 else "H6"})
        write_jsonl(self.aho / "chk.jsonl", rows)
        write_jsonl(self.aho / "index.jsonl", index)
        (self.aho / "MANIFEST.json").write_text(
            json.dumps(
                {
                    "outputs": {
                        "chk.jsonl": {"sha256": file_sha256(self.aho / "chk.jsonl")}
                    }
                }
            )
        )
        self.rows = rows

    def tearDown(self):
        self.tmp.cleanup()

    def probs(self, name: str, answer) -> Path:
        out = self.dir / name
        out.mkdir()
        lines = []
        for r in self.rows:
            n = len(r["options"])
            k = answer(r)
            lines.append(
                {
                    "id": r["id"],
                    "probabilities": [
                        0.9 if j == k else 0.1 / (n - 1) for j in range(n)
                    ],
                }
            )
        write_jsonl(out / "chk.probs.jsonl", lines)
        manifest = out / "PROBS.json"
        manifest.write_text(
            json.dumps(
                {
                    "state_sha256": "s-" + name,
                    "outputs": [
                        {
                            "input_sha256": file_sha256(self.aho / "chk.jsonl"),
                            "output": f"/runs/elsewhere/{name}/chk.probs.jsonl",
                            "output_sha256": file_sha256(out / "chk.probs.jsonl"),
                        }
                    ],
                }
            )
        )
        return manifest

    def test_paired_delta_and_usage(self):
        treatment = self.probs("t", lambda r: r["label"])
        control = self.probs("c", lambda r: len(r["options"]) - 1)
        result = m7_contrast.chk_pair(self.aho, treatment, control)
        l5 = result["L5"]
        self.assertEqual(l5["n"], 30)
        self.assertEqual(l5["treatment"]["accuracy"], 1.0)
        self.assertAlmostEqual(l5["control"]["accuracy"], 0.2)
        self.assertAlmostEqual(l5["delta"]["accuracy"], 0.8)
        self.assertLessEqual(l5["delta"]["accuracy_ci95"][0], 0.8)
        self.assertAlmostEqual(l5["control"]["top_share"], 1.0)
        self.assertAlmostEqual(l5["delta"]["top_share"], 0.2 - 1.0)
        self.assertEqual(l5["delta"]["level_usage"]["4"], 6 - 30)
        self.assertEqual(result["L3"]["n"], 10)
        self.assertNotIn("L4", result)
        out = self.dir / "delta.json"
        quiet(
            m7_contrast.main,
            [
                "chk",
                "--aho-dir",
                str(self.aho),
                "--pair",
                f"m7-m6:mx-s1={treatment}={control}",
                "--output",
                str(out),
            ],
        )
        report = json.loads(out.read_text())
        self.assertEqual(report["pairs"]["m7-m6:mx-s1"]["L5"]["n"], 30)
        self.assertNotIn("r0", out.read_text())

    def test_summary_joins_readouts(self):
        root = self.dir / "arms"
        make(root, "m7-mx-s1", t=0.5, tasks=(0.3, 0.4, 0.5), typed=(300, 200, 160))
        make(root, "m6-mx-s1", t=0.4, tasks=(0.3, 0.4, 0.5), typed=(300, 200, 150))
        contrast = self.dir / "contrast.json"
        contrast.write_text(
            json.dumps(
                {
                    "pairs": {
                        "m7-m6:mx-s1": {
                            "delta": {"P": 1.0, "P_mean3": 2.0},
                            "ci95": {"P": [0.1, 2.0], "P_mean3": [0.5, 3.0]},
                            "draws": 10000,
                        }
                    }
                }
            )
        )
        out = self.dir / "summary.json"
        quiet(
            m7_contrast.main,
            [
                "summary",
                "--pair",
                "m7-m6:mx-s1=m7-mx-s1=m6-mx-s1",
                "--contrast",
                str(contrast),
                "--root",
                str(root),
                "--output",
                str(out),
            ],
        )
        row = json.loads(out.read_text())["pairs"]["m7-m6:mx-s1"]
        self.assertEqual(row["delta_readout"]["typed_score"], 10)
        self.assertAlmostEqual(
            row["delta_readout"]["Q"],
            q_of(0.5, (0.3, 0.4, 0.5)) - q_of(0.4, (0.3, 0.4, 0.5)),
        )
        self.assertEqual(row["paired"]["P_mean3"], {"delta": 2.0, "ci95": [0.5, 3.0]})
        with self.assertRaises(ValueError):
            m7_contrast.split_pair("a=b")


if __name__ == "__main__":
    unittest.main()
