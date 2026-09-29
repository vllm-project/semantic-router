"""CPU tests for the decoder Milestone 5 selection script (synthetic readouts)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ops" / "m5" / "m5-select.py"
_spec = importlib.util.spec_from_file_location("m5_select", SCRIPT)
m5_select = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m5_select)

N4XF_TYPED = {"choice": 450, "noul": 220, "score": 370}


def arm(R, typed=None, invalid=0):
    typed = typed or N4XF_TYPED
    return {
        "proxy": R,
        "proxy_mean_H": R - 1.0,
        "T": 0.8,
        "H": (R / 100) ** 2 / 0.8,
        "H_mean": (R / 100) ** 2 / 0.8 - 0.01,
        "by_type": {
            k: {"correct": v, "invalid": invalid if k == "noul" else 0, "n": 600}
            for k, v in typed.items()
        },
    }


def predictions(path: Path, constant_noul=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(6):
        lines.append(
            {"answers": {"decision": {"type": "choice", "choice": "ab"[i % 2]}}}
        )
        noul = 0.9 if constant_noul or i % 2 else 0.1
        lines.append({"answers": {"decision": {"type": "noul", "noul": noul}}})
        probs = (
            {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}
            if i % 2
            else {"0": 0.7, "1": 0.1, "2": 0.1, "3": 0.1}
        )
        lines.append(
            {"answers": {"decision": {"type": "score", "probabilities": probs}}}
        )
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))


def setup_arm(
    root: Path,
    name,
    seeds,
    soup,
    n4xf=60.0,
    typed=None,
    invalid=0,
    ci=(0.01, 0.05),
    constant=False,
):
    arms = {"nox1": arm(55.0), "n4xf": arm(n4xf)}
    for s, R in zip(m5_select.SEEDS, seeds):
        arms[s] = arm(R, typed, invalid)
        predictions(
            root
            / "arms"
            / "full"
            / f"m5-{name}-{s}-post"
            / "dev"
            / "dev.predictions.jsonl",
            constant,
        )
    arms["soup"] = arm(soup, typed, invalid)
    d = root / "soup" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "readout.json").write_text(json.dumps({"arms": arms}))
    predictions(d / "dev" / "dev.predictions.jsonl", constant)
    chosen = (
        "soup" if soup >= sum(seeds) / 3 else sorted(zip(seeds, m5_select.SEEDS))[1][1]
    )
    if ci is not None:
        mlx = root / "mlxdev" / "readouts" / f"m5-{name}-{chosen}"
        mlx.mkdir(parents=True, exist_ok=True)
        diff = (ci[0] + ci[1]) / 2
        (mlx / "vs-n4xf-soup.json").write_text(
            json.dumps({"metrics": {"noul_ml": {"diff": diff, "ci95": list(ci)}}})
        )
        (mlx / "score.json").write_text(json.dumps({"m_dev": 0.7}))


class SelectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_soup_chosen_and_fix_verdict(self):
        setup_arm(self.root, "A", [60, 61, 62], 62.5)
        r = m5_select.select(self.root, ["A"])["rows"][0]
        self.assertEqual(r["artifact"], "soup")
        self.assertTrue(
            r["eligible"] and r["finalist"] and r["multilingual_noul_fix_dev"]
        )
        self.assertAlmostEqual(r["P_mean3"], 61.5)

    def test_median_seed_when_soup_below_mean(self):
        setup_arm(self.root, "A", [63, 58, 61], 60.0)
        r = m5_select.select(self.root, ["A"])["rows"][0]
        self.assertEqual(r["artifact"], "s3")
        self.assertEqual(r["mlxdev_name"], "m5-A-s3")
        self.assertTrue(r["finalist"])

    def test_missing_mlx_is_pending(self):
        setup_arm(self.root, "A", [63, 58, 61], 60.0, ci=None)
        out = m5_select.select(self.root, ["A"])
        r = out["rows"][0]
        self.assertFalse(r["finalist"])
        self.assertEqual(out["pending"], ["A"])
        self.assertTrue(any("m5-mlx.sh" in p for p in r["pending"]))

    def test_band_drop_is_inclusive(self):
        setup_arm(self.root, "A", [52, 52, 52], 52.0, n4xf=60.0)
        setup_arm(self.root, "B", [52.1, 52.1, 52.1], 52.1, n4xf=60.0)
        rows = {r["arm"]: r for r in m5_select.select(self.root, ["A", "B"])["rows"]}
        self.assertTrue(rows["A"]["dropped"])
        self.assertFalse(rows["B"]["dropped"])

    def test_noul_upper_bound_drop_and_no_fix(self):
        setup_arm(self.root, "A", [60, 60, 60], 61, ci=(-0.05, -0.001))
        setup_arm(self.root, "B", [60, 60, 60], 61, ci=(-0.01, 0.02))
        rows = {r["arm"]: r for r in m5_select.select(self.root, ["A", "B"])["rows"]}
        self.assertTrue(rows["A"]["dropped"])
        self.assertFalse(rows["B"]["dropped"])
        self.assertTrue(rows["B"]["finalist"])
        self.assertFalse(rows["B"]["multilingual_noul_fix_dev"])

    def test_eligibility_floors_invalid_constant(self):
        setup_arm(
            self.root,
            "A",
            [60, 60, 60],
            61,
            typed={"choice": 344, "noul": 220, "score": 370},
        )
        setup_arm(self.root, "B", [60, 60, 60], 61, invalid=1)
        setup_arm(self.root, "C", [60, 60, 60], 61, constant=True)
        setup_arm(
            self.root,
            "D",
            [60, 60, 60],
            61,
            typed={"choice": 345, "noul": 171, "score": 284},
        )
        rows = {
            r["arm"]: r
            for r in m5_select.select(self.root, ["A", "B", "C", "D"])["rows"]
        }
        self.assertEqual(rows["A"]["reasons"], ["choice 344 < 345"])
        self.assertEqual(rows["B"]["reasons"], ["1 invalid"])
        self.assertEqual(rows["C"]["reasons"], ["noul constant"])
        self.assertTrue(rows["D"]["eligible"])
        self.assertTrue(rows["D"]["finalist"])
        self.assertFalse(rows["D"]["multilingual_noul_fix_dev"])

    def test_typed_ratio_boundary(self):
        typed = {k: -(-95 * v // 100) for k, v in N4XF_TYPED.items()}
        setup_arm(self.root, "A", [60, 60, 60], 61, typed=typed)
        typed_low = dict(typed, noul=typed["noul"] - 1)
        setup_arm(self.root, "B", [60, 60, 60], 61, typed=typed_low)
        rows = {r["arm"]: r for r in m5_select.select(self.root, ["A", "B"])["rows"]}
        self.assertTrue(rows["A"]["multilingual_noul_fix_dev"])
        self.assertFalse(rows["B"]["multilingual_noul_fix_dev"])

    def test_table_renders(self):
        setup_arm(self.root, "A", [60, 61, 62], 62.5)
        md = m5_select.table(m5_select.select(self.root, ["A"])["rows"])
        self.assertIn("| A | soup | 62.50 |", md)


if __name__ == "__main__":
    unittest.main()
