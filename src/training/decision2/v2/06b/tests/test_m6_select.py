import importlib
import json
import math
import tempfile
import unittest
from pathlib import Path

select = importlib.import_module("v2.06b.m6_select")


def metrics(noul_correct: int) -> dict:
    return {
        "by_type": {
            "noul": {"accuracy": noul_correct / 290, "correct": noul_correct, "n": 290}
        }
    }


def make(
    root: Path,
    name: str,
    *,
    t: float,
    tasks: tuple[float, float, float],
    typed: tuple[int, int, int] = (300, 200, 150),
    levels: dict | None = None,
    soup: list[str] | None = None,
    select_noul: int = 270,
    cal_noul: int | None = 272,
    stopped: bool = False,
) -> None:
    readout, full = root / name / "readout", root / name / "full"
    readout.mkdir(parents=True)
    full.mkdir(parents=True)
    names = ("discourse", "implicit_hate", "semeval_stance")
    body = {
        "T_dev": t,
        "H_pilot": sorted(tasks)[1],
        "css_pilot_tasks": dict(zip(names, tasks)),
        "typed_by_type": {
            k: {"correct": c, "n": n, "invalid": 0}
            for k, c, n in zip(("choice", "noul", "score"), typed, (800, 400, 400))
        },
        "score_levels_predicted": (
            levels if levels is not None else {"0": 300, "1": 20, "2": 80}
        ),
    }
    (readout / "READOUT.json").write_text(json.dumps(body))
    if soup is not None:
        record = {
            "ingredients": [{"arm": a} for a in soup],
            "select": {"metrics": metrics(select_noul)},
        }
        if cal_noul is not None:
            record["cal"] = {"metrics": metrics(cal_noul)}
        (full / "SOUP.json").write_text(json.dumps(record))
    else:
        (full / "BEST.json").write_text(json.dumps({"metrics": metrics(select_noul)}))
    if stopped:
        (full / "STOPPED.json").write_text("{}")


class SelectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_proxies_and_guards(self):
        make(self.root, "z", t=0.45, tasks=(0.33, 0.34, 0.53), soup=["m4-t-a7-s1"])
        c = select.candidate("z", self.root)
        h3 = (0.33 + 0.34 + 0.53) / 3
        self.assertAlmostEqual(c["H3"], h3)
        self.assertAlmostEqual(c["Q"], 100 * math.sqrt(0.45 * h3))
        self.assertAlmostEqual(c["P"], 100 * math.sqrt(0.45 * 0.34))
        self.assertEqual(c["typed"], {"choice": 300, "noul": 200, "score": 150})
        self.assertEqual(c["score_levels"], 3)
        self.assertTrue(c["guards_pass"])
        self.assertFalse(c["contains_new_family"])

    def test_guard_failures(self):
        make(
            self.root,
            "c254",
            t=0.4,
            tasks=(0.3, 0.3, 0.3),
            typed=(254, 200, 150),
            soup=[],
        )
        make(
            self.root,
            "lvl2",
            t=0.4,
            tasks=(0.3, 0.3, 0.3),
            levels={"0": 390, "2": 10, "1": 0},
            soup=[],
        )
        make(
            self.root,
            "s100",
            t=0.4,
            tasks=(0.3, 0.3, 0.3),
            typed=(300, 200, 100),
            soup=[],
        )
        make(self.root, "noul", t=0.4, tasks=(0.3, 0.3, 0.3), soup=[], cal_noul=240)
        make(self.root, "seed", t=0.4, tasks=(0.3, 0.3, 0.3))
        failed = {
            n: select.candidate(n, self.root)["guards_failed"]
            for n in ("c254", "lvl2", "s100", "noul", "seed")
        }
        self.assertEqual(failed["c254"], ["choice_ge_255"])
        self.assertEqual(failed["lvl2"], ["score_levels_ge_3"])
        self.assertEqual(failed["s100"], ["score_ge_101"])
        self.assertEqual(failed["noul"], ["cal_noul_ge_085"])
        self.assertEqual(failed["seed"], ["cal_noul_ge_085"])
        seed = select.candidate("seed", self.root)
        self.assertIsNone(seed["guards"]["cal_noul_ge_085"])
        self.assertEqual(seed["in_distribution_source"], "BEST.json")

    def test_edges_pass(self):
        make(
            self.root,
            "edge",
            t=0.4,
            tasks=(0.3, 0.3, 0.3),
            typed=(255, 180, 101),
            soup=[],
            select_noul=247,
            cal_noul=247,
        )
        c = select.candidate("edge", self.root)
        self.assertAlmostEqual(c["select_noul"]["accuracy"], 247 / 290)
        self.assertTrue(c["guards_pass"])

    def test_table_finalists(self):
        make(
            self.root,
            "m5-z-soup",
            t=0.45,
            tasks=(0.33, 0.34, 0.53),
            soup=["m4-t-a7-s1"],
        )
        make(
            self.root,
            "m6-cx-soup",
            t=0.44,
            tasks=(0.33, 0.34, 0.40),
            soup=["m6-cx-s1", "m6-cx-s2"],
        )
        make(self.root, "far", t=0.20, tasks=(0.2, 0.2, 0.2), soup=["m6-mx-s1"])
        result = select.table(
            ["m5-z-soup", "m6-cx-soup", "far"], self.root, "m5-z-soup"
        )
        eligible = {c["name"]: c["finalist_eligible"] for c in result["candidates"]}
        self.assertEqual(
            eligible, {"m5-z-soup": False, "m6-cx-soup": True, "far": False}
        )
        self.assertEqual(result["guard_passing_by_Q"][0], "m5-z-soup")

    def test_seedmean_soup_or_median(self):
        for i, t in enumerate((0.30, 0.40, 0.50), 1):
            make(self.root, f"f-s{i}", t=t, tasks=(0.4, 0.4, 0.4))
        make(
            self.root,
            "f-soup",
            t=0.42,
            tasks=(0.4, 0.4, 0.4),
            soup=["f-s1", "f-s2", "f-s3"],
        )
        make(
            self.root,
            "g-soup",
            t=0.38,
            tasks=(0.4, 0.4, 0.4),
            soup=["f-s1", "f-s2", "f-s3"],
        )
        seeds = ["f-s1", "f-s2", "f-s3"]
        self.assertEqual(
            select.seedmean("f-soup", seeds, self.root)["artifact"], "f-soup"
        )
        low = select.seedmean("g-soup", seeds, self.root)
        self.assertEqual(low["artifact"], "f-s2")
        self.assertFalse(low["artifact_is_soup"])

    def test_seedmean_excludes_collapsed(self):
        make(self.root, "f-s1", t=0.30, tasks=(0.4, 0.4, 0.4))
        make(self.root, "f-s2", t=0.40, tasks=(0.4, 0.4, 0.4))
        make(self.root, "f-s3", t=0.10, tasks=(0.4, 0.4, 0.4), stopped=True)
        make(self.root, "f-soup", t=0.36, tasks=(0.4, 0.4, 0.4), soup=["f-s1", "f-s2"])
        result = select.seedmean("f-soup", ["f-s1", "f-s2", "f-s3"], self.root)
        self.assertEqual([e["name"] for e in result["excluded_seeds"]], ["f-s3"])
        self.assertTrue(result["median_ambiguous"])
        self.assertEqual(result["artifact"], "f-soup")
        with self.assertRaises(ValueError):
            select.seedmean("f-soup", ["f-s1", "f-s3"], self.root)

    def test_greedy_rule(self):
        make(
            self.root,
            "S",
            t=0.45,
            tasks=(0.35, 0.35, 0.50),
            typed=(350, 200, 150),
            soup=[],
        )
        make(
            self.root,
            "ok",
            t=0.46,
            tasks=(0.34, 0.35, 0.50),
            typed=(331, 200, 131),
            soup=[],
        )
        make(self.root, "lowq", t=0.44, tasks=(0.35, 0.35, 0.50), soup=[])
        make(self.root, "h3", t=0.60, tasks=(0.30, 0.35, 0.50), soup=[])
        make(
            self.root,
            "choice",
            t=0.50,
            tasks=(0.35, 0.35, 0.50),
            typed=(329, 200, 150),
            soup=[],
        )
        make(
            self.root,
            "guard",
            t=0.50,
            tasks=(0.35, 0.35, 0.50),
            levels={"0": 1, "1": 2},
            soup=[],
        )
        self.assertTrue(select.greedy_check("S", "ok", self.root)["accept"])
        verdicts = {
            n: select.greedy_check("S", n, self.root)["checks"]
            for n in ("lowq", "h3", "choice", "guard")
        }
        self.assertFalse(verdicts["lowq"]["Q_not_lower"])
        self.assertFalse(verdicts["h3"]["H3_within_0.010"])
        self.assertTrue(verdicts["h3"]["Q_not_lower"])
        self.assertFalse(verdicts["choice"]["choice_within_20"])
        self.assertFalse(verdicts["guard"]["guards_hold"])
        for n in ("lowq", "h3", "choice", "guard"):
            self.assertFalse(select.greedy_check("S", n, self.root)["accept"])

    def test_path_refs_and_wrong_tasks(self):
        make(self.root, "p", t=0.4, tasks=(0.3, 0.3, 0.3), soup=[])
        c = select.candidate(str(self.root / "p" / "readout"), self.root)
        self.assertEqual(c["name"], "p")
        body = json.loads((self.root / "p" / "readout" / "READOUT.json").read_text())
        body["css_pilot_tasks"] = {"discourse": 0.3}
        (self.root / "p" / "readout" / "READOUT.json").write_text(json.dumps(body))
        with self.assertRaises(ValueError):
            select.candidate("p", self.root)


if __name__ == "__main__":
    unittest.main()
