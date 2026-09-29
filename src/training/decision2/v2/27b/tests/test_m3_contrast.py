import contextlib
import hashlib
import io
import importlib
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path

from training.model.calibration import metrics, probabilities
from transfer.score import macro_f1
from v2.eval.dev_readout import select_summary

m3 = importlib.import_module("v2.27b.m3_contrast")

FAMILIES = {"d0": "fa", "d1": "fa", "d2": "fa", "d3": "fa", "d4": "fb", "d5": "fb"}
GROUPS = {"d0": "g0", "d1": "g0", "d2": "g1", "d3": "g1", "d4": "g2", "d5": "g2"}
TYPES = {"d0": "choice", "d1": "noul", "d2": "score", "d3": "choice"}
TYPES.update({"d4": "noul", "d5": "score"})
CSS = [(f"{t}-{i}", t, "AB"[i % 2]) for t in ("t1", "t2") for i in range(4)]
CAL_LOGITS = [[2.0, 0.1], [0.3, 1.2], [1.5, 1.4], [0.2, 2.2], [3.0, -1.0], [0.9, 1.0]]
AHO_ROWS = [("h0", "k0"), ("h1", "k0"), ("h2", "k1"), ("h3", "k1")]


def write_jsonl(path: Path, rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question(kind: str) -> tuple[dict, dict]:
    if kind == "choice":
        labels = {"x": "first", "y": "second"}
        return {"type": "choice", "criteria": labels}, {
            "value": "x",
            "label_to_semantic": {"x": "x", "y": "y"},
        }
    if kind == "noul":
        return {"type": "noul", "criteria": {}}, {"value": True}
    return {"type": "score", "criteria": ["l0", "l1", "l2"]}, {"value": 1}


def answer(kind: str, correct: bool) -> dict:
    if kind == "choice":
        pick = "x" if correct else "y"
        other = "y" if correct else "x"
        return {
            "type": "choice",
            "choice": pick,
            "probabilities": {pick: 0.9, other: 0.1},
        }
    if kind == "noul":
        return {"type": "noul", "noul": 0.8 if correct else 0.2}
    return {"type": "score", "score": 1 if correct else 2}


class World:
    def __init__(self, root: Path) -> None:
        self.root = root
        dev = []
        for item, kind in TYPES.items():
            q, g = question(kind)
            dev.append(
                {
                    "id": item,
                    "family": FAMILIES[item],
                    "group_id": GROUPS[item],
                    "questions": {"q": q},
                    "gold": {"q": g},
                }
            )
        self.dev_gold = write_jsonl(root / "gold" / "dev.jsonl", dev)
        self.css_gold = write_jsonl(
            root / "gold" / "css.jsonl",
            [
                {"id": i, "role": "pilot", "task": t, "labels": ["A", "B"], "gold": g}
                for i, t, g in CSS
            ],
        )
        self.cal_rows = [
            {
                "id": f"c{i}",
                "group_id": f"cg{i // 2}",
                "family": "cf" if i < 3 else "cg",
                "task_type": "choice",
                "options": [{"key": "a"}, {"key": "b"}],
                "label": 0,
            }
            for i in range(len(CAL_LOGITS))
        ]
        self.cal = write_jsonl(root / "gold" / "cal.jsonl", self.cal_rows)
        self.aho = write_jsonl(
            root / "gold" / "aho-A7.jsonl",
            [{"id": i, "group_id": g} for i, g in AHO_ROWS],
        )

    def candidate(
        self,
        name: str,
        dev_wrong=(),
        dev_missing=(),
        css_wrong=(),
        temperature=1.0,
        cal_logits=CAL_LOGITS,
        aho_correct=("h0", "h1", "h2"),
        best=None,
        select=(0.8, 0.1),
    ) -> Path:
        out = self.root / name
        typed = write_jsonl(
            out / "output" / "typed-dev.predictions.jsonl",
            [
                {"id": i, "answers": {"q": answer(k, i not in dev_wrong)}}
                for i, k in TYPES.items()
                if i not in dev_missing
            ],
        )
        choices = {
            i: ("B" if g == "A" else "A") if i in css_wrong else g for i, _, g in CSS
        }
        css = write_jsonl(
            out / "output" / "css-pilot.predictions.jsonl",
            [
                {
                    "id": i,
                    "answers": {
                        "label": {
                            "type": "choice",
                            "choice": choices[i],
                            "probabilities": {
                                choices[i]: 0.7,
                                "B" if choices[i] == "A" else "A": 0.3,
                            },
                        }
                    },
                }
                for i, _, _ in CSS
            ],
        )
        fam = {"fa": [], "fb": []}
        for item in TYPES:
            fam[FAMILIES[item]].append(
                item not in dev_wrong and item not in dev_missing
            )
        t_dev = statistics.fmean(sum(v) / len(v) for v in fam.values())
        h = statistics.median(
            macro_f1(
                [g for _, tt, g in CSS if tt == t],
                [choices[i] for i, tt, _ in CSS if tt == t],
                ["A", "B"],
            )
            for t in ("t1", "t2")
        )
        probs = write_jsonl(
            out / "cal" / "cal.probs.jsonl",
            [
                {"id": r["id"], "probabilities": probabilities(logits, temperature)}
                for r, logits in zip(self.cal_rows, cal_logits)
            ],
        )
        summary = select_summary(self.cal_rows, probs)
        (out / "cal698.summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (out / "cal" / "calibration.json").write_text(
            json.dumps({"temperature_by_type": {"choice": temperature}}),
            encoding="utf-8",
        )
        records = [{"id": i, "correct": i in aho_correct} for i, _ in AHO_ROWS[:3]]
        write_jsonl(out / "aho" / "aho-A7-predictions.jsonl", records)
        (out / "aho" / "aho-summary.json").write_text(
            json.dumps({"slices": {"A7": {"micro_accuracy": len(aho_correct) / 4}}}),
            encoding="utf-8",
        )
        readout = {
            "label": name,
            "typed_dev": {
                "predictions_sha256": sha(typed),
                "T_dev": t_dev,
                "invalid_or_missing": len(dev_missing),
            },
            "css_pilot": {
                "predictions_sha256": sha(css),
                "H_pilot": h,
                "invalid_or_missing": 0,
            },
            "development_proxy": 100 * math.sqrt(t_dev * h),
        }
        (out / "READOUT.json").write_text(json.dumps(readout), encoding="utf-8")
        if best is not None:
            run = out / "trainer"
            run.mkdir()
            (run / "BEST.json").write_text(
                json.dumps({"checkpoint": f"checkpoint-{best:07d}"}), encoding="utf-8"
            )
            (run / f"select-step-{best:07d}-metrics.json").write_text(
                json.dumps(
                    {
                        "n": 700,
                        "correct": 600,
                        "family_macro_accuracy": select[0],
                        "family_macro_brier": select[1],
                    }
                ),
                encoding="utf-8",
            )
        return out

    def args(self, candidates: dict, extra: list | None = None, draws=200) -> list:
        argv = [
            "--dev-gold", str(self.dev_gold),
            "--css-gold", str(self.css_gold),
            "--cal-rows", str(self.cal),
            "--cal-sha256", sha(self.cal),
            "--aho", f"A7={self.aho}",
            "--draws", str(draws),
            "--output", str(self.root / "out.json"),
        ]  # fmt: skip
        for name, path in candidates.items():
            argv += ["--candidate", f"{name}={path}"]
            if (path / "trainer").is_dir():
                argv += ["--train-run", f"{name}={path / 'trainer'}"]
        return argv + (extra or [])

    def run(self, candidates: dict, extra: list | None = None, draws=200) -> dict:
        with contextlib.redirect_stdout(io.StringIO()):
            m3.main(self.args(candidates, extra, draws))
        return json.loads((self.root / "out.json").read_text(encoding="utf-8"))


class M3ContrastTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.world = World(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_point_summary(self):
        a = self.world.candidate(
            "A", dev_wrong={"d2"}, dev_missing={"d5"}, css_wrong={"t1-0"}
        )
        out = self.world.run({"A": a})
        point = out["points"]["A"]
        self.assertAlmostEqual(point["T_dev"], (3 / 4 + 1 / 2) / 2)
        self.assertEqual(point["choice_accuracy"], 1.0)
        self.assertEqual(point["score_accuracy"], 0.0)
        self.assertEqual(point["noul_accuracy"], 1.0)
        self.assertEqual((point["invalid_typed"], point["invalid_css"]), (1, 0))
        self.assertAlmostEqual(
            point["P_dev"], 100 * math.sqrt(point["T_dev"] * point["H_pilot"])
        )
        self.assertAlmostEqual(point["aho:A7"], 3 / 4)
        logits = [dict(r, logits=l) for r, l in zip(self.world.cal_rows, CAL_LOGITS)]
        reference = metrics(logits, 1.0)
        self.assertAlmostEqual(point["cal_brier"], reference["brier"], places=12)
        self.assertAlmostEqual(point["cal_ece_10"], reference["ece_10"], places=12)
        self.assertEqual(
            out["details"]["A"]["cal698_temperature_by_type"], {"choice": 1.0}
        )
        files = out["inputs_sha256"]
        self.assertEqual(files[str(a / "READOUT.json")], sha(a / "READOUT.json"))
        for name in (
            "cal698.summary.json",
            "cal/cal.probs.jsonl",
            "aho/aho-A7-predictions.jsonl",
        ):
            self.assertIn(str(a / name), files)
        self.assertIn(str(self.world.cal), files)
        self.assertTrue(any(k.endswith("m3_contrast.py") for k in files))

    def test_recorded_values_must_agree(self):
        a = self.world.candidate("A")
        readout = json.loads((a / "READOUT.json").read_text())
        readout["development_proxy"] += 0.01
        (a / "READOUT.json").write_text(json.dumps(readout))
        with self.assertRaises(ValueError):
            self.world.run({"A": a})
        b = self.world.candidate("B")
        with (b / "output" / "typed-dev.predictions.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaises(ValueError):
            self.world.run({"B": b})

    def test_output_is_written_once(self):
        a = self.world.candidate("A")
        self.world.run({"A": a})
        with self.assertRaises(FileExistsError):
            self.world.run({"A": a})

    def test_contrasts_and_floors(self):
        w = self.world
        cands = {
            "A1": w.candidate("A1", cal_logits=[[-3.0, 3.0]] * 6, dev_missing={"d5"}),
            "A2": w.candidate("A2", dev_wrong={"d1"}),
            "S1": w.candidate("S1", dev_wrong={"d1"}),
            "S2": w.candidate("S2", dev_wrong={"d1"}),
        }
        specs = ["A1:S1", "A2:S2", "A1+A2:S1+S2", "S1:S2"]
        extra = [a for s in specs for a in ("--contrast", s)]
        out = w.run(cands, extra)
        same = out["contrasts"]["S1:S2"]
        for key in ("T_dev", "P_dev", "cal_brier", "aho:A7"):
            self.assertEqual(same["bootstrap"][key]["ci95"], [0.0, 0.0])
        self.assertTrue(same["retention_floors"]["all_pass"])
        self.assertEqual(out["contrasts"]["A2:S2"]["delta"]["T_dev"], 0.0)
        pooled = out["contrasts"]["A1+A2:S1+S2"]
        p = out["points"]
        self.assertAlmostEqual(
            pooled["delta"]["cal_brier"],
            (
                p["A1"]["cal_brier"]
                + p["A2"]["cal_brier"]
                - p["S1"]["cal_brier"]
                - p["S2"]["cal_brier"]
            )
            / 2,
        )
        floors = pooled["retention_floors"]
        self.assertFalse(floors["invalid_not_increased"])
        self.assertFalse(floors["cal_brier_worse_at_most_0.010"])
        self.assertFalse(floors["all_pass"])
        self.assertFalse(floors["gates"])
        low, high = pooled["bootstrap"]["cal_brier"]["ci95"]
        self.assertTrue(0 < low <= pooled["delta"]["cal_brier"] <= high)
        (w.root / "out.json").unlink()
        again = w.run(cands, extra)
        self.assertEqual(again["contrasts"], out["contrasts"])

    def test_soup_rule_and_proxy_screen(self):
        points = {
            "soup": {"P_dev": 70.0},
            "s1": {"P_dev": 71.0},
            "s2": {"P_dev": 69.0},
        }

        class Seed:
            def __init__(self, accuracy, brier):
                self.select = {
                    "family_macro_accuracy": accuracy,
                    "family_macro_brier": brier,
                }

        cands = {"s1": Seed(0.80, 0.12), "s2": Seed(0.80, 0.10)}
        rule = m3.soup_rule(points, cands, "soup", ["s1", "s2"])
        self.assertEqual(rule["artifact"], "soup")
        points["soup"]["P_dev"] = 69.99
        rule = m3.soup_rule(points, cands, "soup", ["s1", "s2"])
        self.assertEqual(rule["artifact"], "s2")
        self.assertFalse(rule["select700_tie"])
        cands["s1"] = Seed(0.81, 0.30)
        self.assertEqual(
            m3.soup_rule(points, cands, "soup", ["s1", "s2"])["artifact"], "s1"
        )

        screen = m3.proxy_screen(
            {"F1": {"P_dev": 72.0}, "F2": {"P_dev": 64.0}, "F3": {"P_dev": 64.01}},
            ["F1", "F2", "F3"],
        )
        kept = {n: c["kept"] for n, c in screen["candidates"].items()}
        self.assertEqual(kept, {"F1": True, "F2": False, "F3": True})

    def test_soup_rule_end_to_end_reads_trainer_select(self):
        w = self.world
        cands = {
            "S1": w.candidate("S1", best=94, select=(0.70, 0.1)),
            "S2": w.candidate("S2", best=188, select=(0.75, 0.2)),
            "Ssoup": w.candidate("Ssoup", dev_wrong={"d0", "d3"}),
        }
        out = w.run(cands, ["--soup", "S=Ssoup:S1+S2"])
        rule = out["soup_rule"]["S"]
        self.assertEqual(rule["artifact"], "S2")
        self.assertEqual(rule["select700"]["S2"]["best"], "checkpoint-0000188")
        self.assertEqual(out["proxy_screen"]["pool"], ["S2"])
        self.assertIn(
            str(cands["S2"] / "trainer" / "select-step-0000188-metrics.json"),
            out["inputs_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
