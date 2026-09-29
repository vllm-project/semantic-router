import contextlib
import hashlib
import importlib
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from benchmark.generate import FINAL_FAMILIES

guard = importlib.import_module("v2.27b.m4_guard")
m4c = importlib.import_module("v2.27b.m4_contrast")

HERE = Path(__file__).resolve().parents[1]
M4 = HERE / "m4"
KINDS = ("choice", "noul", "score")
DEV_FAMILIES = ("fa", "fb", "fc", "fd")
LABELS = "xyz"


def write_jsonl(path: Path, rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question(kind: str, value, levels: int = 3) -> tuple[dict, dict]:
    if kind == "choice":
        criteria = {label: f"option {label}" for label in LABELS}
        return {"type": "choice", "criteria": criteria}, {
            "value": value,
            "label_to_semantic": {label: f"s-{label}" for label in LABELS},
        }
    if kind == "noul":
        return {"type": "noul", "criteria": {}}, {"value": value}
    return {"type": "score", "criteria": [f"l{i}" for i in range(levels)]}, {
        "value": value
    }


def answer(kind: str, value, correct: bool, levels: int = 3, force=None) -> dict:
    if kind == "choice":
        pick = force or (value if correct else LABELS[(LABELS.index(value) + 1) % 3])
        return {
            "type": "choice",
            "choice": pick,
            "probabilities": {label: 0.8 if label == pick else 0.1 for label in LABELS},
        }
    if kind == "noul":
        return {"type": "noul", "noul": 0.8 if value == correct else 0.2}
    return {"type": "score", "score": value if correct else (value + 1) % levels}


class GuardWorld:
    """Typed DEV (24 one-question items, 8 per type) and a two-task CSS pilot."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.dev = []
        for i in range(24):
            kind = KINDS[i % 3]
            value = {"choice": LABELS[(i // 3) % 3], "noul": (i // 3) % 2 == 0}.get(
                kind, (i // 3) % 3
            )
            q, g = question(kind, value)
            self.dev.append(
                {
                    "id": f"d{i}",
                    "family": DEV_FAMILIES[i % 4],
                    "group_id": f"g{i // 4}",
                    "questions": {"decision": q},
                    "gold": {"decision": g},
                }
            )
        self.dev_gold = write_jsonl(root / "gold" / "dev.jsonl", self.dev)
        self.css = [
            (f"{t}-{i}", t, "AB"[i % 2]) for t in ("t1", "t2") for i in range(4)
        ]
        self.css_gold = write_jsonl(
            root / "gold" / "css.jsonl",
            [
                {"id": i, "role": "pilot", "task": t, "labels": ["A", "B"], "gold": g}
                for i, t, g in self.css
            ],
        )

    def readout(
        self,
        name,
        wrong=(),
        missing=(),
        force_choice=None,
        css_wrong=(),
        css_missing=(),
        cal_brier=0.09,
    ) -> Path:
        out = self.root / name
        rows, fam, typ = (
            [],
            {f: [0, 0] for f in DEV_FAMILIES},
            {k: [0, 0] for k in KINDS},
        )
        for item in self.dev:
            q = item["questions"]["decision"]
            value = item["gold"]["decision"]["value"]
            correct = item["id"] not in wrong and item["id"] not in missing
            if force_choice and q["type"] == "choice":
                correct = value == force_choice
            fam[item["family"]][1] += 1
            typ[q["type"]][1] += 1
            fam[item["family"]][0] += correct
            typ[q["type"]][0] += correct
            if item["id"] in missing:
                continue
            a = answer(q["type"], value, correct, force=force_choice)
            rows.append({"id": item["id"], "answers": {"decision": a}})
        typed = write_jsonl(out / "output" / "typed-dev.predictions.jsonl", rows)
        choices = {}
        for i, _, g in self.css:
            if i not in css_missing:
                choices[i] = ("B" if g == "A" else "A") if i in css_wrong else g
        css = write_jsonl(
            out / "output" / "css-pilot.predictions.jsonl",
            [
                {
                    "id": i,
                    "answers": {
                        "label": {
                            "type": "choice",
                            "choice": c,
                            "probabilities": {c: 0.7, "B" if c == "A" else "A": 0.3},
                        }
                    },
                }
                for i, c in choices.items()
            ],
        )
        f1 = []
        for task in ("t1", "t2"):
            items = [(g, choices.get(i)) for i, t, g in self.css if t == task]
            scores = []
            for label in "AB":
                tp = sum(p == g == label for g, p in items)
                fp = sum(p == label and g != label for g, p in items)
                fn = sum(g == label and p != label for g, p in items)
                scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
            f1.append(statistics.fmean(scores))
        h = statistics.median(f1)
        t_dev = statistics.fmean(c / n for c, n in fam.values())
        probs = write_jsonl(
            out / "cal" / "cal.probs.jsonl", [{"id": "c0", "probabilities": [1, 0]}]
        )
        write_json(
            out / "cal698.summary.json",
            {
                "predictions_sha256": sha(probs),
                "n": 10,
                "by_type": {
                    "choice": {"brier": cal_brier, "n": 6},
                    "noul": {"brier": cal_brier, "n": 4},
                },
                "family_macro_brier": cal_brier,
                "family_macro_accuracy": 0.8,
            },
        )
        return write_json(
            out / "READOUT.json",
            {
                "label": name,
                "typed_dev": {
                    "predictions_sha256": sha(typed),
                    "T_dev": t_dev,
                    "by_type": {k: {"correct": c, "n": n} for k, (c, n) in typ.items()},
                    "invalid_or_missing": len(missing),
                },
                "css_pilot": {
                    "predictions_sha256": sha(css),
                    "H_pilot": h,
                    "invalid_or_missing": len(css_missing),
                },
                "development_proxy": 100 * math.sqrt(t_dev * h),
            },
        )

    def run(self, candidates: dict, incumbent: Path, extra=(), output=True) -> dict:
        argv = ["--dev-gold", str(self.dev_gold), "--css-gold", str(self.css_gold)]
        for name, path in candidates.items():
            argv += ["--candidate", f"{name}={path}"]
        argv += ["--incumbent", f"F1={incumbent}", *extra]
        out = self.root / "guard.json"
        if output:
            argv += ["--output", str(out)]
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(
            io.StringIO()
        ):
            guard.main(argv)
        return json.loads(out.read_text() if output else stream.getvalue())


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.world = GuardWorld(Path(self.tmp.name))
        self.f1 = self.world.readout("F1", wrong={"d0", "d1"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_healthy_soups_are_finalists_and_values_are_recomputed(self):
        w = self.world
        cands = {
            n: w.readout(n, wrong={"d0", "d1"}) for n in ("M4-A20-soup", "M4-Ar-soup")
        }
        out = w.run(cands, self.f1)
        self.assertEqual(out["finalists"], ["M4-A20-soup", "M4-Ar-soup"])
        entry = out["candidates"]["M4-A20-soup"]
        self.assertEqual(entry["typed_dev"]["by_type"]["choice"]["correct"], 7)
        self.assertEqual(
            entry["typed_dev"]["by_type"]["noul"]["distinct_categories"], 2
        )
        self.assertAlmostEqual(
            entry["typed_dev"]["by_type"]["choice"]["panel_chance_report_only"], 1 / 3
        )
        self.assertEqual(entry["typed_dev"]["by_type"]["choice"]["chance"], 0.267)
        self.assertEqual(entry["proxy"]["gap"], 0.0)
        self.assertTrue(entry["retention_vs_incumbent"]["all_pass"])
        self.assertAlmostEqual(entry["cal698"]["brier"], 0.09)
        self.assertIn(str(w.dev_gold), out["inputs_sha256"])
        self.assertEqual(out["incumbent"]["name"], "F1")

    def test_collapse_rules(self):
        w = self.world
        noul = [f"d{i}" for i in range(24) if i % 3 == 1][:4]
        cands = {
            "at-chance": w.readout("at-chance", wrong=set(noul)),
            "one-category": w.readout("one-category", force_choice="x"),
            "typed-invalid": w.readout("typed-invalid", missing={"d5"}),
            "css-invalid": w.readout("css-invalid", css_missing={"t1-0"}),
            "fine": w.readout("fine"),
        }
        out = w.run(cands, self.f1)
        flags = {n: c["collapse"]["flags"] for n, c in out["candidates"].items()}
        self.assertIn(
            "noul accuracy 0.5000 is at or below chance", flags["at-chance"][0]
        )
        self.assertEqual(len(flags["one-category"]), 1)
        self.assertIn("uses 1 answer category", flags["one-category"][0])
        self.assertIn("typed_dev invalid answers", flags["typed-invalid"][0])
        self.assertIn("css_pilot invalid answers", flags["css-invalid"][0])
        self.assertEqual(flags["fine"], [])
        self.assertEqual(out["finalists"], ["fine"])

    def test_proxy_drop_is_at_least_8_below_the_best_of_incumbent_and_soups(self):
        base = {
            "typed_dev": {"by_type": {}},
            "H_pilot": 0.6,
            "cal698": {"brier": 0.1},
            "invalid_total": 0,
        }
        cells = {
            k: {"accuracy": 0.9, "distinct_categories": 3, "answer_categories": {}}
            for k in KINDS
        }
        rates = {
            "typed_dev": {"by_type": cells, "invalid_rate": 0.0},
            "css_pilot": {"invalid_rate": 0.0},
        }

        def value(p_dev):
            return {**base, **rates, "P_dev": p_dev}

        out = guard.decide(
            "F1", value(77.0), {"a": value(80.0), "b": value(72.0), "c": value(72.01)}
        )
        self.assertEqual(out["proxy_pool"]["best"], "a")
        dropped = {n: c["proxy"]["dropped"] for n, c in out["candidates"].items()}
        self.assertEqual(dropped, {"a": False, "b": True, "c": False})
        self.assertEqual(out["finalists"], ["a", "c"])
        out = guard.decide("F1", value(81.0), {"a": value(73.0)})
        self.assertEqual((out["proxy_pool"]["best"], out["finalists"]), ("F1", []))

    def test_amended_collapse_rules(self):
        def value(accuracy, categories, invalid=0.0):
            cells = {
                k: {
                    "accuracy": accuracy.get(k, 0.9),
                    "distinct_categories": len(categories.get(k, {"a": 5, "b": 5})),
                    "answer_categories": categories.get(k, {"a": 5, "b": 5}),
                }
                for k in KINDS
            }
            return {
                "typed_dev": {"by_type": cells, "invalid_rate": invalid},
                "css_pilot": {"invalid_rate": 0.0},
            }

        f1 = value({"choice": 1.0, "noul": 0.5725, "score": 0.9225}, {})
        at_chance = value({"noul": 0.49, "score": 0.20}, {})
        flags = guard.collapse(at_chance, f1)
        self.assertEqual(len(flags), 1)
        self.assertIn("score accuracy 0.2000 is at or below chance", flags[0])
        near_constant = value({}, {"noul": {"true": 96, "false": 4}})
        flags = guard.collapse(near_constant, f1)
        self.assertEqual(len(flags), 1)
        self.assertIn("noul modal answer share 0.9600", flags[0])
        self.assertEqual(
            guard.collapse(value({}, {"noul": {"true": 94, "false": 6}}), f1), []
        )

    def test_retention_is_report_only(self):
        w = self.world
        cands = {
            "type-drop": w.readout(
                "type-drop", wrong={"d0", "d1", "d4"}, cal_brier=0.2
            ),
            "h-drop": w.readout("h-drop", wrong={"d0", "d1"}, css_wrong={"t1-0"}),
        }
        out = w.run(cands, self.f1)
        checks = {
            n: c["retention_vs_incumbent"]["checks"]
            for n, c in out["candidates"].items()
        }
        self.assertFalse(checks["type-drop"]["no_type_drop_over_3_points"])
        self.assertFalse(checks["type-drop"]["cal_brier_worse_at_most_0.010"])
        self.assertTrue(checks["type-drop"]["h_pilot_drop_at_most_1.5_points"])
        self.assertFalse(checks["h-drop"]["h_pilot_drop_at_most_1.5_points"])
        self.assertTrue(checks["h-drop"]["no_type_drop_over_3_points"])
        self.assertTrue(all(c["invalid_not_increased"] for c in checks.values()))
        self.assertEqual(out["finalists"], ["type-drop", "h-drop"])

    def test_recorded_values_must_agree(self):
        w = self.world
        a = w.readout("a")
        readout = json.loads(a.read_text())
        readout["typed_dev"]["by_type"]["score"]["correct"] -= 1
        a.write_text(json.dumps(readout))
        with self.assertRaises(ValueError):
            w.run({"a": a}, self.f1)
        b = w.readout("b")
        with (b.parent / "output" / "typed-dev.predictions.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaises(ValueError):
            w.run({"b": b}, self.f1)
        c = w.readout("c")
        readout = json.loads(c.read_text())
        readout["development_proxy"] += 0.01
        c.write_text(json.dumps(readout))
        with self.assertRaises(ValueError):
            w.run({"c": c}, self.f1)

    def test_output_is_written_once_and_stdout_mode(self):
        w = self.world
        a = w.readout("a")
        w.run({"a": a}, self.f1)
        with self.assertRaises(FileExistsError):
            w.run({"a": a}, self.f1)
        printed = w.run({"a": a}, self.f1, output=False)
        self.assertEqual(printed["finalists"], ["a"])

    def test_seed_select_and_absent_arms_are_recorded(self):
        w = self.world
        run = w.root / "trainer"
        write_json(run / "BEST.json", {"checkpoint": "checkpoint-0000446"})
        write_json(
            run / "select-step-0000446-metrics.json",
            {
                "n": 700,
                "correct": 600,
                "family_macro_accuracy": 0.85,
                "family_macro_brier": 0.1,
            },
        )
        out = w.run(
            {"a": w.readout("a")},
            self.f1,
            extra=["--seed-run", f"a-s1={run}", "--absent", "M4-A20r-soup"],
        )
        self.assertEqual(
            out["seeds_select700_at_best"]["a-s1"]["best"], "checkpoint-0000446"
        )
        self.assertEqual(out["absent"], ["M4-A20r-soup"])


class ContrastWorld:
    """Typed FINAL: the four families, GROUPS groups of four variants; evidence join has two questions."""

    GROUPS = 3

    def __init__(self, root: Path) -> None:
        self.root = root
        self.items = []
        for family in FINAL_FAMILIES:
            for g in range(self.GROUPS):
                for v in range(4):
                    k = g * 4 + v
                    if family == "constraint_competition":
                        spec = {"decision": ("choice", LABELS[k % 3], 3)}
                    elif family == "exception_stack":
                        spec = {"decision": ("noul", k % 2 == 0, 3)}
                    elif family == "evidence_join":
                        spec = {
                            "a": ("choice", LABELS[k % 3], 3),
                            "b": ("noul", k % 3 == 0, 3),
                        }
                    else:
                        spec = {"decision": ("score", k % 5, 5)}
                    questions, gold = {}, {}
                    for key, (kind, value, levels) in spec.items():
                        questions[key], gold[key] = question(kind, value, levels)
                    self.items.append(
                        {
                            "id": f"{family}-{g}-{v}",
                            "family": family,
                            "group_id": f"{family}-g{g}",
                            "questions": questions,
                            "gold": gold,
                        }
                    )
        self.gold = write_jsonl(root / "gold" / "typed-final.jsonl", self.items)

    def run(self, name, wrong=(), v3=67.0, loaded=25_746_591_744) -> Path:
        out = self.root / name
        rows = []
        fam = {f: [0, 0] for f in FINAL_FAMILIES}
        typ = {k: [0, 0] for k in KINDS}
        for item in self.items:
            answers = {}
            for key, q in item["questions"].items():
                g = item["gold"][key]
                levels = len(q["criteria"]) if q["type"] == "score" else 3
                correct = (item["id"], key) not in wrong
                answers[key] = answer(q["type"], g["value"], correct, levels)
                fam[item["family"]][0] += correct
                fam[item["family"]][1] += 1
                typ[q["type"]][0] += correct
                typ[q["type"]][1] += 1
            rows.append({"id": item["id"], "answers": answers})
        predictions = write_jsonl(
            out / "output" / "typed-final.predictions.jsonl", rows
        )
        write_json(
            out / "SEAL.json",
            {"panels": {"typed-final": {"predictions_sha256": sha(predictions)}}},
        )
        write_json(
            out / "REPORT.json",
            {
                "panels": {
                    "typed-final": {
                        "T": statistics.fmean(c / n for c, n in fam.values()),
                        "by_family": {f: c / n for f, (c, n) in fam.items()},
                        "by_type": {
                            k: {"accuracy": c / n, "correct": c, "n": n}
                            for k, (c, n) in typ.items()
                        },
                    }
                },
                "parameters": {"loaded": loaded},
                "v3": {"score": v3},
            },
        )
        return out

    def main(self, runs: dict, extra=()) -> dict:
        argv = ["--gold", str(self.gold), "--draws", "300"]
        for name, path in runs.items():
            argv += ["--run", f"{name}={path}"]
        out = self.root / "contrast.json"
        if out.exists():
            out.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            m4c.main([*argv, *extra, "--output", str(out)])
        return json.loads(out.read_text())


def paired(left, right, v3_left, delta, v3_ci, t_ci=(0.01, 0.05), h_ci=(-0.02, 0.03)):
    return {
        "models": {"left": left, "right": right},
        "point": {
            "left": {"score": v3_left, "T": 0.8, "H": 0.58},
            "right": {"score": v3_left - delta, "T": 0.78, "H": 0.58},
            "delta": {"score": delta, "T": 0.02, "H": 0.0},
        },
        "ci95": {"low": v3_ci[0], "high": v3_ci[1]},
        "axis_ci95": {
            "T": {"delta": {"low": t_ci[0], "high": t_ci[1]}},
            "H": {"delta": {"low": h_ci[0], "high": h_ci[1]}},
        },
    }


class ContrastTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.world = ContrastWorld(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_family_type_points_and_paired_intervals(self):
        w = self.world
        cc = [
            (f"constraint_competition-{g}-{v}", "decision")
            for g in range(3)
            for v in range(4)
        ]
        left = w.run("L")
        right = w.run("R", wrong=set(cc[:6]) | {("evidence_join-0-0", "b")})
        out = w.main(
            {"L": left, "R": right, "R2": w.run("R2", wrong=set(cc[:6]))},
            ["--pair", "L:R", "--pair", "R:R2", "--finalist", "L"],
        )
        pair = out["pairs"]["L - R"]
        self.assertAlmostEqual(pair["delta"]["family:constraint_competition"], 0.5)
        self.assertAlmostEqual(pair["delta"]["family:evidence_join"], 1 / 24)
        self.assertAlmostEqual(pair["delta"]["family:exception_stack"], 0.0)
        self.assertAlmostEqual(pair["delta"]["type:choice"], 6 / 24)
        self.assertAlmostEqual(pair["delta"]["type:noul"], 1 / 24)
        self.assertAlmostEqual(pair["delta"]["T"], (0.5 + 1 / 24) / 4)
        self.assertEqual(pair["item_units_per_family"]["evidence_join"], 12)
        self.assertEqual(pair["group_units_per_family"]["evidence_join"], 3)
        for key in ("ci95", "ci95_group"):
            ci = pair[key]["family:constraint_competition"]
            self.assertLessEqual(ci["low"], 0.5)
            self.assertGreaterEqual(ci["high"], 0.5)
            self.assertEqual(
                pair[key]["family:exception_stack"], {"low": 0.0, "high": 0.0}
            )
            self.assertEqual(
                pair[key]["family:resource_ledger"], {"low": 0.0, "high": 0.0}
            )
        self.assertEqual(
            out["pairs"]["R - R2"]["ci95"]["family:constraint_competition"],
            {"low": 0.0, "high": 0.0},
        )
        score = out["score_levels"]["L"]["score"]
        self.assertEqual(score["predicted_distribution"], score["gold_distribution"])
        self.assertEqual(set(score["recall_by_level"].values()), {1.0})
        again = w.main({"L": left, "R": right}, ["--pair", "L:R"])
        self.assertEqual(again["pairs"]["L - R"]["ci95"], pair["ci95"])

    def test_report_values_must_agree(self):
        w = self.world
        run = w.run("L")
        report = json.loads((run / "REPORT.json").read_text())
        report["panels"]["typed-final"]["by_family"]["evidence_join"] -= 0.01
        (run / "REPORT.json").write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            w.main({"L": run})

    def gates(self, entries: dict, dose=None, capacity=None) -> Path:
        root = self.world.root / "gates"
        for name, spec in entries.items():
            folder = root / name
            write_json(
                folder / "paired-vs-F1.json",
                paired(
                    name,
                    "F1",
                    spec["v3"],
                    spec["delta"],
                    spec["ci"],
                    h_ci=spec.get("h_f1", (-0.02, 0.03)),
                ),
            )
            for key, label in m4c.PEERS.items():
                write_json(
                    folder / f"paired-vs-{key}.json",
                    paired(
                        name,
                        label,
                        spec["v3"],
                        -3.0,
                        (-5.0, -1.0),
                        h_ci=spec.get("h_peer", (-0.02, 0.03)),
                    ),
                )
            write_json(
                folder / "types.json",
                {
                    "types": {
                        k: {
                            "verdict": spec.get("verdicts", {}).get(k, "OK"),
                            "accuracy": 0.8,
                            "accuracy_ci95": [0.75, 0.85],
                            "chance": 0.3,
                            "predicted_distinct": 3,
                            "predicted_top_share": 0.4,
                        }
                        for k in KINDS
                    }
                },
            )
            write_json(
                root / f"exposure-{name}.json", {"groups": spec.get("groups", [])}
            )
        for label, value in (("dose", dose), ("capacity", capacity)):
            if value is not None:
                write_json(root / f"contrast-{label}.json", value)
        return root

    def test_successor_rule_tie_break_and_attribution(self):
        w = self.world
        runs = {
            "F1": w.run("F1"),
            "A": w.run("A", v3=69.0),
            "B": w.run("B", v3=69.0, loaded=26_096_775_168),
            "C": w.run("C", v3=68.0),
            "D": w.run("D", v3=66.0),
        }
        specs = {
            "A": {"v3": 69.0, "delta": 1.8, "ci": (0.2, 3.5)},
            "B": {"v3": 69.0, "delta": 1.8, "ci": (0.2, 3.5)},
            "C": {"v3": 68.0, "delta": 0.8, "ci": (0.5, 1.2), "groups": ["g1"]},
            "D": {
                "v3": 66.0,
                "delta": -1.2,
                "ci": (-2.0, 0.5),
                "h_f1": (-0.09, -0.01),
                "h_peer": (-0.1, -0.02),
                "verdicts": {"score": "COLLAPSED: x"},
            },
        }
        gates = self.gates(
            specs,
            dose=paired("A", "C", 69.0, 1.0, (-0.5, 2.0), t_ci=(0.004, 0.03)),
            capacity=paired("B", "A", 69.0, 0.0, (-1.0, 1.0), t_ci=(-0.01, 0.02)),
        )
        extra = [
            "--gates",
            str(gates),
            "--incumbent",
            "F1",
            "--pair",
            "A:C",
            "--pair",
            "A:F1",
        ]
        for name in specs:
            extra += [
                "--finalist",
                name,
                "--exposure",
                f"{name}={gates / f'exposure-{name}.json'}",
            ]
        rule = w.main(runs, extra)["successor_rule"]
        self.assertEqual(rule["passing"], ["A", "B"])
        self.assertEqual(rule["tie_break_order"], ["A", "B"])
        self.assertEqual(rule["successor"], "A")
        self.assertFalse(rule["finalists"]["C"]["checks"]["4_no_overlap_exposure"])
        d = rule["finalists"]["D"]["checks"]
        self.assertEqual(
            [k for k, v in d.items() if not v],
            [
                "1_v3_minus_F1_lower_bound_above_0",
                "2_H_not_significantly_below_F1",
                "3b_H_not_significantly_below_peers",
                "3c_no_type_collapsed",
            ],
        )
        self.assertTrue(d["3a_v3_at_least_64.92"])
        attribution = rule["attribution"]
        self.assertTrue(attribution["dose"]["claimed"])
        self.assertFalse(attribution["capacity"]["claimed"])
        self.assertIsNotNone(attribution["dose"]["families"])
        self.assertIsNotNone(attribution["descriptive_vs_F1"]["A"]["families"])
        specs["A"]["ci"] = (0.1, 3.5)
        gates = self.gates(specs)
        (gates / "contrast-dose.json").unlink()
        rule = w.main(runs, extra)["successor_rule"]
        self.assertEqual(rule["tie_break_order"], ["B", "A"])
        self.assertFalse(rule["attribution"]["dose"]["run"])


class ShellSyntaxTest(unittest.TestCase):
    def test_every_m4_script_parses(self):
        for script in sorted(M4.glob("*.sh")):
            with self.subTest(script.name):
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_overlap_spec_has_three_candidate_tiers(self):
        spec = json.loads((M4 / "overlap-spec-27b-m4.json").read_text())
        candidates = sorted(t["candidate"] for t in spec["tiers"].values())
        self.assertEqual(candidates, ["M4-A20-soup", "M4-A20r-soup", "M4-Ar-soup"])
        for tier in spec["tiers"].values():
            self.assertEqual(
                tier["internal_peers"], ["DEV2.0-27B (F1)", "DEV2.0-27B (F2)"]
            )
            self.assertEqual(
                tier["peers"], ["AutoJev-27B", "Eikos-27B", "Jebadiah-27B"]
            )
            for name in (tier["candidate"], *tier["peers"], *tier["internal_peers"]):
                self.assertIn(name, spec["models"])
        exposures = {
            n: m.get("exposure")
            for n, m in spec["models"].items()
            if n.startswith("M4-")
        }
        self.assertEqual(exposures["M4-A20r-soup"], exposures["M4-A20-soup"])
        self.assertTrue(exposures["M4-Ar-soup"][0].endswith("exposure-m4-ar.json"))
        self.assertEqual(len(spec["reproduce"]), 12)


EVAL_SHA = "c8a5504b5b39049390fccfbe3bcac5ef65e1e710"
ITEM_SHA = "5553298a1a7e1c8b192eab1bb7c2699689b7a3dd"
OVERLAP_SHA = "1bbebc2fe3136858bcda63a12c2cde13c45d9238"
RELAYED = ("M4-A20-s2", "M4-Ar-s1", "M4-Ar-s2")

RUN_FINALIST = r"""#!/usr/bin/env bash
exec python3 - "$@" <<'EOF'
import hashlib, json, os, pathlib, sys
name, gpu, src, *members = sys.argv[1:]
runs = pathlib.Path(os.environ["M4_RUNS"])
keys = ("STAGES", "LABEL", "FROZEN", "CACHE_SHA", "LIMIT", "EXTRA_COMPARATOR")
with open(os.environ["M4_TEST_CALLS"], "a") as log:
    log.write(json.dumps({"tool": "run_finalist", "argv": sys.argv[1:], "env": {k: os.environ.get(k) for k in keys}}) + "\n")
stages, out = os.environ["STAGES"].split(","), runs / name
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if "soup" in stages:
    entries = []
    for member in members:
        full = runs / member / "full"
        run = full / (full / "RUN_DIR").read_text().strip()
        ckpt = run / json.loads((run / "BEST.json").read_text())["checkpoint"]
        files = {p.relative_to(ckpt).as_posix(): digest(p) for p in sorted(ckpt.rglob("*")) if p.is_file()}
        entries.append({"path": str(ckpt), "files_sha256": files})
    lora = {"members": len(members), "rank": int(os.environ.get("STUB_RANK", "16")), "alpha": int(os.environ.get("STUB_ALPHA", "32"))}
    check = {"verify_adapter_config": True, "max_relative_diff": 2e-7, "tolerance_relative": 1e-6, "projections": 496}
    path = out / "soup" / "checkpoint" / "soup_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"lora": lora, "verification": check, "members": entries, "output": {"model_sha256": "0" * 64}}))
if "readout" in stages:
    path = out / "readout-kernel-32768" / "READOUT.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"development_proxy": 77.0, "typed_dev": {"T_dev": 0.87}, "css_pilot": {"H_pilot": 0.68}}))
if "adopt" in stages:
    (out / "cal698").mkdir(parents=True)
    (out / "ADOPTION.json").write_text(json.dumps({"decision": "T = 1", "worsened": ["typed-dev ECE"]}))
if "formal" in stages:
    formal = out / "formal"
    (formal / "triton-cache").mkdir(parents=True)
    (out / "package").mkdir()
    (formal / "REPORT.json").write_text(json.dumps({"v3": {"score": 68.0}, "invalid": {"typed-final": {"invalid_or_missing": 0}}}))
    (formal / "PAIRED-vs-M3-A-soup.json").write_text(json.dumps({"point": {"delta": {"score": 0.8}}, "ci95": {"low": -1.0, "high": 2.6}}))
    (formal / "triton-cache.post.json").write_text(json.dumps({"post_sha256": "ab" * 32}))
    (formal / "SEAL.json").write_text("{}")
EOF
"""

F2_MLX = r"""#!/usr/bin/env bash
exec python3 - "$@" <<'EOF'
import json, os, pathlib, sys
name, gpu, mode = sys.argv[1], sys.argv[2], (sys.argv[3:] or ["full"])[0]
keys = ("SRC", "FROZEN", "CACHE_SHA", "MLX_ROOT")
with open(os.environ["M4_TEST_CALLS"], "a") as log:
    log.write(json.dumps({"tool": "f2-mlx", "argv": sys.argv[1:], "env": {k: os.environ.get(k) for k in keys}}) + "\n")
out = pathlib.Path(os.environ["MLX_ROOT"]) / (name + ("-smoke" if mode == "smoke" else ""))
out.mkdir(parents=True)
(out / ("SMOKE.json" if mode == "smoke" else "COLLECT.json")).write_text("{}")
print(f"mlx-diag {name} ({mode}) collected into {out}")
EOF
"""

GUARD_STUB = """import json, os, sys
with open(os.environ["M4_TEST_CALLS"], "a") as log:
    log.write(json.dumps({"tool": "m4_guard", "argv": sys.argv[1:]}) + "\\n")
finalists = [n for n in os.environ.get("STUB_FINALISTS", "").split(",") if n]
with open(sys.argv[sys.argv.index("--output") + 1], "x") as stream:
    json.dump({"finalists": finalists}, stream)
print(json.dumps({"finalists": finalists}))
"""


def stub(tool: str, payload) -> str:
    return f"""import json, os, sys
with open(os.environ["M4_TEST_CALLS"], "a") as log:
    log.write(json.dumps({{"tool": {tool!r}, "argv": sys.argv[1:]}}) + "\\n")
if "--output" in sys.argv:
    with open(sys.argv[sys.argv.index("--output") + 1], "x") as stream:
        json.dump({payload!r}, stream)
print(json.dumps({payload!r}))
"""


def write_text(path: Path, text: str, mode: int = 0o644) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


def member(runs: Path, name: str) -> Path:
    root = runs / name
    run = root / "full" / "run"
    ckpt = run / "checkpoint-0000446"
    write_text(root / "full" / "RUN_DIR", "run\n")
    write_json(run / "BEST.json", {"checkpoint": "checkpoint-0000446"})
    write_json(
        run / "COMPLETE.json", {"status": "complete", "best": "checkpoint-0000446"}
    )
    write_json(
        run / "select-step-0000446-metrics.json",
        {"n": 700, "family_macro_accuracy": 0.86},
    )
    (ckpt / "adapter").mkdir(parents=True)
    (ckpt / "adapter" / "adapter_model.safetensors").write_bytes(name.encode() * 64)
    write_json(ckpt / "decision_config.json", {"lora": {"rank": 8}})
    if name in RELAYED:
        files = {
            p.relative_to(root).as_posix(): sha(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }
        write_json(
            root / "RELAY.json",
            {"run_dir": "run", "best": "checkpoint-0000446", "files_sha256": files},
        )
    return root


class TailSandbox:
    """A fake mirror holding m4-tail.sh with stub stages, fake eval mirrors and a runs root."""

    def __init__(
        self, root: Path, runs: Path | None = None, members: bool = True
    ) -> None:
        self.root, self.runs = root, runs or root / "runs"
        self.srcroot, self.calls = root / "src", root / "calls.jsonl"
        self.src = f"{'f' * 40}-src_training_decision2"
        d2 = self.srcroot / self.src / "src" / "training" / "decision2"
        self.tail = write_text(
            d2 / "v2/27b/m4/m4-tail.sh", (M4 / "m4-tail.sh").read_text()
        )
        write_text(d2 / "v2/__init__.py", "")
        write_text(d2 / "v2/27b/__init__.py", "")
        write_text(d2 / "v2/27b/run_finalist.sh", RUN_FINALIST)
        write_text(d2 / "v2/27b/m3f2/f2-mlx.sh", F2_MLX)
        write_text(d2 / "v2/27b/m4_guard.py", GUARD_STUB)
        write_text(d2 / "v2/27b/m4_contrast.py", stub("m4_contrast", {"ok": True}))
        tools = {
            EVAL_SHA: {"gates": {}, "panels": {}},
            OVERLAP_SHA: {"overlap_effects": {"groups": [], "methods_agree": True}},
        }
        for mirror, modules in tools.items():
            e = (
                self.srcroot
                / f"{mirror}-src_training_decision2/src/training/decision2/v2"
            )
            write_text(e / "__init__.py", "")
            write_text(e / "eval/__init__.py", "")
            for tool, payload in modules.items():
                write_text(e / f"eval/{tool}.py", stub(tool, payload))
        item = f"{ITEM_SHA}-src_training_decision2/src/training/decision2/v2/eval/records/m4-dev2-27b-f1-gates/item_cis.py"
        write_text(self.srcroot / item, stub("item_cis", {}))
        if members:
            for arm in ("M4-A20", "M4-A20r", "M4-Ar"):
                for seed in ("s1", "s2"):
                    member(self.runs, f"{arm}-{seed}")
        write_json(self.runs / "M3-A-soup/readout-kernel-32768/READOUT.json", {})
        write_json(self.runs / "M3-A-soup/formal/SEAL.json", {})

    def run(self, *args, **env) -> subprocess.CompletedProcess:
        full = {
            "PATH": os.environ["PATH"],
            "HOME": os.environ.get("HOME", "/"),
            "M4_RUNS": str(self.runs),
            "M4_SRCROOT": str(self.srcroot),
            "M4_TEST_CALLS": str(self.calls),
            **env,
        }
        return subprocess.run(
            ["bash", str(self.tail), *args], env=full, capture_output=True, text=True
        )

    def calls_of(self, tool: str) -> list:
        if not self.calls.exists():
            return []
        rows = [json.loads(line) for line in self.calls.read_text().splitlines()]
        return [r for r in rows if r["tool"] == tool]


class TailTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = TailSandbox(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_soup_checks_relays_rank_and_refuses_overwrite(self):
        s = self.s
        adapter = (
            s.runs
            / "M4-Ar-s1/full/run/checkpoint-0000446/adapter/adapter_model.safetensors"
        )
        original = adapter.read_bytes()
        adapter.write_bytes(original + b"x")
        r = s.run("soup", "M4-Ar")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("differs from RELAY.json", r.stderr)
        self.assertEqual(s.calls_of("run_finalist"), [])
        adapter.write_bytes(original)
        r = s.run("soup", "M4-Ar", STUB_RANK="64")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("expected 2 members, rank 16, alpha 32", r.stderr)
        r = s.run("soup", "M4-A20")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("relay verified: M4-A20-s2", r.stdout)
        self.assertIn('"rank": 16', r.stdout)
        call = s.calls_of("run_finalist")[-1]
        self.assertEqual(
            call["argv"], ["M4-A20-soup", "5", s.src, "M4-A20-s1", "M4-A20-s2"]
        )
        self.assertEqual(
            (call["env"]["STAGES"], call["env"]["LABEL"]), ("soup", "M4-A20-soup")
        )
        self.assertEqual(s.run("soup", "M4-A20-soup").returncode, 66)
        r = s.run("soup", "M4-A20r", STUB_RANK="64", STUB_ALPHA="128")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("relay verified", r.stdout)
        (s.runs / "M4-Ar-s2/RELAY.json").unlink()
        shutil.rmtree(s.runs / "M4-Ar-soup")
        self.assertNotEqual(s.run("soup", "M4-Ar").returncode, 0)
        self.assertEqual(s.run("soup", "M4-X").returncode, 2)
        self.assertIn("=== ", (s.runs / "M4-A20-soup.driver.log").read_text())

    def test_full_tail_sequence_with_stubs(self):
        s = self.s
        env = {"M4_LEFT": "5"}
        for arm in ("M4-A20", "M4-A20r"):
            self.assertEqual(
                s.run(
                    "soup",
                    arm,
                    STUB_RANK="64" if arm == "M4-A20r" else "16",
                    STUB_ALPHA="128" if arm == "M4-A20r" else "32",
                ).returncode,
                0,
            )
        self.assertEqual(s.run("readout", "M4-A20", "5").returncode, 3)
        r = s.run("readout", "M4-A20", "5", M4_LEFT="0.2")
        self.assertEqual(r.returncode, 3)
        self.assertIn("expected GPU-hours for soup readout M4-A20-soup: 0.26", r.stdout)
        self.assertEqual(s.run("readout", "M4-A20", "4", **env).returncode, 2)
        self.assertEqual(s.run("readout", "M4-Ar", "5", **env).returncode, 2)
        for arm, gpu in (("M4-A20", "5"), ("M4-A20r", "6")):
            r = s.run("readout", arm, gpu, **env)
            self.assertEqual(r.returncode, 0, r.stderr)
        call = s.calls_of("run_finalist")[-1]
        self.assertEqual(call["env"]["STAGES"], "readout")
        self.assertTrue(call["env"]["FROZEN"].endswith("m3-warm-32768/triton-cache"))
        self.assertEqual(call["env"]["CACHE_SHA"][:8], "583241fb")
        self.assertEqual(call["env"]["LIMIT"], "32768")
        self.assertEqual(s.run("readout", "M4-A20", "5", **env).returncode, 66)

        self.assertEqual(s.run("guard").returncode, 2)
        r = s.run(
            "guard", M4_ABSENT="M4-Ar-soup", STUB_FINALISTS="M4-A20-soup,M4-A20r-soup"
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        argv = s.calls_of("m4_guard")[-1]["argv"]
        self.assertIn(
            f"M4-A20-soup={s.runs}/M4-A20-soup/readout-kernel-32768/READOUT.json", argv
        )
        self.assertIn("M4-Ar-soup", argv[argv.index("--absent") + 1])
        self.assertIn(f"M4-A20-s2={s.runs}/M4-A20-s2/full/run", argv)
        self.assertIn(
            f"DEV2.0-27B (F1)={s.runs}/M3-A-soup/readout-kernel-32768/READOUT.json",
            argv,
        )
        self.assertEqual(s.run("guard", M4_ABSENT="M4-Ar").returncode, 66)

        self.assertEqual(s.run("prepkg", "M4-Ar", "5", **env).returncode, 2)
        self.assertEqual(s.run("formal", "M4-A20", "7", **env).returncode, 2)
        r = s.run("prepkg", "M4-A20", "6", **env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(
            s.calls_of("run_finalist")[-1]["env"]["STAGES"], "cal698,adopt"
        )
        self.assertEqual(s.run("prepkg", "M4-A20", "6", **env).returncode, 66)
        self.assertEqual(s.run("formal", "M4-A20", "7", M4_LEFT="0.5").returncode, 3)
        r = s.run("formal", "M4-A20", "7", **env)
        self.assertEqual(r.returncode, 0, r.stderr)
        call = s.calls_of("run_finalist")[-1]
        self.assertEqual(call["env"]["STAGES"], "package,formal")
        self.assertTrue(call["env"]["FROZEN"].endswith("m3-f2/f1-scored-cache"))
        self.assertEqual(call["env"]["CACHE_SHA"][:8], "03b172f1")
        self.assertEqual(
            call["env"]["EXTRA_COMPARATOR"], f"M3-A-soup={s.runs}/M3-A-soup/formal"
        )
        self.assertIn('"minus_F1"', r.stdout)

        self.assertEqual(s.run("mlx", "M4-A20-soup", "5", **env).returncode, 2)
        self.assertEqual(
            s.run("mlx", "M4-A20r-soup", "5", "smoke", **env).returncode, 2
        )
        r = s.run("mlx", "M4-A20-soup", "5", "smoke", M4_LEFT="0.6")
        self.assertEqual(r.returncode, 0, r.stderr)
        call = s.calls_of("f2-mlx")[-1]
        self.assertEqual(call["argv"], ["M4-A20-soup", "5", "smoke"])
        self.assertEqual(
            call["env"]["FROZEN"], f"{s.runs}/M4-A20-soup/formal/triton-cache"
        )
        self.assertEqual(call["env"]["CACHE_SHA"], "ab" * 32)
        self.assertEqual(call["env"]["MLX_ROOT"], f"{s.runs}/m4-mlx")
        self.assertEqual(call["env"]["SRC"], s.src)
        r = s.run("mlx", "M4-A20-soup", "5", M4_LEFT="0.6")
        self.assertEqual(r.returncode, 3)
        self.assertIn(
            "keeps 0.54 GPU-hours free for each of 1 formal run(s) to come: needs 0.66",
            r.stdout,
        )

        self.assertEqual(s.run("gates").returncode, 2)
        self.assertEqual(s.run("prepkg", "M4-A20r", "5", **env).returncode, 0)
        self.assertEqual(s.run("formal", "M4-A20r", "5", **env).returncode, 0)
        self.assertEqual(s.run("mlx", "M4-A20-soup", "5", M4_LEFT="0.6").returncode, 0)
        self.assertEqual(s.run("mlx", "M4-A20-soup", "5", **env).returncode, 66)
        self.assertEqual(s.run("contrast").returncode, 2)

        r = s.run("gates")
        self.assertEqual(r.returncode, 0, r.stderr)
        paired_calls = [
            c["argv"] for c in s.calls_of("gates") if c["argv"][0] == "paired"
        ]
        self.assertEqual(len(paired_calls), 13)
        outputs = {
            Path(a[a.index("--output") + 1]).relative_to(s.runs / "m4-gates").as_posix()
            for a in paired_calls
        }
        self.assertIn("M4-A20r-soup/paired-F1-minus-cand.json", outputs)
        self.assertIn("contrast-capacity.json", outputs)
        self.assertNotIn("contrast-dose.json", outputs)
        capacity = next(
            a for a in paired_calls if a[-1].endswith("contrast-capacity.json")
        )
        self.assertEqual(capacity[capacity.index("--left-name") + 1], "M4-A20r-soup")
        self.assertEqual(
            len([c for c in s.calls_of("gates") if c["argv"][0] == "types"]), 2
        )
        cfg = json.loads(s.calls_of("item_cis")[-1]["argv"][0])
        self.assertEqual(len(cfg["pairs"]), 11)
        self.assertIn(["M4-A20r-soup", "M4-A20-soup"], cfg["pairs"])
        self.assertEqual(len(cfg["models"]), 7)
        self.assertTrue((s.runs / "m4-gates/SHA256SUMS.txt").is_file())
        self.assertEqual(s.run("gates").returncode, 66)

        r = s.run("overlap", str(M4 / "overlap-spec-27b-m4.json"))
        self.assertEqual(r.returncode, 0, r.stderr)
        spec = json.loads((s.runs / "m4-overlap/spec.json").read_text())
        self.assertEqual(sorted(spec["tiers"]), ["27B M4-A20", "27B M4-A20r"])
        self.assertNotIn("M4-Ar-soup", spec["models"])
        self.assertEqual(len(spec["reproduce"]), 8)
        exposures = [
            c["argv"]
            for c in s.calls_of("overlap_effects")
            if c["argv"][0] == "exposure"
        ]
        self.assertEqual(
            [a[a.index("--expect-sha256") + 1][:8] for a in exposures],
            ["4aa0dc96", "aadeef1a"],
        )
        self.assertEqual(
            s.run("overlap", str(M4 / "overlap-spec-27b-m4.json")).returncode, 66
        )

        r = s.run("contrast")
        self.assertEqual(r.returncode, 0, r.stderr)
        argv = s.calls_of("m4_contrast")[-1]["argv"]
        pairs = [argv[i + 1] for i, a in enumerate(argv) if a == "--pair"]
        self.assertEqual(
            pairs,
            [
                "M4-A20-soup:DEV2.0-27B (F1)",
                "M4-A20r-soup:DEV2.0-27B (F1)",
                "M4-A20r-soup:M4-A20-soup",
            ],
        )
        self.assertIn(f"M4-A20r-soup={s.runs}/m4-overlap/exposure-m4-a20.json", argv)
        self.assertEqual(s.run("contrast").returncode, 66)


FAKE_SSH = r"""#!/usr/bin/env bash
while [ "${1:-}" = -o ]; do shift 2; done
host=$1
shift
cmd="$*"
cmd=${cmd//\/data\/dev2\//$FAKE_SANDBOX/$host/data/dev2/}
export FAKE_HOST=$host
exec bash -c "$cmd"
"""

FAKE_DOCKER = r"""#!/usr/bin/env bash
case "$1" in
  ps) cat "$FAKE_SANDBOX/$FAKE_HOST/docker-ps.txt" 2>/dev/null || true ;;
  inspect) cat "$FAKE_SANDBOX/$FAKE_HOST/docker-started.txt" 2>/dev/null || true ;;
esac
"""


class Nodes:
    """Two fake nodes under one directory: ssh HOST CMD runs CMD locally with /data/dev2 mapped to HOST's tree."""

    def __init__(self, root: Path) -> None:
        self.root = root
        write_text(root / "bin/ssh", FAKE_SSH, 0o755)
        write_text(root / "bin/docker", FAKE_DOCKER, 0o755)
        write_text(root / "nodes.env", "node-a=hosta\nnode-b=hostb\n")
        self.env = {
            **os.environ,
            "PATH": f"{root / 'bin'}:{os.environ['PATH']}",
            "FAKE_SANDBOX": str(root),
            "DEV2_NODES_FILE": str(root / "nodes.env"),
        }

    def runs(self, host: str) -> Path:
        return self.root / host / "data/dev2/runs/27b"

    def run(self, script: str, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(M4 / script), *args],
            env=self.env,
            capture_output=True,
            text=True,
        )


def node_a_arm(runs: Path, arm: str) -> Path:
    root = runs / arm
    run = root / "full" / "run"
    write_text(
        root / "driver.log",
        f"=== launch {arm}\narm {arm} stages admit,onestep,reload,full complete\n",
    )
    write_text(root / "full/RUN_DIR", "run\n")
    write_json(run / "BEST.json", {"checkpoint": "checkpoint-0000446"})
    write_json(
        run / "COMPLETE.json", {"status": "complete", "best": "checkpoint-0000446"}
    )
    write_json(run / "LATEST.json", {"checkpoint": "checkpoint-0000892"})
    write_json(run / "provenance.json", {"seed": 20260928})
    for step in ("0000446", "0000892"):
        write_json(
            run / f"select-step-{step}-metrics.json", {"family_macro_accuracy": 0.8}
        )
        write_jsonl(run / f"select-step-{step}-predictions.jsonl", [{"id": "s"}])
        ckpt = run / f"checkpoint-{step}"
        write_text(ckpt / "adapter/adapter_model.safetensors", f"{arm}{step}" * 50)
        write_json(ckpt / "adapter/adapter_config.json", {"r": 8})
        write_text(ckpt / "adapter/README.md", "# adapter\n")
        for name in (
            "decision_head.safetensors",
            "decision_config.json",
            "checkpoint.json",
            "tokenizer.json",
        ):
            write_text(ckpt / name, f"{name}{step}")
        write_text(ckpt / "trainer_state.pt", "optimizer" * 100)
    write_jsonl(run / "train-metrics.jsonl", [{"step": 1}])
    for stage, hours in (("onestep", 0.065), ("reload", 0.018), ("full", 10.0)):
        write_json(
            root / f"receipts/{stage}.json",
            {"schema_version": "decision2-27b-launch-receipt/1", "gpu_hours": hours},
        )
        write_text(root / f"receipts/{stage}.container.log", f"{stage} log\n")
    write_json(root / "triton-cache.copy.json", {"copy_sha256": "19" * 32})
    write_json(root / "triton-cache.post.json", {"post_sha256": "19" * 32})
    write_text(root / "triton-cache/entry.autotune.json", "{}")
    write_json(root / "admit/admission.json", {"all_admitted": True})
    return root


class RelayTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.nodes = Nodes(Path(self.tmp.name))
        self.a = node_a_arm(self.nodes.runs("hosta"), "M4-A20-s2")
        self.b = self.nodes.runs("hostb") / "M4-A20-s2"

    def tearDown(self):
        self.tmp.cleanup()

    def test_relay_copies_only_what_the_soup_needs_and_is_idempotent(self):
        r = self.nodes.run("m4-relay-best.sh", "M4-A20-s2")
        self.assertEqual(r.returncode, 0, r.stderr)
        got = sorted(
            p.relative_to(self.b).as_posix() for p in self.b.rglob("*") if p.is_file()
        )
        ckpt = "full/run/checkpoint-0000446"
        expected = sorted(
            [
                "RELAY.json",
                "driver.log",
                "full/RUN_DIR",
                "triton-cache.copy.json",
                "triton-cache.post.json",
            ]
            + [
                f"full/run/{n}.json"
                for n in ("BEST", "COMPLETE", "LATEST", "provenance")
            ]
            + [f"full/run/select-step-{s}-metrics.json" for s in ("0000446", "0000892")]
            + [
                f"{ckpt}/{n}"
                for n in (
                    "adapter/adapter_model.safetensors",
                    "adapter/adapter_config.json",
                    "adapter/README.md",
                    "decision_head.safetensors",
                    "decision_config.json",
                    "checkpoint.json",
                    "tokenizer.json",
                )
            ]
            + [
                f"receipts.node-a/{s}{e}"
                for s in ("onestep", "reload", "full")
                for e in (".json", ".container.log")
            ]
        )
        self.assertEqual(got, expected)
        relay = json.loads((self.b / "RELAY.json").read_text())
        for name, digest in relay["files_sha256"].items():
            source = self.a / name.replace("receipts.node-a/", "receipts/")
            self.assertEqual(sha(source), digest)
            self.assertEqual(sha(self.b / name), digest)
        self.assertEqual(
            (relay["run_dir"], relay["best"]), ("run", "checkpoint-0000446")
        )
        self.assertIn("23 files (0.0 MB): 23 sent", r.stdout)
        self.assertIn("SHA-256 lists equal", r.stdout)
        self.assertFalse((self.b.parent / "M4-A20-s2.relay-pending").exists())
        before = sha(self.b / "RELAY.json")
        again = self.nodes.run("m4-relay-best.sh", "M4-A20-s2")
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("0 sent", again.stdout)
        self.assertEqual(sha(self.b / "RELAY.json"), before)

        tail = TailSandbox(
            self.nodes.root / "tail", runs=self.nodes.runs("hostb"), members=False
        )
        member(tail.runs, "M4-A20-s1")
        r = tail.run("soup", "M4-A20")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(
            "relay verified: M4-A20-s2 checkpoint-0000446 (23 files)", r.stdout
        )

    def test_relay_refuses_differing_bytes_and_unfinished_runs(self):
        self.assertEqual(self.nodes.run("m4-relay-best.sh", "M4-A20-s2").returncode, 0)
        (self.b / "driver.log").write_text("changed\n")
        r = self.nodes.run("m4-relay-best.sh", "M4-A20-s2")
        self.assertEqual(r.returncode, 1)
        self.assertIn("nothing overwritten", r.stderr)
        self.assertEqual((self.b / "driver.log").read_text(), "changed\n")
        write_text(self.b / "stray.txt", "x")
        self.assertEqual(self.nodes.run("m4-relay-best.sh", "M4-A20-s2").returncode, 1)

        other = node_a_arm(self.nodes.runs("hosta"), "M4-Ar-s1")
        write_text(other / "driver.log", "=== launch\n")
        r = self.nodes.run("m4-relay-best.sh", "M4-Ar-s1")
        self.assertEqual(r.returncode, 1)
        self.assertIn("has not finished", r.stderr)
        node_a_arm(self.nodes.runs("hosta"), "M4-Ar-s1")
        write_text(self.nodes.root / "hosta/docker-ps.txt", "abc123\n")
        self.assertEqual(self.nodes.run("m4-relay-best.sh", "M4-Ar-s1").returncode, 1)
        (self.nodes.root / "hosta/docker-ps.txt").unlink()
        write_json(
            other / "full/run/COMPLETE.json",
            {"status": "complete", "best": "checkpoint-0000892"},
        )
        r = self.nodes.run("m4-relay-best.sh", "M4-Ar-s1")
        self.assertEqual(r.returncode, 1)
        self.assertIn("frozen BEST", r.stderr)
        self.assertFalse((self.nodes.runs("hostb") / "M4-Ar-s1").exists())
        self.assertEqual(self.nodes.run("m4-relay-best.sh", "M4-A20-s1").returncode, 2)


class BudgetTest(unittest.TestCase):
    def test_receipts_are_summed_once_across_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            nodes = Nodes(Path(tmp))
            a, b = nodes.runs("hosta"), nodes.runs("hostb")
            node_a_arm(a, "M4-A20-s2")
            write_json(
                a / "M4-xnode-A20-s1/receipts/onestep.json",
                {
                    "schema_version": "decision2-27b-launch-receipt/1",
                    "gpu_hours": 0.071,
                },
            )

            def receipt(path, hours, gpu_time=False):
                key = (
                    ("schema", "dev2-gpu-time/1")
                    if gpu_time
                    else ("schema_version", "decision2-27b-launch-receipt/1")
                )
                write_json(path, {key[0]: key[1], "gpu_hours": hours})

            receipt(b / "M4-A20-s1/receipts/onestep.json", 0.077)
            receipt(b / "M4-A20-s1/receipts/full.json", 10.2)
            relayed = b / "M4-A20-s2"
            receipt(relayed / "receipts.node-a/full.json", 10.0)
            receipt(relayed / "receipts/full.json", 10.0)
            write_json(relayed / "RELAY.json", {"files_sha256": {}})
            soup = b / "M4-A20-soup"
            receipt(soup / "readout-kernel-32768/GPU-TIME.json", 0.172, True)
            receipt(soup / "readout-kernel-32768/receipts/cal.json", 0.079)
            receipt(soup / "receipts/cal698.json", 0.044)
            receipt(soup / "formal-smoke/GPU-TIME.json", 0.108, True)
            receipt(soup / "formal/GPU-TIME.json", 0.424, True)
            write_json(soup / "receipts/notes.json", {"gpu_hours": 5.0})
            write_json(soup / "soup/checkpoint/soup_manifest.json", {"lora": {}})
            receipt(b / "m4-mlx/M4-A20-soup/GPU-TIME.json", 0.11, True)
            receipt(b / "m4-mlx/M4-A20-soup-smoke/GPU-TIME.json", 0.03, True)
            started = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
                "%Y-%m-%dT%H:%M:%S.000000000Z"
            )
            write_text(
                nodes.root / "hostb/docker-ps.txt",
                "d2-27b-M4-A20r-s1-full\nunrelated\n",
            )
            write_text(nodes.root / "hostb/docker-started.txt", started + "\n")
            write_json(
                b / "m4-logs/heartbeat-b.json",
                {
                    "arms": [
                        {
                            "running": {"container": "d2-27b-M4-A20r-s1-full"},
                            "projected_full_gpu_hours": 11.0,
                        }
                    ]
                },
            )
            r = nodes.run("m4-budget.sh", "--json")
            self.assertEqual(r.returncode, 0, r.stderr)
            out = json.loads(r.stdout)
            expected = (
                0.065
                + 0.018
                + 10.0
                + 0.071
                + 0.077
                + 10.2
                + 0.172
                + 0.079
                + 0.044
                + 0.108
                + 0.424
                + 0.11
                + 0.03
            )
            self.assertAlmostEqual(out["receipts_gpu_hours"], expected)
            self.assertEqual(
                out["relayed_copies_skipped"], {"a": [], "b": ["M4-A20-s2"]}
            )
            self.assertEqual(
                [r["container"] for r in out["running"]], ["d2-27b-M4-A20r-s1-full"]
            )
            self.assertAlmostEqual(out["running"][0]["hours"], 1.0, places=1)
            self.assertAlmostEqual(out["M4_LEFT"], 70 - expected - 11.0)
            kinds = {(e["kind"], e["owner"]) for e in out["by_owner"]}
            self.assertIn(("probe", "M4-xnode-A20-s1"), kinds)
            self.assertIn(("mlx-diag", "M4-A20-soup-smoke"), kinds)
            text = nodes.run("m4-budget.sh")
            self.assertIn(f"M4_LEFT={70 - expected - 11.0:.2f}", text.stdout)
            receipt(a / "M4-A20-s1/receipts/full.json", 9.9)
            clash = nodes.run("m4-budget.sh")
            self.assertEqual(clash.returncode, 1)
            self.assertIn("differs between node", clash.stderr)


class MlxScoreTest(unittest.TestCase):
    def test_wrapper_scores_the_m4_root_on_node_a(self):
        with tempfile.TemporaryDirectory() as tmp:
            nodes = Nodes(Path(tmp))
            col = nodes.runs("hostb") / "m4-mlx" / "M4-A20-soup"
            for name in (
                "COLLECT.json",
                "GPU-TIME.json",
                "triton-cache.copy.json",
                "triton-cache.post.json",
            ):
                write_json(col / name, {})
            write_jsonl(col / "output/mlx-diag.predictions.jsonl", [{"id": "m1"}])
            write_text(col / "logs/mlx-diag.log", "ok\n")
            score = {
                "type_macro_accuracy": 0.8,
                "english_type_macro_accuracy": 0.85,
                "non_english_type_macro_accuracy": 0.79,
                "cross_language_consistency": 0.7,
                "invalid_or_missing": 0,
                "per_language_mean_accuracy": {"en": 0.85},
                "by_type": {
                    "choice": {
                        "languages": {"en": {"correct": 1, "n": 1}},
                        "gap_vs_english": 0.0,
                    }
                },
            }
            mirror = (
                nodes.root
                / f"hosta/data/dev2/src/{EVAL_SHA}-src_training_decision2/src/training/decision2/v2"
            )
            write_text(mirror / "__init__.py", "")
            write_text(mirror / "eval/__init__.py", "")
            write_text(
                mirror / "eval/multilingual_panel.py",
                "import json, sys\n"
                f"open(sys.argv[sys.argv.index('--output') + 1], 'x').write(json.dumps({score!r}))\n",
            )
            r = nodes.run("m4-mlx-score.sh", "M4-A20-soup")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('"type_macro_accuracy": 0.8', r.stdout)
            copied = (
                nodes.runs("hosta")
                / "m4-mlx/M4-A20-soup/output/mlx-diag.predictions.jsonl"
            )
            self.assertEqual(
                sha(copied), sha(col / "output/mlx-diag.predictions.jsonl")
            )


if __name__ == "__main__":
    unittest.main()
