"""CPU tests for decoder M15: the multilingual-preserving TRAIN (m15_data), the MLX-DEV-M15 panel and the MLX guard."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, HERE / "ops" / "m15" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_lines(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class DataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = load("m15_data")
        self.tmp = Path(tempfile.mkdtemp())
        rows, ids = [], []

        def add(rid, group, lang, block, tokens, family="fam"):
            rows.append(
                {
                    "id": rid,
                    "group_id": group,
                    "family": family,
                    "language": lang,
                    "label": "1",
                }
            )
            ids.append({"id": rid, "block": block, "tokens": tokens})

        for g in range(400):
            add(f"en{g}", f"ge{g}", "en", "base", 3)
        for g in range(300):
            add(f"zh{g}", f"gz{g}", "zh", "base", 2)
        for g in range(100):
            add(f"ja{g}a", f"gj{g}", "ja", "base", 1)
            add(f"ja{g}b", f"gj{g}", "ja", "base", 1)
        add("mix0", "gm", "en", "base", 5)
        add("mix1", "gm", "de", "base", 5)
        for g in range(30):
            for k in ("", "~c2"):
                add(f"ib{g}{k}", f"ib:g{g}", "en", "ib1", 4, family=f"f{g % 3}")
        self.train = write_lines(self.tmp / "train.jsonl", rows)
        self.ids = write_lines(self.tmp / "train.ids.jsonl", ids)
        base_ids = [r["id"] for r in rows if not r["id"].startswith("ib")]
        self.teacher = write_lines(
            self.tmp / "teacher.jsonl",
            [{"id": i, "teacher_probs": {"K1": 0.5, "K2": 0.5}} for i in base_ids],
        )

    def build(self, name: str, share: float | None = None) -> dict:
        args = argparse.Namespace(
            arm_train=self.train,
            arm_sha=sha(self.train),
            arm_ids=self.ids,
            teacher=self.teacher,
            teacher_sha=sha(self.teacher),
            name=name,
            ib_share=share,
            seed=7,
            output=self.tmp / "out",
        )
        return self.data.build(args)

    def test_full_dose_keeps_arm_and_restores_share(self) -> None:
        r = self.build("full")
        out = self.tmp / "out" / "full"
        lines = (out / "train.jsonl").read_bytes().splitlines(keepends=True)
        arm = self.train.read_bytes().splitlines(keepends=True)
        self.assertEqual(lines[: len(arm)], arm)
        copies = [json.loads(x) for x in lines[len(arm) :]]
        self.assertTrue(copies)
        self.assertTrue(all(c["id"].endswith("~m2") for c in copies))
        self.assertTrue(all(c["language"] != "en" for c in copies))
        self.assertNotIn("mix1~m2", {c["id"] for c in copies})
        groups = {}
        for c in copies:
            groups.setdefault(c["group_id"], set()).add(c["id"])
        for g, members in groups.items():
            if g.startswith("gj"):
                self.assertEqual(len(members), 2)
        self.assertAlmostEqual(
            r["train"]["ml_share"], r["base"]["ml_share"], delta=self.data.TOL
        )
        self.assertEqual(r["ib"]["rows"], 60)

    def test_teacher_extended_to_copies(self) -> None:
        r = self.build("full")
        out = self.tmp / "out" / "full"
        teacher = [
            json.loads(x) for x in (out / "teacher-ml.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            (out / "teacher-ml.jsonl").read_bytes()[: self.teacher.stat().st_size],
            self.teacher.read_bytes(),
        )
        copy_ids = {
            json.loads(x)["id"]
            for x in (out / "train.jsonl").read_text().splitlines()
            if "~m2" in json.loads(x)["id"]
        }
        self.assertEqual({t["id"] for t in teacher if "~m2" in t["id"]}, copy_ids)
        self.assertEqual(r["upsample"]["teacher_copies"], len(copy_ids))

    def test_lower_dose_subsets_ib_by_level(self) -> None:
        r = self.build("low", share=0.08)
        T = r["base"]["tokens"]
        self.assertLessEqual(r["ib"]["tokens"], int(0.08 * T))
        self.assertGreater(r["ib"]["tokens"], 120)
        self.assertEqual(r["ib"]["by_level"].get(1), 30)
        self.assertLess(r["ib"]["by_level"].get(2, 0), 30)
        self.assertLess(
            r["upsample"]["tokens"], self.build("full2")["upsample"]["tokens"]
        )
        self.assertAlmostEqual(
            r["train"]["ml_share"], r["base"]["ml_share"], delta=self.data.TOL
        )

    def test_rejects_wrong_hash(self) -> None:
        args = argparse.Namespace(
            arm_train=self.train,
            arm_sha="0" * 64,
            arm_ids=self.ids,
            teacher=self.teacher,
            teacher_sha=sha(self.teacher),
            name="x",
            ib_share=None,
            seed=7,
            output=self.tmp / "out",
        )
        with self.assertRaises(ValueError):
            self.data.build(args)


class PanelTest(unittest.TestCase):
    def test_drops_overlapping_groups(self) -> None:
        mod = load("m15_mlxpanel")
        tmp = Path(tempfile.mkdtemp())
        panel = [
            {"id": "p1", "input_sha256": "h1", "group_id": "g1"},
            {"id": "p2", "input_sha256": "h2", "group_id": "g1"},
            {"id": "p3", "input_sha256": "h3", "group_id": "g2"},
            {"id": "p4", "input_sha256": "h4", "group_id": "g3"},
            {"id": "p5", "input_sha256": "h5", "group_id": "g4"},
        ]
        index = [
            {"id": p["id"], "group_id": p["group_id"], "cell": c, "language": "zh"}
            for p, c in zip(panel, ("a", "a", "a", "b", "b"))
        ]
        train = [
            {"id": "t1", "input_sha256": "h2", "group_id": "x"},
            {"id": "p4", "input_sha256": "z", "group_id": "y"},
        ]
        pp = write_lines(tmp / "panel.jsonl", panel)
        ip = write_lines(tmp / "index.jsonl", index)
        tp = write_lines(tmp / "train.jsonl", train)
        args = argparse.Namespace(
            panel=pp,
            panel_sha=sha(pp),
            index=ip,
            index_sha=sha(ip),
            train=[f"{tp}={sha(tp)}"],
            name="t",
            output=tmp / "out",
        )
        r = mod.build(args)
        kept = [
            json.loads(x)["id"]
            for x in (tmp / "out" / "panel.jsonl").read_text().splitlines()
        ]
        self.assertEqual(kept, ["p3", "p5"])
        self.assertEqual(r["dropped_groups_total"], 2)
        train.append({"id": "q", "input_sha256": "q", "group_id": "g4"})
        tp2 = write_lines(tmp / "train2.jsonl", train)
        args.train, args.output = [f"{tp2}={sha(tp2)}"], tmp / "out2"
        with self.assertRaises(ValueError):
            mod.build(args)


class GuardTest(unittest.TestCase):
    def test_guard_on_upper_bounds(self) -> None:
        rules = load("m15_rules")
        tmp = Path(tempfile.mkdtemp())
        ref = tmp / "4b-R" / "mlxdev"
        ref.mkdir(parents=True)
        (ref / "mlxdev-predictions.jsonl").write_text('{"id": "x"}\n')

        def compare(point: str, noul_hi: float, choice_hi: float) -> None:
            d = tmp / point / "mlxcmp"
            d.mkdir(parents=True)
            metric = lambda hi: {
                "a": 0.5,
                "b": 0.5,
                "diff": hi - 0.01,
                "ci95": [hi - 0.02, hi],
            }
            (d / f"{point}.mlxdev.json").write_text(
                json.dumps(
                    {
                        "a_sha256": sha(ref / "mlxdev-predictions.jsonl"),
                        "metrics": {
                            "noul_ml": metric(noul_hi),
                            "choice_ml": metric(choice_hi),
                            "score_ml": metric(-0.1),
                            "m_dev": metric(-0.1),
                        },
                    }
                )
            )

        compare("4b-A", 0.01, 0.0)
        compare("4b-B", 0.01, -0.001)
        row, reasons = rules.mlx_guard(tmp, "4b-A", "4b-R")
        self.assertEqual(reasons, [])
        self.assertEqual(row["score_ml"]["ci95"][1], -0.1)
        _, reasons = rules.mlx_guard(tmp, "4b-B", "4b-R")
        self.assertEqual(len(reasons), 1)
        self.assertIn("choice_ml", reasons[0])
        _, reasons = rules.mlx_guard(tmp, "4b-C", "4b-R")
        self.assertIn("no compare", reasons[0])
        _, reasons = rules.mlx_guard(tmp, "4b-A", "4b-other")
        self.assertIn("not against", reasons[0])


if __name__ == "__main__":
    unittest.main()
