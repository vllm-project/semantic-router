"""CPU tests for decoder M17: the IB-swap TRAIN at LH's token count (m17_data)."""

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
        name, HERE / "ops" / "m17" / f"{name}.py"
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
        self.data = load("m17_data")
        self.tmp = Path(tempfile.mkdtemp())
        base, base_tok = [], []

        def row(rid, group, lang, family, ttype="choice"):
            return {
                "id": rid,
                "group_id": group,
                "family": family,
                "language": lang,
                "task_type": ttype,
                "label": "1",
            }

        def add(rid, group, lang, tokens, family, ttype="choice"):
            base.append(row(rid, group, lang, family, ttype))
            base_tok.append(tokens)

        for g in range(300):
            add(
                f"en{g}",
                f"ge{g}",
                "en",
                1 + g % 5,
                f"fam{g % 4}",
                ("choice", "noul", "score")[g % 3],
            )
        for g in range(200):
            add(f"zh{g}", f"gz{g}", "zh", 3, "famz")
        add("mix-en", "gmix", "en", 4, "fam0")
        add("mix-de", "gmix", "de", 4, "fam0")
        add("mf-a", "gmf", "en", 2, "fam1")
        add("mf-b", "gmf", "en", 2, "fam2")
        self.base_rows = base
        self.T = sum(base_tok)
        self.base = write_lines(self.tmp / "lh.jsonl", base)
        ib, ib_meta = [], []
        for g in range(120):
            fam = ("sentfin", "args", "hover", "fc_rel")[g % 4]
            blk = "ib1" if fam in ("sentfin", "args") else "ib2"
            ib.append(row(f"ib{g}", f"ib:{g // 2}", "en", fam))
            ib_meta.append({"id": f"ib{g}", "block": blk, "tokens": 2 + g % 3})
        base_meta = [
            {"id": r["id"], "block": "base", "tokens": t}
            for r, t in zip(base, base_tok)
        ]
        self.counted = write_lines(self.tmp / "a10.jsonl", base + ib[:40])
        self.counted_ids = write_lines(
            self.tmp / "a10.ids.jsonl", base_meta + ib_meta[:40]
        )
        self.pool = write_lines(self.tmp / "a25.jsonl", base + ib)
        self.pool_ids = write_lines(self.tmp / "a25.ids.jsonl", base_meta + ib_meta)
        self.teacher = write_lines(
            self.tmp / "teacher.jsonl",
            [
                {"id": r["id"], "teacher_probs": {"K1": 0.5, "K2": 0.5}}
                for r in reversed(base)
            ],
        )

    def args(
        self, out: str, teacher: Path | None = None, arms=("s10=0.10", "s17=0.17")
    ) -> argparse.Namespace:
        teacher = teacher or self.teacher
        return argparse.Namespace(
            base=self.base,
            base_sha=sha(self.base),
            base_train=self.counted,
            base_train_sha=sha(self.counted),
            base_ids=self.counted_ids,
            pool_train=self.pool,
            pool_sha=sha(self.pool),
            pool_ids=self.pool_ids,
            teacher=teacher,
            teacher_sha=sha(teacher),
            drop=["sentfin"],
            arm=list(arms),
            seed=20261002,
            output=self.tmp / out,
        )

    def test_swap_keeps_tokens_and_non_english_rows(self) -> None:
        report = self.data.build(self.args("out"))
        self.assertEqual(report["T"], self.T)
        for name in ("s10", "s17"):
            arm = report["arms"][name]
            out = self.tmp / "out" / name
            rows = [
                json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()
            ]
            meta = [
                json.loads(x)
                for x in (out / "train.ids.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["id"] for r in rows], [m["id"] for m in meta])
            self.assertEqual(sum(m["tokens"] for m in meta), arm["train"]["tokens"])
            self.assertGreaterEqual(arm["train"]["tokens"], self.T)
            self.assertLessEqual(
                arm["train"]["tokens"] - self.T, self.data.TOL * self.T
            )
            ids = {r["id"] for r in rows}
            for r in self.base_rows:
                if r["language"] != "en":
                    self.assertIn(r["id"], ids)
            self.assertIn("mix-en", ids)
            self.assertFalse(any(r["family"] == "sentfin" for r in rows))
            ib = [m for m in meta if m["block"] != "base"]
            self.assertEqual(sum(m["tokens"] for m in ib), arm["ib"]["tokens"])
            self.assertEqual(arm["removed"]["removed_tokens"], arm["ib"]["tokens"])
            self.assertLessEqual(arm["ib"]["tokens"], arm["ib"]["target"])
            n_base = sum(1 for m in meta if m["block"] == "base")
            self.assertTrue(all(m["block"] == "base" for m in meta[:n_base]))
            lh = self.base.read_bytes().splitlines(keepends=True)
            kept = (out / "train.jsonl").read_bytes().splitlines(keepends=True)[:n_base]
            self.assertEqual(kept, [x for x in lh if json.loads(x)["id"] in ids])

    def test_teacher_covers_exactly_the_kept_rows_in_teacher_order(self) -> None:
        self.data.build(self.args("out"))
        out = self.tmp / "out" / "s17"
        kept = [
            json.loads(x)["id"]
            for x in (out / "train.ids.jsonl").read_text().splitlines()
            if json.loads(x)["block"] == "base"
        ]
        lines = (out / "teacher-s.jsonl").read_bytes().splitlines(keepends=True)
        self.assertEqual(sorted(json.loads(x)["id"] for x in lines), sorted(kept))
        full = self.teacher.read_bytes().splitlines(keepends=True)
        self.assertEqual(lines, [x for x in full if json.loads(x)["id"] in set(kept)])

    def test_builds_are_deterministic(self) -> None:
        a = self.data.build(self.args("a"))
        b = self.data.build(self.args("b"))
        for name in ("s10", "s17"):
            self.assertEqual(
                a["arms"][name]["train"]["sha256"], b["arms"][name]["train"]["sha256"]
            )
            self.assertEqual(
                a["arms"][name]["teacher"]["sha256"],
                b["arms"][name]["teacher"]["sha256"],
            )

    def test_incomplete_teacher_fails(self) -> None:
        part = write_lines(
            self.tmp / "part.jsonl",
            [{"id": r["id"], "teacher_probs": {"K1": 1.0}} for r in self.base_rows[1:]],
        )
        with self.assertRaises(ValueError):
            self.data.build(self.args("bad", teacher=part))

    def test_wrong_hash_fails(self) -> None:
        args = self.args("bad")
        args.pool_sha = "0" * 64
        with self.assertRaises(ValueError):
            self.data.build(args)


class RulesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = load("m17_rules")
        self.tmp = Path(tempfile.mkdtemp())

    def row(self, point, transfer, gain=False, eligible=True, gates=7, card=0.0):
        line, alpha = self.rules.line_alpha(point)
        return {
            "point": point,
            "line": line,
            "alpha": alpha,
            "eligible": eligible,
            "gates_passed": gates,
            "ib_dev": {"transfer_delta": transfer},
            "htdev2": {"verdict": "GAIN" if gain else "TIE"},
            "mlx_dev2": {"card_delta": card},
        }

    def test_gate_mapping(self) -> None:
        g = self.rules.gate_of
        self.assertEqual(g("type floor choice: 700 < 728 - 0.03*1000"), 1)
        self.assertEqual(g("family floor x: 1/2 < 2/2 - 0.10"), 1)
        self.assertEqual(g("Noul floor: rule_precedence 3 < 9 - 0.01*400"), 1)
        self.assertEqual(g("Score floor: Score5-typed-DEV check half COLLAPSE"), 2)
        self.assertEqual(g("Score5-typed-DEV readout missing"), 2)
        self.assertEqual(g("HT-DEV v2 FLAG (-0.0300)"), 3)
        self.assertEqual(
            g("retention: probe macro delta -0.03 CI [-0.05, -0.01] below 0"), 4
        )
        self.assertEqual(
            g("yes-bias guard: hs1-dev false-yes 0.4 > 4b-LH-f 0.2 + 0.1"), 5
        )
        self.assertEqual(g("breadth: transfer macro delta -0.01 < 0 vs 4b-LH-f"), 6)
        self.assertEqual(
            g("MLX-DEV2 guard: card delta -0.0200, upper bound -0.0100 < 0 vs 4b-LH-f"),
            7,
        )

    def test_points_and_alpha(self) -> None:
        self.assertEqual(self.rules.line_alpha("4b-LHS17SD"), ("4b-LHS17SD", 1.0))
        self.assertAlmostEqual(self.rules.line_alpha("4b-LHS10SD-a33")[1], 1 / 3)
        with self.assertRaises(ValueError):
            self.rules.line_alpha("4b-LHS10SD-a50")

    def test_mlx2_guard(self) -> None:
        lines, root = self.tmp / "lines", self.tmp / "mlx2"
        (root / "4b-LH-f").mkdir(parents=True)
        ref = root / "4b-LH-f" / "mlx-dev2.predictions.jsonl"
        ref.write_text('{"id": "a", "answers": ["x"]}\n')
        out = lines / "4b-LHS10SD" / "mlx2cmp"
        out.mkdir(parents=True)

        read_sha = sha(ref)

        def cmp(high):
            (out / "4b-LHS10SD.mlx2.json").write_text(
                json.dumps(
                    {
                        "candidate_name": "4b-LHS10SD",
                        "reference_name": "4b-LH-f",
                        "predictions_sha256": {"candidate": "c", "reference": read_sha},
                        "card_eligible": {
                            "delta": high - 0.01,
                            "ci95": {"low": high - 0.02, "high": high},
                        },
                        "per_type": {},
                        "guard_pass": high >= 0,
                    }
                )
            )
            return self.rules.mlx2_guard(lines, root, "4b-LHS10SD", "4b-LH-f")

        self.assertEqual(cmp(0.004)[1], [])
        self.assertTrue(cmp(-0.001)[1][0].startswith("MLX-DEV2 guard"))
        ref.write_text('{"id": "a", "answers": ["y"]}\n')
        self.assertIn("not against", cmp(0.004)[1][0])

    def test_better_arm_and_pick(self) -> None:
        r = self.rules
        a = self.row("4b-LHS10SD", 0.03, eligible=False, gates=6, card=0.01)
        b = self.row("4b-LHS17SD", 0.05, eligible=False, gates=5)
        self.assertEqual(sorted([a, b], key=r.better_key)[0]["point"], "4b-LHS10SD")
        c = self.row("4b-LHS17SD", 0.03, eligible=False, gates=6, card=0.02)
        self.assertEqual(sorted([a, c], key=r.better_key)[0]["point"], "4b-LHS17SD")
        rows = [
            self.row("4b-LHS10SD", 0.040),
            self.row("4b-LHS17SD", 0.050),
            self.row("4b-LHS17SD-a67", 0.050, gain=True),
            self.row("4b-LHS17SD-a33", 0.020),
        ]
        self.assertEqual(r.pick(rows), ["4b-LHS17SD-a67", "4b-LHS10SD"])
        rows[1]["eligible"] = rows[2]["eligible"] = False
        rows[0]["eligible"] = False
        self.assertEqual(r.pick(rows), ["4b-LHS17SD-a33"])


class SuccessorTest(unittest.TestCase):
    def test_index_path_items(self) -> None:
        succ = load("m17_successor")
        bars = ["bar-lh", "bar-f"]

        def out(v3_low, v3_high, red_high, rest=True):
            items = {
                "1_v3_vs_bar": {
                    "pass": v3_low > 0,
                    **{b: {"ci95": {"low": v3_low, "high": v3_high}} for b in bars},
                },
                "6b_reduced_panels": {
                    "pass": v3_low > 0,
                    **{
                        b: {
                            "rule1_v3_vs_bar": {"ci95": {"low": -1.0, "high": red_high}}
                        }
                        for b in bars
                    },
                },
            }
            for k in succ.REST:
                items[k] = {"pass": rest}
            return {"items": items, "bars": bars}

        ip = succ.index_path(out(-1.8, 2.6, 2.4), None)
        self.assertTrue(ip["candidate"])
        self.assertIsNone(ip["1p_b_index_significantly_positive"])
        self.assertFalse(ip["items_1p_7_pass"])
        self.assertTrue(succ.index_path(out(-1.8, 2.6, 2.4), "PASS")["items_1p_7_pass"])
        self.assertFalse(
            succ.index_path(out(-1.8, 2.6, 2.4), "FAIL")["items_1p_7_pass"]
        )
        self.assertFalse(succ.index_path(out(-3.0, -0.1, 2.4), "PASS")["candidate"])
        self.assertFalse(succ.index_path(out(-1.8, 2.6, -0.2), "PASS")["candidate"])
        self.assertFalse(
            succ.index_path(out(-1.8, 2.6, 2.4, rest=False), "PASS")["candidate"]
        )
        self.assertFalse(succ.index_path(out(0.4, 2.6, 2.4), "PASS")["candidate"])


if __name__ == "__main__":
    unittest.main()
