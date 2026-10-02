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


if __name__ == "__main__":
    unittest.main()
