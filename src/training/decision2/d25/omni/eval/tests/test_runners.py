from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from d25.omni.eval import run_rows, run_suite, shards
from d25.omni.model import tiny


def suite_rows(rng: random.Random, images: list[str]) -> list[dict]:
    rows = []
    for index in range(10):
        refs = rng.sample(images, index % 3)
        questions = {
            "q1": {
                "type": "choice",
                "instructions": "Which one ?",
                "criteria": {"A": "cat", "B": "dog", "C": "red"},
            },
            "q2": {"type": "noul", "instructions": "Is it shown ?"},
        }
        if index % 4 == 0:
            questions.pop("q2")
        rows.append(
            {
                "id": f"bench-{index}",
                "family": "tiny-bench",
                "split": "test",
                "images": refs,
                "state": "picture of a cat",
                "questions": questions,
                "expected": (
                    {"q1": "A", "q2": True} if "q2" in questions else {"q1": "A"}
                ),
                "metadata": {
                    "benchmark": "CV-Bench" if index % 2 else "BLINK",
                    "chance": 0.33,
                },
            }
        )
    rows[-1]["images"] = ["images/missing.png"]
    return rows


class RunnersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = tiny.fixtures()
        cls.tmp = Path(tempfile.mkdtemp(prefix="d25-omni-runners-"))
        cls.suite = cls.tmp / "suite"
        shutil.copytree(cls.paths["data"] / "images", cls.suite / "images")
        names = sorted(f"images/{p.name}" for p in (cls.suite / "images").glob("*.png"))
        cls.rows = suite_rows(random.Random(2), names)
        tiny.write_rows(cls.suite / "rows.jsonl.gz", cls.rows)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def suite_args(self, **overrides) -> argparse.Namespace:
        values = dict(
            ckpt=str(self.paths["init_vega"]),
            suite=str(self.suite),
            rows=None,
            out=str(self.tmp / "out"),
            num_shards=2,
            shard=0,
            device="cpu",
            max_pixels=1_638_400,
            token_budget=8192,
            max_batch_size=8,
            readout_dtype="float32",
            chunk=3,
            limit=None,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_suite_shards_resume_and_merge(self) -> None:
        first = run_suite.run(self.suite_args(shard=0, limit=2))
        self.assertEqual(first["ran"], 2)
        path = shards.shard_path(self.tmp / "out", "results", 0, 2)
        with path.open("a") as stream:
            stream.write('{"id": "torn')
        for shard in (0, 1):
            run_suite.run(self.suite_args(shard=shard))
        again = run_suite.run(self.suite_args(shard=0))
        self.assertEqual(again["ran"], 1)
        records = shards.merge(self.tmp / "out", "results", 2)
        self.assertEqual(set(records), {row["id"] for row in self.rows})
        self.assertEqual(records["bench-9"]["status"], "error")
        with self.assertRaises(SystemExit):
            run_suite.merge_outputs(self.suite_args())
        ok = records["bench-1"]
        self.assertEqual(ok["status"], "ok")
        self.assertEqual(set(ok["response"]["answers"]), {"q1", "q2"})
        self.assertEqual(ok["response"]["answers"]["q1"]["type"], "choice")
        self.assertAlmostEqual(sum(ok["probabilities"]["q1"]), 1.0, places=5)

    def test_suite_merge_writes_results_in_suite_order(self) -> None:
        rows = [row for row in self.rows if row["id"] != "bench-9"]
        tiny.write_rows(self.tmp / "rows-ok.jsonl.gz", rows)
        args = self.suite_args(
            rows=str(self.tmp / "rows-ok.jsonl.gz"), out=str(self.tmp / "out-ok")
        )
        for shard in (0, 1):
            run_suite.run(self.suite_args(rows=args.rows, out=args.out, shard=shard))
        summary = run_suite.merge_outputs(args)
        merged = [
            json.loads(line)
            for line in (self.tmp / "out-ok" / "results.jsonl").read_text().splitlines()
        ]
        self.assertEqual([r["id"] for r in merged], [r["id"] for r in rows])
        self.assertEqual(summary["status"], {"ok": len(rows)})
        self.assertEqual(set(summary["diagnostics"]), {"CV-Bench", "BLINK"})
        self.assertEqual(summary["scorer"], "d25.omni.suite.score.score_suite")
        self.assertEqual(set(summary["scores"]["benchmarks"]), {"CV-Bench", "BLINK"})
        self.assertFalse(summary["scores"]["complete"])
        self.assertTrue((self.tmp / "out-ok" / "scores.json").exists())

    def test_run_rows(self) -> None:
        files = [
            str(self.paths["data"] / "mm.jsonl"),
            str(self.paths["data"] / "text.jsonl"),
        ]
        out = self.tmp / "rows-out"
        base = dict(
            ckpt=str(self.paths["init_vega"]),
            rows=files,
            out=str(out),
            num_shards=3,
            device="cpu",
            max_pixels=1_638_400,
            token_budget=8192,
            max_batch_size=8,
            readout_dtype="float32",
            chunk=7,
            limit=None,
        )
        for shard in range(3):
            run_rows.run(argparse.Namespace(shard=shard, **base))
        self.assertEqual(run_rows.run(argparse.Namespace(shard=1, **base))["ran"], 0)
        summary = run_rows.merge_outputs(argparse.Namespace(shard=0, **base))
        self.assertEqual(summary["rows"], 64)
        self.assertEqual(summary["status"], {"ok": 64})
        records = [
            json.loads(line) for line in (out / "probs.jsonl").read_text().splitlines()
        ]
        expected_ids = [
            json.loads(line)["id"]
            for name in files
            for line in Path(name).read_text().splitlines()
        ]
        self.assertEqual([r["id"] for r in records], expected_ids)
        self.assertTrue(all(abs(sum(r["probs"]) - 1) < 1e-5 for r in records))


if __name__ == "__main__":
    unittest.main()
