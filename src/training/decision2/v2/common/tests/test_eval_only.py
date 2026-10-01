from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from v2.common import eval_only

V2 = Path(__file__).resolve().parents[2]
QUAL_PROMPTS_SHA256 = "c1d6d9222b488347dc1616a42ef4f3ae9b6669c648d860fd8e0c4478c3d4bb80"
# Every module that assembles training mixtures (or teacher targets / replay rows for them).
BUILDERS = {
    "data/m2/mixtures.py": ("guard", "check_rows"),
    "data/m3/xl.py": ("guard", "check_rows"),
    "data/m3/xl_r2.py": ("guard",),
    "data/m3/teacher_targets.py": ("guard",),
    "data/m3/xl_prompts.py": ("guard",),
    "data/replay_targets.py": ("guard", "check_rows"),
    "data/hr2/build.py": ("guard", "check_rows"),
    "dec/build_mixture.py": ("guard", "check_file", "check_rows"),
    "dec/m5_block.py": ("guard", "check_rows"),
    "06b/mixture.py": ("guard", "check_rows"),
    "06b/m6_partition.py": ("guard", "check_rows"),
    "9b/lux9b/mix.py": ("guard", "check_rows"),
    "27b/build_mixtures.py": ("guard", "check_file", "check_rows"),
    "27b/build_replay.py": ("guard", "check_file", "check_rows"),
}
BUILDER_NAME = re.compile(
    r"(mixture|mixtures|^mix|partition|^xl|_block|replay)\w*\.py$"
)


def pool(files=(), ids=()):
    return eval_only.Pool(
        "test-pool",
        (re.compile(r"(^|[/:])m3a2?/qual(/|$)"),),
        frozenset(files),
        frozenset(hashlib.sha256(i.encode()).hexdigest() for i in ids),
    )


class RegistryTest(unittest.TestCase):
    def test_qualification_pools_match_and_training_paths_do_not(self):
        hits = [
            "/data/dev2/private/htdev/iso/training/local/m3a/qual/qual.prompts.jsonl",
            "/data/dev2/private/htdev/iso/training/local/m3a2/qual/out/p1.jsonl",
            "/data/dev2/runs/eval/m4/c1-event2-recheck2/local-snapshot/m3a2/qual/warm-v2.prompts.jsonl",
            "/data/dev2/private/htdev/iso/training/local/m3a/qualification.json",
            "/data/dev2/private/htdev/iso/training/local/m3a2/qualify.events.jsonl",
            "/data/dev2/private/htdev/iso/training/local/m3a2/upload-aj-m/qualification-v2.report.json",
            "local:m3a2/qual/sets.json",
            "/data/dev2/private/panels/goldfree/typed-final.prompts.jsonl",
        ]
        misses = [
            "/data/dev2/private/htdev/iso/training/local/m3a/waves/aj-m.prompts.jsonl",
            "/data/dev2/private/htdev/iso/training/local/m3a2/upload-aj-m/rp-v2/aj-m.targets.jsonl",
            "/data/dev2/private/htdev/iso/training/local/m3a2/rows/aj-m.rows.jsonl",
            "/data/dev2/runs/dec/m4/data/m4-xl-full-29m/train.jsonl",
            "/data/dev2/private/panels/gold/select.jsonl",
            "m3/teachers/autojev27/rp-v2/aj-m.targets.jsonl",
        ]
        for path in hits:
            self.assertIsNotNone(eval_only.match_path(path), path)
        for path in misses:
            self.assertIsNone(eval_only.match_path(path), path)
        pools = eval_only.load()
        self.assertIn(QUAL_PROMPTS_SHA256, pools[0].file_sha256)
        self.assertEqual(len(pools[0].item_id_sha256), 487)


class GuardTest(unittest.TestCase):
    def test_guard_walks_arguments_specs_and_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "renamed.jsonl"
            copy.write_text('{"id": "x"}\n')
            digest = hashlib.sha256(copy.read_bytes()).hexdigest()
            fine = Path(tmp) / "train.jsonl"
            fine.write_text('{"id": "y"}\n')
            pools = (pool(files=[digest]),)
            eval_only.guard(
                argparse.Namespace(rows=fine, seed=3), {"a": [str(fine)]}, pools=pools
            )
            for bad in (
                argparse.Namespace(rows=copy),
                {"pools": {"x": {"rows": [str(copy)]}}},
                f"lux1={fine},{copy}",
                "/elsewhere/m3a/qual/out/p2.jsonl",
                {"files": ["local:m3a2/qual/qual.prompts.jsonl"]},
            ):
                with self.assertRaises(eval_only.EvalOnlyInputError):
                    eval_only.guard(bad, pools=pools)
            eval_only.check_file(
                fine, hashlib.sha256(fine.read_bytes()).hexdigest(), pools
            )
            with self.assertRaises(eval_only.EvalOnlyInputError):
                eval_only.check_file(fine, digest, pools)

    def test_check_rows(self):
        pools = (pool(ids=["td_abc"]),)
        self.assertEqual(eval_only.check_rows([{"id": "a"}, {"id": "b#r1"}], pools), 2)
        for ident in ("td_abc", "td_abc#r2"):
            with self.assertRaises(eval_only.EvalOnlyInputError):
                eval_only.check_rows([{"id": "a"}, {"id": ident}], pools)


class RelabelTest(unittest.TestCase):
    def test_corpora_and_training_manifests(self):
        corpora = {
            "schema": "c1-corpora/1",
            "labels": {
                "local-m3a": {
                    "kind": "training",
                    "files": [
                        {
                            "path": "/x/m3a/qual/qual.prompts.jsonl",
                            "sha256": "a",
                            "bytes": 1,
                        },
                        {"path": "/x/m3a/waves/w.jsonl", "sha256": "b", "bytes": 2},
                        {
                            "path": "/x/elsewhere/copy.jsonl",
                            "sha256": "c" * 64,
                            "bytes": 3,
                        },
                    ],
                },
                "hf-head": {
                    "kind": "training",
                    "files": [{"path": "/y/t.jsonl", "sha256": "d"}],
                },
            },
        }
        pools = (pool(files=["c" * 64]),)
        out, receipt = eval_only.relabel(corpora, pools)
        self.assertEqual(
            [f["path"] for f in out["labels"]["local-m3a"]["files"]],
            ["/x/m3a/waves/w.jsonl"],
        )
        moved = out["labels"]["local-m3a:evaluation-only"]
        self.assertEqual(moved["kind"], "evaluation-only")
        self.assertEqual(len(moved["files"]), 2)
        self.assertEqual(receipt["moved_by_label"], {"local-m3a": 2})
        self.assertEqual(out["labels"]["hf-head"], corpora["labels"]["hf-head"])
        again, receipt2 = eval_only.relabel(out, pools)
        self.assertEqual(again, out)
        self.assertEqual(receipt2["moved_by_label"], {})
        training = {
            "schema": "htdev-training-manifest/1",
            "labels": {
                "local-m3a": {
                    "root": "/x/m3a",
                    "files": [
                        {
                            "path": "/x/m3a/qual/out/p1.jsonl",
                            "sha256": "a",
                            "bytes": 5,
                            "rows": 2,
                        },
                        {
                            "path": "/x/m3a/hf/a.jsonl",
                            "sha256": "b",
                            "bytes": 7,
                            "rows": 3,
                        },
                    ],
                    "file_count": 2,
                    "bytes": 12,
                    "rows": 5,
                }
            },
            "totals": {"labels": 1},
        }
        out, _ = eval_only.relabel(training, pools)
        self.assertEqual(out["labels"]["local-m3a"]["rows"], 3)
        self.assertEqual(out["labels"]["local-m3a:evaluation-only"]["file_count"], 1)
        self.assertEqual(out["totals"]["labels"], 2)

    def test_cli_writes_new_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "m.json"
            src.write_text(json.dumps({"schema": "c1-corpora/1", "labels": {}}))
            args = [
                "relabel",
                "--manifest",
                str(src),
                "--out",
                str(Path(tmp) / "o.json"),
                "--receipt",
                str(Path(tmp) / "r.json"),
            ]
            self.assertEqual(eval_only.main(args), 0)
            with self.assertRaises(FileExistsError):
                eval_only.main(args)


class BuilderCoverageTest(unittest.TestCase):
    def test_every_mixture_builder_calls_the_guard(self):
        for rel, calls in BUILDERS.items():
            text = (V2 / rel).read_text(encoding="utf-8")
            self.assertIn("from v2.common import eval_only", text, rel)
            for call in calls:
                self.assertIn(f"eval_only.{call}(", text, f"{rel} does not call {call}")

    def test_new_builders_are_registered(self):
        found = {
            str(p.relative_to(V2))
            for p in V2.rglob("*.py")
            if "tests" not in p.parts
            and BUILDER_NAME.search(p.name)
            and re.search(r"^def main\(", p.read_text(encoding="utf-8"), re.M)
        }
        unguarded = sorted(
            rel
            for rel in found - set(BUILDERS)
            if "eval_only." not in (V2 / rel).read_text(encoding="utf-8")
        )
        self.assertEqual(
            unguarded, [], "register these builders in BUILDERS and call the guard"
        )

    def test_builders_refuse_a_qualification_pool(self):
        from v2.data.m2 import mixtures
        from v2.data.m3 import xl

        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / "pools.json"
            spec.write_text(
                json.dumps(
                    {"A0s": {"rows": ["/n/m3a2/qual/qual.prompts.jsonl"], "tokens": []}}
                )
            )
            for main in (mixtures.main, xl.main):
                with self.assertRaises(eval_only.EvalOnlyInputError):
                    main(["--pools", str(spec), "--out-dir", str(Path(tmp) / "out")])
            self.assertFalse((Path(tmp) / "out").exists())


if __name__ == "__main__":
    unittest.main()
