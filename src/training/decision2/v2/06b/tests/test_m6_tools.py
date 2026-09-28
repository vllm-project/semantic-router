import hashlib
import importlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

cache = importlib.import_module("v2.06b.m6_cache")
answers = importlib.import_module("v2.06b.m6_answers_diff")
stage = importlib.import_module("v2.06b.m6_stage")


def rows(path: Path, values: list[tuple[str, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for item, choice, p in values:
            answer = {
                "type": "choice",
                "choice": choice,
                "probabilities": {"a": p, "b": 1 - p},
            }
            stream.write(
                json.dumps(
                    {"id": item, "input_sha256": item, "answers": {"q1": answer}}
                )
                + "\n"
            )


class CacheTest(unittest.TestCase):
    def test_tree_matches_sha256sum_listing_and_records_additions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "src"
            (src / "B").mkdir(parents=True)
            (src / "a_b").mkdir()
            (src / "B" / "k.autotune.json").write_text("{}")
            (src / "a_b" / "x.hsaco").write_bytes(b"\x00\x01")
            listing = cache.listing(src)
            expected = "".join(
                f"{hashlib.sha256((src / r).read_bytes()).hexdigest()}  ./{r}\n"
                for r in sorted(listing, key=lambda r: ("./" + r).encode())
            )
            self.assertEqual(
                cache.tree_sha256(listing),
                hashlib.sha256(expected.encode()).hexdigest(),
            )
            if os.path.exists("/usr/bin/sha256sum"):
                shell = subprocess.run(
                    "find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum",
                    shell=True,
                    cwd=src,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.split()[0]
                self.assertEqual(cache.tree_sha256(listing), shell)
            snap, manifest = tmp / "snap", tmp / "snap.MANIFEST.json"
            record = cache.snapshot(src, snap, manifest, cache.tree_sha256(listing))
            self.assertEqual(record["files"], 2)
            for path in (snap, snap / "B", snap / "B" / "k.autotune.json"):
                self.assertEqual(path.stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                cache.snapshot(src, snap, manifest, None)
            run = tmp / "run.triton-cache"
            cache.seed(snap, manifest, run)
            (run / "NEW").mkdir()
            (run / "NEW" / "f.autotune.json").write_text("{}")
            out = cache.record(manifest, run, tmp / "M6-CACHE.json")
            self.assertFalse(out["unchanged"])
            self.assertEqual(out["added"], ["NEW/f.autotune.json"])
            self.assertEqual(out["autotune_entries_added"], ["NEW/f.autotune.json"])
            self.assertNotEqual(out["before_tree_sha256"], out["after_tree_sha256"])
            with self.assertRaises(FileExistsError):
                cache.seed(snap, manifest, run)


class AnswersDiffTest(unittest.TestCase):
    def test_counts_without_item_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows(
                tmp / "a" / "output" / "typed-final.predictions.jsonl",
                [("i1", "a", 0.9), ("i2", "b", 0.2)],
            )
            rows(
                tmp / "b" / "output" / "typed-final.predictions.jsonl",
                [("i1", "a", 0.8), ("i2", "a", 0.6)],
            )
            rows(
                tmp / "a-mlx" / "output" / "mlx-diag.predictions.jsonl",
                [("m1", "a", 0.7)],
            )
            rows(
                tmp / "b-mlx" / "output" / "mlx-diag.predictions.jsonl",
                [("m1", "a", 0.7)],
            )
            same = answers.diff_runs(tmp / "a", tmp / "a", ["typed-final", "mlx-diag"])
            self.assertTrue(same["answers_equal"])
            self.assertEqual(same["total"]["max_abs_numeric_drift"], 0.0)
            result = answers.diff_runs(
                tmp / "a", tmp / "b", ["typed-final", "mlx-diag", "css15"]
            )
            typed = result["panels"]["typed-final"]
            self.assertEqual(typed["items_compared"], 2)
            self.assertEqual(typed["answers_differ"], 1)
            self.assertAlmostEqual(typed["max_abs_numeric_drift"], 0.4)
            self.assertEqual(result["panels"]["mlx-diag"]["answers_differ"], 0)
            self.assertEqual(result["missing_panels"], ["css15"])
            self.assertFalse(result["answers_equal"])
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                answers.main(
                    [str(tmp / "a"), str(tmp / "b"), "--panels", "typed-final"]
                )
            self.assertNotIn("i1", buffer.getvalue())
            self.assertNotIn("i2", buffer.getvalue())


class StageTest(unittest.TestCase):
    def test_build_and_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            full = tmp / "arms" / "m6-x-soup" / "full"
            export = full / "best-export"
            (export / "backbone").mkdir(parents=True)
            (export / "backbone" / "model.safetensors").write_bytes(b"w" * 10)
            (export / "decision_config.json").write_text("{}")
            files = {
                rel: {
                    "bytes": (export / rel).stat().st_size,
                    "sha256": hashlib.sha256((export / rel).read_bytes()).hexdigest(),
                }
                for rel in ("backbone/model.safetensors", "decision_config.json")
            }
            (full / "best-export.MANIFEST.json").write_text(
                json.dumps({"files": files})
            )
            manifest_sha = hashlib.sha256(
                (full / "best-export.MANIFEST.json").read_bytes()
            ).hexdigest()
            (full / "SOUP.json").write_text(
                json.dumps(
                    {
                        "best_export_manifest_sha256": manifest_sha,
                        "state_sha256": "s" * 64,
                        "ingredients": [{"arm": "m6-cx-s1"}],
                        "status": "COMPLETE",
                    }
                )
            )
            formal = tmp / "formal" / "m6-x-soup"
            formal.mkdir(parents=True)
            (formal / "M6-SUMMARY.json").write_text(json.dumps({"v3": 46.0}))
            numbers = tmp / "numbers.json"
            numbers.write_text(
                json.dumps(
                    {"development": {"Q": 44.1}, "post_key_same_panel": {"v3": 46.0}}
                )
            )
            dest = tmp / "stage"
            info = stage.build(full, dest, formal, numbers, "m6-x-soup", "abc123")
            self.assertEqual(
                info["repo"], "llm-semantic-router/dev2-staging-06bm6-m6-x-soup"
            )
            report = stage.validate(dest, full)
            self.assertTrue(report["ok"], report)
            staging = json.loads((dest / "dev2-staging" / "STAGING.json").read_text())
            self.assertEqual(staging["export_manifest_sha256"], manifest_sha)
            self.assertEqual(staging["state_sha256"], "s" * 64)
            readme = (dest / "README.md").read_text()
            self.assertIn("not a release", readme)
            self.assertIn("Qwen/Qwen3-0.6B-Base", readme)
            (dest / "decision_config.json").write_text('{"x": 1}')
            self.assertFalse(stage.validate(dest, full)["ok"])
            with self.assertRaises(FileExistsError):
                stage.build(full, dest, formal, numbers, "m6-x-soup", "abc123")


if __name__ == "__main__":
    unittest.main()
