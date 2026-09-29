import hashlib
import importlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

m7 = importlib.import_module("v2.06b.m7_teacher")

POOLS = {
    "a1": "BASE",
    "a2": "BASE",
    "q1": "A7q",
    "q2": "A7q",
    "h1": "V1:A6h",
    "r1": "R",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.recipe = root / "recipe.ids.jsonl"
        self.recipe.write_text(
            "".join(json.dumps({"id": k, "pool": p}) + "\n" for k, p in POOLS.items())
        )
        # Unusual spacing and key order: kept lines must survive byte for byte.
        self.lines = [
            b'{"teacher_probs": {"A": 1.0}, "id": "a1", "input_sha256": "x-a1"}\n',
            b'{"id":"q1","input_sha256":"x-q1","teacher_probs":{"A":0.5,"B":0.5}}\n',
            b'{"id": "r1",  "input_sha256": "x-r1", "teacher_probs": {"A": 0.25}}\n',
            b'{"id": "h1", "input_sha256": "x-h1", "teacher_probs": {"A": 1}}\n',
            b'{"id": "a2", "input_sha256": "x-a2", "teacher_probs": {"\\u00e9": 1}}\n',
            b'{"id": "q2", "input_sha256": "x-q2", "teacher_probs": {"A": 1}}\n',
        ]
        self.teacher = root / "teacher.jsonl"
        self.teacher.write_bytes(b"".join(self.lines))
        self.mix1 = self.mixture("m1.train.jsonl", ["a1", "q1", "r1", "h1"])
        self.mix2 = self.mixture("m2.train.jsonl", ["a2", "q2", "a1"])
        self.output = root / "out" / "t.jsonl"
        self.report = root / "out" / "t.report.json"
        self.output.parent.mkdir()

    def mixture(self, name: str, ids: list[str], changed: str = "") -> Path:
        path = self.root / name
        path.write_text(
            "".join(
                json.dumps(
                    {"id": k, "input_sha256": "changed" if k == changed else f"x-{k}"}
                )
                + "\n"
                for k in ids
            )
        )
        return path

    def argv(self, covered=(2, 2), kept=3, teacher_sha=None, mixes=None) -> list[str]:
        mixes = mixes or [self.mix1, self.mix2]
        argv = [
            "--teacher", str(self.teacher),
            "--teacher-sha256", teacher_sha or sha(self.teacher),
            "--recipe", str(self.recipe),
            "--drop-pool", "A7q", "--drop-pool", "V1:A6h", "--drop-pool", "H6",
            "--expect-kept", str(kept),
            "--output", str(self.output), "--report", str(self.report),
        ]  # fmt: skip
        for path, n in zip(mixes, covered):
            argv += ["--mixture", f"{path}={sha(path)}={n}"]
        return argv

    def run(self, argv: list[str]) -> int:
        with redirect_stdout(io.StringIO()):
            return m7.main(argv)


class TeacherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = Fixture(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_keeps_lines_byte_identical_in_order_and_reports_counts(self):
        f = self.f
        self.assertEqual(f.run(f.argv()), 0)
        self.assertEqual(f.output.read_bytes(), f.lines[0] + f.lines[2] + f.lines[4])
        report = json.loads(f.report.read_text())
        self.assertTrue(report["pass"])
        self.assertEqual(report["input"]["lines"], 6)
        self.assertEqual(report["input"]["sha256"], sha(f.teacher))
        self.assertEqual(report["output"]["sha256"], sha(f.output))
        self.assertEqual(report["output"]["lines"], 3)
        self.assertEqual(report["removed_by_pool"], {"A7q": 2, "V1:A6h": 1})
        self.assertEqual(report["kept_by_pool"], {"BASE": 2, "R": 1})
        m1 = report["mixtures"]["m1.train.jsonl"]
        self.assertEqual((m1["rows"], m1["covered"], m1["uncovered"]), (4, 2, 2))
        self.assertEqual(m1["covered_by_pool"], {"BASE": 1, "R": 1})
        self.assertEqual(m1["uncovered_by_pool"], {"A7q": 1, "V1:A6h": 1})
        self.assertEqual(report["mixtures"]["m2.train.jsonl"]["covered"], 2)

    def test_count_mismatch_writes_failing_report_and_no_teacher(self):
        f = self.f
        self.assertEqual(f.run(f.argv(covered=(2, 3))), 1)
        self.assertFalse(f.output.exists())
        report = json.loads(f.report.read_text())
        self.assertFalse(report["pass"])
        self.assertFalse(report["checks"]["covered_match"])
        self.assertTrue(report["checks"]["kept_matches"])

    def test_kept_count_mismatch_fails(self):
        f = self.f
        self.assertEqual(f.run(f.argv(kept=4)), 1)
        self.assertFalse(f.output.exists())

    def test_uncovered_row_in_kept_pool_fails(self):
        f = self.f
        changed = f.mixture("m3.train.jsonl", ["a1", "r1", "q1"], changed="r1")
        self.assertEqual(f.run(f.argv(covered=(2, 1), mixes=[f.mix1, changed])), 1)
        report = json.loads(f.report.read_text())
        self.assertFalse(report["checks"]["uncovered_only_in_dropped_pools"])
        self.assertTrue(report["checks"]["covered_match"])

    def test_unknown_teacher_id_stops_before_output(self):
        f = self.f
        f.teacher.write_bytes(
            b"".join(f.lines)
            + b'{"id": "zz", "input_sha256": "x", "teacher_probs": {}}\n'
        )
        with self.assertRaises(ValueError):
            f.run(f.argv())
        self.assertFalse(f.output.exists() or f.report.exists())

    def test_unknown_mixture_id_stops(self):
        f = self.f
        bad = f.mixture("m4.train.jsonl", ["a1", "zz"])
        with self.assertRaises(ValueError):
            f.run(f.argv(mixes=[f.mix1, bad]))
        self.assertFalse(f.output.exists() or f.report.exists())

    def test_repeated_teacher_id_stops(self):
        f = self.f
        f.teacher.write_bytes(b"".join(f.lines) + f.lines[0])
        with self.assertRaises(ValueError):
            f.run(f.argv())

    def test_teacher_hash_mismatch_stops(self):
        f = self.f
        with self.assertRaises(ValueError):
            f.run(f.argv(teacher_sha="0" * 64))
        self.assertFalse(f.output.exists() or f.report.exists())

    def test_mixture_hash_mismatch_stops(self):
        f = self.f
        argv = f.argv()
        argv[-1] = f"{f.mix2}={'0' * 64}=2"
        with self.assertRaises(ValueError):
            f.run(argv)

    def test_refuses_to_overwrite(self):
        f = self.f
        self.assertEqual(f.run(f.argv()), 0)
        before = f.output.read_bytes()
        with self.assertRaises(FileExistsError):
            f.run(f.argv())
        f.report.unlink()
        with self.assertRaises(FileExistsError):
            f.run(f.argv())
        self.assertEqual(f.output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
