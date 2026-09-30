import copy
import hashlib
import importlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

teachers = importlib.import_module("v2.27b.m4b.build_teachers")


def row(i, source, task_type="choice", keys=("a", "b"), label=0, **extra):
    return {
        "id": f"r{i}",
        "source": source,
        "task_type": task_type,
        "options": [{"key": k, "description": k} for k in keys],
        "label": label,
        "input_sha256": hashlib.sha256(f"r{i}".encode()).hexdigest(),
        "audit_metadata": extra.pop("audit_metadata", {}),
        **extra,
    }


ROWS = [
    row(0, "google_goemotions_official_train"),
    row(1, "decision2_rule_x"),
    row(2, "argq30k_train", "score", ("s0", "s1", "s2"), 2),
    row(3, "legacy:stage3_replay", upstream_label="card_arrival"),
    row(4, "legacy:stage3_replay"),
    row(5, "banking77_train", audit_metadata={"a7": {"sub_arm": "A7i"}}),
    row(6, "decision2_verifiable_v2_a6", "score", ("s0", "s1")),
    row(7, "legacy:snli", keys=("e", "n", "c"), label=1),
    row(8, "saf_de_train", "score", ("s0", "s1")),
    row(
        9,
        "klue_sts_train",
        "score",
        ("s0", "s1"),
        audit_metadata={"a7": {"sub_arm": "A7o"}},
    ),
]
S_A0S = ["r0", "r3", "r7"]
S_A6H = ["r2", "r8"]


def probs(r, winner=None):
    keys = [o["key"] for o in r["options"]]
    top = keys[r["label"]] if winner is None else winner
    rest = [k for k in keys if k != top]
    return {top: 0.7, **{k: 0.3 / len(rest) for k in rest}}


class TeacherBuildTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.train = self.write_jsonl("train.jsonl", ROWS)
        by_source = Counter(r["source"] for r in ROWS)
        self.mixtures = self.write(
            "MIXTURES.json",
            {
                "mixtures": {
                    "a7": {
                        "by_source": dict(by_source),
                        "parts": {"A6": {"rows": 3}, "A7": {"rows": 2}},
                    }
                }
            },
        )
        by_id = {r["id"]: r for r in ROWS}
        lux_a0s = [self.record(by_id[i]) for i in S_A0S] + [
            self.record(by_id["r1"]),
            {"id": "gone", "teacher_probs": {}},
        ]
        self.files = {
            "lux-a0s": lux_a0s,
            "lux-w1": [self.record(by_id["r2"])],
            "lux-w2": [self.record(by_id["r8"])],
            "aj-a0s": [
                self.record(by_id[i], winner="b" if i == "r0" else None) for i in S_A0S
            ],
            "aj-m": [self.record(by_id[i]) for i in S_A6H],
        }

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, r, winner=None):
        return {
            "id": r["id"],
            "input_sha256": r["input_sha256"],
            "teacher_probs": probs(r, winner),
        }

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def write_jsonl(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return path

    def spec(self, files=None):
        files = files or self.files
        paths = {k: self.write_jsonl(f"{k}.jsonl", v) for k, v in files.items()}
        entry = lambda k, pool: {
            "pool": pool,
            "path": str(paths[k]),
            "sha256": teachers.sha_file(paths[k]),
        }  # noqa: E731
        return {
            "expect_s_rows": 5,
            "train": {"path": str(self.train), "sha256": teachers.sha_file(self.train)},
            "mixtures": {"name": "a7", "path": str(self.mixtures), "sha256": teachers.sha_file(self.mixtures)},
            "teachers": {
                "lux": {"model": "lux", "files": [entry("lux-a0s", "A0s-strict"), entry("lux-w1", "A6h"),
                                                  entry("lux-w2", "A6h")]},
                "aj": {"model": "aj", "files": [entry("aj-a0s", "A0s-strict"), entry("aj-m", "A6h")]},
            },
        }  # fmt: skip

    def test_selection_rule(self):
        self.assertEqual(
            [r["id"] for r in ROWS if teachers.in_s(r)], ["r0", "r2", "r3", "r7", "r8"]
        )
        pools = [teachers.pool_of(r) for r in ROWS]
        self.assertEqual(pools[5], "A7")
        self.assertEqual(pools[6], "A6g")
        self.assertEqual(pools[9], "A7")
        self.assertEqual(pools[4], "A0s-strict")

    def test_build_outputs_in_train_order_and_is_deterministic(self):
        spec = self.spec()
        first = teachers.build(spec, self.root / "out1")
        second = teachers.build(spec, self.root / "out2")
        order = ["r0", "r2", "r3", "r7", "r8"]
        self.assertEqual((self.root / "out1" / "S.ids.txt").read_text().split(), order)
        lux = [
            json.loads(line)
            for line in (self.root / "out1" / "teacher-lux.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual([r["id"] for r in lux], order)
        self.assertEqual(set(lux[0]), {"id", "input_sha256", "teacher_probs"})
        for name in (
            "S.ids.txt",
            "teacher-lux.jsonl",
            "teacher-aj.jsonl",
            "MANIFEST.json",
        ):
            self.assertEqual(
                (self.root / "out1" / name).read_bytes(),
                (self.root / "out2" / name).read_bytes(),
            )
        self.assertEqual(first, second)
        self.assertEqual(first["s"]["by_pool"], {"A0s-strict": 3, "A6h": 2})
        counts = first["teachers"]["lux"]["files"][0]["counts"]
        self.assertEqual(
            counts,
            {"in_train_outside_s": 1, "not_in_train": 1, "records": 5, "used": 3},
        )
        self.assertEqual(
            first["teachers"]["aj"]["agreement_with_gold"]["all"]["agree"], 4
        )
        self.assertEqual(
            first["teachers"]["lux"]["agreement_with_gold"]["all"]["agree"], 5
        )
        self.assertEqual(
            first["train_pools"]["rows_by_pool"],
            {"A0s-strict": 5, "A6g": 1, "A6h": 2, "A7": 2},
        )
        self.assertFalse((self.root / "out1.pending").exists())
        with self.assertRaises(FileExistsError):
            teachers.build(spec, self.root / "out1")

    def fails(self, mutate, message):
        files = copy.deepcopy(self.files)
        mutate(files)
        with self.assertRaisesRegex(ValueError, message):
            teachers.build(self.spec(files), self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_join_failures(self):
        self.fails(lambda f: f["aj-m"].pop(), "1 S rows have no teacher record")
        self.fails(
            lambda f: f["aj-m"].append(dict(f["aj-m"][0])), "repeated teacher id"
        )
        self.fails(
            lambda f: f["lux-w2"].append(dict(f["lux-w1"][0])), "already covered"
        )
        self.fails(lambda f: f["aj-a0s"][1].update(input_sha256="0" * 64), "input hash")
        self.fails(
            lambda f: f["aj-a0s"][1].update(teacher_probs={"a": 1.0}), "keys differ"
        )
        self.fails(
            lambda f: f["aj-a0s"][1].update(teacher_probs={"a": 0.5, "b": 0.49}),
            "invalid teacher",
        )
        self.fails(
            lambda f: f["aj-a0s"][1].update(teacher_probs={"a": 1.2, "b": -0.2}),
            "invalid teacher",
        )
        self.fails(
            lambda f: f["aj-a0s"][1].update(teacher_probs={"a": True, "b": 0}),
            "invalid teacher",
        )
        self.fails(
            lambda f: f["aj-a0s"].append(dict(f["aj-m"][0])),
            "pool A6h in a A0s-strict file",
        )

    def test_sum_tolerance(self):
        files = copy.deepcopy(self.files)
        files["aj-a0s"][1]["teacher_probs"] = {"a": 0.7 + 5e-7, "b": 0.3}
        teachers.build(self.spec(files), self.root / "ok")

    def test_pinned_inputs(self):
        spec = self.spec()
        bad = copy.deepcopy(spec)
        bad["teachers"]["lux"]["files"][1]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "expected"):
            teachers.build(bad, self.root / "out")
        bad = copy.deepcopy(spec)
        bad["train"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "TRAIN differs"):
            teachers.build(bad, self.root / "out")
        bad = copy.deepcopy(spec)
        bad["expect_s_rows"] = 6
        with self.assertRaisesRegex(ValueError, "expects 6"):
            teachers.build(bad, self.root / "out")
        bad = copy.deepcopy(spec)
        bad["teachers"]["aj"]["files"][0]["pool"] = "A7"
        with self.assertRaisesRegex(ValueError, "names pool"):
            teachers.build(bad, self.root / "out")

    def test_mixture_mismatch(self):
        report = json.loads(self.mixtures.read_text())
        report["mixtures"]["a7"]["parts"]["A7"]["rows"] = 3
        self.mixtures.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "A7: 2 rows"):
            teachers.build(self.spec(), self.root / "out")
        report["mixtures"]["a7"]["parts"]["A7"]["rows"] = 2
        report["mixtures"]["a7"]["by_source"]["legacy:snli"] = 2
        self.mixtures.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "per source"):
            teachers.build(self.spec(), self.root / "out")

    def test_committed_spec_shape(self):
        spec = json.loads(
            (Path(teachers.__file__).parent / "teacher-sources-m4b.json").read_text()
        )
        self.assertEqual(spec["expect_s_rows"], 5744)
        self.assertEqual(sorted(spec["teachers"]), ["aj", "lux"])
        for teacher in spec["teachers"].values():
            for entry in teacher["files"]:
                self.assertIn(entry["pool"], teachers.S_POOLS)
                self.assertRegex(entry["sha256"], "^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
