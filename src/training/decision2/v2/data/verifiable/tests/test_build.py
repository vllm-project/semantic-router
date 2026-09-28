import collections
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from training.model.data import check_partition_isolation, load_partition
from v2.data.verifiable import build, core
from v2.data.verifiable.tests import checks

ROOT = Path(__file__).resolve().parents[4]
SMALL = 12
SEEDS = {"a2": "unit-a2", "a4": "unit-a4", "a6": "unit-a6"}


def _files(arm):
    names = ("a4h", "a4r") if arm == "a4" else (arm,)
    return [f"{n}.{part}.jsonl" for n in names for part in ("train", "aho")] + [
        f"{arm}.build.json"
    ]


class BuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "first"
        cls.manifests = {}
        for arm, seed in SEEDS.items():
            files, manifest = build.generate(arm, seed, SMALL, 0.3)
            build.write(cls.out, arm, files, manifest)
            cls.manifests[arm] = manifest
        cls.partitions = {}
        for name in ("a2", "a6", "a4h", "a4r"):
            cls.partitions[name] = (
                load_partition(cls.out / f"{name}.train.jsonl", "train"),
                load_partition(cls.out / f"{name}.aho.jsonl", "select"),
            )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, name):
        train, aho = self.partitions[name]
        return train + aho

    def test_same_seed_gives_identical_bytes(self):
        second = Path(self.tmp.name) / "second"
        for arm, seed in SEEDS.items():
            files, manifest = build.generate(arm, seed, SMALL, 0.3)
            build.write(second, arm, files, manifest)
            for name in _files(arm):
                self.assertEqual(
                    (self.out / name).read_bytes(), (second / name).read_bytes(), name
                )
        other = build.generate("a2", "another-seed", SMALL, 0.3)[0]["a2.train.jsonl"]
        self.assertNotEqual(other[0]["state"], self.rows("a2")[0]["state"])

    def test_bytes_do_not_depend_on_process_hash_seed(self):
        for arm in ("a2", "a4"):
            for hash_seed in ("0", "4242"):
                out = Path(self.tmp.name) / f"proc-{arm}-{hash_seed}"
                env = {**os.environ, "PYTHONHASHSEED": hash_seed}
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "v2.data.verifiable.build",
                        "--arm",
                        arm,
                        "--seed",
                        SEEDS[arm],
                        "--out-dir",
                        str(out),
                        "--groups-per-family",
                        str(SMALL),
                    ],
                    cwd=ROOT,
                    env=env,
                    check=True,
                    capture_output=True,
                )
                for name in _files(arm):
                    self.assertEqual(
                        (self.out / name).read_bytes(),
                        (out / name).read_bytes(),
                        f"{name} {hash_seed}",
                    )

    def test_rows_pass_contract_and_partition_isolation(self):
        for name, (train, aho) in self.partitions.items():
            with self.subTest(file=name):
                self.assertTrue(train)
                check_partition_isolation({"train": train, "select": aho})
                for row in train + aho:
                    self.assertEqual(
                        row["split"] == "select",
                        core.held_out(row["group_id"]),
                        row["id"],
                    )
                    checks.assert_row(self, row)

    def test_identifier_formats(self):
        for name in ("a2", "a6", "a4h", "a4r"):
            arm = "a4" if name.startswith("a4") else name
            for row in self.rows(name):
                self.assertRegex(
                    row["id"], rf"^{name}-{row['family']}-(en|zh)-\d{{5}}-v\d+$"
                )
                self.assertRegex(
                    row["group_id"],
                    rf"^{arm}:{row['family']}:{row['language']}:[0-9a-f]{{16}}$",
                )
                self.assertEqual(row["source"], f"decision2_verifiable_v2_{name}")
                for key in (
                    "variant_index",
                    "operative_edit",
                    "facts_sha256",
                    "version",
                    "family",
                ):
                    self.assertIn(key, row["audit_metadata"])
                if row["task_type"] == "choice":
                    self.assertEqual(
                        [o["key"] for o in row["options"]],
                        [f"o{i}" for i in range(1, len(row["options"]) + 1)],
                    )
                if row["task_type"] == "noul":
                    expected = (
                        ["No", "Yes"] if row["language"] == "en" else ["否", "是"]
                    )
                    self.assertEqual(
                        [o["description"] for o in row["options"]], expected
                    )

    def test_counterfactual_groups_cover_every_gold_once(self):
        for name in ("a2", "a6"):
            groups = checks.assert_counterfactual_groups(self, self.rows(name))
            self.assertGreater(groups, 30)
            rows = self.rows(name)
            same = sum(r["audit_metadata"]["variant_index"] == r["label"] for r in rows)
            self.assertLess(
                same / len(rows), 0.6, "variant index should not encode the label"
            )

    def test_a2_families_languages_and_tasks(self):
        rows = self.rows("a2")
        families = collections.Counter(r["family"] for r in rows)
        self.assertEqual(set(families), {m.FAMILY for m in build.A2_FAMILIES})
        zh = sum(r["language"] == "zh" for r in rows) / len(rows)
        self.assertTrue(0.2 < zh < 0.4, zh)
        self.assertEqual({r["task_type"] for r in rows}, {"choice", "noul"})

    def test_a6_level_counts_and_grades_are_balanced(self):
        files, manifest = build.generate("a6", "unit-a6-balance", 46, 0.3)
        rows = files["a6.train.jsonl"] + files["a6.aho.jsonl"]
        target = manifest["parameters"]["rows_per_level_target"]
        by_levels = collections.Counter(len(r["options"]) for r in rows)
        self.assertEqual(set(by_levels), set(build.LEVELS))
        for levels, count in by_levels.items():
            self.assertLessEqual(
                abs(count - target), 0.05 * target, (levels, count, target)
            )
            grades = collections.Counter(
                r["label"] for r in rows if len(r["options"]) == levels
            )
            self.assertEqual(set(grades), set(range(levels)))
            self.assertEqual(len(set(grades.values())), 1, (levels, grades))
        evidence = {
            len(r["options"]) for r in rows if r["family"] == "a6_evidence_status"
        }
        self.assertEqual(evidence, {3})
        self.assertEqual(manifest["oracle_disagreements"], 0)

    def test_default_sizing_plans(self):
        plan = build.a6_plan(build.DEFAULT_A6_ROWS_PER_LEVEL)
        rows = collections.Counter()
        for levels_list in plan.values():
            for levels in levels_list:
                rows[levels] += levels
        for levels, count in rows.items():
            self.assertLessEqual(abs(count - 600), 30, (levels, count))
        self.assertEqual(build.rows_per_level(None), 600)

    def test_a4_files_are_aligned_and_differ_only_in_distractors(self):
        hard = {r["id"].removeprefix("a4h-"): r for r in self.rows("a4h")}
        rand = {r["id"].removeprefix("a4r-"): r for r in self.rows("a4r")}
        self.assertEqual(set(hard), set(rand))
        differing = 0
        for key, h in hard.items():
            r = rand[key]
            self.assertEqual(h["state"], r["state"], key)
            self.assertEqual(h["group_id"], r["group_id"], key)
            self.assertEqual(h["label"], r["label"], key)
            self.assertEqual(len(h["options"]), len(r["options"]), key)
            self.assertEqual(h["options"][h["label"]], r["options"][r["label"]], key)
            if h["task_type"] == "choice":
                self.assertEqual(h["instructions"], r["instructions"], key)
                near = {
                    o["description"]
                    for i, o in enumerate(h["options"])
                    if i != h["label"]
                }
                far = {
                    o["description"]
                    for i, o in enumerate(r["options"])
                    if i != r["label"]
                }
                self.assertNotEqual(near, far, key)
                differing += 1
            elif h["label"] == 1:
                self.assertEqual(h["instructions"], r["instructions"], key)
            else:
                self.assertNotEqual(h["instructions"], r["instructions"], key)
        self.assertGreater(differing, 50)
        groups = collections.Counter(r["group_id"] for r in self.rows("a4h"))
        self.assertTrue(set(groups.values()) <= {1, 2})
        positions = collections.Counter(
            r["label"] for r in self.rows("a4h") if r["task_type"] == "choice"
        )
        self.assertLessEqual(
            max(positions.values()) - min(positions.values()), 6, positions
        )

    def test_a4_groups_are_disjoint_from_a2(self):
        a2_groups = {r["group_id"].split(":", 1)[1] for r in self.rows("a2")}
        a4_groups = {r["group_id"].split(":", 1)[1] for r in self.rows("a4h")}
        self.assertFalse(a2_groups & a4_groups)
        a2_states = {r["state"] for r in self.rows("a2")}
        self.assertFalse(a2_states & {r["state"] for r in self.rows("a4h")})

    def test_manifest_counts_and_code_hashes(self):
        package = Path(build.__file__).resolve().parent
        modules = {p.name for p in package.glob("*.py")}
        for arm, manifest in self.manifests.items():
            self.assertEqual(set(manifest["code_sha256"]), modules)
            self.assertEqual(manifest["oracle_disagreements"], 0)
            self.assertEqual(manifest["dropped_rows"], 0)
            self.assertEqual(manifest["seed"], SEEDS[arm])
            for name, info in manifest["files"].items():
                lines = (self.out / name).read_text(encoding="utf-8").splitlines()
                self.assertEqual(info["rows"], len(lines), name)
                self.assertRegex(info["file_sha256"], r"^[0-9a-f]{64}$")

    def test_refuses_to_overwrite(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            code = build.main(
                [
                    "--arm",
                    "a2",
                    "--seed",
                    SEEDS["a2"],
                    "--out-dir",
                    str(self.out),
                    "--groups-per-family",
                    "2",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("refusing to overwrite", err.getvalue())
        with self.assertRaises(FileExistsError):
            build.write(self.out, "a2", {"a2.train.jsonl": []}, {"files": {}})

    def test_state_lengths_vary_across_groups(self):
        words = [
            core.words(r["state"], r["language"])
            for name in ("a2", "a6")
            for r in self.rows(name)
        ]
        words.sort()
        self.assertLess(words[0], 120)
        self.assertGreater(words[-1], 400)
        median = words[len(words) // 2]
        self.assertTrue(120 <= median <= 320, median)


class ChoiceTextTest(unittest.TestCase):
    def test_option_keys_do_not_encode_order_letters(self):
        options = core.choice_options(["x", "y", "z"])
        self.assertTrue(all(re.fullmatch(r"o\d+", o["key"]) for o in options))


if __name__ == "__main__":
    unittest.main()
