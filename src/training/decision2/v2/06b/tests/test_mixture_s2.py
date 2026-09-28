import importlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from training.model.data import INPUT_FIELDS, digest

mixture = importlib.import_module("v2.06b.mixture")
common = importlib.import_module("v2.06b.common")


def row(
    rid,
    *,
    family="fam",
    group=None,
    kind="choice",
    keys=("a", "b"),
    label=0,
    tokens=10,
    state=None,
):
    out = {
        "id": rid,
        "state": state if state is not None else f"state {rid}",
        "instructions": "pick one",
        "options": [{"key": k, "description": f"option {k}"} for k in keys],
        "label": label,
        "task_type": kind,
        "family": family,
        "group_id": group or f"g-{rid}",
        "language": "en",
        "split": "train",
        "source": "src",
        "evaluation_role": "train",
        "render_template": "t",
        "audit_metadata": {},
        "_tokens": tokens,
    }
    out["input_sha256"] = digest({field: out[field] for field in INPUT_FIELDS})
    return out


def count(r):
    return r["_tokens"]


class ArmFiles:
    def __init__(self):
        self.dir = tempfile.TemporaryDirectory()

    def arm(self, name, rows):
        path = Path(self.dir.name) / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return {
            "arm": name,
            "path": str(path),
            "sha256": common.file_sha256(path),
            "rows": len(rows),
        }

    def view(self, members):
        path = Path(self.dir.name) / "view.json"
        path.write_text(json.dumps({"members": members}))
        return {"path": str(path), "sha256": common.file_sha256(path)}


class BaseTest(unittest.TestCase):
    def test_excluded_families_and_positional_renumbering(self):
        files = ArmFiles()
        leaky = row("r1", keys=("result_2", "result_0", "result_1"), label=0)
        rows = [leaky, row("r2", family="drop"), row("r3")]
        entry = {
            **files.arm("A0s", rows),
            "exclude_families": ["drop"],
            "renumber_option_keys": True,
        }
        kept, tokens, seen, report = mixture.filtered_base(entry, count, 100)
        self.assertEqual([r["id"] for r in kept], ["r1", "r3"])
        self.assertEqual(
            [o["key"] for o in kept[0]["options"]], ["result_0", "result_1", "result_2"]
        )
        self.assertNotEqual(kept[0]["input_sha256"], leaky["input_sha256"])
        self.assertIn(leaky["input_sha256"], seen)
        self.assertIn(kept[0]["input_sha256"], seen)
        self.assertIn(rows[1]["input_sha256"], seen)
        self.assertEqual(
            (report["excluded_family_rows"], report["renumbered_rows"]), (1, 1)
        )

    def test_base_row_over_cap_is_an_error(self):
        files = ArmFiles()
        with self.assertRaises(ValueError):
            mixture.filtered_base(files.arm("A0s", [row("r1", tokens=50)]), count, 49)


class ViewComponentTest(unittest.TestCase):
    def setUp(self):
        self.files = ArmFiles()
        self.base = row("b1", state="shared")
        big = [row(f"x{i}", tokens=10) for i in range(40)]
        small = [row(f"y{i}", tokens=10) for i in range(3)]
        dup = row("x-dup", state="shared")
        excluded = row("x-ex", family="bad")
        huge = row("x-huge", tokens=500)
        self.comp = {
            "name": "A7",
            "policy": "view_then_equal_shares",
            "exclude_families": ["bad"],
            "arms": [
                self.files.arm("X", big + [dup, excluded, huge]),
                self.files.arm("Y", small),
            ],
            "view": self.files.view(
                [
                    {"id": "x0", "part": "train"},
                    {"id": "x-ex", "part": "train"},
                    {"id": "x-dup", "part": "train"},
                    {"id": "x-huge", "part": "train"},
                    {"id": "x1", "part": "aho"},
                ]
            ),
        }

    def build(self, target):
        return mixture.view_component(
            self.comp, count, 100, {self.base["input_sha256"]}, target, "seed"
        )

    def test_view_first_then_water_filled_equal_shares(self):
        rows, tokens, report = self.build(210)
        ids = [r["id"] for r in rows]
        self.assertEqual(ids[0], "x0")
        self.assertNotIn("x-ex", ids)
        self.assertNotIn("x-dup", ids)
        self.assertNotIn("x-huge", ids)
        self.assertEqual(sum(i.startswith("y") for i in ids), 3)
        self.assertEqual(sum(tokens), 210)
        self.assertEqual(report["view_tokens"], 10)
        self.assertEqual(
            report["dropped"],
            {
                "X:excluded_family": 1,
                "X:over_cap": 1,
                "X:repeats_earlier_input": 1,
                "view:excluded_or_absent": 1,
            },
        )
        self.assertEqual(ids, [r["id"] for r in self.build(210)[0]])

    def test_target_below_view_takes_only_the_view(self):
        rows, tokens, _ = self.build(5)
        self.assertEqual([r["id"] for r in rows], ["x0"])


class ResampleAndBuildTest(unittest.TestCase):
    def test_matched_token_resample_copies(self):
        base = [row(f"b{i}", group=f"g{i // 2}", tokens=10) for i in range(10)]
        rows, tokens, report = mixture.resample_component(
            {"tokens": 250}, base, [10] * 10, "seed"
        )
        self.assertEqual(report["whole_copies"], 2)
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        self.assertTrue(
            all(r["teacher_source_id"] == r["id"].split("#")[0] for r in rows)
        )
        self.assertLessEqual(sum(tokens), 250)
        self.assertGreaterEqual(sum(tokens), 240)

    def test_build_s2_end_to_end(self):
        files = ArmFiles()
        base = [row(f"b{i}", tokens=10) for i in range(6)]
        extra = [row(f"e{i}", tokens=10) for i in range(10)]
        spec = {
            "template": "S2",
            "unit": mixture.UNIT,
            "seed": "s",
            "max_row_tokens": 100,
            "base": [files.arm("A0s", base)],
            "components": [
                {
                    "name": "A6",
                    "policy": "token_budget",
                    "tokens": 30,
                    "seed": "m3",
                    "arms": [files.arm("A6g", extra)],
                },
            ],
        }
        rows, tokens, report = mixture.build_s2(spec, count)
        self.assertEqual(report["train_rows"], 9)
        self.assertEqual(report["components"]["A6"]["rows"], 3)
        spec["expected"] = {"train_rows": 8}
        with self.assertRaises(ValueError):
            mixture.build_s2(spec, count)


class ManifestComponentTest(unittest.TestCase):
    def test_join_by_id_with_group_cap_and_base_dedup(self):
        files = ArmFiles()
        base = row("b1", state="shared")
        pool = [
            row("p1", group="g1"),
            row("p2", group="g1", tokens=500),
            row("p3", group="g2"),
            row("p4", group="g3", state="shared"),
            row("p5", group="g4"),
        ]
        manifest_path = Path(files.dir.name) / "recipe.ids.jsonl"
        entries = [{"id": "b1", "pool": "A0s"}] + [
            {"id": r["id"], "pool": "P"} for r in pool[:4]
        ]
        manifest_path.write_text("".join(json.dumps(e) + "\n" for e in entries))
        comp = {
            "manifest": {
                "path": str(manifest_path),
                "sha256": common.file_sha256(manifest_path),
                "rows": len(entries),
            },
            "skip_pools": ["A0s"],
            "pools": {"P": [files.arm("P", pool)]},
        }
        rows, tokens, report = mixture.manifest_component(
            comp, count, 100, {base["input_sha256"]}
        )
        self.assertEqual([r["id"] for r in rows], ["p3"])
        self.assertEqual(
            report["dropped"],
            {"P:over_cap_group_rows": 2, "P:repeats_earlier_input": 1},
        )
        self.assertEqual(report["skipped_pool_rows"], {"A0s": 1})
        entries.append({"id": "absent", "pool": "P"})
        manifest_path.write_text("".join(json.dumps(e) + "\n" for e in entries))
        comp["manifest"].update(
            sha256=common.file_sha256(manifest_path), rows=len(entries)
        )
        with self.assertRaises(ValueError):
            mixture.manifest_component(comp, count, 100, set())


class MaterializedComponentTest(unittest.TestCase):
    def setUp(self):
        self.files = ArmFiles()
        self.rows = [row(f"a7:x{i}", group=f"g{i // 2}", tokens=10) for i in range(8)]
        self.rows += [row("a7:n1", kind="noul", keys=("true", "false"), tokens=10)]
        self.rows += [row("a6-s1", kind="score", keys=("0", "1", "2"), tokens=10)]
        self.rows += [row("base1", tokens=10)]
        path = Path(self.files.dir.name) / "mix.train.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        self.source = {
            "path": str(path),
            "sha256": common.file_sha256(path),
            "rows": len(self.rows),
        }

    def comp(self, **extra):
        return {
            "name": "T-A7",
            "policy": "from_materialized",
            "mixture": self.source,
            "id_prefixes": ["a7:"],
            "seed": "s",
            **extra,
        }

    def test_prefix_and_type_filter_keeps_rows_identical(self):
        rows, tokens, report = mixture.materialized_component(
            self.comp(task_types=["choice"]), count, 100, set()
        )
        self.assertEqual([r["id"] for r in rows], [f"a7:x{i}" for i in range(8)])
        self.assertEqual(rows, self.rows[:8])
        self.assertEqual(report["rows_matching_prefix"], 9)
        self.assertEqual(report["task_type_tokens"], {"choice": 80})

    def test_subsample_takes_whole_groups_in_hash_order(self):
        rows, tokens, _ = mixture.materialized_component(
            self.comp(subsample_tokens=45), count, 100, set()
        )
        self.assertLessEqual(sum(tokens), 45)
        self.assertGreaterEqual(sum(tokens), 30)
        groups = Counter(r["group_id"] for r in rows if r["id"].startswith("a7:x"))
        self.assertTrue(all(n == 2 for n in groups.values()))
        again, _, _ = mixture.materialized_component(
            self.comp(subsample_tokens=45), count, 100, set()
        )
        self.assertEqual(rows, again)

    def test_repeated_input_hash_or_changed_file_is_an_error(self):
        with self.assertRaises(ValueError):
            mixture.materialized_component(
                self.comp(), count, 100, {self.rows[0]["input_sha256"]}
            )
        with self.assertRaises(ValueError):
            mixture.materialized_component(
                {**self.comp(), "mixture": {**self.source, "sha256": "0" * 64}},
                count,
                100,
                set(),
            )


class ManifestBudgetTest(unittest.TestCase):
    def test_budget_is_split_by_pool_tokens_in_whole_groups(self):
        files = ArmFiles()
        p = [row(f"p{i}", group=f"pg{i // 2}", tokens=10) for i in range(20)]
        q = [row(f"q{i}", group=f"qg{i}", tokens=10) for i in range(10)]
        manifest_path = Path(files.dir.name) / "recipe.ids.jsonl"
        entries = [{"id": r["id"], "pool": "P"} for r in p]
        entries += [{"id": r["id"], "pool": "Q"} for r in q]
        manifest_path.write_text("".join(json.dumps(e) + "\n" for e in entries))
        comp = {
            "seed": "s",
            "manifest": {
                "path": str(manifest_path),
                "sha256": common.file_sha256(manifest_path),
                "rows": len(entries),
            },
            "pools": {"P": [files.arm("P", p)], "Q": [files.arm("Q", q)]},
        }
        rows, tokens, report = mixture.manifest_component(comp, count, 100, set(), 150)
        by_pool = Counter(r["id"][0] for r in rows)
        self.assertEqual(by_pool, {"p": 10, "q": 5})
        self.assertEqual(report["budget_tokens"], 150)
        self.assertEqual(report["dropped"], {"budget:groups_not_selected": 10})
        full, _, full_report = mixture.manifest_component(comp, count, 100, set())
        self.assertEqual(len(full), 30)
        self.assertNotIn("budget_tokens", full_report)


class TeacherAlignTest(unittest.TestCase):
    def test_only_rows_every_teacher_covers_are_kept(self):
        align = importlib.import_module("v2.06b.teacher_align")
        rows = [row("c1", keys=("a", "b")), row("c2", keys=("a", "b")), row("c3")]
        copy = {**rows[0], "id": "c1#resample1", "teacher_source_id": "c1"}

        def entry(r, sha=None):
            return {
                "id": r["id"],
                "input_sha256": sha or r["input_sha256"],
                "teacher_probs": {"a": 0.4, "b": 0.6},
            }

        lux = {e["id"]: e for e in (entry(rows[0]), entry(rows[1]))}
        aj = {
            e["id"]: e
            for e in (entry(rows[0]), entry(rows[1], "0" * 64), entry(rows[2]))
        }
        out, report = align.align(rows + [copy], {"lux": lux, "aj": aj})
        self.assertEqual([e["id"] for e in out["lux"]], ["c1"])
        self.assertEqual([e["id"] for e in out["aj"]], ["c1"])
        self.assertEqual(report["aligned_rows"], 2)
        self.assertEqual(report["covered_by_teacher"]["lux"]["rows"], 3)
        self.assertEqual(report["covered_by_teacher"]["aj"]["rows"], 3)
        bad = {"c1": {**entry(rows[0]), "teacher_probs": {"x": 1.0}}}
        with self.assertRaises(ValueError):
            align.align(rows, {"bad": bad})


class TeacherMergeTest(unittest.TestCase):
    def test_merge_checks_hashes_repeats_and_distributions(self):
        merge = importlib.import_module("v2.06b.teacher_merge")
        with tempfile.TemporaryDirectory() as tmp:

            def write(name, entries):
                path = Path(tmp) / name
                path.write_text("".join(json.dumps(e) + "\n" for e in entries))
                return path, common.file_sha256(path)

            good = {"input_sha256": "0" * 64, "teacher_probs": {"a": 0.25, "b": 0.75}}
            a = write("a.jsonl", [{"id": "x", **good}])
            b = write("b.jsonl", [{"id": "y", **good}])
            rows, report = merge.merge([a, b])
            self.assertEqual([r["id"] for r in rows], ["x", "y"])
            self.assertEqual(report["rows"], 2)
            with self.assertRaises(ValueError):
                merge.merge([a, a])
            with self.assertRaises(ValueError):
                merge.merge([(a[0], "1" * 64)])
            bad = write(
                "c.jsonl",
                [
                    {
                        "id": "z",
                        "input_sha256": "0" * 64,
                        "teacher_probs": {"a": 0.5, "b": 0.6},
                    }
                ],
            )
            with self.assertRaises(ValueError):
                merge.merge([bad])


class OptionKeyTeacherTest(unittest.TestCase):
    def test_native_order_hash_guard_and_copies(self):
        train = importlib.import_module("v2.06b.train")
        noul = row("n1", kind="noul", keys=("true", "false"), label=0)
        choice = row("c1", keys=("a", "b", "c"))
        changed = row("c2", keys=("a", "b"))
        copy = {**choice, "id": "c1#resample1", "teacher_source_id": "c1"}
        entries = [
            {
                "id": "n1",
                "input_sha256": noul["input_sha256"],
                "teacher_probs": {"true": 0.8, "false": 0.2},
            },
            {
                "id": "c1",
                "input_sha256": choice["input_sha256"],
                "teacher_probs": {"a": 0.1, "b": 0.2, "c": 0.7},
            },
            {
                "id": "c2",
                "input_sha256": "0" * 64,
                "teacher_probs": {"a": 0.5, "b": 0.5},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teacher.jsonl"
            path.write_text("".join(json.dumps(e) + "\n" for e in entries))
            out, counts = train.option_key_teacher(
                [noul, choice, changed, copy, row("z")], path
            )
        self.assertEqual(out[0], [0.2, 0.8])
        self.assertEqual(out[1], [0.1, 0.2, 0.7])
        self.assertIsNone(out[2])
        self.assertEqual(out[3], [0.1, 0.2, 0.7])
        self.assertIsNone(out[4])
        self.assertEqual(counts, {"covered": 3, "no_entry": 1, "input_changed": 1})


if __name__ == "__main__":
    unittest.main()
