import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

from training.model.data import INPUT_FIELDS, digest, file_sha256  # noqa: E402
from lux9b import m3_data  # noqa: E402

REGISTRY = {
    "sources": [
        {"dataset_id": "Bekhouche/HalluTruthQA-4K", "grade": "A"},
        {"dataset_id": "Hplss/wb-review-dataset", "grade": "B"},
        {
            "dataset_id": "Arena-90K / SHP / SQuAD / SNLI / IMDB derivatives",
            "grade": "C",
        },
    ]
}


def row(rid, group, source="h1_train", family="fam", split="train", keys=("a", "b")):
    out = {
        "id": rid,
        "state": f"state {rid}",
        "instructions": "Pick one.",
        "options": [{"key": k, "description": f"option {k}"} for k in keys],
        "label": 0,
        "task_type": "choice",
        "family": family,
        "group_id": group,
        "language": "en",
        "split": split,
        "source": source,
        "evaluation_role": split,
        "render_template": "t",
        "audit_metadata": {},
    }
    out["input_sha256"] = digest({f: out[f] for f in INPUT_FIELDS})
    return out


def teacher(r):
    keys = [o["key"] for o in r["options"]]
    return {
        "id": r["id"],
        "input_sha256": r["input_sha256"],
        "teacher_probs": dict.fromkeys(keys, 1 / len(keys)),
    }


class SourceGuardTest(unittest.TestCase):
    def test_candidates_are_denied_and_rejected_entries_are_not(self):
        keys = m3_data.c1_keys(REGISTRY)
        self.assertTrue(m3_data.denied_hits("hallutruthqa_train", keys, []))
        self.assertTrue(m3_data.denied_hits("wb-review-dataset", keys, []))
        self.assertFalse(m3_data.denied_hits("legacy:snli", keys, ["xnli"]))
        self.assertFalse(m3_data.denied_hits("squad2_train", keys, []))
        self.assertFalse(
            m3_data.denied_hits("google_goemotions_official_train", keys, [])
        )
        self.assertTrue(m3_data.denied_hits("massive_intents", keys, ["massive"]))


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.recipe_rows = [
            row("a0s-1", "g1", "legacy:snli"),
            row("h1-1", "g2"),
            row("h1-2", "g2"),
        ]
        a0_dup = row("a7:dup", "g9", "legacy:snli")
        a0_dup["state"] = self.recipe_rows[0]["state"]
        a0_dup["input_sha256"] = self.recipe_rows[0]["input_sha256"]
        renamed = row("a7:renum", "g8", "legacy:snli", keys=("x", "y"))
        a0_extra = row("a0-only", "g7", "legacy:snli", keys=("p", "q"))
        renamed["audit_metadata"] = {
            "a7": {"original_input_sha256": a0_extra["input_sha256"]}
        }
        self.replay = (
            [a0_dup, renamed]
            + [row(f"a7:r{i}", f"rg{i}", "legacy:stage4") for i in range(6)]
            + [row("a7:cosmos", "cg", "legacy:cosmos_qa", family="natural_cosmos_qa")]
        )
        self.files = {}
        self.write(
            "recipe.jsonl",
            [
                {
                    "id": r["id"],
                    "pool": "A0s" if r["id"].startswith("a0s") else "H1",
                    "source": r["source"],
                    "native": 5,
                }
                for r in self.recipe_rows
            ],
        )
        self.write("a0s.jsonl", self.recipe_rows[:1])
        self.write("h1.jsonl", self.recipe_rows[1:])
        self.write("teacher.jsonl", [teacher(r) for r in self.recipe_rows])
        self.write("a0.jsonl", self.recipe_rows[:1] + [a0_extra])
        self.write("a7.jsonl", self.replay)
        self.write("select.jsonl", [row("s-1", "sg", split="select")])
        (self.dir / "view.json").write_text(
            json.dumps(
                {"members": [{"id": r["id"], "part": "train"} for r in self.replay]}
            )
        )
        (self.dir / "c1.json").write_text(json.dumps(REGISTRY))

    def write(self, name, records):
        (self.dir / name).write_text("".join(json.dumps(r) + "\n" for r in records))

    def ref(self, name):
        return {"file": f"t:{name}", "sha256": file_sha256(self.dir / name)}

    def spec(self, **changes):
        spec = {
            "name": "t",
            "seed": "s",
            "max_length": 100,
            "recipe": self.ref("recipe.jsonl"),
            "pools": {"A0s": [self.ref("a0s.jsonl")], "H1": [self.ref("h1.jsonl")]},
            "teachers": [self.ref("teacher.jsonl")],
            "a0_for_dedupe": self.ref("a0.jsonl"),
            "replay": {
                "view": self.ref("view.json"),
                "files": [self.ref("a7.jsonl")],
                "exclude_families": ["natural_cosmos_qa"],
                "budget_tokens": 20,
            },
            "isolation": [{"role": "select", **self.ref("select.jsonl")}],
            "denied_sources": {"c1_registry": "t:c1.json", "extra": ["massive"]},
        }
        spec.update(changes)
        return spec

    def build(self, spec):
        lengths = lambda rows, *_: [5] * len(rows)  # noqa: E731
        with mock.patch.object(m3_data, "token_lengths", lengths):
            return m3_data.build(spec, {"t": self.dir}, Path("."), 1)

    def test_recipe_teacher_and_replay_dedupe(self):
        train, teachers, manifest = self.build(self.spec())
        ids = {r["id"] for r in train}
        self.assertTrue({"a0s-1", "h1-1", "h1-2"} <= ids)
        self.assertNotIn("a7:dup", ids)
        self.assertNotIn("a7:renum", ids)
        self.assertNotIn("a7:cosmos", ids)
        self.assertEqual(manifest["replay"]["duplicate_of_a0_or_recipe"], 2)
        self.assertEqual(manifest["replay"]["excluded_family"], 1)
        self.assertEqual(manifest["replay"]["tokens"], 20)
        self.assertEqual([t["id"] for t in teachers], ["a0s-1", "h1-1", "h1-2"])

    def test_extra_components_dedupe_against_earlier_rows(self):
        repeat = row("a7g:copy", "cg2", "legacy:stage4")
        repeat["state"] = self.replay[2]["state"]
        repeat["input_sha256"] = self.replay[2]["input_sha256"]
        extra = [repeat] + [row(f"v1:{i}", f"vg{i}", "a2_verifiable") for i in range(4)]
        self.write("v1.jsonl", extra)
        spec = self.spec()
        spec["replay"]["budget_tokens"] = 10**6
        spec["extras"] = [
            {"name": "v1", "files": [self.ref("v1.jsonl")], "budget_tokens": 10}
        ]
        train, _, manifest = self.build(spec)
        ids = {r["id"] for r in train}
        self.assertNotIn("a7g:copy", ids)
        self.assertEqual(manifest["extras"]["v1"]["duplicate_of_a0_or_recipe"], 1)
        self.assertEqual(manifest["extras"]["v1"]["rows"], 2)
        self.assertEqual(sum(i.startswith("v1:") for i in ids), 2)

    def test_missing_teacher_or_bad_hash_fails(self):
        self.write("teacher.jsonl", [teacher(r) for r in self.recipe_rows[:2]])
        with self.assertRaises(ValueError):
            self.build(self.spec(teachers=[self.ref("teacher.jsonl")]))
        spec = self.spec()
        spec["recipe"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.build(spec)

    def test_denied_source_fails(self):
        bad = row("a7:bad", "bg", "hallutruthqa_train")
        self.write("a7.jsonl", self.replay + [bad])
        (self.dir / "view.json").write_text(
            json.dumps(
                {
                    "members": [
                        {"id": r["id"], "part": "train"} for r in self.replay + [bad]
                    ]
                }
            )
        )
        spec = self.spec()
        spec["replay"].update(
            view=self.ref("view.json"),
            files=[self.ref("a7.jsonl")],
            budget_tokens=10**6,
        )
        with self.assertRaises(ValueError):
            self.build(spec)


class RecipeBudgetTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.rows, self.recipe = [], []
        for pool, source in (("P1", "s1"), ("P2", "s2"), ("P3", "s3")):
            for g in range(30):
                for k in range(1 + g % 3):
                    score = (g // 2) % 2
                    keys = ("0", "1") if score else ("a", "b")
                    r = row(f"{pool}-{g}-{k}", f"{pool}-g{g}", source, keys=keys)
                    r["language"] = ("en", "de")[g % 2]
                    r["task_type"] = ("choice", "score")[score]
                    r["input_sha256"] = digest({f: r[f] for f in INPUT_FIELDS})
                    self.rows.append(r)
                    self.recipe.append(
                        {
                            "id": r["id"],
                            "pool": pool,
                            "source": source,
                            "native": 10 + (g * 7 + k) % 23,
                        }
                    )
        self.native = {e["id"]: e["native"] for e in self.recipe}
        self.total = sum(self.native.values())
        self.write("recipe.jsonl", self.recipe)
        for pool in ("P1", "P2", "P3"):
            self.write(f"{pool}.jsonl", [r for r in self.rows if r["id"][:2] == pool])
        self.write("teacher.jsonl", [teacher(r) for r in self.rows])
        self.write("select.jsonl", [row("s-1", "sg", split="select")])
        (self.dir / "c1.json").write_text(json.dumps(REGISTRY))

    def write(self, name, records):
        (self.dir / name).write_text("".join(json.dumps(r) + "\n" for r in records))

    def ref(self, name):
        return {"file": f"t:{name}", "sha256": file_sha256(self.dir / name)}

    def spec(self, **changes):
        spec = {
            "name": "t",
            "seed": "20260929",
            "max_length": 100,
            "recipe": self.ref("recipe.jsonl"),
            "recipe_budget_tokens": self.total // 3,
            "recipe_budget_tolerance": 0.5,
            "pools": {p: [self.ref(f"{p}.jsonl")] for p in ("P1", "P2", "P3")},
            "teachers": [self.ref("teacher.jsonl")],
            "isolation": [{"role": "select", **self.ref("select.jsonl")}],
            "denied_sources": {"c1_registry": "t:c1.json", "extra": ["massive"]},
        }
        spec.update(changes)
        return spec

    def build(self, spec):
        lengths = lambda rows, *_: [self.native[r["id"]] for r in rows]  # noqa: E731
        with mock.patch.object(m3_data, "token_lengths", lengths):
            return m3_data.build(spec, {"t": self.dir}, Path("."), 1)

    def test_budget_whole_groups_and_manifest(self):
        train, teachers, manifest = self.build(self.spec())
        budget = manifest["recipe"]["recipe_budget"]
        realized = sum(self.native[r["id"]] for r in train)
        self.assertEqual(budget["native_tokens"], realized)
        self.assertEqual(manifest["recipe"]["recipe_native_tokens"], realized)
        self.assertGreaterEqual(realized, self.total // 3)
        self.assertLessEqual(realized, self.total // 3 * 1.5)
        chosen = {r["group_id"] for r in train}
        self.assertEqual(
            {r["id"] for r in train},
            {r["id"] for r in self.rows if r["group_id"] in chosen},
        )
        pools = manifest["recipe"]["recipe_selected_by_pool"]
        self.assertEqual(set(pools), {"P1", "P2", "P3"})
        self.assertEqual(sum(p["rows"] for p in pools.values()), len(train))
        self.assertEqual(sum(p["tokens"] for p in pools.values()), realized)
        self.assertEqual([t["id"] for t in teachers], sorted(r["id"] for r in train))
        self.assertEqual(manifest["recipe"]["recipe_languages"], 2)

    def test_excluded_pools_are_dropped_and_unknown_pool_fails(self):
        train, _, manifest = self.build(self.spec(recipe_exclude_pools=["P3"]))
        self.assertFalse(any(r["id"].startswith("P3") for r in train))
        self.assertIn("P3", manifest["recipe"]["recipe_excluded_pools"])
        self.assertEqual(
            set(manifest["recipe"]["recipe_selected_by_pool"]), {"P1", "P2"}
        )
        with self.assertRaises(ValueError):
            self.build(self.spec(recipe_exclude_pools=["V2:P3"]))

    def test_same_seed_same_output(self):
        first = self.build(self.spec())
        second = self.build(self.spec())
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        other = self.build(self.spec(seed="other"))
        self.assertNotEqual({r["id"] for r in first[0]}, {r["id"] for r in other[0]})

    def test_selected_row_without_target_or_with_wrong_hash_fails(self):
        self.write(
            "teacher.jsonl", [teacher(r) for r in self.rows if r["id"][:2] != "P2"]
        )
        with self.assertRaises(ValueError):
            self.build(self.spec(teachers=[self.ref("teacher.jsonl")]))
        bad = [teacher(r) for r in self.rows]
        for t in bad:
            t["input_sha256"] = "0" * 64
        self.write("teacher.jsonl", bad)
        with self.assertRaises(ValueError):
            self.build(self.spec(teachers=[self.ref("teacher.jsonl")]))

    def test_budget_over_available_fails(self):
        with self.assertRaises(ValueError):
            self.build(self.spec(recipe_budget_tokens=self.total))


if __name__ == "__main__":
    unittest.main()
