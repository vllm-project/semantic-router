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


if __name__ == "__main__":
    unittest.main()
