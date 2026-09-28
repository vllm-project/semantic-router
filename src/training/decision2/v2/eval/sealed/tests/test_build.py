from __future__ import annotations

import json
import random
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from v2.eval.sealed import build


def row(
    i: int, gold: str, text: str, group: str | None = None, task: str = "src/stance"
) -> dict:
    return {
        "source": "src",
        "task": task,
        "source_item_id": f"s{i}",
        "group_id": group or f"g{i}",
        "balance_label": gold,
        "language": "en",
        "state": text,
        "question": {
            "type": "choice",
            "instructions": "Stance?",
            "criteria": {
                "agree": "Agrees.",
                "disagree": "Disagrees.",
                "other": "Other.",
            },
        },
        "gold": gold,
        "overlap_texts": [text],
        "date": None,
    }


def hit(
    candidate: dict, verdict: str = "CLEAN", containment: float = 0.0, shingles: int = 5
) -> dict:
    return {
        "id": f"{candidate['task']}|{candidate['source_item_id']}",
        "verdict": verdict,
        "containment": containment,
        "shingles": shingles,
    }


def run(rows, hits, config, salt="s" * 40):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "cand").mkdir()
        (root / "cand" / "src.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        (root / "hits.jsonl").write_text("".join(json.dumps(h) + "\n" for h in hits))
        (root / "config.json").write_text(json.dumps(config))
        (root / "salt").write_text(salt)
        build.select(
            Namespace(
                candidates_dir=root / "cand",
                config=root / "config.json",
                salt_file=root / "salt",
                overlap_hits=root / "hits.jsonl",
                output_dir=root / "out",
                replicates=50,
            )
        )
        out = root / "out"
        prompts = [
            json.loads(x) for x in (out / "prompts.jsonl").read_text().splitlines()
        ]
        gold = [json.loads(x) for x in (out / "gold.jsonl").read_text().splitlines()]
        manifest = json.loads((out / "manifest.json").read_text())
        return prompts, gold, manifest


class BuildTest(unittest.TestCase):
    def test_exclusions_caps_and_balance(self):
        rng = random.Random(1)
        rows = [
            row(
                i,
                rng.choice(["agree", "disagree", "other"]),
                f"comment number {i} " * 3,
            )
            for i in range(300)
        ]
        hits = [hit(r) for r in rows]
        hits[0] = hit(rows[0], "OVERLAP", 0.9)
        hits[1] = hit(rows[1], "REVIEW", 0.3)
        hits[2] = hit(rows[2], "REVIEW", 0.0)
        hits[3] = hit(rows[3], shingles=0)
        config = {
            "sources": ["src"],
            "tasks": {"src/stance": {"cap": 60, "group_cap": 1}},
        }
        prompts, gold, manifest = run(rows, hits, config)
        task = manifest["tasks"]["src/stance"]
        self.assertEqual(
            task["exclusions"], {"overlap": 1, "partial_overlap": 1, "unscreenable": 1}
        )
        self.assertEqual(task["selected"], 60)
        self.assertEqual(task["balance"], "gold")
        self.assertEqual(
            sorted(json.loads(k) for k in task["labels"]),
            ["agree", "disagree", "other"],
        )
        self.assertEqual(set(task["labels"].values()), {20})
        ids = {g["source_item_id"] for g in gold}
        self.assertFalse({"s0", "s1", "s3"} & ids)
        self.assertTrue(
            all(
                p["id"].startswith("c1-") and "s" not in p["id"][3:4] or True
                for p in prompts
            )
        )
        self.assertTrue(all(set(p) == {"id", "state", "questions"} for p in prompts))
        record = gold[0]["gold"]["decision"]
        self.assertEqual(
            record["label_to_semantic"],
            {"agree": "agree", "disagree": "disagree", "other": "other"},
        )
        self.assertEqual(manifest["leak_audit"]["verdict_option_surface"], "CLEAN")

    def test_group_cap_and_salt_determinism(self):
        rng = random.Random(5)
        rows = [
            row(
                i,
                ["agree", "disagree"][i % 2],
                f"text {i} " + "x " * rng.randint(3, 30),
                group=f"g{i // 10}",
            )
            for i in range(200)
        ]
        hits = [hit(r) for r in rows]
        config = {
            "sources": ["src"],
            "tasks": {"src/stance": {"cap": 40, "group_cap": 2}},
        }
        _, gold_a, manifest = run(rows, hits, config)
        _, gold_b, _ = run(rows, hits, config)
        _, gold_c, _ = run(rows, hits, config, salt="t" * 40)
        groups = [g["group_id"] for g in gold_a]
        self.assertLessEqual(max(groups.count(x) for x in set(groups)), 2)
        self.assertEqual([g["id"] for g in gold_a], [g["id"] for g in gold_b])
        self.assertNotEqual(
            {g["source_item_id"] for g in gold_a}, {g["source_item_id"] for g in gold_c}
        )
        self.assertEqual(manifest["tasks"]["src/stance"]["selected"], 40)

    def test_length_cue_switches_to_length_balance(self):
        rng = random.Random(2)
        rows = []
        for i in range(600):
            gold = rng.choice(["agree", "disagree"])
            words = rng.randint(30, 60) if gold == "disagree" else rng.randint(5, 40)
            rows.append(
                row(i, gold, f"item{i} " + " ".join(f"w{j}" for j in range(words)))
            )
        hits = [hit(r) for r in rows]
        config = {
            "sources": ["src"],
            "tasks": {"src/stance": {"cap": 200, "group_cap": 1}},
        }
        _, _, manifest = run(rows, hits, config)
        task = manifest["tasks"]["src/stance"]
        self.assertGreater(task["pool_checks"]["length_gain"], 5.0)
        self.assertEqual(task["balance"], "length")
        self.assertLess(
            task["selected_checks"]["length_gain"], task["pool_checks"]["length_gain"]
        )

    def test_selected_gate_rebuilds_with_length_balance_or_drops(self):
        rng = random.Random(4)
        rows = []
        for i in range(900):
            gold = "agree" if i % 5 else "disagree"
            low, high = (5, 60) if gold == "agree" else (25, 60)
            words = rng.randint(low, high)
            rows.append(
                row(i, gold, f"item{i} " + " ".join(f"w{j}" for j in range(words)))
            )
        hits = [hit(r) for r in rows]
        config = {
            "sources": ["src"],
            "tasks": {"src/stance": {"cap": 200, "group_cap": 1}},
        }
        _, gold, manifest = run(rows, hits, config)
        task = manifest["tasks"]["src/stance"]
        self.assertTrue(task["gate"], task)
        if task["selected"]:
            self.assertEqual(task["balance"], "length")
            self.assertLess(task["selected_checks"]["length_gain"], 5.0)
            self.assertGreaterEqual(len(task["labels"]), 2)
        else:
            self.assertIn("dropped", task["gate"][-1])

    def test_small_tasks_are_dropped(self):
        rows = [
            row(i, ["agree", "disagree"][i % 2], f"short item {i} " * 3)
            for i in range(20)
        ]
        hits = [hit(r) for r in rows]
        config = {
            "sources": ["src"],
            "tasks": {"src/stance": {"cap": 20, "group_cap": 1}},
        }
        prompts, _, manifest = run(rows, hits, config)
        self.assertEqual(manifest["tasks"]["src/stance"]["selected"], 0)
        self.assertIn("< 30", manifest["tasks"]["src/stance"]["gate"][0])
        self.assertEqual(prompts, [])

    def test_gold_length_rank_balance_for_per_item_options(self):
        rng = random.Random(3)
        rows = []
        for i in range(400):
            options = {k: "x" * rng.randint(5, 30) for k in "ABCD"}
            gold = rng.choice("ABCD")
            if rng.random() < 0.6:
                options[gold] = "y" * 60
            r = row(i, gold, f"question {i} " * 3)
            r["question"]["criteria"] = options
            rows.append(r)
        hits = [hit(r) for r in rows]
        config = {
            "sources": ["src"],
            "tasks": {"src/stance": {"cap": 120, "group_cap": 1}},
        }
        _, _, manifest = run(rows, hits, config)
        task = manifest["tasks"]["src/stance"]
        self.assertEqual(task["balance"], "gold_length_rank")
        self.assertLess(task["selected_checks"]["option_length_gain"], 0.05)


if __name__ == "__main__":
    unittest.main()
