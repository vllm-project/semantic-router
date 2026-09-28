from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from v2.eval.htdev import build
from v2.eval.htdev.tests.fixtures import WRITERS
from v2.eval.sealed.overlap import load_protected

NOUL = {
    "type": "noul",
    "instructions": "Is it so?",
    "criteria": {"true": "Yes: so.", "false": "No: not so."},
}
PAIR = {
    "type": "choice",
    "instructions": "Which one?",
    "criteria": {"A": "Option A.", "B": "Option B."},
}


def noul_rows(task: str, source: str, count: int, split: str = "test", share=0.5):
    rows = []
    for i in range(count):
        rows.append(
            {
                "id": f"{task}|{i}",
                "task": task,
                "source": source,
                "split": split,
                "source_item_id": str(i),
                "group_id": f"g{i // 2}",
                "language": "en",
                "balance_label": json.dumps(i < count * share),
                "state": {"text": f"item {i} " + "x" * (i % 7)},
                "question": NOUL,
                "gold": i < count * share,
                "overlap_texts": [],
                "option_texts": None,
                "provenance": {"file": "f", "row": i},
            }
        )
    return rows


def pair_rows(task: str, source: str, count: int, longer_wins: bool, always=False):
    rows = []
    for i in range(count):
        short, long = f"s{i}", f"a much longer option text {i}"
        gold = "AB"[i % 2]
        wins = longer_wins and (always or i % 4 != 0)
        texts = [long, short] if (gold == "A") == wins else [short, long]
        rows.append(
            {
                "id": f"{task}|{i}",
                "task": task,
                "source": source,
                "split": "test",
                "source_item_id": str(i),
                "group_id": f"g{i}",
                "language": "en",
                "balance_label": json.dumps(gold),
                "state": {"option_a": texts[0], "option_b": texts[1]},
                "question": PAIR,
                "gold": gold,
                "overlap_texts": texts,
                "option_texts": texts,
                "provenance": {"file": "f", "row": i},
            }
        )
    return rows


def config(slots, **extra):
    return {
        "schema": "test",
        "seed_prefix": "ht-dev/1",
        "cap": 20,
        "floor": 10,
        "min_tasks": 2,
        "split_preference": ["test", "validation", "train"],
        "choice_length_margin": 0.10,
        "slots": slots,
        **extra,
    }


def slot(kind, task, source, backups=(), group_cap=1):
    return {
        "kind": kind,
        "primary": {"task": task, "source": source, "group_cap": group_cap},
        "backups": [
            {"task": t, "source": s, "group_cap": group_cap} for t, s in backups
        ],
    }


class SelectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name: str, value) -> Path:
        path = self.dir / name
        if isinstance(value, list):
            path.write_text("".join(json.dumps(r) + "\n" for r in value))
        else:
            path.write_text(value if isinstance(value, str) else json.dumps(value))
        return path

    def run_select(self, pool, cfg, admitted, exclusions="", out="out", decisions=None):
        admission = admitted if isinstance(admitted, dict) else {"admitted": admitted}
        args = [
            "select",
            "--pool",
            str(self.write("pool.jsonl", pool)),
            "--admission",
            str(self.write("ADMISSION.json", admission)),
            "--exclusions",
            str(self.write("exclusions.txt", exclusions)),
            "--exclusions",
            str(self.write("exclusions2.txt", "")),
            "--output-dir",
            str(self.dir / out),
            "--config",
            str(self.write("config.json", cfg)),
        ]
        if decisions is not None:
            args += ["--decisions", str(self.write("decisions.json", decisions))]
        return build.main(args)

    def manifest(self, out="out"):
        return json.loads((self.dir / out / "MANIFEST.json").read_text())

    def test_refuses_without_admission_file(self):
        pool = self.write("pool.jsonl", noul_rows("a/x", "a", 30))
        with self.assertRaises(SystemExit):
            build.main(
                [
                    "select",
                    "--pool",
                    str(pool),
                    "--admission",
                    str(self.dir / "missing.json"),
                    "--exclusions",
                    str(self.write("ex.txt", "")),
                    "--output-dir",
                    str(self.dir / "out"),
                ]
            )
        self.assertFalse((self.dir / "out").exists())

    def test_balanced_deterministic_and_write_new(self):
        pool = noul_rows("a/x", "a", 60, share=0.3) + noul_rows("b/y", "b", 60)
        cfg = config(
            [slot("a", "a/x", "a", group_cap=2), slot("b", "b/y", "b", group_cap=2)]
        )
        self.assertEqual(self.run_select(pool, cfg, ["a", "b"], out="one"), 0)
        self.assertEqual(self.run_select(pool, cfg, ["a", "b"], out="two"), 0)
        for name in ("ht-dev.prompts.jsonl", "ht-dev.gold.jsonl"):
            self.assertEqual(
                (self.dir / "one" / name).read_bytes(),
                (self.dir / "two" / name).read_bytes(),
            )
        manifest = self.manifest("one")
        self.assertEqual(
            manifest["tasks"]["a/x"]["selected"]["by_label"], {"false": 10, "true": 10}
        )
        prompts = (self.dir / "one" / "ht-dev.prompts.jsonl").read_text()
        self.assertNotIn('"gold"', prompts)
        self.assertNotIn("a/x", prompts)
        row = json.loads(prompts.splitlines()[0])
        self.assertEqual(set(row), {"id", "state", "questions"})
        with self.assertRaises(FileExistsError):
            self.run_select(pool, cfg, ["a", "b"], out="one")

    def test_order_is_the_prereg_hash(self):
        pool = noul_rows("a/x", "a", 60) + noul_rows("b/y", "b", 60)
        cfg = config(
            [slot("a", "a/x", "a", group_cap=2), slot("b", "b/y", "b", group_cap=2)]
        )
        self.run_select(pool, cfg, ["a", "b"])
        gold = [
            json.loads(l)
            for l in (self.dir / "out" / "ht-dev.gold.jsonl").read_text().splitlines()
        ]
        chosen = {g["source_item_id"] for g in gold if g["task"] == "a/x"}
        ranked = sorted(
            (r for r in pool if r["task"] == "a/x"),
            key=lambda r: hashlib.sha256(
                f"ht-dev/1:a/x:{r['source_item_id']}".encode()
            ).hexdigest(),
        )
        taken, groups, labels = [], Counter(), Counter()
        for r in ranked:
            if groups[r["group_id"]] < 2 and labels[r["gold"]] < 10:
                taken.append(r["source_item_id"])
                groups[r["group_id"]] += 1
                labels[r["gold"]] += 1
        self.assertEqual(chosen, set(taken))

    def test_backup_replaces_a_primary_below_floor_or_not_admitted(self):
        pool = (
            noul_rows("a/x", "a", 8)
            + noul_rows("a/backup", "ab", 40)
            + noul_rows("b/y", "b", 40)
            + noul_rows("b/backup", "bb", 40)
        )
        cfg = config(
            [
                slot("a", "a/x", "a", [("a/backup", "ab")], group_cap=2),
                slot("b", "b/y", "b", [("b/backup", "bb")], group_cap=2),
            ]
        )
        self.assertEqual(self.run_select(pool, cfg, ["a", "ab", "bb"]), 0)
        manifest = self.manifest()
        self.assertEqual(
            [s["task"] for s in manifest["slots"]], ["a/backup", "b/backup"]
        )
        self.assertEqual(manifest["slots"][0]["tried"][0]["result"], "below floor")
        self.assertEqual(manifest["slots"][1]["tried"][0]["result"], "not admitted")
        self.assertEqual(manifest["tasks"]["a/backup"]["role"], "backup1")

    def test_stops_below_min_tasks(self):
        pool = noul_rows("a/x", "a", 40) + noul_rows("b/y", "b", 5)
        cfg = config(
            [slot("a", "a/x", "a", group_cap=2), slot("b", "b/y", "b", group_cap=2)]
        )
        self.assertEqual(self.run_select(pool, cfg, ["a", "b"]), 3)
        self.assertTrue((self.dir / "out" / "STOPPED.json").exists())
        self.assertFalse((self.dir / "out" / "ht-dev.prompts.jsonl").exists())

    def test_train_only_when_needed_for_the_floor(self):
        pool = (
            noul_rows("a/x", "a", 30, "test")
            + [
                dict(r, id=r["id"] + "t", source_item_id=r["source_item_id"] + "t")
                for r in noul_rows("a/x", "a", 30, "train")
            ]
            + noul_rows("b/y", "b", 6, "validation")
            + [
                dict(r, id=r["id"] + "t", source_item_id=r["source_item_id"] + "t")
                for r in noul_rows("b/y", "b", 30, "train")
            ]
        )
        cfg = config(
            [slot("a", "a/x", "a", group_cap=4), slot("b", "b/y", "b", group_cap=4)]
        )
        self.assertEqual(self.run_select(pool, cfg, ["a", "b"]), 0)
        tasks = self.manifest()["tasks"]
        self.assertEqual(tasks["a/x"]["selected"]["by_split"], {"test": 20})
        self.assertEqual(tasks["b/y"]["stage"], "test+validation+train")
        self.assertIn("train", tasks["b/y"]["selected"]["by_split"])

    def test_validation_only_when_test_cannot_fill_the_cap(self):
        def val(rows):
            return [
                dict(r, id=r["id"] + "v", source_item_id=r["source_item_id"] + "v")
                for r in rows
            ]

        pool = (
            noul_rows("a/x", "a", 16, "test")
            + val(noul_rows("a/x", "a", 30, "validation"))
            + noul_rows("b/y", "b", 40, "test")
            + val(noul_rows("b/y", "b", 30, "validation"))
        )
        cfg = config(
            [slot("a", "a/x", "a", group_cap=4), slot("b", "b/y", "b", group_cap=4)]
        )
        self.assertEqual(self.run_select(pool, cfg, ["a", "b"]), 0)
        tasks = self.manifest()["tasks"]
        self.assertEqual(tasks["a/x"]["stage"], "test+validation")
        self.assertEqual(tasks["a/x"]["selected"]["items"], 20)
        self.assertEqual(tasks["b/y"]["stage"], "test")
        self.assertEqual(tasks["b/y"]["selected"]["by_split"], {"test": 20})

    def test_isolation_admission_format_and_decisions(self):
        pool = (
            noul_rows("a/x", "a", 40)
            + noul_rows("a/backup", "ab", 40)
            + noul_rows("b/y", "b_b", 40)
        )
        cfg = config(
            [
                slot("a", "a/x", "a", [("a/backup", "ab")], group_cap=2),
                slot("b", "b/y", "b_b", group_cap=2),
            ],
            admission_keys={"bb": "b_b"},
        )
        admission = {
            "sources": {
                "a": {"admitted": True},
                "ab": {"admitted": True},
                "bb": {"admitted": True},
                "zz": {"admitted": False},
            }
        }
        decisions = {"not_admitted": {"a": "lineage"}}
        self.assertEqual(self.run_select(pool, cfg, admission, decisions=decisions), 0)
        manifest = self.manifest()
        self.assertEqual([s["task"] for s in manifest["slots"]], ["a/backup", "b/y"])
        self.assertEqual(manifest["slots"][0]["tried"][0]["reason"], "lineage")
        self.assertEqual(manifest["admitted_sources"], ["ab", "b_b"])

    def test_length_gate_failure_after_redraw_excludes_the_task(self):
        pool = (
            pair_rows("a/pair", "a", 80, longer_wins=True, always=True)
            + noul_rows("a/backup", "ab", 40)
            + noul_rows("b/y", "b", 40)
        )
        cfg = config(
            [
                slot("a", "a/pair", "a", [("a/backup", "ab")]),
                slot("b", "b/y", "b", group_cap=2),
            ]
        )
        self.assertEqual(self.run_select(pool, cfg, ["a", "ab", "b"]), 0)
        manifest = self.manifest()
        self.assertEqual(manifest["slots"][0]["task"], "a/backup")
        self.assertEqual(
            manifest["slots"][0]["tried"][0]["result"],
            "length gate fails after re-draw",
        )
        gate = manifest["tasks"]["a/pair"]["length_gate"]
        self.assertEqual(len(gate), 1)
        self.assertFalse(gate[0]["passes_after"])

    def test_gold_rows_carry_the_cluster(self):
        rows = noul_rows("a/x", "a", 40)
        for r in rows:
            r["cluster_id"] = "c" + r["group_id"][1:2]
        pool = rows + noul_rows("b/y", "b", 40)
        cfg = config(
            [slot("a", "a/x", "a", group_cap=2), slot("b", "b/y", "b", group_cap=2)]
        )
        self.run_select(pool, cfg, ["a", "b"])
        gold = build.read_jsonl(self.dir / "out" / "ht-dev.gold.jsonl")
        self.assertTrue(
            all(g["cluster_id"].startswith("c") for g in gold if g["task"] == "a/x")
        )
        self.assertTrue(
            all(g["cluster_id"] == g["group_id"] for g in gold if g["task"] == "b/y")
        )
        prompts = (self.dir / "out" / "ht-dev.prompts.jsonl").read_text()
        self.assertNotIn("cluster", prompts)

    def test_exclusions_drop_items(self):
        pool = noul_rows("a/x", "a", 30) + noul_rows("b/y", "b", 30)
        cfg = config(
            [slot("a", "a/x", "a", group_cap=2), slot("b", "b/y", "b", group_cap=2)]
        )
        excluded = "\n".join(["a/x|0", json.dumps({"id": "a/x|1"})])
        self.run_select(pool, cfg, ["a", "b"], exclusions=excluded)
        tasks = self.manifest()["tasks"]
        self.assertEqual(tasks["a/x"]["exclusions"], {"excluded": 2})
        gold = build.read_jsonl(self.dir / "out" / "ht-dev.gold.jsonl")
        chosen = {g["source_item_id"] for g in gold if g["task"] == "a/x"}
        self.assertFalse(chosen & {"0", "1"})

    def test_length_cue_is_balanced_away(self):
        pool = pair_rows("a/pair", "a", 80, longer_wins=True) + noul_rows(
            "b/y", "b", 40
        )
        cfg = config([slot("a", "a/pair", "a"), slot("b", "b/y", "b", group_cap=2)])
        self.assertEqual(self.run_select(pool, cfg, ["a", "b"]), 0)
        task = self.manifest()["tasks"]["a/pair"]
        self.assertEqual(task["balance"], "gold_length_rank")
        self.assertLessEqual(task["length_only"]["macro_f1"], 0.6)
        self.assertGreater(task["pool_checks"]["option_length_gain"], 0.2)


class PoolAndExportTest(unittest.TestCase):
    def test_pool_checks_pins_and_export_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            WRITERS["claim_stance"](tmp / "sources" / "claim_stance", False)
            root = tmp / "sources" / "claim_stance"
            files = {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*")
                if p.is_file()
            }
            pins = {
                "sources": {
                    "claim_stance": {
                        "kind": "hf",
                        "repo": "r",
                        "revision": "x",
                        "licence": "l",
                        "licence_evidence": "e",
                        "files": files,
                    }
                }
            }
            (tmp / "pins.json").write_text(json.dumps(pins))
            args = [
                "pool",
                "--sources-dir",
                str(tmp / "sources"),
                "--pins",
                str(tmp / "pins.json"),
            ]
            self.assertEqual(build.main(args + ["--output-dir", str(tmp / "pool")]), 0)
            rows = build.read_jsonl(tmp / "pool" / "pool.jsonl")
            self.assertEqual(len(rows), 6)
            self.assertEqual(
                rows[0]["id"], f"{rows[0]['task']}|{rows[0]['source_item_id']}"
            )
            build.main(
                [
                    "export-scan",
                    "--pool",
                    str(tmp / "pool" / "pool.jsonl"),
                    "--output-dir",
                    str(tmp / "scan"),
                ]
            )
            protected = load_protected(tmp / "scan" / "overlap-protected.jsonl")
            self.assertEqual(protected.ids, [r["id"] for r in rows])
            embed = build.read_jsonl(
                tmp / "scan" / "embed" / "stance__claim_stance.jsonl"
            )
            self.assertEqual(set(embed[0]), {"id", "group_id", "state"})
            self.assertTrue(all(e["group_id"] == e["id"] for e in embed))
            inventory = json.loads((tmp / "scan" / "embed-inventory.json").read_text())
            self.assertEqual(inventory[0]["role"], "ht-dev:stance/claim_stance")
            (root / "test.csv").write_text("tampered")
            with self.assertRaises(ValueError):
                build.main(args + ["--output-dir", str(tmp / "pool2")])


class DisplayOrderTest(unittest.TestCase):
    def test_written_criteria_keep_the_display_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            WRITERS["circa"](tmp / "sources" / "circa", False)
            root = tmp / "sources" / "circa"
            files = {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*")
                if p.is_file()
            }
            pins = {
                "sources": {
                    "circa": {
                        "kind": "hf",
                        "repo": "r",
                        "revision": "x",
                        "licence": "l",
                        "licence_evidence": "e",
                        "files": files,
                    }
                }
            }
            (tmp / "pins.json").write_text(json.dumps(pins))
            build.main(
                [
                    "pool",
                    "--sources-dir",
                    str(tmp / "sources"),
                    "--pins",
                    str(tmp / "pins.json"),
                    "--output-dir",
                    str(tmp / "pool"),
                ]
            )
            from v2.eval.htdev.sources import circa

            by_id = {c.source_item_id: c for c in circa.candidates(root)}
            orders = set()
            for row in build.read_jsonl(tmp / "pool" / "pool.jsonl"):
                keys = list(row["question"]["criteria"])
                self.assertEqual(
                    keys, list(by_id[row["source_item_id"]].question["criteria"])
                )
                orders.add(tuple(keys))
            self.assertGreater(len(orders), 1)


class FlaggedAndScanDropsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        rows = noul_rows("a/x", "a", 6)
        rows[0]["state"] = {
            "text": "A flagged sentence that is long enough.",
            "t": "shared context string here",
        }
        rows[1]["state"] = {
            "text": "Another different sentence, long enough.",
            "t": "shared context string here",
        }
        rows[2]["option_texts"] = ["An option text that was flagged at source."]
        self.pool = self.dir / "pool.jsonl"
        self.pool.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def tearDown(self):
        self.tmp.cleanup()

    def admission(self):
        def leaves(*texts):
            return [build.leaf_digest(t) for t in texts]

        data = {
            "sources": {
                "aa": {
                    "admitted": True,
                    "flagged_rows": [
                        {
                            "file": "f",
                            "row": 0,
                            "leaf_sha256": leaves(
                                "a FLAGGED   sentence that is long enough.",
                                "shared context string here",
                            ),
                        },
                        {
                            "file": "f",
                            "row": 2,
                            "leaf_sha256": leaves(
                                "An option text that was flagged at source.",
                                "a leaf the builder never shows anywhere",
                            ),
                        },
                    ],
                }
            }
        }
        path = self.dir / "ADMISSION.json"
        path.write_text(json.dumps(data))
        cfg = self.dir / "config.json"
        cfg.write_text(json.dumps({"admission_keys": {"aa": "a"}}))
        return path, cfg

    def test_flagged_any_and_row_rules(self):
        admission, cfg = self.admission()
        for match, expected in (
            ("any", {"a/x|0", "a/x|1", "a/x|2"}),
            ("row", {"a/x|0"}),
            ("rarest", {"a/x|0", "a/x|2"}),
        ):
            out = self.dir / match
            build.main(
                [
                    "flagged",
                    "--pool",
                    str(self.pool),
                    "--admission",
                    str(admission),
                    "--config",
                    str(cfg),
                    "--match",
                    match,
                    "--output-dir",
                    str(out),
                ]
            )
            ids = {r["id"] for r in build.read_jsonl(out / "flagged-ids.jsonl")}
            self.assertEqual(ids, expected)
            receipt = json.loads((out / "FLAGGED.json").read_text())
            self.assertEqual(receipt["tasks"]["a/x"]["match_any"], 3)
            self.assertEqual(receipt["tasks"]["a/x"]["match_row"], 1)
            self.assertEqual(receipt["tasks"]["a/x"]["match_rarest"], 2)

    def test_scan_drops_collects_lexical_and_embedding(self):
        excluded = self.dir / "flagged.jsonl"
        excluded.write_text(json.dumps({"id": "a/x|5"}) + "\n")
        verdicts = ["CLEAN", "OVERLAP", "REVIEW", "CLEAN", "CLEAN"]
        hits = self.dir / "hits.jsonl"
        hits.write_text(
            "".join(
                json.dumps(
                    {"id": f"a/x|{i}", "verdict": v, "shingles": 0 if i == 3 else 5}
                )
                + "\n"
                for i, v in enumerate(verdicts)
            )
        )
        private = self.dir / "embed.private.json"
        private.write_text(
            json.dumps(
                {
                    "quarantined": [
                        {
                            "row_id": "a/x|4",
                            "protected_role": "css15_goldfree",
                            "cosine": 0.95,
                        }
                    ],
                    "public": {
                        "by_file": {
                            "a__x.jsonl": {
                                "groups": 5,
                                "review_band_groups": 1,
                                "quarantined_groups": 1,
                                "roles_css15_goldfree": 2,
                            }
                        }
                    },
                }
            )
        )
        out = self.dir / "drops"
        build.main(
            [
                "scan-drops",
                "--pool",
                str(self.pool),
                "--exclude",
                str(excluded),
                "--lexical",
                f"train={hits}",
                "--embed",
                f"panels={private}",
                "--output-dir",
                str(out),
            ]
        )
        ids = {r["id"] for r in build.read_jsonl(out / "scan-ids.jsonl")}
        self.assertEqual(ids, {"a/x|1", "a/x|2", "a/x|3", "a/x|4"})
        task = json.loads((out / "SCANS.json").read_text())["tasks"]["a/x"]
        self.assertEqual(task["scanned"], 5)
        self.assertEqual(task["train:OVERLAP"], 1)
        self.assertEqual(task["train:no_shingles"], 1)
        self.assertEqual(task["panels:quarantine"], 1)
        self.assertEqual(task["panels:band"], 1)
        self.assertEqual(task["dropped"], 4)
        hits.write_text(
            json.dumps({"id": "a/x|0", "verdict": "CLEAN", "shingles": 3}) + "\n"
        )
        with self.assertRaises(SystemExit):
            build.main(
                [
                    "scan-drops",
                    "--pool",
                    str(self.pool),
                    "--exclude",
                    str(excluded),
                    "--lexical",
                    f"train={hits}",
                    "--output-dir",
                    str(self.dir / "d2"),
                ]
            )


class WaterfillTest(unittest.TestCase):
    def test_leftover_goes_to_larger_strata(self):
        self.assertEqual(
            build.waterfill({"a": 3, "b": 50, "c": 50}, 21), {"a": 3, "b": 9, "c": 9}
        )
        self.assertEqual(sum(build.waterfill({"a": 3, "b": 50}, 20).values()), 20)


if __name__ == "__main__":
    unittest.main()
