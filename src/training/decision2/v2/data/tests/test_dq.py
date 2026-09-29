import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from v2.data.dq import blind_review as br
from v2.data.dq import embed_manifest

NOUL = [{"key": "false", "description": "No"}, {"key": "true", "description": "Yes"}]


def pn1_row(ident, language, family, label, group=None, p=None):
    return {
        "id": ident,
        "group_id": group or f"g:{ident}",
        "language": language,
        "family": family,
        "label": label,
        "options": NOUL,
        "task_type": "noul",
        "split": "train",
        "instructions": "Do the two sentences have the same meaning?",
        "state": f"Sentence A: {ident[-5:]} one\nSentence B: {ident[-5:]} two",
        "audit_metadata": {"judge": {"label_p_yes": p}, "edited_position": 1},
    }


def pn1_population(per=12):
    rows = []
    for language in ("ja", "de"):
        for family, label in (
            ("pn-hop", 1),
            ("pn-near", 0),
            ("pn-name", 1),
            ("pn-twin", 0),
        ):
            for k in range(per):
                rows.append(
                    pn1_row(
                        f"m4pn1-{family}-{language}{k:03d}", language, family, label
                    )
                )
    return rows


def answers(items, fn):
    return {
        i["rid"]: {
            "rid": i["rid"],
            "answer": fn(i),
            "confidence": "high",
            "fluency": ["ok", "ok"],
        }
        for i in items
    }


class StatsTest(unittest.TestCase):
    def test_known_intervals(self):
        self.assertAlmostEqual(br.wilson(0, 10)[1], 3.8415 / 13.8415, places=4)
        self.assertAlmostEqual(
            br.clopper_pearson(0, 51)[1], 1 - 0.025 ** (1 / 51), places=6
        )
        self.assertAlmostEqual(
            br.clopper_pearson(51, 51)[0], 0.025 ** (1 / 51), places=6
        )
        lo, hi = br.clopper_pearson(5, 20)
        self.assertAlmostEqual(lo, 0.086571, places=4)
        self.assertAlmostEqual(hi, 0.491046, places=4)

    def test_threshold_arithmetic(self):
        self.assertLessEqual(br.clopper_pearson(9, 224)[1], 0.08)
        self.assertGreater(br.clopper_pearson(10, 224)[1], 0.08)
        self.assertGreater(br.clopper_pearson(4, 16)[0], 0.05)
        self.assertLess(br.clopper_pearson(3, 16)[0], 0.05)

    def test_kappa(self):
        self.assertEqual(br.cohen_kappa(["yes", "no"], ["yes", "no"]), 1.0)
        a = ["yes"] * 10 + ["no"] * 10
        b = ["yes"] * 8 + ["no"] * 2 + ["no"] * 8 + ["yes"] * 2
        self.assertAlmostEqual(br.cohen_kappa(a, b), 0.6)


class Pn1SampleTest(unittest.TestCase):
    def test_cells_groups_and_determinism(self):
        rows = pn1_population()
        rows.append(
            pn1_row(
                "m4pn1-pn-twin-dup", "ja", "pn-twin", 1, group="g:m4pn1-pn-twin-ja000"
            )
        )
        picked, population = br.sample_pn1(rows)
        again, _ = br.sample_pn1(list(reversed(rows)))
        self.assertEqual([r["id"] for r in picked], [r["id"] for r in again])
        cells = {}
        for row in picked:
            cells[br.pn1_cell(row)] = cells.get(br.pn1_cell(row), 0) + 1
        self.assertTrue(all(v <= br.PN1_PER_CELL for v in cells.values()))
        self.assertEqual(len({r["group_id"] for r in picked}), len(picked))
        self.assertEqual(population["ja|natural|yes"], 12)

    def test_packet_is_blind(self):
        built = br.build_pn1(pn1_population(), "0" * 64)
        self.assertEqual(len(built["packet_r1"]), 2 * 2 * 2 * br.PN1_PER_CELL)
        self.assertEqual(
            {i["rid"] for i in built["packet_r1"]},
            {i["rid"] for i in built["packet_r2"]},
        )
        self.assertNotEqual(built["packet_r1"], built["packet_r2"])
        for item in built["packet_r1"]:
            self.assertEqual(sorted(item), sorted(br.PN1_PACKET_FIELDS))
            self.assertEqual(item["options"], ["No", "Yes"])
        with self.assertRaises(ValueError):
            br.assert_blind(
                [{"rid": "p1", "state": "pn-hop leak"}], ("rid", "state"), {"pn-hop"}
            )


class Pn1ScoreTest(unittest.TestCase):
    def setUp(self):
        self.built = br.build_pn1(pn1_population(), "0" * 64)
        self.key = self.built["key"]

    def test_majority_rule_and_r3(self):
        gold = {k["rid"]: k["gold"] for k in self.key}
        flip = lambda v: "no" if v == "yes" else "yes"
        rids = sorted(gold)
        both, split_wrong, split_right = rids[0], rids[1], rids[2]
        r1 = answers(
            self.key,
            lambda i: (
                flip(gold[i["rid"]])
                if i["rid"] in (both, split_wrong, split_right)
                else gold[i["rid"]]
            ),
        )
        r2 = answers(
            self.key,
            lambda i: flip(gold[i["rid"]]) if i["rid"] == both else gold[i["rid"]],
        )
        with self.assertRaises(ValueError):
            br.pn1_items(self.key, r1, r2, None)
        r3 = {
            split_wrong: {"rid": split_wrong, "answer": flip(gold[split_wrong])},
            split_right: {"rid": split_right, "answer": gold[split_right]},
        }
        report, private = br.score_pn1(self.built["sample"], self.key, r1, r2, r3)
        self.assertEqual({e["rid"] for e in private["errors"]}, {both, split_wrong})
        self.assertEqual(report["inter_reviewer"]["splits"], 2)
        self.assertEqual(report["error_unweighted"]["k"], 2)

    def test_verdict_thresholds(self):
        def items(errors, n=224, cells=None):
            out = []
            for k in range(n):
                cell = cells[k] if cells else f"l{k % 14}|natural|yes"
                language, group, _ = cell.split("|")
                out.append(
                    {
                        "error": k < errors,
                        "cell": cell,
                        "language": language,
                        "group": group,
                    }
                )
            return out

        pop = {f"l{k}|natural|yes": 100 for k in range(14)}
        cells = [f"l{k % 14}|natural|yes" for k in range(224)]
        self.assertTrue(br.pn1_verdict(items(9, cells=cells), pop)["P1"])
        self.assertFalse(br.pn1_verdict(items(10, cells=cells), pop)["P1"])
        clustered = ["l0|natural|yes"] * 16 + [
            f"l{1 + k % 13}|natural|yes" for k in range(208)
        ]
        verdict = br.pn1_verdict(items(4, cells=clustered), pop)
        self.assertFalse(verdict["P3"])
        self.assertEqual(verdict["failing_language_group_cells"], ["l0|natural"])


class Pn1FixTest(unittest.TestCase):
    def test_failing_cell_dropped_and_balanced(self):
        rows = pn1_population(per=12)
        built = br.build_pn1(rows, "0" * 64)
        key = built["key"]
        gold = {k["rid"]: k["gold"] for k in key}
        flip = lambda v: "no" if v == "yes" else "yes"
        bad = [k["rid"] for k in key if k["cell"].startswith("de|natural")][:4]
        r = answers(
            key, lambda i: flip(gold[i["rid"]]) if i["rid"] in bad else gold[i["rid"]]
        )
        report, private = br.score_pn1(built["sample"], key, r, r, None)
        self.assertEqual(report["verdict"]["verdict"], "FAIL")
        fixed, receipt = br.fix_pn1(rows, private, built["sample"]["population"])
        self.assertEqual(receipt["verdict"], "FIXED-PASS")
        self.assertEqual(receipt["steps"]["F2_cells"], ["de|natural"])
        self.assertFalse(
            any(
                r["language"] == "de" and r["family"] in ("pn-hop", "pn-near")
                for r in fixed
            )
        )
        for language in ("ja", "de"):
            labels = [br.noul_gold(r) for r in fixed if r["language"] == language]
            self.assertEqual(labels.count("yes"), labels.count("no"))


def hs1_rows():
    rows = []
    for family in ("hs1_quote_check", "hs1_policy_packet"):
        for task, options in (
            (
                "choice",
                [{"key": f"o{i}", "description": f"option {i}"} for i in (1, 2, 3)],
            ),
            ("noul", NOUL),
        ):
            for k in range(20):
                rows.append(
                    {
                        "id": f"hs1-{family}-{task}-{k:03d}",
                        "group_id": f"hs1:{family}:{task}:{k // 2}",
                        "family": family,
                        "task_type": task,
                        "label": k % len(options),
                        "options": options,
                        "split": "train",
                        "instructions": "Decide.",
                        "state": f"world {family} {task} {k}",
                        "render_template": f"hs1/{family}/kind/v1",
                        "audit_metadata": {
                            "kind": "kind",
                            "subtype": "s",
                            "option_refs": {},
                        },
                    }
                )
    return rows


class Hs1Test(unittest.TestCase):
    def test_sample_adjudicate_score(self):
        built = br.build_hs1(hs1_rows(), "0" * 64)
        key = built["key"]
        self.assertEqual(len(key), 4 * 10)
        packet = {i["rid"]: i for items in built["packets"].values() for i in items}
        for item in packet.values():
            self.assertEqual(sorted(item), sorted(br.HS1_PACKET_FIELDS))
        wrong, flagged = key[0]["rid"], key[1]["rid"]
        answers_ = {}
        for row in key:
            other = next(
                o["key"]
                for o in packet[row["rid"]]["options"]
                if o["key"] != row["gold_key"]
            )
            answers_[row["rid"]] = {
                "rid": row["rid"],
                "answer": other if row["rid"] == wrong else row["gold_key"],
                "flags": ["ambiguous"] if row["rid"] == flagged else [],
                "note": "two readings" if row["rid"] == flagged else "",
            }
        items, adj_key = br.adjudication_hs1(key, packet, answers_)
        self.assertEqual({e["rid"] for e in adj_key}, {wrong, flagged})
        for item, entry in zip(items, adj_key):
            if entry["type"] == "disagree":
                self.assertEqual(
                    {item["candidates"]["A"], item["candidates"]["B"]},
                    {entry["A"], entry["B"]},
                )
                self.assertNotIn("gold_is", item)
        verdicts = {
            e["aid"]: {
                "aid": e["aid"],
                "verdict": e["gold_is"] if e["type"] == "disagree" else "defect",
            }
            for e in adj_key
        }
        with self.assertRaises(ValueError):
            br.score_hs1(built["sample"], key, answers_, adj_key, verdicts, {})
        confirmed = {flagged: {"status": "not_defect", "evidence": "state decides"}}
        report, _ = br.score_hs1(
            built["sample"], key, answers_, adj_key, verdicts, confirmed
        )
        self.assertEqual(report["verdict"], "PASS")
        confirmed = {flagged: {"status": "defect", "cause": "template"}}
        report, _ = br.score_hs1(
            built["sample"], key, answers_, adj_key, verdicts, confirmed
        )
        self.assertEqual(report["verdict"], "DEFECTS-FOUND")


class EmbedManifestTest(unittest.TestCase):
    def test_localize(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.jsonl").write_text('{"id": "a", "state": "x"}\n')
            (root / "b-new.jsonl").write_text('{"id": "b", "state": "y"}\n')
            (root / "c.jsonl").write_text('{"id": "c", "state": "z"}\n')
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            entries = [
                {
                    "role": "a",
                    "path": str(root / "a.jsonl"),
                    "sha256": digest(root / "a.jsonl"),
                },
                {
                    "role": "b",
                    "path": "/missing/b.jsonl",
                    "sha256": digest(root / "b-new.jsonl"),
                },
            ]
            out, receipt = embed_manifest.localize(
                entries,
                {"/missing/b.jsonl": str(root / "b-new.jsonl")},
                {"ext_c": str(root / "c.jsonl")},
            )
            self.assertEqual([e["role"] for e in out], ["a", "b", "ext_c"])
            self.assertEqual(receipt["same_path"], 1)
            self.assertEqual(out[1]["path"], str(root / "b-new.jsonl"))
            with self.assertRaises(ValueError):
                embed_manifest.localize(entries, {}, {})
            with self.assertRaises(ValueError):
                embed_manifest.localize(
                    entries, {"/missing/b.jsonl": str(root / "c.jsonl")}, {}
                )


class CliTest(unittest.TestCase):
    def test_sample_cli_refuses_wrong_hash_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = root / "rows.jsonl"
            rows.write_text("".join(json.dumps(r) + "\n" for r in pn1_population()))
            with self.assertRaises(ValueError):
                br.main(
                    [
                        "sample-pn1",
                        "--rows",
                        str(rows),
                        "--sha256",
                        "0" * 64,
                        "--out-dir",
                        str(root / "o"),
                    ]
                )
            digest = hashlib.sha256(rows.read_bytes()).hexdigest()
            args = [
                "sample-pn1",
                "--rows",
                str(rows),
                "--sha256",
                digest,
                "--out-dir",
                str(root / "o"),
            ]
            self.assertEqual(br.main(args), 0)
            self.assertTrue((root / "o" / "pn1.key.jsonl").exists())
            with self.assertRaises(FileExistsError):
                br.main(args)


if __name__ == "__main__":
    unittest.main()
