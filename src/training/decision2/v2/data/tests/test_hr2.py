import collections
import json
import tempfile
import unittest
from pathlib import Path

from training.model.data import canonical
from v2.data.dq.blind_review import read_jsonl
from v2.data.hr2 import audit, build, review
from v2.data.hr2 import families as fam

SCREEN = (set(), set())


def turn(text):
    return [{"role": "user", "content": text}]


def pref(prompt, first, second, overall, scores):
    return {
        "domain": "general",
        "language": "english",
        "context": turn(prompt),
        "response1": first,
        "response2": second,
        "overall_preference": overall,
        "individual_preference": [{"score": score} for score in scores],
    }


def feedback(prompt, levels_one, levels_two, first="r1", second="r2"):
    def texts(levels):
        return [f"The response is {level} helpful. More detail." for level in levels]

    return {
        "domain": "general",
        "language": "english",
        "context": turn(prompt),
        "response1": first,
        "response2": second,
        "feedback1": texts(levels_one),
        "feedback2": texts(levels_two),
    }


def step(*completions, chosen=None):
    return {
        "completions": [
            {"text": text, "rating": rating} for text, rating in completions
        ],
        "chosen_completion": chosen,
        "human_completion": None,
    }


def prm(problem, *steps):
    return {
        "question": {"problem": problem},
        "label": {"steps": list(steps), "finish_reason": "found_error"},
    }


class HelpSteer3Test(unittest.TestCase):
    def test_key_ignores_response_order(self):
        one = pref("q", "x", "y", -3, [-3, -2])
        two = pref("q", "y", "x", 3, [3, 2])
        self.assertEqual(fam.hs3_key(one), fam.hs3_key(two))

    def test_pref_keeps_agreeing_repeats_once_and_drops_every_conflicting_copy(self):
        records = [
            pref("q1", "good!", "bad", -3, [-3, -2]),
            pref("q1", "bad", "good!", 2, [2, 3]),
            pref("q2", "p", "q", -2, [-2, -3]),
            pref("q2", "p", "q", 1, [1, 1]),
            pref("q3", "m", "n", -3, [-3, 1]),
            pref("q4", "ok", "longer", -2, [-2, -2]),
        ]
        report = collections.Counter()
        rows = fam.hs3_pref(records, SCREEN, report)
        self.assertEqual(report["drop_exact_duplicates"], 1)
        self.assertEqual(report["drop_conflicting_duplicates"], 1)
        self.assertEqual(report["drop_tie_or_slight"], 1)
        self.assertEqual(report["drop_annotator_split"], 1)
        self.assertEqual(len(rows), 2)
        for row in rows:
            gold = row["state"]["response_a" if row["label"] == 0 else "response_b"]
            self.assertIn(gold, ("good!", "ok"))

    def test_help_requires_three_parsed_close_levels_agreeing_across_copies(self):
        self.assertEqual(fam.help_levels(["The response is mostly helpful."]), [3])
        self.assertIsNone(fam.help_levels(["Helpful overall."]))
        records = [
            feedback("a", ["mostly", "mostly", "perfectly"], ["mostly"] * 3),
            feedback(
                "b", ["not", "mostly", "perfectly"], ["not", "mostly", "perfectly"]
            ),
            feedback("c", ["mostly"] * 3, ["mostly"] * 3),
            feedback("c", ["slightly"] * 3, ["slightly"] * 3),
            feedback("d", ["partially"] * 2, ["partially"] * 2),
        ]
        report = collections.Counter()
        fam.hs3_help(records, SCREEN, report)
        self.assertEqual(report["eligible"], 1)
        self.assertEqual(report["drop_disagreement"], 1)
        self.assertEqual(report["drop_conflicting_duplicates"], 2)
        self.assertEqual(report["drop_unparsed_or_not_three"], 1)


class PrmTest(unittest.TestCase):
    def test_walk_records_alternative_ratings(self):
        record = prm("p", step(("a", 1), ("b", -1), chosen=0), step(("c", -1)))
        yes, no, rated = fam.prm_walk(record)
        self.assertEqual(yes, [(0, [], "a")])
        self.assertEqual(no, (1, ["a"], "c"))
        self.assertIn(((0, [], "b"), -1), rated)

    def test_conflicting_ratings_of_one_state_drop_the_row(self):
        records = [
            prm("p1", step(("s1", 1)), step(("s2", -1))),
            prm("p1", step(("s1", 0))),
            prm("p2", step(("a", 1), ("b", -1), chosen=0), step(("c", -1))),
            prm("p2", step(("x", 1), ("a", 0), chosen=0)),
        ]
        report = collections.Counter()
        fam.prm_step(records, report)
        self.assertEqual(report["drop_conflicting_ratings"], 2)
        self.assertEqual(report["eligible"], 3)


class CommonRulesTest(unittest.TestCase):
    def test_resolve_and_dedup(self):
        row = {"id": "x", "input_sha256": "h1", "label": 0, "family": "f"}
        report = collections.Counter()
        kept = fam.resolve([row, dict(row), dict(row, id="y", label=1)], report)
        self.assertEqual(sorted(r["id"] for r in kept), ["x", "y"])
        self.assertEqual(report["drop_exact_duplicates"], 1)
        kept = fam.resolve([row, dict(row, label=1)], collections.Counter())
        self.assertEqual(kept, [])
        rows = [
            row,
            dict(row),
            dict(row, id="w", family="g"),
            dict(row, id="y", input_sha256="h2", label=1),
            dict(row, id="z", input_sha256="h2", family="g"),
        ]
        unique, dup = build.dedup(rows)
        self.assertEqual([r["id"] for r in unique], ["w"])
        self.assertEqual(dup["conflicting"], {"f": 1, "g": 1})

    def test_dev_slice_is_deterministic_and_about_a_tenth(self):
        groups = [f"hr2:src:{i}" for i in range(20000)]
        share = sum(build.is_dev(g) for g in groups) / len(groups)
        self.assertTrue(0.09 < share < 0.11, share)
        self.assertEqual(
            [build.is_dev(g) for g in groups[:50]],
            [build.is_dev(g) for g in groups[:50]],
        )

    def test_balancers(self):
        rows = [
            {"id": str(i), "label": int(i < 70), "audit_metadata": {"hr2": {}}}
            for i in range(100)
        ]
        for row in rows:
            row["audit_metadata"]["hr2"]["hash_key"] = row["id"]
            row["audit_metadata"]["hr2"]["cell"] = str(
                row["label"] + int(row["id"]) % 3
            )
        kept = fam.balance_yes_no(rows, None, "t")
        self.assertEqual(collections.Counter(r["label"] for r in kept), {0: 30, 1: 30})
        capped = fam.cap_share(rows, 100, 0.30, "t")
        cells = collections.Counter(r["audit_metadata"]["hr2"]["cell"] for r in capped)
        self.assertLessEqual(max(cells.values()), 0.30 * len(capped))


def review_row(i, family, task_type, label, levels=2):
    keys = {"choice": ["a", "b"], "noul": ["false", "true"]}.get(
        task_type, [str(k) for k in range(levels)]
    )
    return {
        "id": f"hr2-{family}-{i}",
        "group_id": f"hr2:{family}:{i}",
        "family": family,
        "source": f"{family}_train",
        "language": "en",
        "split": "train",
        "task_type": task_type,
        "label": label,
        "instructions": "Q?",
        "state": {"text": f"item {i}"},
        "options": [{"key": k, "description": k} for k in keys],
    }


class ReviewTest(unittest.TestCase):
    def rows(self):
        rows = [review_row(i, "fam_noul", "noul", i % 2) for i in range(60)]
        rows += [review_row(i, "fam_score", "score", i % 5, 5) for i in range(60)]
        return rows

    def test_sample_is_blind_stratified_and_one_per_group(self):
        built = review.build(self.rows(), "0" * 64)
        key = built["key"]
        self.assertEqual(len(key), 48)
        cells = collections.Counter((k["family"], k["gold"]) for k in key)
        self.assertEqual(cells[("fam_noul", "true")], 12)
        self.assertEqual(
            sorted(v for (f, _), v in cells.items() if f == "fam_score"),
            [4, 5, 5, 5, 5],
        )
        self.assertEqual(len({k["group_id"] for k in key}), 48)
        packet = [item for chunk in built["packets_r1"] for item in chunk]
        self.assertEqual(sorted(packet[0]), sorted(review.FIELDS))
        self.assertEqual(
            sorted(i["rid"] for c in built["packets_r2"] for i in c),
            sorted(i["rid"] for i in packet),
        )

    def test_sample_cli_keeps_line_separators_inside_strings(self):
        rows = self.rows()
        rows[0]["state"]["text"] = "a\u2028b"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(
                "".join(canonical(r) + "\n" for r in rows), encoding="utf-8"
            )
            self.assertEqual(
                review.main(["sample", "--train", str(path), "--out-dir", f"{tmp}/s"]),
                0,
            )
            self.assertEqual(len(read_jsonl(Path(tmp) / "s" / "key.jsonl")), 48)
            packets = "".join(
                (Path(tmp) / "s" / f"packet.r1.{i}.jsonl").read_text(encoding="utf-8")
                for i in (1, 2)
            )
            self.assertNotIn("\u2028", packets)
            self.assertEqual(len(packets.split("\n")) - 1, 48)
            self.assertIn(
                "a\u2028b",
                [json.loads(x)["state"]["text"] for x in packets.split("\n") if x],
            )

    def test_errors_splits_and_score_tolerance(self):
        key = review.build(self.rows(), "0" * 64)["key"]
        r1, r2 = {}, {}
        for row in key:
            gold = row["gold"]
            if row["task_type"] == "score":
                r1[row["rid"]] = {"answer": str((int(gold) + 1) % 5)}
                r2[row["rid"]] = {"answer": gold}
            else:
                other = "false" if gold == "true" else "true"
                r1[row["rid"]] = {"answer": "yes" if gold == "true" else "no"}
                r2[row["rid"]] = {"answer": other}
        splits = review.split_rids(key, r1, r2)
        noul = [k for k in key if k["task_type"] == "noul"]
        wrapped = [k for k in key if k["task_type"] == "score" and k["gold"] == "4"]
        self.assertEqual(len(splits), len(noul) + len(wrapped))
        r3 = {rid: {"answer": r2[rid]["answer"]} for rid in splits}
        sample = {
            "population": {"fam_noul": 60, "fam_score": 60},
            "rows_sha256": "",
            "rows": 120,
            "salt": review.SALT,
            "n": len(key),
        }
        report, private = review.score(sample, key, r1, r2, r3)
        self.assertEqual(report["verdict"]["errors"], len(noul))
        self.assertEqual(report["verdict"]["failing_families"], ["fam_noul"])
        self.assertIsNotNone(report["fix_rule_f1"])
        self.assertEqual(report["fix_rule_f1"]["errors"], 0)
        self.assertEqual(len(private["errors"]), len(noul))


class AuditTest(unittest.TestCase):
    def test_names_pass_on_the_registries(self):
        self.assertEqual(audit.names()["hits"], {})

    def test_quarantine_lists(self):
        rows = [
            {"group_id": "g1", "family": "f", "split": "train"},
            {"group_id": "g2", "family": "f", "split": "select"},
            {"group_id": "g3", "family": "h", "split": "select"},
        ]
        hit = {"methods": ["N"], "roles": ["panel"]}
        drop, dev_drop, public = audit.quarantine(
            rows,
            [{"groups": {"g1": hit}}],
            {"groups": {"g2": dict(hit, roles=["train_role"])}},
            {"panel"},
            {"pairs": [{"a": "g3", "b": "g1"}, {"a": "g1", "b": "g2"}]},
        )
        self.assertEqual(drop, {"g1"})
        self.assertEqual(dev_drop, {"g3"})
        self.assertEqual(public["report_only_roles_hit"], {"train_role": 1})


if __name__ == "__main__":
    unittest.main()
