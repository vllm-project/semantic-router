import collections
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from training.model.data import canonical, validate_row
from v2.data.ib1 import audit, build, index_guard, review
from v2.data.ib1 import families as fam


def counter():
    return collections.Counter()


class FamiliesTest(unittest.TestCase):
    def test_noul_families_are_balanced_and_valid(self):
        rows = fam.sumedit(
            [
                {
                    "domain": d,
                    "id": f"x{i}",
                    "doc": f"Doc {i}.",
                    "summary": f"S {i}.",
                    "label": i % 2,
                }
                for i in range(40)
                for d in ("billsum", "news")
            ],
            counter(),
        )
        self.assertTrue(
            all(r["audit_metadata"]["ib1"]["domain"] == "billsum" for r in rows)
        )
        self.assertEqual(collections.Counter(r["label"] for r in rows), {0: 20, 1: 20})
        sms = fam.sms(
            [{"sms": f"m {i}", "label": int(i < 3)} for i in range(10)], counter()
        )
        self.assertEqual(collections.Counter(r["label"] for r in sms), {0: 3, 1: 3})
        for row in rows + sms:
            validate_row(row, "train")

    def test_expertqa_keeps_complete_and_missing_with_evidence(self):
        claims = [
            {"claim_string": "a", "evidence": ["[1] e"], "support": "Complete"},
            {"claim_string": "b", "evidence": ["[1] e"], "support": "Missing"},
            {"claim_string": "c", "evidence": ["[1] e"], "support": "Partial"},
            {"claim_string": "d", "evidence": [], "support": "Missing"},
        ]
        report = counter()
        rows = fam.expqa(
            [{"question": "q", "answers": {"s": {"claims": claims}}}], report
        )
        self.assertEqual(sorted(r["state"]["claim"] for r in rows), ["a", "b"])
        self.assertEqual(report["drop_support_Partial"], 1)
        self.assertEqual(report["drop_no_evidence"], 1)

    def test_args_twins_and_portals(self):
        lines = []
        for i in range(6):
            for stance in ("PRO", "CON"):
                lines.append(
                    json.dumps(
                        {
                            "id": f"{i}{stance}",
                            "conclusion": f"statement number {i} holds",
                            "premises": [
                                {"text": f"premise {i} {stance}", "stance": stance}
                            ],
                            "context": {
                                "sourceUrl": (
                                    "https://idebate.org/x"
                                    if i
                                    else "https://www.debate.org/x"
                                )
                            },
                        }
                    )
                )
        rows = fam.args(lines, counter())
        groups = collections.Counter(r["group_id"] for r in rows)
        self.assertEqual(len(groups), 5)
        self.assertEqual(set(groups.values()), {2})
        self.assertEqual(collections.Counter(r["label"] for r in rows), {0: 5, 1: 5})

    def test_rotated_snips_and_relevance(self):
        files = {
            k: {k: [{"data": [{"text": f"{k} request {i}"}]} for i in range(40)]}
            for k in fam.SNIPS_FUNCTIONS
        }
        sel, rel = fam.snips(files, counter(), counter())
        for row in sel:
            gold = row["options"][row["label"]]["description"]
            intent = row["audit_metadata"]["ib1"]["intent"]
            self.assertEqual(gold, fam.snips_function(intent))
            validate_row(row, "train")
        positions = collections.Counter(r["label"] for r in sel)
        self.assertEqual(len(positions), 4)
        for row in rel:
            names = {f["name"] for f in json.loads(row["state"]["functions"])}
            gold = fam.SNIPS_FUNCTIONS[row["audit_metadata"]["ib1"]["intent"]]["name"]
            self.assertEqual(row["label"] == 1, gold in names)
        self.assertEqual(
            collections.Counter(r["label"] for r in rel)[0],
            collections.Counter(r["label"] for r in rel)[1],
        )

    def test_medmcqa_explanation_filter(self):
        options = ["Atrophy", "Hyperplasia", "Dysplasia", "Metaplasia"]
        self.assertTrue(
            fam.medmcqa_consistent(options, 0, "leads to atrophy of the kidney")
        )
        self.assertFalse(fam.medmcqa_consistent(options, 0, "atrophy, not hyperplasia"))
        self.assertFalse(fam.medmcqa_consistent(options, 1, "leads to atrophy"))

    def test_wanli_unanimous_only(self):
        records = [
            {"id": i, "premise": f"p{i}", "hypothesis": "h", "gold": "neutral"}
            for i in range(3)
        ]
        notes = [
            {"id": 0, "label": "neutral"},
            {"id": 0, "label": "neutral"},
            {"id": 1, "label": "neutral"},
            {"id": 1, "label": "entailment"},
        ]
        report = counter()
        fam.wanli(records, notes, report)
        self.assertEqual(report["drop_not_unanimous"], 2)
        self.assertEqual(report["eligible"], 1)

    def test_poem_perturbs_the_rhyme_word_only(self):
        stanza = [
            "the morning sun was shining bright",
            "upon the hills of yesterday",
            "and all the birds were singing",
            "along the green and quiet day",
        ]
        other = [
            "a different line of verse here",
            "we wandered through the cold",
            "the stars were cold and far",
            "until the coming of the gold",
        ]
        lines = [(line, 1) for line in stanza] + [(line, 2) for line in other]
        rows = fam.poem(lines, counter())
        self.assertEqual(len(rows), 2)
        for row in rows:
            original = row["state"][
                "stanza_a" if row["label"] == 0 else "stanza_b"
            ].split("\n")
            changed = row["state"][
                "stanza_b" if row["label"] == 0 else "stanza_a"
            ].split("\n")
            self.assertEqual(original[:3], changed[:3])
            self.assertNotEqual(original[3], changed[3])
            self.assertEqual(
                original[3].rsplit(" ", 1)[0], changed[3].rsplit(" ", 1)[0]
            )


class RebalanceTest(unittest.TestCase):
    def test_twins_and_position_band(self):
        rows = fam.args(
            [
                json.dumps(
                    {
                        "id": f"{i}{s}",
                        "conclusion": f"statement number {i} holds",
                        "premises": [{"text": f"p {i} {s}", "stance": s}],
                        "context": {"sourceUrl": "https://idebate.org/x"},
                    }
                )
                for i in range(4)
                for s in ("PRO", "CON")
            ],
            counter(),
        )
        kept = build.twins(rows[1:])
        self.assertEqual(len(kept), 6)
        options = [{"key": "o1", "description": "a"}, {"key": "o2", "description": "b"}]
        skewed = [
            {"id": str(i), "label": int(i < 70), "options": options} for i in range(100)
        ]
        banded = build.position_band(skewed, "s")
        share = sum(r["label"] for r in banded) / len(banded)
        self.assertLessEqual(share, 0.55 + 1e-9)
        six = [{"key": f"o{i}", "description": str(i)} for i in range(6)]
        mixed = skewed + [
            {"id": f"x{i}", "label": i % 6, "options": six} for i in range(120)
        ]
        kept = build.position_band(mixed, "s")
        self.assertEqual(sum(len(r["options"]) == 6 for r in kept), 120)

    def test_round2_constructions(self):
        three = [{"key": f"o{i}", "description": str(i)} for i in range(3)]

        def row(i, family="sentfin", label=None, domain=None):
            return {
                "id": f"{family}{i}",
                "family": family,
                "label": i % 3 if label is None else label,
                "options": three,
                "audit_metadata": {"ib1": {"domain": domain}},
            }

        neutral = build.CONSTRUCTIONS["sentfin-neutral"][2]
        rows = [row(i) for i in range(30)]
        self.assertEqual(
            build.construction(rows[neutral], ["sentfin-neutral"]), "sentfin-neutral"
        )
        self.assertIsNone(build.construction(rows[neutral], ["wands-partial"]))
        shakespeare = row(0, "sumedit", 0, "shakespeare")
        self.assertEqual(
            build.construction(shakespeare, sorted(build.CONSTRUCTIONS)),
            "sumedit-shakespeare",
        )
        kept = [r for r in rows if r["label"] != neutral]
        self.assertEqual(build.equal_labels(kept, "s"), [])
        balanced = build.rebalance(kept, ["sentfin-neutral"])
        self.assertEqual(len(balanced), 20)
        self.assertNotIn(neutral, {r["label"] for r in balanced})
        result, fails = audit.balance(balanced, ["sentfin-neutral"])
        self.assertEqual(fails, [])
        result, fails = audit.balance(
            balanced + rows[neutral : neutral + 1], ["sentfin-neutral"]
        )
        self.assertEqual(fails, ["sentfin"])
        self.assertEqual(
            result["sentfin"]["dropped_constructions_present"], ["sentfin-neutral"]
        )


class IndexGuardTest(unittest.TestCase):
    def test_rules_and_controls(self):
        long_text = "The quick brown fox jumps over the lazy dog while the farmer watches from the porch and smiles."
        suite = [
            {
                "id": "b:1",
                "benchmark": "Bench",
                "state": {"passage": long_text},
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Pick one option now please.",
                        "criteria": {"A": "x"},
                    }
                },
            },
            {"id": "b:2", "benchmark": "Other", "state": "short text", "questions": {}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                for row in suite:
                    stream.write(json.dumps(row) + "\n")
            hashes, codes, names, stats = index_guard.build_reference([path], 2)
            ref = Path(tmp) / "ref"
            index_guard.save_reference(ref, hashes, codes, names, stats)
            self.assertEqual(stats["rows_with_a_13gram_leaf"], 1)
            loaded, loaded_codes, meta = index_guard.load_reference(ref)
            index_guard._WORK.update(hashes=loaded, codes=loaded_codes)
            self.assertIn("E", index_guard.hits([long_text]))
            perturbed = "NOTE: " + long_text.upper().replace(" ", "   ")
            found = index_guard.hits([perturbed])
            self.assertIn("G", found)
            self.assertNotIn("E", found)
            self.assertEqual(
                index_guard.hits(
                    ["An unrelated sentence about pears and apples in the market."]
                ),
                {},
            )
            self.assertEqual(meta["benchmarks"][found["G"][0]], "Bench")
            result = index_guard.controls([path], ref)
            self.assertEqual(result["controls"], 1)
            self.assertEqual(result["exact_copies_flagged"], 1)
            self.assertEqual(result["perturbed_copies_flagged"], 1)

    def test_templates_are_reported_apart(self):
        row = fam.sms(
            [{"sms": "hello there", "label": 1}, {"sms": "buy now", "label": 0}],
            counter(),
        )[0]
        data, template = index_guard.candidate_leaves(row)
        self.assertEqual(data, [row["state"]["message"]])
        self.assertIn(fam.SMS_INSTRUCTIONS, template)


class ReviewTest(unittest.TestCase):
    def rows(self, families=("a", "b"), per=40):
        out = []
        for family in families:
            for i in range(per):
                row = fam.noul(
                    yes=bool(i % 2),
                    source="src",
                    family=family,
                    language="en",
                    group_key=f"{family}{i}",
                    key=f"{family}{i}",
                    state={"text": f"{family} text {i}"},
                    instructions="Q?",
                    template="t",
                )
                out.append(row)
        return out

    def test_screen_and_review_samples_are_disjoint(self):
        rows = self.rows()
        screen = review.screen_sample(rows, "x", set())
        self.assertEqual(screen["sample"]["sampled"], {"a": 10, "b": 10})
        sample = review.review_sample(rows, "x", screen["key"], set())
        self.assertEqual(sample["sample"]["per_family"], 105)
        screened = {k["id"] for k in screen["key"]}
        self.assertFalse(screened & {k["id"] for k in sample["key"]})
        text = "".join(
            canonical(item) for packet in sample["packets_r1"] for item in packet
        )
        self.assertNotIn(rows[0]["source"] + '"', text)

    def test_round2_sample_is_fresh(self):
        rows = self.rows(per=300)
        first = review.review_sample(rows, "x", [], set())
        second = review.review_sample(rows, "x", first["key"], set(), 2)
        self.assertEqual(second["sample"]["round"], 2)
        self.assertEqual(second["sample"]["per_family"], 108)
        self.assertTrue(all(k["rid"].startswith("t") for k in second["key"]))
        self.assertFalse(
            {k["group_id"] for k in first["key"]}
            & {k["group_id"] for k in second["key"]}
        )
        short = self.rows(("a",), 300) + self.rows(("b",), 10)
        topped = review.review_sample(short, "x", [], set(), 2)
        self.assertEqual(topped["sample"]["n"], 216)
        self.assertEqual(topped["sample"]["sampled"], {"a": 206, "b": 10})

    def test_screen_drop_rule_and_verdict(self):
        rows = self.rows(("a",), 20)
        screen = review.screen_sample(rows, "x", set())
        answers = {
            k["rid"]: {"rid": k["rid"], "answer": "false" if i < 3 else k["gold"]}
            for i, k in enumerate(screen["key"])
        }
        public, private = review.screen_score(screen["key"], answers)
        wrong = sum(1 for k in screen["key"][:3] if k["gold"] != "false")
        self.assertEqual(public["disagreements_by_family"]["a"], wrong)
        scored = [{"family": "a", "error": i < 4} for i in range(14)]
        self.assertEqual(review.verdict(scored, {"a": 100})["failing_families"], ["a"])
        scored = [{"family": "a", "error": i < 3} for i in range(14)] + [
            {"family": "b", "error": False} for _ in range(200)
        ]
        result = review.verdict(scored, {"a": 100, "b": 100})
        self.assertEqual(result["failing_families"], [])
        self.assertTrue(result["P1"])


if __name__ == "__main__":
    unittest.main()
